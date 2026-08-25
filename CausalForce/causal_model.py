"""
Causal GCN Model for CausalForce.

Replaces the original GCN_model with causal inference components:
  - CausalDisentangleModule  (replaces direct ego-object concatenation)
  - CausalMessagePassing     (replaces GCN message_passing)
  - CausalRiskHead           (replaces score_head + risk_type_head)

Backbones (PDResNet50, Riskbench_backbone), LSTM temporal modeling,
and camera_features remain unchanged for weight transfer compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import Riskbench_backbone
from pdresnet50 import pdresnet50
from model import pad_trackers
from causal_modules import (
    CausalDisentangleModule,
    CausalMessagePassing,
    CausalRiskHead,
)


class CausalGCN_model(nn.Module):
    def __init__(self, time_steps=3, pretrained=True, partialConv=True,
                 NUM_BOX=12, num_confounders=64, num_heads=4, cf_alpha=0.3):
        super().__init__()

        self.time_steps = time_steps
        self.pretrained = pretrained
        self.partialConv = partialConv
        self.num_box = NUM_BOX
        self.hidden_size = 512
        self.num_bin = 8

        # ── Backbones (unchanged, weights transfer from pre-trained) ──
        if self.partialConv:
            self.backbone = pdresnet50(pretrained=self.pretrained)

        self.object_backbone = Riskbench_backbone(
            roi_align_kernel=8, n=self.num_box, pretrained=pretrained)

        self.camera_features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # ── Temporal modeling (unchanged) ──
        self.fusion_size = 512
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)

        # ── Phase 1: Causal Disentanglement ──
        self.causal_disentangle = CausalDisentangleModule(
            feat_dim=self.hidden_size, num_confounders=num_confounders
        )

        # ── Phase 2: Causal Message Passing ──
        self.causal_message_passing = CausalMessagePassing(
            hidden_size=self.hidden_size, num_heads=num_heads, cf_alpha=cf_alpha
        )

        # ── Output projection (unchanged dim: 512→128) ──
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 128, bias=False),
            nn.ReLU(inplace=True),
        )

        # ── Phase 3: Causal Risk Head ──
        self.causal_risk_head = CausalRiskHead(
            feat_dim=128, num_bins=self.num_bin, num_risk_types=4
        )

    def step(self, camera_input, hx, cx):
        fusion_input = camera_input
        hx, cx = self.lstm(self.drop(fusion_input), (hx, cx))
        return hx, cx

    def forward(self, camera_inputs, trackers, mask=None, device='cuda'):
        """
        Args:
            camera_inputs: (B, T, C, H, W)  front-view images
            trackers:      list[list[Tensor(n,4)]]  bounding boxes per frame
            mask:          optional occlusion mask
        Returns:
            dict with keys:
                score_H8     (B, N, 8)   risk scores
                risk_type    (B, N, 4)   risk category logits
                hx_seq       (B, T, N, H) LSTM hidden sequence (for HTSC loss)
                obj_emb      (B, N, H)   final object embeddings
                causal_feat  (B, N, H)   causal features (for L_ortho)
                scene_feat   (B, N, H)   scene features  (for L_ortho)
                direct       (B, N, 8)   NDE logits      (for L_cf)
                indirect     (B, N, 8)   TIE logits      (for L_cf)
        """
        batch_size = camera_inputs.shape[0]
        t = camera_inputs.shape[1]
        c = camera_inputs.shape[2]
        h = camera_inputs.shape[3]
        w = camera_inputs.shape[4]

        if mask is None:
            mask = torch.ones((batch_size, t, c, h, w)).to(device)

        # Initialize LSTM states for ego + N objects
        hx = torch.zeros(
            (batch_size * (1 + self.num_box), self.hidden_size)).to(device)
        cx = torch.zeros(
            (batch_size * (1 + self.num_box), self.hidden_size)).to(device)

        # ═══════════════════════════════════════════════════════
        # Backbone feature extraction (unchanged)
        # ═══════════════════════════════════════════════════════
        camera_inputs = camera_inputs.reshape(-1, c, h, w)

        if self.partialConv:
            ego_features = self.backbone.features(
                camera_inputs, mask.reshape(-1, c, h, w))
        else:
            ego_features = self.backbone.features(camera_inputs)

        c = ego_features.shape[1]
        h = ego_features.shape[2]
        w = ego_features.shape[3]
        ego_features = ego_features.reshape(batch_size, t, c, h, w)

        padded_trackers, tracker_counts = pad_trackers(
            trackers, self.num_box, device)
        tracker = padded_trackers.view(-1, padded_trackers.shape[2], 4)
        _, obj_features = self.object_backbone(camera_inputs, tracker)
        obj_features = obj_features.reshape(
            batch_size, t, self.num_box, -1)

        # ═══════════════════════════════════════════════════════
        # LSTM temporal modeling (unchanged)
        # ═══════════════════════════════════════════════════════
        hx_seq = []
        for l in range(self.time_steps):
            ego_feature = ego_features[:, l].clone()
            obj_feature = obj_features[:, l].clone()

            ego_feature = self.camera_features(
                ego_feature).reshape(batch_size, 1, -1)

            feature_input = torch.cat((ego_feature, obj_feature), 1)
            feature_input = feature_input.reshape(-1, self.fusion_size)

            hx, cx = self.step(feature_input, hx, cx)
            hx_step = hx.view(
                batch_size, self.num_box + 1, self.hidden_size)
            hx_seq.append(hx_step[:, 1:, :])  # B×N×H (objects only)

        hx_seq = torch.stack(hx_seq, dim=1)  # (B, T, N, H)

        # ═══════════════════════════════════════════════════════
        # Causal inference pipeline (NEW — replaces GCN + heads)
        # ═══════════════════════════════════════════════════════

        # Reshape LSTM output → separate ego / objects
        hx_reshaped = hx.view(
            batch_size, self.num_box + 1, self.hidden_size)
        ego_hidden = hx_reshaped[:, 0:1, :]   # (B, 1, H)
        obj_hidden = hx_reshaped[:, 1:, :]     # (B, N, H)

        # Phase 1: Causal Feature Disentanglement
        deconfounded, causal_feat, scene_feat = \
            self.causal_disentangle(obj_hidden, ego_hidden)

        # Build validity mask from tracker bounding boxes
        valid_mask = (padded_trackers[:, -1, :, 2]
                      + padded_trackers[:, -1, :, 3]) != 0  # (B, N)
        # MultiheadAttention key_padding_mask: True = ignore
        ego_valid = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=device)
        kp_mask = torch.cat([ego_valid, ~valid_mask], dim=1)  # (B, 1+N)

        # Phase 2: Causal Message Passing
        causal_effect, attn_w = self.causal_message_passing(
            deconfounded, ego_hidden, key_padding_mask=kp_mask)

        # Output projection
        x = self.out_layer(self.drop(causal_effect))

        # Object validity mask for final outputs
        obj_mask = valid_mask.unsqueeze(-1).float()  # (B, N, 1)

        # Phase 3: Causal Risk Head (TE = NDE + TIE)
        score, risk_type, direct, indirect = \
            self.causal_risk_head(x, return_components=True)
        score = score * obj_mask        # (B, N, 8)
        risk_type = risk_type * obj_mask  # (B, N, 4)

        return {
            'score_H8': score,
            'risk_type': risk_type,
            'hx_seq': hx_seq,
            'obj_emb': causal_effect,
            # Causal components for auxiliary losses
            'causal_feat': causal_feat,
            'scene_feat': scene_feat,
            'direct': direct * obj_mask,
            'indirect': indirect * obj_mask,
        }
