"""
Causal Inference Modules for CausalForce.

Phase 1: CausalDisentangleModule - Backdoor adjustment to separate causal/confounding features
Phase 2: CausalMessagePassing - Counterfactual-aware relational reasoning (replaces GCN)
Phase 3: CausalRiskHead - Total Effect decomposition for risk prediction (replaces score_head + risk_type_head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDisentangleModule(nn.Module):
    """Phase 1: Disentangle causal features from confounding scene context.

    Implements backdoor adjustment via a learnable confounder dictionary:
        P(Y|do(X)) = Σ_s P(Y|X,S=s) P(S=s)

    The confounder dictionary approximates P(S) as a discrete set of
    scene prototypes, enabling tractable marginalization.
    """

    def __init__(self, feat_dim=512, num_confounders=64):
        super().__init__()
        self.feat_dim = feat_dim
        # Learnable confounder dictionary: K prototypes of dim D
        self.confounder_dict = nn.Parameter(
            torch.randn(num_confounders, feat_dim) * 0.01
        )
        # Project out causal vs confounding components
        self.proj_causal = nn.Linear(feat_dim, feat_dim)
        self.proj_confound = nn.Linear(feat_dim, feat_dim)
        # Gated fusion with ego context
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, obj_feat, ego_feat):
        """
        Args:
            obj_feat: (B, N, D) object hidden states from LSTM
            ego_feat: (B, 1, D) ego vehicle hidden state
        Returns:
            deconfounded: (B, N, D) deconfounded object features
            causal_feat:  (B, N, D) pure causal component
            scene_feat:   (B, N, D) confounding scene component
        """
        # Soft-attention over confounder dictionary → approximate E[X|S]
        attn = F.softmax(
            obj_feat @ self.confounder_dict.T / (self.feat_dim ** 0.5),
            dim=-1
        )  # (B, N, K)
        confound_feat = attn @ self.confounder_dict  # (B, N, D)

        # Separate: causal = X - E[X|S],  scene = E[X|S]
        causal_feat = self.proj_causal(obj_feat - confound_feat)
        scene_feat = self.proj_confound(confound_feat)

        # Gated fusion controlled by ego context
        ego_expanded = ego_feat.expand_as(causal_feat)
        gate = self.gate(torch.cat([causal_feat, ego_expanded], dim=-1))
        deconfounded = gate * causal_feat + (1 - gate) * scene_feat

        # Residual connection + layer norm for stable training
        deconfounded = self.layer_norm(deconfounded + obj_feat)

        return deconfounded, causal_feat, scene_feat


class CausalMessagePassing(nn.Module):
    """Phase 2: Causal relational reasoning via factual + counterfactual branches.

    Replaces the original ego-object pairwise GCN attention with:
    1. Multi-head causal attention (factual reasoning)
    2. Intervention MLP estimating do(X_i) effects
    3. Counterfactual projection for Natural Indirect Effect (NIE)

    Total Causal Effect = Factual + α * Counterfactual adjustment
    """

    def __init__(self, hidden_size=512, num_heads=4, cf_alpha=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.cf_alpha = cf_alpha

        # Factual: multi-head attention over ego+objects
        self.causal_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.1
        )
        # Intervention module: estimates do(X_i) effect
        self.intervention_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Counterfactual projection
        self.counterfactual_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, node_features, ego_feature, key_padding_mask=None):
        """
        Args:
            node_features:    (B, N, H) deconfounded object features
            ego_feature:      (B, 1, H) ego vehicle feature
            key_padding_mask: (B, 1+N) bool, True = invalid position
        Returns:
            causal_effect: (B, N, H)
            attn_weights:  (B, num_heads, N, 1+N)
        """
        # Build key/value set: [ego, obj_1, ..., obj_N]
        all_nodes = torch.cat([ego_feature, node_features], dim=1)  # (B, 1+N, H)

        # Factual branch: standard relational attention
        factual_out, attn_w = self.causal_attn(
            node_features, all_nodes, all_nodes,
            key_padding_mask=key_padding_mask
        )  # (B, N, H)

        # Counterfactual branch: "what if object i had no causal influence?"
        intervened = self.intervention_mlp(node_features)
        cf_out = self.counterfactual_proj(factual_out - intervened)

        # Total Causal Effect
        causal_effect = factual_out + self.cf_alpha * cf_out
        causal_effect = self.layer_norm(causal_effect + node_features)  # residual

        return causal_effect, attn_w


class CausalRiskHead(nn.Module):
    """Phase 3: Risk prediction via Total Effect decomposition.

    TE = NDE + TIE
      NDE (Natural Direct Effect):   object features → risk score
      TIE (Total Indirect Effect):   object features → risk type (mediator) → risk score

    This breaks the confounded coupling between risk_type and risk_score
    that exists in the original shared-feature architecture.
    """

    def __init__(self, feat_dim=128, num_bins=8, num_risk_types=4):
        super().__init__()
        self.num_bins = num_bins
        # NDE path: direct prediction
        self.direct_head = nn.Linear(feat_dim, num_bins)
        # TIE path: mediated through risk type
        self.type_head = nn.Linear(feat_dim, num_risk_types)
        self.indirect_head = nn.Linear(num_risk_types, num_bins)
        # Learned fusion of direct + indirect
        self.fusion = nn.Sequential(
            nn.Linear(num_bins * 2, num_bins),
            nn.Sigmoid()
        )

    def forward(self, x, return_components=False):
        """
        Args:
            x: (B, N, feat_dim)
        Returns:
            total_score: (B, N, num_bins) calibrated risk scores in [0,1]
            risk_type:   (B, N, num_risk_types) risk category logits
            [if return_components] direct, indirect raw logits
        """
        direct = self.direct_head(x)                              # NDE
        risk_type = self.type_head(x)                              # mediator
        indirect = self.indirect_head(F.softmax(risk_type, -1))    # TIE

        total = self.fusion(torch.cat([direct, indirect], dim=-1))

        if return_components:
            return total, risk_type, direct, indirect
        return total, risk_type


# ──────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────

def orthogonality_loss(causal_feat, scene_feat):
    """Ensure causal and scene features are statistically independent.

    Minimizes squared cosine similarity between the two feature sets
    so that they encode non-overlapping information.
    """
    c = F.normalize(causal_feat, dim=-1)
    s = F.normalize(scene_feat, dim=-1)
    return (c * s).sum(dim=-1).pow(2).mean()


def counterfactual_loss(direct, indirect, gt_risk_score):
    """Regularize both causal paths to independently predict risk.

    Prevents the model from collapsing all information into one path
    (e.g., ignoring the indirect/mediated path entirely).
    """
    direct_pred = torch.sigmoid(direct)
    indirect_pred = torch.sigmoid(indirect)
    with torch.cuda.amp.autocast(enabled=False):
        L_direct = F.binary_cross_entropy(direct_pred.float(), gt_risk_score.float(), reduction='mean')
        L_indirect = F.binary_cross_entropy(indirect_pred.float(), gt_risk_score.float(), reduction='mean')
    return L_direct + 0.5 * L_indirect


def causal_nonconformity(preds, gts, causal_uncertainty=None, method="causal_weighted"):
    """Phase 4: Causal-aware nonconformity scores for conformal calibration.

    When model relies heavily on confounders (high causal_uncertainty),
    nonconformity is inflated → wider prediction sets → safer coverage.
    """
    if method == "absolute" or causal_uncertainty is None:
        return torch.abs(preds - gts)
    elif method == "causal_weighted":
        return torch.abs(preds - gts) * (1.0 + causal_uncertainty)
    elif method == "class_cond":
        return torch.where(gts == 1, 1.0 - preds, preds)
    return torch.abs(preds - gts)
