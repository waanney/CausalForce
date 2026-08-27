"""
CausalForce Training Script.

Changes vs original train.py:
  1. Uses CausalGCN_model instead of GCN_model
  2. Adds L_ortho (orthogonality loss) and L_cf (counterfactual loss)
  3. Transfers risk_type_head weights → causal_risk_head.type_head
  4. Loss ramp-up schedule for causal losses (stable warm-up)
"""

import argparse
import os
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data import MultipleRisksDataset, custom_collate_fn
from causal_model import CausalGCN_model
from torch.utils.data import DataLoader
from online_conformal.saocp import SAOCP
from train import compute_nonconformity, htsc_loss
from causal_modules import orthogonality_loss, counterfactual_loss
from checkpoint_utils import load_partial_checkpoint


class CausalForce(pl.LightningModule):
    def __init__(self, lr, num_confounders=64, num_heads=4, cf_alpha=0.3,
                 w_score=1.0, w_htsc=10.0, w_ortho=0.1, w_cf=0.5,
                 ortho_ramp_epochs=4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.coverage = 0.7
        self.nc_method = "absolute"

        self.model = CausalGCN_model(
            num_confounders=num_confounders,
            num_heads=num_heads,
            cf_alpha=cf_alpha,
        )

        self.class_cps = {
            'OBS': SAOCP(model=None, train_data=None, max_scale=1.0,
                         coverage=self.coverage, horizon=8),
            'OCC': SAOCP(model=None, train_data=None, max_scale=1.0,
                         coverage=self.coverage, horizon=8),
            'I':   SAOCP(model=None, train_data=None, max_scale=1.0,
                         coverage=self.coverage, horizon=8),
            'C':   SAOCP(model=None, train_data=None, max_scale=1.0,
                         coverage=self.coverage, horizon=8),
        }
        self.index_to_class = {0: 'OBS', 1: 'OCC', 2: 'I', 3: 'C'}

        # Loss weights
        self.w_score = w_score
        self.w_htsc = w_htsc
        self.w_ortho = w_ortho
        self.w_cf = w_cf
        self.ortho_ramp_epochs = ortho_ramp_epochs

    def _get_ortho_weight(self):
        """Ramp up orthogonality loss from 0 → w_ortho over first N epochs."""
        if self.ortho_ramp_epochs <= 0:
            return self.w_ortho
        progress = min(self.current_epoch / self.ortho_ramp_epochs, 1.0)
        return self.w_ortho * progress

    def _compute_step_losses(self, batch, prefix=""):
        """Shared logic for training_step and validation_step."""
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_interval_H8 = batch['risk_interval_H8']

        outputs = self.model(front_imgs, all_objs_bbs)

        B, T, C, H, W = front_imgs.shape

        total_loss = torch.tensor(0.0, device=self.device)
        total_loss_score = torch.tensor(0.0, device=self.device)
        total_loss_htsc = torch.tensor(0.0, device=self.device)
        total_loss_ortho = torch.tensor(0.0, device=self.device)
        total_loss_cf = torch.tensor(0.0, device=self.device)

        # ── Per-batch orthogonality loss (global, not per-sample) ──
        loss_ortho = orthogonality_loss(
            outputs['causal_feat'], outputs['scene_feat'])
        total_loss_ortho = loss_ortho

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]
            pred_risk_type = outputs["risk_type"][i]
            pred_direct = outputs["direct"][i]
            pred_indirect = outputs["indirect"][i]
            gt_risk_ids = label_risk_ids[i]
            gt_risk_score_H8 = label_risk_interval_H8[i]
            all_objs_id = all_objs_ids[i][-1]

            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_scoreH8_i = pred_risk_score_H8.sum() * 0.0
                loss_htsc_i = pred_risk_score_H8.sum() * 0.0
                loss_cf_i = pred_risk_score_H8.sum() * 0.0

            elif len(gt_risk_ids) == 0:
                gt_zeros = torch.zeros_like(pred_risk_score_H8[:N])
                with torch.cuda.amp.autocast(enabled=False):
                    loss_scoreH8_i = F.binary_cross_entropy(
                        pred_risk_score_H8[:N].float(), gt_zeros.float(), reduction='mean')
                loss_htsc_i = htsc_loss(
                    hx_seq=outputs["hx_seq"][i],
                    pred_risk_type=outputs["risk_type"][i],
                    all_objs_id=all_objs_ids[i], sim_thr=0.0)
                # Counterfactual loss on non-risk objects
                loss_cf_i = counterfactual_loss(
                    pred_direct[:N], pred_indirect[:N], gt_zeros)

                # Update conformal predictors
                for type_pred, obj_pred, obj_gt in zip(
                        pred_risk_type[:N], pred_risk_score_H8[:N], gt_zeros):
                    pred_cls = self.index_to_class[type_pred.argmax().item()]
                    nc = compute_nonconformity(
                        obj_pred, obj_gt, method=self.nc_method)
                    for t in range(8):
                        nc_t = pd.Series([nc[t].item()], dtype=float)
                        self.class_cps[pred_cls].update(
                            ground_truth=nc_t,
                            forecast=pd.Series([0], dtype=float),
                            horizon=t + 1)

            else:
                matched_pred, matched_gt = [], []
                matched_direct, matched_indirect = [], []
                match_pred_type = []
                non_risk_pred, non_risk_pred_type = [], []
                non_risk_direct, non_risk_indirect = [], []

                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_score_H8[j])
                        match_pred_type.append(pred_risk_type[j])
                        matched_direct.append(pred_direct[j])
                        matched_indirect.append(pred_indirect[j])
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_score_H8[idx])
                    else:
                        non_risk_pred.append(pred_risk_score_H8[j])
                        non_risk_pred_type.append(pred_risk_type[j])
                        non_risk_direct.append(pred_direct[j])
                        non_risk_indirect.append(pred_indirect[j])

                if len(matched_pred) == 0:
                    gt_zeros = torch.zeros_like(pred_risk_score_H8[:N])
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_scoreH8_i = F.binary_cross_entropy(
                            pred_risk_score_H8[:N].float(), gt_zeros.float(), reduction='mean')
                    loss_cf_i = counterfactual_loss(
                        pred_direct[:N], pred_indirect[:N], gt_zeros)

                    for type_pred, obj_pred, obj_gt in zip(
                            pred_risk_type[:N], pred_risk_score_H8[:N],
                            gt_zeros):
                        pred_cls = self.index_to_class[
                            type_pred.argmax().item()]
                        nc = compute_nonconformity(
                            obj_pred, obj_gt, method=self.nc_method)
                        for t in range(8):
                            nc_t = pd.Series([nc[t].item()], dtype=float)
                            self.class_cps[pred_cls].update(
                                ground_truth=nc_t,
                                forecast=pd.Series([0], dtype=float),
                                horizon=t + 1)
                else:
                    preds = torch.stack(matched_pred)
                    gts = torch.stack(matched_gt)
                    d_stack = torch.stack(matched_direct)
                    i_stack = torch.stack(matched_indirect)

                    if len(non_risk_pred) == 0:
                        with torch.cuda.amp.autocast(enabled=False):
                            loss_scoreH8_i = F.binary_cross_entropy(
                                preds.float(), gts.float(), reduction='mean')
                        loss_cf_i = counterfactual_loss(
                            d_stack, i_stack, gts)
                    else:
                        nr_preds = torch.stack(non_risk_pred)
                        nr_gt = torch.zeros_like(nr_preds)
                        nr_d = torch.stack(non_risk_direct)
                        nr_i = torch.stack(non_risk_indirect)

                        with torch.cuda.amp.autocast(enabled=False):
                            loss_scoreH8_i = (
                                F.binary_cross_entropy(preds.float(), gts.float(), reduction='mean')
                                + F.binary_cross_entropy(nr_preds.float(), nr_gt.float(), reduction='mean'))
                        loss_cf_i = (
                            counterfactual_loss(d_stack, i_stack, gts)
                            + counterfactual_loss(nr_d, nr_i, nr_gt))

                    # Update conformal for matched
                    for type_pred, obj_pred, obj_gt in zip(
                            match_pred_type, matched_pred, matched_gt):
                        pred_cls = self.index_to_class[
                            type_pred.argmax().item()]
                        nc = compute_nonconformity(
                            obj_pred, obj_gt, method=self.nc_method)
                        for t in range(8):
                            nc_t = pd.Series([nc[t].item()], dtype=float)
                            self.class_cps[pred_cls].update(
                                ground_truth=nc_t,
                                forecast=pd.Series([0], dtype=float),
                                horizon=t + 1)

                    # Update conformal for non-risk
                    if len(non_risk_pred) > 0:
                        for type_pred, obj_pred, obj_gt in zip(
                                non_risk_pred_type,
                                [p.detach() for p in non_risk_pred],
                                [torch.zeros_like(p) for p in non_risk_pred]):
                            pred_cls = self.index_to_class[
                                type_pred.argmax().item()]
                            nc = compute_nonconformity(
                                obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)
                                self.class_cps[pred_cls].update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t + 1)

                # HTSC loss (same as original)
                loss_htsc_i = htsc_loss(
                    hx_seq=outputs["hx_seq"][i],
                    pred_risk_type=outputs["risk_type"][i],
                    all_objs_id=all_objs_ids[i], sim_thr=0.0)

            # Accumulate per-sample losses
            w_ortho_cur = self._get_ortho_weight()
            total_loss += (self.w_score * loss_scoreH8_i
                           + self.w_htsc * loss_htsc_i
                           + self.w_cf * loss_cf_i)
            total_loss_score += self.w_score * loss_scoreH8_i
            total_loss_htsc += self.w_htsc * loss_htsc_i
            total_loss_cf += self.w_cf * loss_cf_i

        # Average over batch
        total_loss = total_loss / B
        total_loss_score = total_loss_score / B
        total_loss_htsc = total_loss_htsc / B
        total_loss_cf = total_loss_cf / B

        # Add global ortho loss (with ramp-up)
        w_ortho_cur = self._get_ortho_weight()
        total_loss = total_loss + w_ortho_cur * total_loss_ortho

        # Logging
        is_train = (prefix == "")
        self.log(f"{prefix}total_loss", total_loss, on_step=is_train, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}loss_score", total_loss_score, on_step=is_train, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}loss_htsc", total_loss_htsc, on_step=is_train, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}loss_ortho", total_loss_ortho, on_step=is_train, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}loss_cf", total_loss_cf, on_step=is_train, on_epoch=True, prog_bar=False, sync_dist=True)

        return total_loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get("total_loss_epoch", metrics.get("total_loss", torch.tensor(0.0))).item()
        val_loss = metrics.get("val_total_loss", metrics.get("val_total_loss_epoch", torch.tensor(0.0))).item()
        loss_score = metrics.get("loss_score", torch.tensor(0.0)).item()
        loss_htsc = metrics.get("loss_htsc", torch.tensor(0.0)).item()
        loss_cf = metrics.get("loss_cf", torch.tensor(0.0)).item()
        print(
            f"\n==================== [Epoch {self.current_epoch:02d}] Train: {train_loss:.5f} | Val: {val_loss:.5f} | Score: {loss_score:.5f} | HTSC: {loss_htsc:.5f} | CF: {loss_cf:.5f} ====================\n",
            flush=True,
        )

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        return self._compute_step_losses(batch, prefix="")

    def validation_step(self, batch, batch_idx):
        self._compute_step_losses(batch, prefix="val_")

    def configure_optimizers(self):
        trainable = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable, lr=self.lr, weight_decay=1e-7)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
        return [optimizer], [lr_scheduler]

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint['saocp_class'] = self.class_cps
        return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str,
                        default='CausalForce',
                        help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val_every', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--pretrain_ckpt', type=str,
                        default='/path/to/your/pretrained_risk_category_cls.ckpt',
                        help='Path to pre-trained risk category classifier.')
    # Causal hyperparameters
    parser.add_argument('--num_confounders', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--cf_alpha', type=float, default=0.3)
    parser.add_argument('--w_ortho', type=float, default=0.1)
    parser.add_argument('--w_cf', type=float, default=0.5)
    parser.add_argument('--ortho_ramp_epochs', type=int, default=4)

    parser.add_argument('--train_data', type=str, default=os.path.expanduser('~/data/MCR_Dataset/Risk-Datasets-Venue/train/'), help='Path to train data')
    parser.add_argument('--val_data', type=str, default=os.path.expanduser('~/data/MCR_Dataset/Risk-Datasets-Venue/val/'), help='Path to val data')

    parser.add_argument('--num_workers', type=int, default=12, help='Number of dataloader workers')
    parser.add_argument('--precision', type=int, default=16, help='Precision: 16 (FP16 AMP) or 32 (FP32)')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    train_set = MultipleRisksDataset(data_root=args.train_data)
    print(len(train_set))
    val_set = MultipleRisksDataset(data_root=args.val_data)
    print(len(val_set))

    dataloader_train = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True)
    dataloader_val = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    causal_model = CausalForce(
        lr=args.lr,
        num_confounders=args.num_confounders,
        num_heads=args.num_heads,
        cf_alpha=args.cf_alpha,
        w_ortho=args.w_ortho,
        w_cf=args.w_cf,
        ortho_ramp_epochs=args.ortho_ramp_epochs,
    )

    # ── Load pre-trained classifier weights ──
    def transfer_stage1_key(key):
        if 'risk_type_head.' in key:
            new_key = key.replace(
                'risk_type_head.', 'causal_risk_head.type_head.')
            print(f"Transferred: {key} -> {new_key}", flush=True)
            return new_key
        return key

    load_partial_checkpoint(
        causal_model,
        args.pretrain_ckpt,
        key_transform=transfer_stage1_key,
        allowed_missing_prefixes=(
            'model.causal_disentangle.',
            'model.causal_message_passing.',
            'model.causal_risk_head.direct_head.',
            'model.causal_risk_head.indirect_head.',
            'model.causal_risk_head.fusion.',
            'model.causal_risk_head.fusion_gate.',
        ),
        allowed_unexpected_prefixes=(
            'model.fc_emb_1.',
            'model.fc_emb_2.',
        ),
        required_loaded_prefixes=(
            'model.backbone.',
            'model.object_backbone.',
            'model.camera_features.',
            'model.lstm.',
            'model.out_layer.',
            'model.causal_risk_head.type_head.',
        ),
        map_location='cpu',
    )

    # Freeze the risk type head (same as original)
    for name, param in causal_model.named_parameters():
        if 'causal_risk_head.type_head' in name:
            param.requires_grad = False
            print(f"Freeze: {name}")
        # Skip GCN params that don't exist in causal model
        elif 'fc_emb' in name:
            continue
        else:
            param.requires_grad = True
            print(f"Trainable: {name}")

    frozen_count = sum(
        parameter.numel() for parameter in causal_model.parameters()
        if not parameter.requires_grad)
    trainable_count = sum(
        parameter.numel() for parameter in causal_model.parameters()
        if parameter.requires_grad)
    print(f"Frozen parameter count: {frozen_count}", flush=True)
    print(f"Trainable parameter count: {trainable_count}", flush=True)

    causal_model.cuda()

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False, mode="min", monitor="val_total_loss",
        save_top_k=2, save_last=True,
        dirpath=args.logdir,
        filename="best_{epoch:02d}-{val_total_loss:.3f}")
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.logdir,
        gpus=args.gpus,
        accelerator='gpu',
        precision=args.precision,
        sync_batchnorm=True,
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_every,
        max_epochs=args.epochs,
    )

    trainer.fit(causal_model, dataloader_train, dataloader_val)
