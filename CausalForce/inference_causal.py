"""
CausalForce Inference Script.

Identical evaluation metrics as original inference.py but uses CausalGCN_model.
Metrics: Coverage, Tube Volume (TV), Temporal Consistency (TC),
         Boundary Alignment (BA), Risk-IOU.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import MultipleRisksDataset, custom_collate_fn
from causal_model import CausalGCN_model
import numpy as np
from checkpoint_utils import load_model_checkpoint
from risk_tube_metrics import (
    conformal_risk_mask,
    evaluate_risk_tube,
    mask_interval,
    temporal_consistency,
)


class CausalInferenceModule(pl.LightningModule):
    def __init__(self, model, cp_ckpt, debug_samples=10):
        super().__init__()
        self.model = model
        self.class_cps = cp_ckpt['saocp_class']
        self.index_to_class = {0: 'OBS', 1: 'OCC', 2: 'I', 3: 'C'}
        self.debug_samples = debug_samples
        self.debug_samples_printed = 0
        self._print_conformal_state()

        # Counters (same as original)
        self.covered = 0
        self.total_gt_risk_obj_cnt = 0
        self.total_sample_cnt = 0
        self.total_risk_sample_cnt = 0
        self.total_gt_risk_obj_tube_volume = 0
        self.total_risk_obj_pred_tube_volume = 0
        self.total_non_risk_obj_pred_tube_volume = 0
        self.risk_interval_iou_pic = 0.0
        self.fragmented_prediction_penalty = 0.0
        self.trasition_cnt = 0
        self.release_penalty_pic = 0.0
        self.detect_penalty_pic = 0.0
        self.iou = 0.0

    def _quantiles(self, risk_class):
        if risk_class not in self.class_cps:
            raise KeyError(f"Missing SAOCP state for class {risk_class}")
        values = np.asarray([
            self.class_cps[risk_class].predict(horizon=t + 1)[1]
            for t in range(8)
        ], dtype=float)
        # Fallback for uncalibrated classes (e.g. OBS with 0 calibration count)
        if values.sum() == 0:
            calibrated_vals = []
            for k, cp in self.class_cps.items():
                v = np.asarray([cp.predict(horizon=t + 1)[1] for t in range(8)], dtype=float)
                if v.sum() > 0:
                    calibrated_vals.append(v)
            if calibrated_vals:
                values = np.mean(calibrated_vals, axis=0)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid SAOCP radii for {risk_class}: {values.tolist()}")
        if (values < 0).any():
            raise ValueError(
                f"Negative SAOCP radii for {risk_class}: {values.tolist()}")
        return values

    def _print_conformal_state(self):
        print("SAOCP checkpoint state (upper radii q by horizon):", flush=True)
        coverages = set()
        for risk_class in self.index_to_class.values():
            cp = self.class_cps.get(risk_class)
            if cp is None:
                raise KeyError(f"Checkpoint is missing SAOCP class {risk_class}")
            coverages.add(getattr(cp, "coverage", None))
            values = self._quantiles(risk_class)
            residuals = getattr(cp, "residuals", None)
            horizon_map = getattr(residuals, "horizon2residuals", {})
            counts = [len(horizon_map.get(t + 1, [])) for t in range(8)]
            formatted = " ".join(f"{value:.6f}" for value in values)
            print(f"  {risk_class}: {formatted}", flush=True)
            print(f"    calibration counts: {counts}", flush=True)
            if any(count == 0 for count in counts):
                print(
                    f"  [Notice] SAOCP calibration bucket for {risk_class} has 0 count: {counts}", flush=True)
        print(f"SAOCP target coverage value(s): {sorted(coverages, key=str)}", flush=True)

    def _debug_tube(self, scenario_id, object_id, risk_class, raw, q, pred, gt, metrics):
        if self.debug_samples_printed >= self.debug_samples:
            return
        self.debug_samples_printed += 1
        sample_number = self.debug_samples_printed
        fmt = lambda values: " ".join(f"{float(value):.4f}" for value in values)
        bits = lambda values: " ".join(str(int(value)) for value in values)
        print(f"\n[Risk Tube Debug {sample_number}] scenario={scenario_id} "
              f"object_id={object_id} predicted_class={risk_class}", flush=True)
        print(f"GT:         {bits(gt)}", flush=True)
        print(f"Raw score:  {fmt(raw)}", flush=True)
        print(f"q:          {fmt(q)}", flush=True)
        print(f"threshold:  {fmt(1.0 - q)}", flush=True)
        print(f"Calibrated: {bits(pred)}", flush=True)
        print(f"GT interval:   {mask_interval(gt)}", flush=True)
        print(f"Pred interval: {mask_interval(pred)}", flush=True)
        print(f"Coverage={int(metrics['coverage'])} IoU={metrics['iou']:.4f} "
              f"TC={metrics['temporal_consistency']:.4f} "
              f"BA={metrics['boundary_alignment']:.4f} "
              f"Risk-IoU={metrics['risk_iou']:.4f}", flush=True)

    @torch.no_grad()
    def forward(self, imgs, all_objs):
        return self.model(imgs, all_objs)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_interval_H8 = batch['risk_interval_H8']
        scenario_ids = batch['scenario_id']

        B, T, C, H, W = front_imgs.shape
        outputs = self.model(front_imgs, all_objs_bbs)
        if batch_idx == 0:
            print("Tensor-shape audit (first batch):", flush=True)
            print(f"  input [B,T,C,H,W]: {tuple(front_imgs.shape)}", flush=True)
            print(f"  score_H8 [B,N,H]: {tuple(outputs['score_H8'].shape)}", flush=True)
            print(f"  risk_type [B,N,4]: {tuple(outputs['risk_type'].shape)}", flush=True)
            print(f"  hx_seq [B,T,N,D]: {tuple(outputs['hx_seq'].shape)}", flush=True)
            print(f"  causal_feat [B,N,D]: {tuple(outputs['causal_feat'].shape)}", flush=True)
            print(f"  scene_feat [B,N,D]: {tuple(outputs['scene_feat'].shape)}", flush=True)
            print(f"  direct [B,N,H]: {tuple(outputs['direct'].shape)}", flush=True)
            print(f"  indirect [B,N,H]: {tuple(outputs['indirect'].shape)}", flush=True)
            first_tracker_shapes = [
                tuple(frame.shape) for frame in all_objs_bbs[0]]
            print(f"  first sample tracker frames: {first_tracker_shapes}", flush=True)

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]
            pred_risk_type = outputs["risk_type"][i]
            gt_risk_ids = label_risk_ids[i]
            gt_risk_score_H8 = label_risk_interval_H8[i]
            all_objs_id = all_objs_ids[i][-1][:pred_risk_score_H8.shape[0]]

            if len(all_objs_id) > 0:
                self.total_sample_cnt += 1
            if len(gt_risk_score_H8) > 0:
                self.total_risk_sample_cnt += 1

            for j, obj_id in enumerate(all_objs_id):
                pred_cls = self.index_to_class[
                    pred_risk_type[j].argmax().item()]
                q = self._quantiles(pred_cls)
                pred_np = conformal_risk_mask(pred_risk_score_H8[j], q)
                pred_risk_mask = torch.as_tensor(
                    pred_np, device=pred_risk_score_H8.device)

                if obj_id in gt_risk_ids:
                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx].to(
                        pred_risk_mask.device)

                    if gt_risk_mask.sum() == 0:
                        continue

                    self.total_gt_risk_obj_cnt += 1
                    self.total_risk_obj_pred_tube_volume += \
                        pred_risk_mask.sum().item()
                    self.total_gt_risk_obj_tube_volume += \
                        gt_risk_mask.sum().item()

                    metrics = evaluate_risk_tube(
                        pred_risk_mask, gt_risk_mask)
                    self.iou += metrics["iou"]
                    self.trasition_cnt += 1
                    self.fragmented_prediction_penalty += metrics[
                        "temporal_consistency"]
                    self.detect_penalty_pic += metrics["detect_alignment"]
                    self.release_penalty_pic += metrics["release_alignment"]
                    self.risk_interval_iou_pic += metrics["risk_iou"]
                    self.covered += int(metrics["coverage"])
                    self._debug_tube(
                        scenario_ids[i], obj_id, pred_cls,
                        pred_risk_score_H8[j].detach().cpu().numpy(),
                        q, pred_np,
                        gt_risk_mask.detach().cpu().numpy(), metrics)
                else:
                    self.total_non_risk_obj_pred_tube_volume += \
                        pred_risk_mask.sum().item()
                    self.trasition_cnt += 1
                    self.fragmented_prediction_penalty += temporal_consistency(
                        pred_np, np.zeros_like(pred_np))

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        cnt = self.total_gt_risk_obj_cnt or 1
        r_cnt = self.total_risk_sample_cnt or 1
        s_cnt = self.total_sample_cnt or 1
        t_cnt = self.trasition_cnt or 1

        print(f"Coverage:  {self.covered / cnt:.4f}")
        print(f"GT TV:     {self.total_gt_risk_obj_tube_volume / r_cnt:.4f}")
        print(f"Pred TV:   {(self.total_risk_obj_pred_tube_volume / r_cnt) + (self.total_non_risk_obj_pred_tube_volume / s_cnt):.4f}")
        print(f"Risk-IOU:  {self.risk_interval_iou_pic / cnt:.4f}")
        print(f"IoU:       {self.iou / cnt:.4f}")
        print(f"TC:        {self.fragmented_prediction_penalty / t_cnt:.4f}")
        print(f"BA:        {0.5 * (self.detect_penalty_pic + self.release_penalty_pic) / cnt:.4f}")

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default='/path/to/your/testing_data/')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--debug_samples", type=int, default=10)
    parser.add_argument("--checkpoint", type=str,
                        default='/path/to/your/causal_checkpoint.ckpt')
    args = parser.parse_args()

    test_set = MultipleRisksDataset(data_root=args.data_root)
    print(len(test_set))
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn)

    model = CausalGCN_model(pretrained=False)

    checkpoint = load_model_checkpoint(
        model, args.checkpoint, map_location="cpu", require_conformal=True)
    model.cuda()
    model.eval()

    inference_module = CausalInferenceModule(
        model=model, cp_ckpt=checkpoint, debug_samples=args.debug_samples)
    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, dataloaders=test_loader)
