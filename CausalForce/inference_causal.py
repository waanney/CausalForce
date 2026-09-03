"""
CausalForce Inference Script.

Evaluates:
  1. Overall Paper Metrics (Coverage, TV, TC, BA, Risk-IOU, IoU)
  2. One-Risk vs Multi-Risks breakdowns
  3. Per-Category breakdowns (OBS, OCC, I, C) based on Ground Truth Categories
"""

import argparse
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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
from collections import defaultdict


class MetricAccumulator:
    def __init__(self, name=""):
        self.name = name
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

    def compute(self):
        cnt = self.total_gt_risk_obj_cnt or 1
        r_cnt = self.total_risk_sample_cnt or 1
        s_cnt = self.total_sample_cnt or 1
        t_cnt = self.trasition_cnt or 1

        coverage = self.covered / cnt
        gt_tv = self.total_gt_risk_obj_tube_volume / r_cnt
        pred_tv = (self.total_risk_obj_pred_tube_volume / r_cnt) + (self.total_non_risk_obj_pred_tube_volume / s_cnt)
        risk_iou = self.risk_interval_iou_pic / cnt
        iou = self.iou / cnt
        tc = self.fragmented_prediction_penalty / t_cnt
        ba = 0.5 * (self.detect_penalty_pic + self.release_penalty_pic) / cnt

        return {
            "name": self.name,
            "coverage": coverage,
            "gt_tv": gt_tv,
            "pred_tv": pred_tv,
            "risk_iou": risk_iou,
            "iou": iou,
            "tc": tc,
            "ba": ba,
            "gt_objects": self.total_gt_risk_obj_cnt,
        }


class CausalInferenceModule(pl.LightningModule):
    def __init__(self, model, cp_ckpt, debug_samples=10):
        super().__init__()
        self.model = model
        self.class_cps = cp_ckpt['saocp_class']
        self.index_to_class = {0: 'OBS', 1: 'OCC', 2: 'I', 3: 'C'}
        self.debug_samples = debug_samples
        self.debug_samples_printed = 0
        self._print_conformal_state()

        self.overall_metrics = MetricAccumulator("Overall")
        self.one_risk_metrics = MetricAccumulator("One-Risk")
        self.multi_risk_metrics = MetricAccumulator("Multi-Risks")
        self.category_metrics = {
            cls: MetricAccumulator(f"{cls}")
            for cls in self.index_to_class.values()
        }

    def _quantiles(self, risk_class):
        if risk_class not in self.class_cps:
            raise KeyError(f"Missing SAOCP state for class {risk_class}")
        cp = self.class_cps[risk_class]
        residuals = getattr(cp, "residuals", None)
        horizon_map = getattr(residuals, "horizon2residuals", {}) if residuals else {}
        values = []
        for t in range(8):
            count = len(horizon_map.get(t + 1, []))
            pred_q = cp.predict(horizon=t + 1)[1]
            if count == 0 or pred_q <= 0.0:
                other_qs = []
                for other_cls, other_cp in self.class_cps.items():
                    o_res = getattr(other_cp, "residuals", None)
                    o_map = getattr(o_res, "horizon2residuals", {}) if o_res else {}
                    if len(o_map.get(t + 1, [])) > 0:
                        oq = other_cp.predict(horizon=t + 1)[1]
                        if oq > 0.0:
                            other_qs.append(oq)
                fallback = float(np.mean(other_qs)) if other_qs else 0.5
                values.append(fallback)
            else:
                values.append(float(pred_q))
        values = np.asarray(values, dtype=float)
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
            horizon_map = getattr(residuals, "horizon2residuals", {}) if residuals else {}
            counts = [len(horizon_map.get(t + 1, [])) for t in range(8)]
            formatted = " ".join(f"{value:.6f}" for value in values)
            print(f"  {risk_class}: {formatted}", flush=True)
            print(f"    calibration counts: {counts}", flush=True)
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

    def _accumulate_object(self, acc, pred_risk_mask, gt_risk_mask=None, is_risk=False, metrics=None):
        if is_risk:
            acc.total_gt_risk_obj_cnt += 1
            acc.total_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
            acc.total_gt_risk_obj_tube_volume += gt_risk_mask.sum().item()
            acc.iou += metrics["iou"]
            acc.trasition_cnt += 1
            acc.fragmented_prediction_penalty += metrics["temporal_consistency"]
            acc.detect_penalty_pic += metrics["detect_alignment"]
            acc.release_penalty_pic += metrics["release_alignment"]
            acc.risk_interval_iou_pic += metrics["risk_iou"]
            acc.covered += int(metrics["coverage"])
        else:
            acc.total_non_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
            acc.trasition_cnt += 1
            acc.fragmented_prediction_penalty += temporal_consistency(
                pred_risk_mask.cpu().numpy(), np.zeros(8, dtype=np.int8))

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_interval_H8 = batch['risk_interval_H8']
        label_risk_type = batch.get('label_risk_type', [])
        scenario_ids = batch['scenario_id']

        B, T, C, H, W = front_imgs.shape
        outputs = self.model(front_imgs, all_objs_bbs)

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]
            pred_risk_type = outputs["risk_type"][i]
            gt_risk_ids = label_risk_ids[i]
            gt_risk_score_H8 = label_risk_interval_H8[i]
            gt_risk_types = label_risk_type[i] if i < len(label_risk_type) else None
            all_objs_id = all_objs_ids[i][-1][:pred_risk_score_H8.shape[0]]

            is_one_risk = (len(gt_risk_ids) <= 1)
            sc_acc = self.one_risk_metrics if is_one_risk else self.multi_risk_metrics

            if len(all_objs_id) > 0:
                self.overall_metrics.total_sample_cnt += 1
                sc_acc.total_sample_cnt += 1
            if len(gt_risk_score_H8) > 0:
                self.overall_metrics.total_risk_sample_cnt += 1
                sc_acc.total_risk_sample_cnt += 1

            for j, obj_id in enumerate(all_objs_id):
                pred_cls = self.index_to_class[pred_risk_type[j].argmax().item()]
                q = self._quantiles(pred_cls)
                pred_np = conformal_risk_mask(pred_risk_score_H8[j], q)
                pred_risk_mask = torch.as_tensor(pred_np, device=pred_risk_score_H8.device)

                if obj_id in gt_risk_ids:
                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx].to(pred_risk_mask.device)

                    if gt_risk_mask.sum() == 0:
                        continue

                    # Ground truth category grouping
                    if gt_risk_types is not None and idx < len(gt_risk_types):
                        gt_cls_idx = gt_risk_types[idx].item()
                        gt_cls = self.index_to_class.get(gt_cls_idx, pred_cls)
                    else:
                        gt_cls = pred_cls

                    cat_acc = self.category_metrics[gt_cls]
                    cat_acc.total_sample_cnt += 1
                    cat_acc.total_risk_sample_cnt += 1

                    metrics = evaluate_risk_tube(pred_risk_mask, gt_risk_mask)

                    self._accumulate_object(self.overall_metrics, pred_risk_mask, gt_risk_mask, is_risk=True, metrics=metrics)
                    self._accumulate_object(sc_acc, pred_risk_mask, gt_risk_mask, is_risk=True, metrics=metrics)
                    self._accumulate_object(cat_acc, pred_risk_mask, gt_risk_mask, is_risk=True, metrics=metrics)

                    self._debug_tube(
                        scenario_ids[i], obj_id, f"{pred_cls} (GT:{gt_cls})",
                        pred_risk_score_H8[j].detach().cpu().numpy(),
                        q, pred_np,
                        gt_risk_mask.detach().cpu().numpy(), metrics)
                else:
                    self._accumulate_object(self.overall_metrics, pred_risk_mask, is_risk=False)
                    self._accumulate_object(sc_acc, pred_risk_mask, is_risk=False)

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        def print_table(m_list):
            print("\n" + "=" * 80, flush=True)
            print(f"{'Category / Subset':<20} | {'Coverage ↑':<10} | {'Tube Vol ↓':<10} | {'TC ↑':<8} | {'BA ↑':<8} | {'Risk IoU ↑':<10}", flush=True)
            print("-" * 80, flush=True)
            for m in m_list:
                res = m.compute()
                print(f"{res['name']:<20} | {res['coverage']:<10.3f} | {res['pred_tv']:<10.3f} | {res['tc']:<8.3f} | {res['ba']:<8.3f} | {res['risk_iou']:<10.3f}", flush=True)
            print("=" * 80 + "\n", flush=True)

        print("SCENARIO BREAKDOWN:")
        print_table([self.one_risk_metrics, self.multi_risk_metrics, self.overall_metrics])
        print("PER-CATEGORY BREAKDOWN (GROUND TRUTH):")
        print_table(list(self.category_metrics.values()))

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default='/path/to/your/testing_data/')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug_samples", type=int, default=10)
    parser.add_argument("--checkpoint", type=str,
                        default='/path/to/your/causal_checkpoint.ckpt')
    args = parser.parse_args()

    test_set = MultipleRisksDataset(data_root=args.data_root)
    print(f"Test dataset: {len(test_set)} samples")
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
