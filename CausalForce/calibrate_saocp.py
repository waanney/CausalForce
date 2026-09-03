"""
SAOCP Conformal Calibration Script for CausalForce.

Calibrates the online conformal prediction buckets (OBS, OCC, I, C)
across all 8 forecast horizons on a designated calibration/training stream,
then saves a calibrated checkpoint ready for paper evaluation.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data import MultipleRisksDataset, custom_collate_fn
from causal_model import CausalGCN_model
from online_conformal.saocp import SAOCP
from train import compute_nonconformity


def calibrate(args):
    print(f"Loading checkpoint for calibration: {args.checkpoint}", flush=True)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = CausalGCN_model(pretrained=False)
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = raw_ckpt.get("state_dict", raw_ckpt)
    
    cleaned_sd = {}
    for k, v in state_dict.items():
        clean_k = k[6:] if k.startswith("model.") else k
        cleaned_sd[clean_k] = v
    model.load_state_dict(cleaned_sd, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    model.to(device)
    model.eval()

    index_to_class = {0: 'OBS', 1: 'OCC', 2: 'I', 3: 'C'}
    class_cps = {
        cls: SAOCP(model=None, train_data=None, max_scale=1.0, coverage=args.coverage, horizon=8)
        for cls in index_to_class.values()
    }

    calib_set = MultipleRisksDataset(data_root=args.data_root)
    print(f"Calibration dataset size: {len(calib_set)} samples from {args.data_root}", flush=True)
    calib_loader = DataLoader(
        calib_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    total_updates = {cls: [0] * 8 for cls in index_to_class.values()}

    print(f"Starting SAOCP calibration loop (Target coverage: {args.coverage})...", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            front_imgs = batch['front_imgs'].to(device)
            all_objs_bbs = batch['all_objs_bbs']
            all_objs_ids = batch['all_objs_id']
            label_risk_ids = batch['risk_id']
            label_risk_interval_H8 = batch['risk_interval_H8']

            all_objs_bbs_dev = []
            for b in range(len(all_objs_bbs)):
                sample_bbs = [bb.to(device) for bb in all_objs_bbs[b]]
                all_objs_bbs_dev.append(sample_bbs)

            outputs = model(front_imgs, all_objs_bbs_dev)
            B = front_imgs.shape[0]

            for i in range(B):
                pred_risk_score_H8 = outputs["score_H8"][i]
                pred_risk_type = outputs["risk_type"][i]
                gt_risk_ids = label_risk_ids[i]
                gt_risk_score_H8 = label_risk_interval_H8[i]
                all_objs_id = all_objs_ids[i][-1][:pred_risk_score_H8.shape[0]]

                N = len(all_objs_id)
                if N == 0:
                    continue

                for j, obj_id in enumerate(all_objs_id):
                    pred_cls = index_to_class[pred_risk_type[j].argmax().item()]
                    score_pred = pred_risk_score_H8[j]

                    if obj_id in gt_risk_ids:
                        idx = gt_risk_ids.index(obj_id)
                        gt_score = gt_risk_score_H8[idx].to(score_pred.device)
                    else:
                        gt_score = torch.zeros_like(score_pred)

                    nc = compute_nonconformity(score_pred, gt_score, method="absolute")
                    for t in range(8):
                        nc_val = float(nc[t].item())
                        class_cps[pred_cls].update(
                            ground_truth=pd.Series([nc_val], dtype=float),
                            forecast=pd.Series([0.0], dtype=float),
                            horizon=t + 1
                        )
                        total_updates[pred_cls][t] += 1

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(calib_loader):
                print(f"Batch [{batch_idx + 1}/{len(calib_loader)}] calibrated.", flush=True)

    print("\n=======================================================", flush=True)
    print("CALIBRATION COMPLETE - SAOCP Radii (q) by Class & Horizon:", flush=True)
    print("=======================================================", flush=True)
    for risk_class in index_to_class.values():
        cp = class_cps[risk_class]
        q_vals = [cp.predict(horizon=t + 1)[1] for t in range(8)]
        formatted_q = " ".join(f"{q:.4f}" for q in q_vals)
        counts = total_updates[risk_class]
        print(f"  {risk_class:4s}: q = [{formatted_q}]", flush=True)
        print(f"        counts = {counts}", flush=True)
    print("=======================================================\n", flush=True)

    out_checkpoint = raw_ckpt.copy() if isinstance(raw_ckpt, dict) else {}
    if "state_dict" not in out_checkpoint:
        out_checkpoint["state_dict"] = {f"model.{k}": v for k, v in model.state_dict().items()}
    out_checkpoint["saocp_class"] = class_cps
    out_checkpoint["coverage"] = args.coverage

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)), exist_ok=True)
    torch.save(out_checkpoint, args.output_ckpt)
    print(f"Saved calibrated checkpoint to: {args.output_ckpt}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Data root for calibration (train or calibration split)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained CausalForce checkpoint")
    parser.add_argument("--output_ckpt", type=str, required=True,
                        help="Path to output calibrated checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--coverage", type=float, default=0.9,
                        help="Conformal target coverage (default: 0.9)")
    args = parser.parse_args()
    calibrate(args)
