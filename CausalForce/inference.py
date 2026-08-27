import argparse
import os
import sys
import traceback
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
from checkpoint_utils import load_model_checkpoint
import numpy as np
from risk_tube_metrics import (
    conformal_risk_mask,
    evaluate_risk_tube,
    pic_star,
    temporal_consistency,
)

def run_evaluation(args):
    print(f"Loading checkpoint from: {args.checkpoint}", flush=True)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.checkpoint}")

    test_set = MultipleRisksDataset(data_root=args.data_root)
    print(f"Test dataset total samples: {len(test_set)}", flush=True)
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    model = GCN_model()

    checkpoint = load_model_checkpoint(
        model,
        args.checkpoint,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        require_conformal=True,
    )
    model.cuda()
    model.eval()

    class_cps = checkpoint.get('saocp_class', None)
    index_to_class = {0: 'OBS', 1: 'OCC', 2: 'I', 3: 'C'}

    covered = 0
    total_gt_risk_obj_cnt = 0
    total_sample_cnt = 0 
    total_risk_sample_cnt = 0
    
    total_gt_risk_obj_tube_volume = 0
    total_risk_obj_pred_tube_volume = 0
    total_non_risk_obj_pred_tube_volume = 0
    
    risk_interval_iou_pic = 0.0
    fragmented_prediction_penalty = 0.0
    trasition_cnt = 0
    release_penalty_pic = 0.0
    detect_penalty_pic = 0.0
    iou = 0.0

    print("Starting evaluation loop for Paper Metrics (Coverage, TV, TC, BA, Risk-IoU)...", flush=True)
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                front_imgs = batch['front_imgs'].cuda()
                all_objs_bbs = batch['all_objs_bbs']
                all_objs_ids = batch['all_objs_id']
                label_risk_ids = batch['risk_id']
                label_risk_interval_H8 = batch['risk_interval_H8']

                all_objs_bbs_cuda = []
                for b in range(len(all_objs_bbs)):
                    sample_bbs = []
                    for t in range(len(all_objs_bbs[b])):
                        sample_bbs.append(all_objs_bbs[b][t].cuda())
                    all_objs_bbs_cuda.append(sample_bbs)

                outputs = model(front_imgs, all_objs_bbs_cuda, device='cuda')

                B = front_imgs.shape[0]
                for i in range(B):
                    pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
                    pred_risk_type     = outputs["risk_type"][i]           # (num_preds, 4)
                    gt_risk_ids        = label_risk_ids[i]             # list[int]
                    gt_risk_score_H8   = label_risk_interval_H8[i]  # list of tensors (8,)
                    all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)

                    if len(all_objs_id) > 0:
                        total_sample_cnt += 1
                    if len(gt_risk_score_H8) > 0:
                        total_risk_sample_cnt += 1

                    for j, obj_id in enumerate(all_objs_id):
                        pred_cls = index_to_class[pred_risk_type[j].argmax().item()]
                        
                        if class_cps is not None and pred_cls in class_cps:
                            q_vec = torch.tensor(
                                [class_cps[pred_cls].predict(horizon=t+1)[1] for t in range(8)],
                                device=pred_risk_score_H8.device
                            )
                        else:
                            q_vec = torch.full((8,), 0.5, device=pred_risk_score_H8.device)

                        pred_np = conformal_risk_mask(
                            pred_risk_score_H8[j], q_vec)
                        pred_risk_mask = torch.as_tensor(
                            pred_np, device=pred_risk_score_H8.device)

                        if obj_id in gt_risk_ids:
                            idx = gt_risk_ids.index(obj_id)
                            gt_risk_mask = gt_risk_score_H8[idx].to(pred_risk_mask.device)

                            if gt_risk_mask.sum() == 0:
                                continue

                            total_gt_risk_obj_cnt += 1
                            total_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                            total_gt_risk_obj_tube_volume += gt_risk_mask.sum().item()

                            metrics = evaluate_risk_tube(
                                pred_risk_mask, gt_risk_mask)
                            iou += metrics["iou"]
                            trasition_cnt += 1
                            fragmented_prediction_penalty += metrics[
                                "temporal_consistency"]
                            detect_penalty_pic += metrics[
                                "detect_alignment"]
                            release_penalty_pic += metrics[
                                "release_alignment"]
                            risk_interval_iou_pic += metrics["risk_iou"]
                            covered += int(metrics["coverage"])
                        else:
                            total_non_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                            trasition_cnt += 1
                            fragmented_prediction_penalty += temporal_consistency(
                                pred_np, np.zeros_like(pred_np))

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                    cur_cov = (covered / total_gt_risk_obj_cnt) if total_gt_risk_obj_cnt > 0 else 0.0
                    cur_riou = (risk_interval_iou_pic / total_gt_risk_obj_cnt) if total_gt_risk_obj_cnt > 0 else 0.0
                    print(f"Batch [{batch_idx + 1}/{len(test_loader)}] | Evaluated GT Objects: {total_gt_risk_obj_cnt} | Coverage: {cur_cov:.4f} | Risk-IoU: {cur_riou:.4f}", flush=True)

    except Exception as e:
        print(f"\n[ERROR] Exception occurred during evaluation loop: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e

    cnt = total_gt_risk_obj_cnt or 1
    r_cnt = total_risk_sample_cnt or 1
    s_cnt = total_sample_cnt or 1
    t_cnt = trasition_cnt or 1

    final_cov = covered / cnt
    final_gt_tv = total_gt_risk_obj_tube_volume / r_cnt
    final_pred_tv = (total_risk_obj_pred_tube_volume / r_cnt) + (total_non_risk_obj_pred_tube_volume / s_cnt)
    final_iou = iou / cnt
    final_tc = fragmented_prediction_penalty / t_cnt
    final_ba = 0.5 * (detect_penalty_pic + release_penalty_pic) / cnt
    final_risk_iou = risk_interval_iou_pic / cnt

    print("\n=======================================================", flush=True)
    print("           PAPER EVALUATION METRICS RESULTS            ", flush=True)
    print("=======================================================", flush=True)
    print(f"  Coverage ↑                 : {final_cov:.4f}", flush=True)
    print(f"  Tube Volume (TV) ↓         : {final_pred_tv:.4f} (GT TV: {final_gt_tv:.4f})", flush=True)
    print(f"  Temporal Consistency (TC) ↑ : {final_tc:.4f}", flush=True)
    print(f"  Boundary Alignment (BA) ↑   : {final_ba:.4f}", flush=True)
    print(f"  IoU ↑                      : {final_iou:.4f}", flush=True)
    print(f"  Risk-IoU ↑                 : {final_risk_iou:.4f}", flush=True)
    print("=======================================================\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.expanduser('~/data/MCR_Dataset/Risk-Datasets-Venue/val/'))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=os.path.expanduser('~/CausalForce/log/risk_category_classifier/epoch=9-last.ckpt'))
    args = parser.parse_args()

    run_evaluation(args)
