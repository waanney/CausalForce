import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
import numpy as np

def pic_star(pred, gt, tau=2, mode="detect"):
    gt   = np.asarray(gt.cpu(),   dtype=np.int8)
    pred = np.asarray(pred.cpu(), dtype=np.int8)
    assert gt.shape == pred.shape, "gt & pred must have same length"
    n = len(gt)

    ones = np.where(gt == 1)[0]
    if len(ones) == 0:
        return 1.0

    T_s, T_e = ones[0], ones[-1]
    center   = T_s if mode == "detect" else T_e

    t = np.arange(n)
    w = np.exp(-np.abs(t - center) / tau)
    W = w.sum()

    f1 = 1 - np.abs(gt - pred)
    loss   = (w * (1 - f1)).sum() / W
    score  = 1 - loss
    return score

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

    checkpoint = torch.load(args.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
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
                gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
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

                    pred_risk_mask = (pred_risk_score_H8[j] >= (1 - q_vec)).long()

                    if obj_id in gt_risk_ids:
                        idx = gt_risk_ids.index(obj_id)
                        gt_risk_mask = gt_risk_score_H8[idx]

                        if gt_risk_mask.sum() == 0:
                            continue

                        total_gt_risk_obj_cnt += 1
                        total_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                        total_gt_risk_obj_tube_volume += gt_risk_mask.sum().item()

                        pred = pred_risk_mask.bool()
                        gt   = gt_risk_mask.bool()
                        intersection = (pred & gt).sum().float()
                        union        = (pred | gt).sum().float()
                        cur_iou = intersection / (union + 1e-6)
                        iou += cur_iou.item()

                        transitions_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()
                        transitions_gt   = (gt_risk_mask[1:] != gt_risk_mask[:-1]).sum().item()

                        fragmented_prediction_pen = 1.0 - (np.abs(transitions_pred - transitions_gt) / 7.0)
                        if np.abs(transitions_pred - transitions_gt) >= 0:
                            trasition_cnt += 1
                            fragmented_prediction_penalty += fragmented_prediction_pen

                        d_pen_pic = pic_star(pred_risk_mask, gt_risk_mask, mode="detect")
                        e_pen_pic = pic_star(pred_risk_mask, gt_risk_mask, mode="release")
                        detect_penalty_pic += d_pen_pic
                        release_penalty_pic += e_pen_pic
                        risk_interval_iou_pic += (cur_iou.item() * (0.5 * (d_pen_pic + e_pen_pic) + fragmented_prediction_pen) / 2.0)

                        if torch.all(pred_risk_mask[gt_risk_mask == 1] == 1):
                            covered += 1
                    else:
                        total_non_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                        transitions_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()
                        fragmented_prediction_pen = 1.0 - (np.abs(transitions_pred) / 7.0)
                        if np.abs(transitions_pred) >= 0:
                            trasition_cnt += 1
                            fragmented_prediction_penalty += fragmented_prediction_pen

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                cur_cov = (covered / total_gt_risk_obj_cnt) if total_gt_risk_obj_cnt > 0 else 0.0
                cur_riou = (risk_interval_iou_pic / total_gt_risk_obj_cnt) if total_gt_risk_obj_cnt > 0 else 0.0
                print(f"Batch [{batch_idx + 1}/{len(test_loader)}] | Evaluated GT Objects: {total_gt_risk_obj_cnt} | Coverage: {cur_cov:.4f} | Risk-IoU: {cur_riou:.4f}", flush=True)

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
