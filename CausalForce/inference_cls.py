import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import MultipleRisksDataset, custom_collate_fn
from classifier import GCN_model
from checkpoint_utils import load_model_checkpoint
import numpy as np


CLASS_NAMES = ("OBS", "OCC", "I", "C")


def classification_report(confusion):
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        tp = confusion[index, index]
        predicted = confusion[:, index].sum()
        actual = confusion[index, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((name, int(actual), precision, recall, f1))
    return rows

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

    model = GCN_model(pretrained=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    legacy_window_accuracy_sum = 0.0
    risk_window_count = 0
    matched_correct = 0
    matched_object_count = 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    visible_object_pred_count = np.zeros(4, dtype=np.int64)

    print("Starting evaluation loop...", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            front_imgs = batch['front_imgs'].to(device)
            all_objs_bbs = batch['all_objs_bbs']
            all_objs_ids = batch['all_objs_id']
            label_risk_ids = batch['risk_id']
            label_risk_type = batch['label_risk_type']
            
            # Ensure bounding box tensors are on CUDA device
            all_objs_bbs_cuda = []
            for b in range(len(all_objs_bbs)):
                sample_bbs = []
                for t in range(len(all_objs_bbs[b])):
                    sample_bbs.append(all_objs_bbs[b][t].to(device))
                all_objs_bbs_cuda.append(sample_bbs)

            outputs = model(front_imgs, all_objs_bbs_cuda, device=device)

            B = front_imgs.shape[0]
            for i in range(B):
                pred_risk_type = outputs["risk_type"][i]        # (num_preds, 4)
                gt_risk_ids    = label_risk_ids[i]             # list[int]
                # The model truncates detections to NUM_BOX slots. Keep IDs in
                # the identical order and truncate them at the same boundary.
                all_objs_id = all_objs_ids[i][-1][:pred_risk_type.shape[0]]
                gt_risk_types  = label_risk_type[i]            # list[int] or (num_objs,)
                
                N = len(all_objs_id)
                if len(all_objs_id) == 0 or len(gt_risk_ids) == 0:
                    continue

                matched_pred, matched_gt = [], []
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_type[j])
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_types[idx])
                
                if len(matched_pred) == 0:
                    continue

                preds = torch.stack(matched_pred) # (P, 4)
                if not isinstance(matched_gt[0], torch.Tensor):
                    gts = torch.tensor(matched_gt, device=preds.device)
                else:
                    gts = torch.stack(matched_gt).to(preds.device)

                pred_labels = preds.argmax(dim=1)
                window_correct = (pred_labels == gts).sum().item()
                legacy_window_accuracy_sum += window_correct / len(gts)
                risk_window_count += 1
                matched_correct += window_correct
                matched_object_count += len(gts)
                for gt_label, pred_label in zip(
                        gts.detach().cpu().tolist(),
                        pred_labels.detach().cpu().tolist()):
                    confusion[int(gt_label), int(pred_label)] += 1

                risk_type = pred_risk_type[:N].argmax(dim=1).cpu().numpy()
                for rt in risk_type:
                    visible_object_pred_count[int(rt)] += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                cur_acc = (100 * matched_correct / matched_object_count
                           if matched_object_count else 0.0)
                print(f"Batch [{batch_idx + 1}/{len(test_loader)}] | "
                      f"Risk windows: {risk_window_count} | "
                      f"Matched risk objects: {matched_object_count} | "
                      f"Micro accuracy: {cur_acc:.2f}%", flush=True)

    micro_accuracy = matched_correct / matched_object_count if matched_object_count else 0.0
    legacy_accuracy = (legacy_window_accuracy_sum / risk_window_count
                       if risk_window_count else 0.0)
    report = classification_report(confusion)
    macro_f1 = sum(row[4] for row in report) / len(report)
    print("\n=======================================================", flush=True)
    print(f"Stage 1 matched-object micro accuracy: {micro_accuracy * 100:.2f}%", flush=True)
    print(f"Legacy mean-per-window accuracy: {legacy_accuracy * 100:.2f}%", flush=True)
    print(f"Risk windows evaluated: {risk_window_count}", flush=True)
    print(f"Matched risk-object observations: {matched_object_count}", flush=True)
    print("Per-class metrics on matched risk objects:", flush=True)
    print("  class  support  precision  recall  F1", flush=True)
    for name, support, precision, recall, f1 in report:
        print(f"  {name:>3} {support:8d} {precision:10.4f} {recall:7.4f} {f1:7.4f}", flush=True)
    print(f"Macro-F1: {macro_f1:.4f}", flush=True)
    print("Confusion matrix (rows=GT, columns=prediction; OBS/OCC/I/C):", flush=True)
    for row in confusion:
        print("  " + " ".join(f"{int(value):8d}" for value in row), flush=True)
    print("Predicted classes for all visible object observations", flush=True)
    print("(includes non-risk objects and repeated sliding-window observations):", flush=True)
    for index, name in enumerate(CLASS_NAMES):
        print(f"  - {name}: {int(visible_object_pred_count[index])}", flush=True)
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
