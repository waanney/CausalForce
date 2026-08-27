import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import MultipleRisksDataset, custom_collate_fn
from classifier import GCN_model
import numpy as np

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

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    cls_accuracy = 0.0
    risk_sample_cnt = 0
    risk_type_cnt = {'OBS': 0, 'OCC': 0, 'I': 0, 'C': 0}

    print("Starting evaluation loop...", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            front_imgs = batch['front_imgs'].cuda()
            all_objs_bbs = batch['all_objs_bbs']
            all_objs_ids = batch['all_objs_id']
            label_risk_ids = batch['risk_id']
            label_risk_type = batch['label_risk_type']
            
            outputs = model(front_imgs, all_objs_bbs)

            B = front_imgs.shape[0]
            for i in range(B):
                pred_risk_type = outputs["risk_type"][i]        # (num_preds, 4)
                gt_risk_ids    = label_risk_ids[i]             # list[int]
                all_objs_id    = all_objs_ids[i][-1]           # list[int] or (num_objs,)
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
                gts   = torch.stack(matched_gt)  # (P, 4) if tensor else convert
                if not isinstance(gts, torch.Tensor):
                    gts = torch.tensor(gts, device=preds.device)
                else:
                    gts = gts.to(preds.device)

                accuracy = (preds.argmax(dim=1) == gts).float().mean().item()
                cls_accuracy += accuracy
                risk_sample_cnt += 1

                risk_type = pred_risk_type[:N].argmax(dim=1).cpu().numpy()
                for rt in risk_type:
                    if rt == 0: risk_type_cnt['OBS'] += 1
                    elif rt == 1: risk_type_cnt['OCC'] += 1
                    elif rt == 2: risk_type_cnt['I'] += 1
                    elif rt == 3: risk_type_cnt['C'] += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                cur_acc = (cls_accuracy / risk_sample_cnt * 100) if risk_sample_cnt > 0 else 0.0
                print(f"Batch [{batch_idx + 1}/{len(test_loader)}] | Evaluated Samples: {risk_sample_cnt} | Current Accuracy: {cur_acc:.2f}%", flush=True)

    acc = (cls_accuracy / risk_sample_cnt) if risk_sample_cnt > 0 else 0.0
    print("\n=======================================================", flush=True)
    print(f"FINAL RESULT - Stage 1 Classifier Validation Accuracy: {acc * 100:.2f}% ({acc:.4f})", flush=True)
    print(f"Total Risk Samples Evaluated: {risk_sample_cnt}", flush=True)
    print("Risk Type Predictions Count:", flush=True)
    for risk_type, count in risk_type_cnt.items():
        print(f"  - {risk_type}: {count}", flush=True)
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
