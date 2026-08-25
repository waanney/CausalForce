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
from inference import pic_star
import numpy as np


class CausalInferenceModule(pl.LightningModule):
    def __init__(self, model, cp_ckpt):
        super().__init__()
        self.model = model
        self.coverage = 0.9
        self.class_cps = cp_ckpt['saocp_class']
        print(self.class_cps)
        self.index_to_class = {0: 'OBS', 1: 'OCC', 2: 'I', 3: 'C'}

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

        B, T, C, H, W = front_imgs.shape
        outputs = self.model(front_imgs, all_objs_bbs)

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]
            pred_risk_type = outputs["risk_type"][i]
            gt_risk_ids = label_risk_ids[i]
            gt_risk_score_H8 = label_risk_interval_H8[i]
            all_objs_id = all_objs_ids[i][-1]

            if len(all_objs_id) > 0:
                self.total_sample_cnt += 1
            if len(gt_risk_score_H8) > 0:
                self.total_risk_sample_cnt += 1

            for j, obj_id in enumerate(all_objs_id):
                pred_cls = self.index_to_class[
                    pred_risk_type[j].argmax().item()]
                q_vec = torch.tensor(
                    [self.class_cps[pred_cls].predict(horizon=t + 1)[1]
                     for t in range(8)],
                    device=pred_risk_score_H8.device)

                pred_risk_mask = (
                    pred_risk_score_H8[j] >= (1 - q_vec)).long()

                if obj_id in gt_risk_ids:
                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx]

                    if gt_risk_mask.sum() == 0:
                        continue

                    self.total_gt_risk_obj_cnt += 1
                    self.total_risk_obj_pred_tube_volume += \
                        pred_risk_mask.sum().item()
                    self.total_gt_risk_obj_tube_volume += \
                        gt_risk_mask.sum().item()

                    pred = pred_risk_mask.bool()
                    gt = gt_risk_mask.bool()
                    intersection = (pred & gt).sum().float()
                    union = (pred | gt).sum().float()
                    iou = intersection / (union + 1e-6)
                    self.iou += iou

                    t_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()
                    t_gt = (gt_risk_mask[1:] != gt_risk_mask[:-1]).sum().item()
                    frag_pen = 1.0 - (np.abs(t_pred - t_gt) / 7.0)
                    if np.abs(t_pred - t_gt) >= 0:
                        self.trasition_cnt += 1
                        self.fragmented_prediction_penalty += frag_pen

                    d_pen = pic_star(pred_risk_mask, gt_risk_mask, mode="detect")
                    e_pen = pic_star(pred_risk_mask, gt_risk_mask, mode="release")
                    self.detect_penalty_pic += d_pen
                    self.release_penalty_pic += e_pen
                    self.risk_interval_iou_pic += (
                        iou * (0.5 * (d_pen + e_pen) + frag_pen) / 2.0)

                    if torch.all(pred_risk_mask[gt_risk_mask == 1] == 1):
                        self.covered += 1
                else:
                    self.total_non_risk_obj_pred_tube_volume += \
                        pred_risk_mask.sum().item()
                    t_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()
                    frag_pen = 1.0 - (np.abs(t_pred) / 7.0)
                    if np.abs(t_pred) >= 0:
                        self.trasition_cnt += 1
                        self.fragmented_prediction_penalty += frag_pen

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
    parser.add_argument("--checkpoint", type=str,
                        default='/path/to/your/causal_checkpoint.ckpt')
    args = parser.parse_args()

    test_set = MultipleRisksDataset(data_root=args.data_root)
    print(len(test_set))
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, collate_fn=custom_collate_fn)

    model = CausalGCN_model()

    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    inference_module = CausalInferenceModule(model=model, cp_ckpt=checkpoint)
    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
