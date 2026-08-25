import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
import numpy as np
from typing import Dict, List, Any


def pic_star(pred, gt, tau=2, mode="detect"):
    """
    Parameters
    ----------
    gt   : 1-D array-like of int (0/1)   — ground-truth risk interval
    pred : 1-D array-like of int (0/1)   — model prediction
    tau  : float                         — decay constant (smaller→faster decay)
    mode : "detect" | "release"          — choose center at risk start or end

    Returns
    -------
    score ∈ [0,1]  — bigger is better
    """
    gt   = np.asarray(gt.cpu(),   dtype=np.int8)
    pred = np.asarray(pred.cpu(), dtype=np.int8)
    assert gt.shape == pred.shape, "gt & pred must have same length"
    n = len(gt)

    # find risk begin/end
    ones = np.where(gt == 1)[0]
    if len(ones) == 0:
        return 1.0

    T_s, T_e = ones[0], ones[-1]
    center   = T_s if mode == "detect" else T_e

    # compute weights
    t = np.arange(n)
    w = np.exp(-np.abs(t - center) / tau)
    W = w.sum()

    # compute f1 scores
    f1 = 1 - np.abs(gt - pred)    # gt==pred → 1；else 0

    # compute PIC score
    loss   = (w * (1 - f1)).sum() / W
    score  = 1 - loss
    return score

class EMATubeSmootherTorch:
    def __init__(self, H: int = 8, alpha: float = 0.6, clamp01: bool = True):
        self.H = H
        self.alpha = alpha
        self.clamp01 = clamp01
        self.state: Dict[int, torch.Tensor] = {}  # obj_id -> Tensor(H,)

    def reset(self):
        self.state.clear()

    def step(self, frame_tubes: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        frame_tubes: {obj_id: Tensor(8,)}  
        return:      {obj_id: Tensor(8,)}  
        """
        out: Dict[int, torch.Tensor] = {}
        for oid, z in frame_tubes.items():
            z = z.view(-1)
            H = z.shape[0]
            if H != self.H:
                if H > self.H:
                    z = z[:self.H]
                else:
                    pad = torch.zeros(self.H - H, dtype=z.dtype, device=z.device)
                    z = torch.cat([z, pad], dim=0)

            if oid not in self.state:
                self.state[oid] = z.detach()
            y_prev = self.state[oid].to(z.dtype).to(z.device).detach()  

            y = self.alpha * z + (1.0 - self.alpha) * y_prev  
            if self.clamp01:
                y = torch.clamp(y, 0.0, 1.0)

            out[oid] = y
            self.state[oid] = y.detach()
        
        return out
    
class InferenceModule(pl.LightningModule):
    def __init__(self, model, ckpt):
        super().__init__()
        self.model = model
        self.kalman_filter = ckpt['kalman_filter']
        print(self.kalman_filter)


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
            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)

            risktube_mask = (pred_risk_score_H8 >= 0.5)  # (num_preds, 8)
            obj_has_any_risk = risktube_mask.any(dim=1)
            num_predicted_risky_objs = obj_has_any_risk.sum().item()
            self.total_pred_risky_objs_cnt += num_predicted_risky_objs


            if len(all_objs_id) > 0:
                self.total_obj_cnt += len(all_objs_id)
                self.total_sample_cnt += 1

                frame_dict = {int(obj_id): pred_risk_score_H8[j] for j, obj_id in enumerate(all_objs_id)}
                smoothed_dict = self.kalman_filter.step(frame_dict)
                device = pred_risk_score_H8.device
                smoothed_H8 = torch.stack([smoothed_dict[int(obj_id)] for obj_id in all_objs_id], dim=0).to(device)  # (N, 8)
                pred_risk_score_H8 = smoothed_H8

            if len(gt_risk_score_H8) > 0:
                self.total_risk_sample_cnt += 1

            for j, obj_id in enumerate(all_objs_id):
                if obj_id in gt_risk_ids:                                  
                    pred_risk_mask = (pred_risk_score_H8[j] >= 0.5).long()         # (8,)
                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx]            # (8,)


                    self.total_gt_risk_obj_cnt += 1 
                    self.total_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                    self.total_gt_risk_obj_tube_volume += gt_risk_mask.sum().item()
                    
                    pred = pred_risk_mask.bool()
                    gt   = gt_risk_mask.bool()
                    intersection = (pred & gt).sum().float()      # scalar
                    union        = (pred | gt).sum().float()      # scalar
                    eps = 1e-6
                    iou = (intersection / (union + eps)).item()
                    self.iou += iou 

                    pred_risk_mask = (pred_risk_score_H8[j] >= 0.5).long()
                    transitions_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()    
                    transitions_gt = (gt_risk_mask[1:] != gt_risk_mask[:-1]).sum().item()   
                    
                    
                    fragmented_prediction_pen = 1.0 - (np.abs(transitions_pred-transitions_gt) / 7.0)      # T=8 → divide by 7
                    if np.abs(transitions_pred-transitions_gt) > 0:                     
                        self.trasition_cnt += 1
                        self.fragmented_prediction_penalty += fragmented_prediction_pen

                    
                    d_pen_pic = pic_star(pred_risk_mask, gt_risk_mask, mode="detect")
                    e_pen_pic = pic_star(pred_risk_mask, gt_risk_mask, mode="release")
                    self.detect_penalty_pic += d_pen_pic
                    self.release_penalty_pic += e_pen_pic
                    self.risk_interval_iou_pic += (iou * (0.5*(d_pen_pic + e_pen_pic) + fragmented_prediction_pen) / 2.0)

                    if torch.all(pred_risk_mask[gt_risk_mask == 1] == 1):
                        self.covered += 1

                else:

                    pred_risk_mask = (pred_risk_score_H8[j] >= 0.5).long()
                    self.total_non_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()

                    transitions_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()  
                    transitions_gt = 0.0 
                    
                    fragmented_prediction_pen = 1.0 - (np.abs(transitions_pred-transitions_gt) / 7.0)      # T=8 → divide by 7
                    if np.abs(transitions_pred-transitions_gt) > 0:                     
                        self.trasition_cnt += 1
                        self.fragmented_prediction_penalty += fragmented_prediction_pen
    
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        print("Average Coverage Ratio:", self.covered / self.total_gt_risk_obj_cnt if self.total_gt_risk_obj_cnt > 0 else 0.0)

        print("Average GT Tube Volume:", self.total_gt_risk_obj_tube_volume / self.total_risk_sample_cnt if self.total_risk_sample_cnt > 0 else 0.0)
        print("Average Pred Tube Volume:", ((self.total_risk_obj_pred_tube_volume / self.total_risk_sample_cnt)+(self.total_non_risk_obj_pred_tube_volume / self.total_sample_cnt)) if self.total_sample_cnt > 0 else 0.0)
        # print("Average Pred Tube Volume of Risk Obj:", self.total_risk_obj_pred_tube_volume / self.total_risk_sample_cnt if self.total_risk_sample_cnt > 0 else 0.0)
        # print("Average Tube Volume of Pred Non-Risk Obj:", self.total_non_risk_obj_pred_tube_volume / self.total_sample_cnt if self.total_sample_cnt > 0 else 0.0)

        print("Average IoU:", self.iou / self.total_gt_risk_obj_cnt if self.total_gt_risk_obj_cnt > 0 else 0.0)  
        print("Average TC:", self.fragmented_prediction_penalty / self.trasition_cnt if self.trasition_cnt > 0 else 0.0)
        print("Average BA:", 0.5*(self.detect_penalty_pic+self.release_penalty_pic) / self.total_gt_risk_obj_cnt if self.total_gt_risk_obj_cnt > 0 else 0.0)
        print("Average Risk-IOU:", self.risk_interval_iou_pic / self.total_gt_risk_obj_cnt if self.total_gt_risk_obj_cnt > 0 else 0.0)

        # print("Average Detect-BA:", self.detect_penalty_pic / self.total_gt_risk_obj_cnt if self.total_gt_risk_obj_cnt > 0 else 0.0)
        # print("Average Release-BA:", self.release_penalty_pic / self.total_gt_risk_obj_cnt if self.total_gt_risk_obj_cnt > 0 else 0)
        
    def configure_optimizers(self):
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/path/to/your/testing_data/')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default='/path/to/your/checkpoint.ckpt')
    args = parser.parse_args()

    
    test_set = MultipleRisksDataset(data_root=args.data_root)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    
    model = GCN_model()

    checkpoint = torch.load(args.checkpoint)

    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    inference_module = InferenceModule(model=model, ckpt=checkpoint)

    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
