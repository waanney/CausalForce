import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_yolo_bbox_as_input import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
import numpy as np
from torchvision.ops import box_iou

    
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

def match_boxes(gt_boxes: torch.Tensor,
                pred_boxes: torch.Tensor,
                iou_thresh: float = 0.4,
                method: str = "hungarian"):
    """
    gt_boxes:   (G,4)  [x1,y1,x2,y2] 
    pred_boxes: (P,4)  [x1,y1,x2,y2]  
    iou_thresh: float — IoU threshold for a valid match
    method:     "hungarian" or "greedy"
    """
    device = gt_boxes.device
    G = gt_boxes.size(0)
    P = pred_boxes.size(0)

    if G == 0 or P == 0:
        return [], list(range(G)), list(range(P))

    # IoU matrix (G,P)
    ious = box_iou(gt_boxes, pred_boxes)  # on same device

    matches = []
    if method.lower() == "hungarian":
        
        try:
            from scipy.optimize import linear_sum_assignment
            cost = (1.0 - ious).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                i = float(ious[r, c].item())
                if i >= iou_thresh:
                    matches.append((int(r), int(c), i))
        except ImportError:
            method = "greedy"

    if method.lower() == "greedy":
        iou_mat = ious.clone()
        used_g = set()
        used_p = set()
        while True:
            # find max IoU
            val, idx = torch.max(iou_mat.view(-1), dim=0)
            val = float(val.item())
            if val < iou_thresh:
                break
            r = int(idx.item() // P)
            c = int(idx.item() %  P)
            matches.append((r, c, val))
            used_g.add(r); used_p.add(c)
            # avoid re-using
            iou_mat[r, :] = -1.0
            iou_mat[:, c] = -1.0

    matched_g = {g for g,_,_ in matches}
    matched_p = {p for _,p,_ in matches}
    unmatched_gt   = [i for i in range(G) if i not in matched_g]
    unmatched_pred = [j for j in range(P) if j not in matched_p]
    return matches, unmatched_gt, unmatched_pred

class InferenceModule(pl.LightningModule):
    def __init__(self, model, cp_ckpt):
        super().__init__()
        self.model = model
        self.coverage = 0.9

        self.class_cps = cp_ckpt['saocp_class']
        print(self.class_cps)
        self.index_to_class = {0:'OBS', 1:'OCC', 2:'I', 3:'C'}

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
        label_risk_bbs = batch['label_risk_bbs']
        label_risk_interval_H8 = batch['risk_interval_H8']
            
        B, T, C, H, W = front_imgs.shape

        outputs = self.model(front_imgs, all_objs_bbs)

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            pred_risk_type     = outputs["risk_type"][i]           # (num_preds, 4)
            
            gt_risk_bbs      = label_risk_bbs[i]             # list[[x1,y1,x2,y2], ...]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)

            objs_bbs       = all_objs_bbs[i][-1]           # (num_objs, 4)

            if len(objs_bbs) > 0:
                self.total_sample_cnt += 1

            if len(gt_risk_score_H8) > 0:
                self.total_risk_sample_cnt += 1


            matches, un_g, un_p = match_boxes(gt_risk_bbs, objs_bbs, iou_thresh=0.5, method="hungarian")

            for gt_idx, pred_idx, _ in matches:   
                # risk object                           
                pred_cls = self.index_to_class[pred_risk_type[pred_idx].argmax().item()]

                q_vec = torch.tensor(
                    [self.class_cps[pred_cls].predict(horizon=t+1)[1]  
                    for t in range(8)],
                    device=pred_risk_score_H8.device     # (8,)
                )
                
                pred_risk_mask = (pred_risk_score_H8[pred_idx] >= (1 - q_vec)).long()

                gt_risk_mask = gt_risk_score_H8[gt_idx]            # (8,)
                
                if gt_risk_mask.sum() == 0:
                    continue

                self.total_gt_risk_obj_cnt += 1 
                self.total_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                self.total_gt_risk_obj_tube_volume += gt_risk_mask.sum().item()
                
                pred = pred_risk_mask.bool()
                gt   = gt_risk_mask.bool()
                intersection = (pred & gt).sum().float()      # scalar
                union        = (pred | gt).sum().float()      # scalar
                eps = 1e-6
                iou = intersection / (union + eps)

                self.iou += iou 

                transitions_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()   
                transitions_gt = (gt_risk_mask[1:] != gt_risk_mask[:-1]).sum().item()    
                
                
                fragmented_prediction_pen = 1.0 - (np.abs(transitions_pred-transitions_gt) / 7.0)      # T=8 → divide by 7
                if np.abs(transitions_pred-transitions_gt) >= 0:                     
                    self.trasition_cnt += 1
                    self.fragmented_prediction_penalty += fragmented_prediction_pen

                d_pen_pic = pic_star(pred_risk_mask, gt_risk_mask, mode="detect")
                e_pen_pic = pic_star(pred_risk_mask, gt_risk_mask, mode="release")
                self.detect_penalty_pic += d_pen_pic
                self.release_penalty_pic += e_pen_pic
                self.risk_interval_iou_pic += (iou * (0.5*(d_pen_pic + e_pen_pic) + fragmented_prediction_pen) / 2.0)

                if torch.all(pred_risk_mask[gt_risk_mask == 1] == 1):
                    self.covered += 1

            for pred_idx in un_p:   
                # non risk object
                
                pred_cls = self.index_to_class[pred_risk_type[pred_idx].argmax().item()]

                q_vec = torch.tensor(
                    [self.class_cps[pred_cls].predict(horizon=t+1)[1]  
                    for t in range(8)],
                    device=pred_risk_score_H8.device     # (8,)
                )
                pred_risk_mask = (pred_risk_score_H8[pred_idx] >= (1 - q_vec)).long()

                
                self.total_non_risk_obj_pred_tube_volume += pred_risk_mask.sum().item()
                
                transitions_pred = (pred_risk_mask[1:] != pred_risk_mask[:-1]).sum().item()   
                transitions_gt = 0.0 
                
                fragmented_prediction_pen = 1.0 - (np.abs(transitions_pred-transitions_gt) / 7.0)      # T=8 → divide by 7
                if np.abs(transitions_pred-transitions_gt) >= 0:                     
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

    inference_module = InferenceModule(model=model, cp_ckpt=checkpoint)

    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
