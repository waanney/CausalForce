import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional

def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return a[0]*b[1] - a[1]*b[0]

def _segments_intersect_with_point(p1: np.ndarray, p2: np.ndarray,
                                   q1: np.ndarray, q2: np.ndarray,
                                   eps: float = 1e-9) -> Tuple[bool, Optional[np.ndarray]]:
    r = p2 - p1
    s = q2 - q1
    rxs = _cross(r, s)
    qp = q1 - p1
    qpxr = _cross(qp, r)

    if abs(rxs) > eps:
        t = _cross(qp, s) / rxs
        u = _cross(qp, r) / rxs
        if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
            inter = p1 + t * r
            return True, inter
        return False, None

    if abs(qpxr) > eps:
        return False, None

    r2 = np.dot(r, r)
    if r2 < eps:
        def _on_segment(a, b, x):
            return (min(a[0], b[0]) - eps <= x[0] <= max(a[0], b[0]) + eps and
                    min(a[1], b[1]) - eps <= x[1] <= max(a[1], b[1]) + eps)
        if _on_segment(q1, q2, p1):
            return True, p1.copy()
        return False, None

    t0 = np.dot(q1 - p1, r) / r2
    t1 = np.dot(q2 - p1, r) / r2
    tmin, tmax = sorted([t0, t1])
    if tmax < -eps or tmin > 1 + eps:
        return False, None
    t_mid = np.clip(0.5 * (max(0.0, tmin) + min(1.0, tmax)), 0.0, 1.0)
    inter = p1 + t_mid * r
    return True, inter

def _point_in_rect(pt: np.ndarray,
                   rect_tl: Tuple[float, float],
                   rect_br: Tuple[float, float],
                   eps: float = 1e-9) -> bool:
    x0, y0 = rect_tl
    x1, y1 = rect_br
    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
    return (xmin - eps <= pt[0] <= xmax + eps) and (ymin - eps <= pt[1] <= ymax + eps)

def _segment_intersects_rect(p1: np.ndarray, p2: np.ndarray,
                             rect_tl: Tuple[float, float],
                             rect_br: Tuple[float, float]) -> bool:
    # point in rectangle
    if _point_in_rect(p1, rect_tl, rect_br) or _point_in_rect(p2, rect_tl, rect_br):
        return True
    # edge intersects rectangle
    x0, y0 = rect_tl
    x1, y1 = rect_br

    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
    tl = np.array([xmin, ymin], float)
    tr = np.array([xmax, ymin], float)
    bl = np.array([xmin, ymax], float)
    br = np.array([xmax, ymax], float)

    edges = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
    for q1, q2 in edges:
        ok, _ = _segments_intersect_with_point(p1, p2, q1, q2)
        if ok:
            return True
    return False

def intersects_with_region_single(
    pred_traj_x_H8_normalized_obj: np.ndarray,
    pred_traj_y_H8_normalized_obj: np.ndarray,
    w: int = 900,
    h: int = 256,
    rect_tl: Tuple[float, float] = (300.0, 128.0),  # upper left
    rect_br: Tuple[float, float] = (600.0, 256.0)   # bottom right
) -> bool:
    """
    If object's 2D trajectory hits the rectangular region in front of the ego vehicle, return True/False.
    """
    x_norm = np.clip(np.asarray(pred_traj_x_H8_normalized_obj, dtype=float), 0.0, 1.0)
    y_norm = np.clip(np.asarray(pred_traj_y_H8_normalized_obj, dtype=float), 0.0, 1.0)
    x_px = x_norm * float(w)
    y_px = y_norm * float(h)

    for t in range(len(x_px) - 1):
        p1 = np.array([x_px[t],   y_px[t]], dtype=float)
        p2 = np.array([x_px[t+1], y_px[t+1]], dtype=float)
        if _segment_intersects_rect(p1, p2, rect_tl, rect_br):
            return True
    return False

class InferenceModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.draw_imgs = []
        self.vis_path = 'path/to/your/vis_dir'  # please set your visualization path
        self.IMG_H = 256
        self.IMG_W = 900
        self.scenario_id = None
        self.brake_counts = 0
        self.brake_list = []
        self.mis_align_brake_counts = 0
        self.scenario_counts = 0
        self.mode = 'vis_save'  # 'vis_save' or 'metric'

    @torch.no_grad()
    def forward(self, imgs, all_objs):
        return self.model(imgs, all_objs)
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        target_img = batch['target_img']
        scenario_ids = batch['scenario_id']

        label_risk_interval_H8 = batch['risk_interval_H8']
        all_objs_id_curr = batch['all_objects_id_curr']
        all_obj_dist = batch['all_obj_dist']
        brakes = batch['brake']

        B, T, C, H, W = front_imgs.shape

        outputs = self.model(front_imgs, all_objs_bbs)

        for i in range(B):
            
            target_img_i = target_img[i]   # (3, H, W), tensor
            draw = ImageDraw.Draw(target_img_i)   
            self.scenario_id = scenario_ids[i]
            brake = brakes[i]

            pred_traj_x_H8_normalized = outputs["traj_x"][i]        # (num_preds, 8) 
            pred_traj_y_H8_normalized = outputs["traj_y"][i]        # (num_preds, 8)
 
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
            all_objs_bb = all_objs_bbs[i][-1]           # (num_objs, 4)
            objs_id_curr = all_objs_id_curr[i]
            obj_dist = all_obj_dist[i]
   
            if len(all_objs_id) > 0:
                N = len(all_objs_id) 
               
                self.total_obj_cnt += len(all_objs_id)
                self.total_sample_cnt += 1

            if len(gt_risk_score_H8) > 0:
                self.total_risk_sample_cnt += 1

            isCaculated = False
            for j, obj_id in enumerate(all_objs_id):
                if obj_id in gt_risk_ids:
                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx]            # (8,)
                    if gt_risk_mask.sum() > 0:
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="red", width=5)

                if obj_id in objs_id_curr:
                    pred_traj_x_H8_normalized_obj = pred_traj_x_H8_normalized[j].cpu().numpy()  # (8,)
                    pred_traj_y_H8_normalized_obj = pred_traj_y_H8_normalized[j].cpu().numpy()  # (8,)
                    hit = intersects_with_region_single(
                        pred_traj_x_H8_normalized_obj,
                        pred_traj_y_H8_normalized_obj,
                        w=900, h=256,
                        rect_tl=(230, 100),   
                        rect_br=(670, 256)    
                    )

                    if obj_dist[j] < 10 and hit:
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="green", width=3)
                        if isCaculated == False:
                            isCaculated = True
                            self.brake_counts += 1
                else:
                    if obj_id in objs_id_curr:
                        pred_traj_x_H8_normalized_obj = pred_traj_x_H8_normalized[j].cpu().numpy()  # (8,)
                        pred_traj_y_H8_normalized_obj = pred_traj_y_H8_normalized[j].cpu().numpy()  # (8,)
                        hit = intersects_with_region_single(
                            pred_traj_x_H8_normalized_obj,
                            pred_traj_y_H8_normalized_obj,
                            w=900, h=256,
                            rect_tl=(230, 100),   
                            rect_br=(670, 256)   
                        )

                        if obj_dist[j] < 10 and hit:
                            bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                            draw.rectangle(bbox, outline="green", width=3)
                            if isCaculated == False:
                                isCaculated = True
                                self.brake_counts += 1

            self.draw_imgs.append(target_img_i)
            if isCaculated:
                self.brake_list.append(1)
                if brake == 0:
                    self.mis_align_brake_counts += 1
            else:
                self.brake_list.append(0)
                if brake == 1:
                    self.mis_align_brake_counts += 1
    
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        if self.mode == 'vis_save':
            self.draw_imgs[0].save(
            "/".join([self.vis_path, self.scenario_id+".gif"]),
            save_all=True,
            append_images=self.draw_imgs[1:],
            duration=200,  
            loop=0
            )

            np.save("/".join([self.vis_path, self.scenario_id+"_brake.npy"]), np.array(self.brake_list))

        elif self.mode == 'metric':
            print('Total scenario:', self.scenario_counts)
            print('Average brake Count:', self.brake_counts / self.scenario_counts)
            print('Average misalignment brake Count:', self.mis_align_brake_counts / self.scenario_counts)
    def configure_optimizers(self):
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/path/to/your/vis_scenario/')
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default='/path/to/your/checkpoint.ckpt')

    # one scenario for visualization; all scenarios for metrics calculation
    parser.add_argument("--mode", type=str, default='vis_save', choices=['vis_save', 'metric'], help='Visualization or Metrics calculation')
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

    inference_module = InferenceModule(model=model)
    inference_module.scenario_counts = len(os.listdir(args.data_root))
    inference_module.mode = args.mode   

    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
