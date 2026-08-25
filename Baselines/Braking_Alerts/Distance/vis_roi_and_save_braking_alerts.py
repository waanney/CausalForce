import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import MultipleRisksDataset, custom_collate_fn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
    
class InferenceModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.draw_imgs = []
        self.brake_list = []
        
        self.vis_path = 'path/to/your/vis_dir'  # please set your visualization path
        self.scenario_counts = 0
        self.scenario_id = None
        self.brake_counts = 0
        self.mis_align_brake_counts = 0
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
        all_obj_dist = batch['all_obj_dist']
        brakes = batch['brake']

        B, T, C, H, W = front_imgs.shape
    
        for i in range(B):
            
            self.scenario_id = scenario_ids[i]
            brake = brakes[i]

            target_img_i = target_img[i]   # (3, H, W), tensor
            draw = ImageDraw.Draw(target_img_i)   

            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
            all_objs_bb = all_objs_bbs[i][-1]           # (num_objs, 4)
            obj_dist = all_obj_dist[i]

            isCaculated = False

            for j, obj_id in enumerate(all_objs_id):
                if obj_id in gt_risk_ids:
                    # risk objects                                    
                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx]            # (8,)
                    if gt_risk_mask.sum() > 0:
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="red", width=12)

                    if obj_dist[j] < 10:
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="green", width=6)  
                        if isCaculated == False:
                            isCaculated = True
                            self.brake_counts += 1
                else:
                    # non-risk objects
                    if obj_dist[j] < 10:
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="green", width=6)
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

    # one scenario for visualization; all scenarios for metrics calculation
    parser.add_argument("--mode", type=str, default='vis_save', choices=['vis_save', 'metric'], help='Visualization or Metrics calculation')
    
    args = parser.parse_args()

    
    test_set = MultipleRisksDataset(data_root=args.data_root)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    
    inference_module = InferenceModule()
    inference_module.mode = args.mode   
    inference_module.scenario_counts = len(os.listdir(args.data_root))
    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
