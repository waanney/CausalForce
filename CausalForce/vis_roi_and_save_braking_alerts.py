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
import imageio



def save_gif_with_keyframes(frames, slow_indices, base_dt=0.2, slow_dt=0.6, out="out.gif"):
    durations = [slow_dt if i in set(slow_indices) else base_dt for i in range(len(frames))]
    imageio.mimsave(out, frames, duration=durations, loop=0)  
    return out

class InferenceModule(pl.LightningModule):
    def __init__(self, model, cp_ckpt):
        super().__init__()
        self.model = model
        self.coverage = 0.9

        self.class_cps = cp_ckpt['saocp_class']
        print(self.class_cps)
        self.index_to_class = {0:'OBS', 1:'OCC', 2:'I', 3:'C'}

        self.scenario_id = None
        self.brake_counts = 0
        self.vis_path = '/path/to/your/vis_dir'  # please set your visualization path
        self.draw_imgs = []
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

        label_risk_interval_H8 = batch['risk_interval_H8']
        all_obj_dist = batch['all_obj_dist']
        target_img = batch['target_img']
        scenario_ids = batch['scenario_id']
        brakes = batch['brake']
            
        B, T, C, H, W = front_imgs.shape

        outputs = self.model(front_imgs, all_objs_bbs)



        for i in range(B):
            
            brake = brakes[i]
            target_img_i = target_img[i]   # (3, H, W), tensor

 
            draw = ImageDraw.Draw(target_img_i)   
            self.scenario_id = scenario_ids[i]

            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            pred_risk_type     = outputs["risk_type"][i]           # (num_preds, 4)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
            all_objs_bb = all_objs_bbs[i][-1] 
            obj_dist = all_obj_dist[i]           # (num_objs,)
   
            isCaculated = False
            for j, obj_id in enumerate(all_objs_id):
                if obj_id in gt_risk_ids:
                    # risk objects                              
                    pred_cls = self.index_to_class[pred_risk_type[j].argmax().item()]
                    
                    q_vec = torch.tensor(
                        [self.class_cps[pred_cls].predict(horizon=t+1)[1]   
                        for t in range(8)],
                        device=pred_risk_score_H8.device     # (8,)
                    )
                    
                    pred_risk_mask = (pred_risk_score_H8[j] >= (1 - q_vec)).long()

                    idx = gt_risk_ids.index(obj_id)
                    gt_risk_mask = gt_risk_score_H8[idx]            # (8,)
                    if gt_risk_mask.sum() > 0:
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="red", width=12)
                    

                    if obj_dist[j] < 10 and pred_risk_mask.sum() >= 0 :
                        bbox = all_objs_bb[j].cpu().numpy().astype(int).tolist()
                        draw.rectangle(bbox, outline="green", width=6)
                        if isCaculated == False:
                            isCaculated = True
                            self.brake_counts += 1

                else:
                    # non-risk objects
                    pred_cls = self.index_to_class[pred_risk_type[j].argmax().item()]
                    
                    q_vec = torch.tensor(
                        [self.class_cps[pred_cls].predict(horizon=t+1)[1]  
                        for t in range(8)],
                        device=pred_risk_score_H8.device     # (8,)
                    )
                    
                    pred_risk_mask = (pred_risk_score_H8[j] >= (1 - q_vec)).long()

                    if obj_dist[j] < 10 and pred_risk_mask.sum() >= 0 :
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
            save_gif_with_keyframes(self.draw_imgs[0:], slow_indices=list(range(0)), base_dt=0.2, slow_dt=0.5, out="/".join([self.vis_path, self.scenario_id+"_slow.gif"]))
        
            np.save("/".join([self.vis_path, self.scenario_id+"_brake.npy"]), np.array(self.brake_list))

        elif self.mode == 'metric':
            print('Total scenario:', self.scenario_counts)
            print('Average brake Count:', self.brake_counts / self.scenario_counts)
            print('Average mis-alignment brake Count:', self.mis_align_brake_counts / self.scenario_counts)

        

    def configure_optimizers(self):
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/path/to/your/vis_scenario/') 
    parser.add_argument("--batch_size", type=int, default=10)
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
    #print(checkpoint)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    inference_module = InferenceModule(model=model, cp_ckpt=checkpoint)
    inference_module.scenario_counts = len(os.listdir(args.data_root))
    inference_module.mode = args.mode

    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
