import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import MultipleRisksDataset, custom_collate_fn
from classifier import GCN_model
import numpy as np

class InferenceModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cls_accuracy = 0.0
        self.risk_sample_cnt = 0
        self.risk_type_cnt = {'OBS':0, 'OCC':0, 'I':0, 'C':0}

    @torch.no_grad()
    def forward(self, imgs, all_objs):
        return self.model(imgs, all_objs)
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_type = batch['label_risk_type']
            
        B, T, C, H, W = front_imgs.shape

        outputs = self.model(front_imgs, all_objs_bbs)


        for i in range(B):
            pred_risk_type = outputs["risk_type"][i]        # (num_preds, 4)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
            gt_risk_types =   label_risk_type[i]          # list[int] or (num_objs,)
            
            N = len(all_objs_id)
            
            if len(all_objs_id) == 0:
               continue

            elif len(gt_risk_ids) == 0:
                continue

            else:
                matched_pred, matched_gt = [], []
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_type[j])         
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_types[idx])           
                        
                preds = torch.stack(matched_pred) # (P, 4)                   
                gts   = torch.stack(matched_gt)  # (P, 4)   
                
                accuracy = (preds.argmax(dim=1) == gts).float().mean().item()
                self.cls_accuracy += accuracy
                self.risk_sample_cnt += 1

                # compute risk type counts
                risk_type = pred_risk_type[:N].argmax(dim=1).cpu().numpy()
                for rt in risk_type:
                    if rt == 0:
                        self.risk_type_cnt['OBS'] += 1
                    elif rt == 1:
                        self.risk_type_cnt['OCC'] += 1
                    elif rt == 2:
                        self.risk_type_cnt['I'] += 1
                    elif rt == 3:
                        self.risk_type_cnt['C'] += 1


                        
    
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        print(f"Average classification accuracy: {self.cls_accuracy / self.risk_sample_cnt:.4f}")
        print("Risk type counts:")
        for risk_type, count in self.risk_type_cnt.items():
            print(f"{risk_type}: {count}")

    def configure_optimizers(self):
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/path/to/your/testing_data/')
    parser.add_argument("--batch_size", type=int, default=8)
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

    inference_module = InferenceModule(model=model)

    trainer = pl.Trainer(gpus=args.gpus)
    trainer.test(inference_module, test_dataloaders=test_loader)
