

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
from online_conformal.saocp import SAOCP
import pandas as pd

def compute_nonconformity(preds: torch.Tensor,
                          gts: torch.Tensor,
                          method: str = "class_cond",
                          lam: float = 0.5) -> torch.Tensor:
    """
    preds, gts: shape (T,), preds∈[0,1], gts∈{0,1}
    method: "absolute"    → |p-g|
            "class_cond"  → g==1:1-p, g==0:p
            "adjacent"    → |p-g| + λ*(err_{t-1}+err_{t+1})
    """
    if method == "absolute":
        return torch.abs(preds - gts)

    elif method == "class_cond":
        return torch.where(gts == 1,
                           1.0 - preds,
                           preds)

    elif method == "adjacent":
        err = torch.abs(preds - gts)
        left  = torch.cat([err[:1], err[:-1]])
        right = torch.cat([err[1:], err[-1:]])
        return err + lam * (left + right)

    else:
        raise ValueError(f"Unknown method {method}")


class OnlineCP(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.coverage = 0.9
        
        self.cp = SAOCP(model=None, train_data=None, max_scale=1.0, coverage=self.coverage, horizon=8)
        self.model = GCN_model()
        self.nc_method = "absolute"
        
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_interval_H8 = batch['risk_interval_H8']

        outputs = self.model(front_imgs, all_objs_bbs)
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_loss_score = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)

            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_scoreH8_i = pred_risk_score_H8.sum() * 0.0

            elif len(gt_risk_ids) == 0:
                gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N])  
                loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')

                for obj_pred, obj_gt in zip(pred_risk_score_H8[:N], gt_risk_score_H8):
                    # compute nonconformity vector (8,)
                    nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                    for t in range(8):
                        nc_t = pd.Series([nc[t].item()], dtype=float) 
                        self.cp.update(
                            ground_truth=nc_t,
                            forecast=pd.Series([0], dtype=float),
                            horizon=t+1
                        )

            else:
                matched_pred, matched_gt = [], []
                non_risk_pred =[]
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_score_H8[j])          # (8,)
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_score_H8[idx])            # (8,)
                    else:
                        non_risk_pred.append(pred_risk_score_H8[j])
                        
                if len(matched_pred) == 0:
                    gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N]) 
                    loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')
                    
                    for obj_pred, obj_gt in zip(pred_risk_score_H8[:N], gt_risk_score_H8):
                        # compute nonconformity vector (8,)
                        nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                        for t in range(8):
                            nc_t = pd.Series([nc[t].item()], dtype=float)
                            self.cp.update(
                                ground_truth=nc_t,
                                forecast=pd.Series([0], dtype=float),
                                horizon=t+1
                            )
                   
                else:
                    if len(non_risk_pred) == 0:
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean')

                        for obj_pred, obj_gt in zip(matched_pred, matched_gt):
                            # compute nonconformity vector (8,)
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)
                                self.cp.update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )
                    else:
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        
                        non_risk_preds = torch.stack(non_risk_pred)
                        non_risk_gt = torch.zeros_like(non_risk_preds)

                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean') + \
                                         F.binary_cross_entropy(non_risk_preds, non_risk_gt, reduction='mean')
                        

                        for obj_pred, obj_gt in zip(non_risk_preds, non_risk_gt):
                            # compute nonconformity vector (8,)
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float) 
                                self.cp.update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

                        for obj_pred, obj_gt in zip(matched_pred, matched_gt):
                            # compute nonconformity vector (8,)
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float) 
                                self.cp.update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

            total_loss += (loss_scoreH8_i) 
            total_loss_score += loss_scoreH8_i
         
        total_loss = total_loss / B
        total_loss_score = total_loss_score / B

        self.log("total_loss", total_loss, prog_bar=True)
        self.log("total_loss_score", total_loss_score, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_interval_H8 = batch['risk_interval_H8']

        outputs = self.model(front_imgs, all_objs_bbs)
        
        val_total_loss = torch.tensor(0.0, device=self.device)
        val_total_loss_score = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape
        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)

            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_scoreH8_i = pred_risk_score_H8.sum() * 0.0

            elif len(gt_risk_ids) == 0:
                gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N]) 
                loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')

                for obj_pred, obj_gt in zip(pred_risk_score_H8[:N], gt_risk_score_H8):
                    # compute nonconformity vector (8,)
                    nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                    for t in range(8):
                        nc_t = pd.Series([nc[t].item()], dtype=float)  
                        self.cp.update(
                            ground_truth=nc_t,
                            forecast=pd.Series([0], dtype=float),
                            horizon=t+1
                        )

            else:
                matched_pred, matched_gt = [], []
                non_risk_pred =[]
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_score_H8[j])          # (8,)
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_score_H8[idx])            # (8,)
                    else:
                        non_risk_pred.append(pred_risk_score_H8[j])
                        
                if len(matched_pred) == 0:
                    gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N]) 
                    loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')

                    for obj_pred, obj_gt in zip(pred_risk_score_H8[:N], gt_risk_score_H8):
                        # compute nonconformity vector (8,)
                        nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                        for t in range(8):
                            nc_t = pd.Series([nc[t].item()], dtype=float) 
                            self.cp.update(
                                ground_truth=nc_t,
                                forecast=pd.Series([0], dtype=float),
                                horizon=t+1
                            )
                
                else:

                    if len(non_risk_pred) == 0:
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean')

                        for obj_pred, obj_gt in zip(matched_pred, matched_gt):
                            # compute nonconformity vector (8,)
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)
                                self.cp.update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )
                    else:
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        
                        non_risk_preds = torch.stack(non_risk_pred)
                        non_risk_gt = torch.zeros_like(non_risk_preds)

                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean') + \
                                         F.binary_cross_entropy(non_risk_preds, non_risk_gt, reduction='mean')
                        

                        for obj_pred, obj_gt in zip(non_risk_preds, non_risk_gt):
                            # compute nonconformity vector (8,)
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float) 
                                self.cp.update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

                        for obj_pred, obj_gt in zip(matched_pred, matched_gt):
                            # compute nonconformity vector (8,)
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)
                                self.cp.update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

            val_total_loss += (loss_scoreH8_i)
            val_total_loss_score += loss_scoreH8_i

  
        val_total_loss = val_total_loss / B
        val_total_loss_score = val_total_loss_score / B


        self.log("val_total_loss", val_total_loss, prog_bar=True)
        self.log("val_total_loss_score", val_total_loss_score, prog_bar=True)

	
    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint['saocp'] = self.cp
        
        return checkpoint

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='OnlineCP', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    train_set = MultipleRisksDataset(data_root='/path/to/your/training_data/')
    print(len(train_set))
    val_set = MultipleRisksDataset(data_root='/path/to/your/validation_data/')
    print(len(val_set))

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    OnlineCP_Model = OnlineCP(args.lr)
    checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_total_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_total_loss:.3f}")
	
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=args.logdir,
                                            gpus = args.gpus,
                                            accelerator='ddp',
                                            sync_batchnorm=True,
                                            plugins=DDPPlugin(find_unused_parameters=False),
                                            profiler='simple',
                                            benchmark=True,
                                            log_every_n_steps=1,
                                            flush_logs_every_n_steps=2,
                                            callbacks=[checkpoint_callback,],
                                            check_val_every_n_epoch = args.val_every,
                                            max_epochs = args.epochs
                                            )

   
    trainer.fit(OnlineCP_Model, dataloader_train, dataloader_val)



