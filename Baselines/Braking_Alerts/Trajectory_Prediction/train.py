

import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin 
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model

class TP(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.model = GCN_model()
        self.w_traj = 1.0
        
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        all_objs_id_curr = batch['all_objects_id_curr']

        label_traj_H8_normalized = batch['traj_H8_normalized']

        outputs = self.model(front_imgs, all_objs_bbs)
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_loss_traj = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape
        for i in range(B):
            pred_traj_x_H8_normalized = outputs["traj_x"][i]        # (num_preds, 8)
            pred_traj_y_H8_normalized = outputs["traj_y"][i]        # (num_preds, 8)

            gt_traj_H8_normalized = label_traj_H8_normalized[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
            objs_id_curr = all_objs_id_curr[i]           # list[int] or (num_objs,)

            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_traj_i = pred_traj_x_H8_normalized.sum() * 0.0

            else:
                loss_traj_i = torch.tensor(0.0, device=self.device)
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in objs_id_curr:
                        idx = objs_id_curr.index(obj_id)
                        gt_traj_H8_normalized_curr_obj = gt_traj_H8_normalized[idx]
                        pred_H = gt_traj_H8_normalized_curr_obj.shape[0]
                        pred_traj_x_H8_normalized_obj = pred_traj_x_H8_normalized[j][:pred_H]  # Truncate to the length of ground truth
                        pred_traj_y_H8_normalized_obj = pred_traj_y_H8_normalized[j][:pred_H]
                        pred_traj = torch.stack((pred_traj_x_H8_normalized_obj, pred_traj_y_H8_normalized_obj), dim=1)  # (pred_H, 2)
                        loss_traj_i += F.mse_loss(pred_traj, gt_traj_H8_normalized_curr_obj, reduction='mean')

            total_loss += (self.w_traj*loss_traj_i)
            total_loss_traj += self.w_traj*loss_traj_i
    
        total_loss = total_loss / B
        total_loss_traj = total_loss_traj / B

        self.log("val_total_loss", total_loss, prog_bar=True)
        self.log("val_total_loss_traj", total_loss_traj, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        all_objs_id_curr = batch['all_objects_id_curr']

        label_traj_H8_normalized = batch['traj_H8_normalized']

        outputs = self.model(front_imgs, all_objs_bbs)
        
        val_total_loss = torch.tensor(0.0, device=self.device)
        val_total_loss_traj = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape
        for i in range(B):
            pred_traj_x_H8_normalized = outputs["traj_x"][i]        # (num_preds, 8) 
            pred_traj_y_H8_normalized = outputs["traj_y"][i]        # (num_preds, 8)

            gt_traj_H8_normalized = label_traj_H8_normalized[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
            objs_id_curr = all_objs_id_curr[i]           # list[int] or (num_objs,)

            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_traj_i = pred_traj_x_H8_normalized.sum() * 0.0

            else:
                loss_traj_i = torch.tensor(0.0, device=self.device)
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in objs_id_curr:
                        idx = objs_id_curr.index(obj_id)
                        gt_traj_H8_normalized_curr_obj = gt_traj_H8_normalized[idx]
                        pred_H = gt_traj_H8_normalized_curr_obj.shape[0]
                        pred_traj_x_H8_normalized_obj = pred_traj_x_H8_normalized[j][:pred_H]  # Truncate to the length of ground truth
                        pred_traj_y_H8_normalized_obj = pred_traj_y_H8_normalized[j][:pred_H]
                        pred_traj = torch.stack((pred_traj_x_H8_normalized_obj, pred_traj_y_H8_normalized_obj), dim=1)  # (pred_H, 2) in (x, y) format and normalized to 0~1 (W=900,H=256)
                        loss_traj_i += F.mse_loss(pred_traj, gt_traj_H8_normalized_curr_obj, reduction='mean')


            val_total_loss += (self.w_traj*loss_traj_i)
            val_total_loss_traj += self.w_traj*loss_traj_i
 

        val_total_loss = val_total_loss / B
        val_total_loss_traj = val_total_loss_traj / B

        self.log("val_total_loss", val_total_loss, prog_bar=True)
        self.log("val_total_loss_traj", val_total_loss_traj, prog_bar=True)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='trajectory_prediction', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=12, help='Number of train epochs.')
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

    TP_Model = TP(args.lr)
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

    trainer.fit(TP_Model, dataloader_train, dataloader_val)



