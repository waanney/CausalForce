

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


class BP(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.model = GCN_model()
        self.w_bp = 1.0
        
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        front_imgs = batch['front_imgs']
        all_objs_bbs = batch['all_objs_bbs']
        all_objs_ids = batch['all_objs_id']
        label_risk_ids = batch['risk_id']
        label_risk_interval = batch['risk_interval_H8']

        outputs = self.model(front_imgs, all_objs_bbs)
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_loss_bp = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape
        for i in range(B):
            isInfluenced = outputs["isInfluenced"][i]        # (num_preds, 8)

            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score   = label_risk_interval[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)

            if len(all_objs_id) == 0:
                loss_bp_i = isInfluenced.sum() * 0.0

            elif len(gt_risk_ids) == 0:
                gt_bp = torch.ones_like(isInfluenced)  # No risks -> go
                loss_bp_i = F.binary_cross_entropy(isInfluenced, gt_bp, reduction='mean')

            else:
                behavior = [(s > 0).any() for s in gt_risk_score]
                behavior_tensor = torch.tensor(behavior, device=self.device)
                if behavior_tensor.sum() > 0:   # stop
                    gt_bp = torch.zeros_like(isInfluenced)
                    loss_bp_i = F.binary_cross_entropy(isInfluenced, gt_bp, reduction='mean')
                else:   # go
                    gt_bp = torch.ones_like(isInfluenced)
                    loss_bp_i = F.binary_cross_entropy(isInfluenced, gt_bp, reduction='mean')

            total_loss += (self.w_bp*loss_bp_i)
            total_loss_bp += self.w_bp*loss_bp_i
   
        total_loss = total_loss / B
        total_loss_bp = total_loss_bp / B

        self.log("total_loss", total_loss, prog_bar=True)
        self.log("total_loss_bp", total_loss_bp, prog_bar=True)

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
        label_risk_interval = batch['risk_interval_H8']

        outputs = self.model(front_imgs, all_objs_bbs)
        
        val_total_loss = torch.tensor(0.0, device=self.device)
        val_total_loss_bp = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape
        for i in range(B):
            isInfluenced = outputs["isInfluenced"][i]        # (num_preds, 8)

            gt_risk_ids        = label_risk_ids[i]           # list[int]
            gt_risk_score   = label_risk_interval[i]         # (num_gt, 8)

            all_objs_id        = all_objs_ids[i][-1]         # list[int] or (num_objs,)


            if len(all_objs_id) == 0:
                loss_bp_i = isInfluenced.sum() * 0.0

            elif len(gt_risk_ids) == 0:
                gt_bp = torch.ones_like(isInfluenced)  # No risks -> go
                loss_bp_i = F.binary_cross_entropy(isInfluenced, gt_bp, reduction='mean')

            else:
                behavior = [(s > 0).any() for s in gt_risk_score]
                behavior_tensor = torch.tensor(behavior, device=self.device)
                if behavior_tensor.sum() > 0: # stop
                    gt_bp = torch.zeros_like(isInfluenced)
                    loss_bp_i = F.binary_cross_entropy(isInfluenced, gt_bp, reduction='mean')
                else:  # go
                    gt_bp = torch.ones_like(isInfluenced)
                    loss_bp_i = F.binary_cross_entropy(isInfluenced, gt_bp, reduction='mean')

            val_total_loss += (self.w_bp*loss_bp_i)
            val_total_loss_bp += self.w_bp*loss_bp_i

        val_total_loss =  val_total_loss / B
        val_total_loss_bp = val_total_loss_bp / B

        self.log("val_total_loss", val_total_loss, prog_bar=True)
        self.log("val_total_loss_bp", val_total_loss_bp, prog_bar=True)
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='behavior_prediction', help='Unique experiment identifier.')
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

    BP_Model = BP(args.lr)
    checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_total_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_total_loss:.3f}")
	
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=args.logdir,
                                            gpus = args.gpus,
                                            accelerator='ddp',
                                            sync_batchnorm=True,
                                            plugins=DDPPlugin(find_unused_parameters=True),
                                            profiler='simple',
                                            benchmark=True,
                                            log_every_n_steps=1,
                                            flush_logs_every_n_steps=2,
                                            callbacks=[checkpoint_callback,],
                                            check_val_every_n_epoch = args.val_every,
                                            max_epochs = args.epochs
                                            )

    trainer.fit(BP_Model, dataloader_train, dataloader_val)



