

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
from model import GCN_LSTM_BetaBernoulli_model

def ranking_margin_loss(pos_scores: torch.Tensor,
                        neg_scores: torch.Tensor,
                        margin: float = 0.2) -> torch.Tensor:
    """
    pos_scores: (P,)  scores of risk objects (logits or raw scores)
    neg_scores: (N,)  scores of non-risk objects
    Note: if P == 0 or N == 0, return 0 while keeping the computation graph.
    """
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        # Keep the computation graph to avoid errors during .backward()
        return (pos_scores.sum() + neg_scores.sum()) * 0.0

    # Form all (p, n) pairwise margins: shape → (P, N)
    diff = margin - pos_scores.view(-1, 1) + neg_scores.view(1, -1)

    loss = F.relu(diff)      # max(0, diff)
    return loss.mean()       # can also use .sum() or take the max



def evidential_BB_loss(alpha, beta, y, eps=1e-6):
    S = alpha + beta                     # shape=(B,N,T)
    # 1) Predictive Bernoulli NLL
    nll = - ( y * torch.log(alpha/(S+eps))
            + (1-y) * torch.log(beta/(S+eps)) ).mean()
    # 2) Optional KL regularizer
    kl = ( torch.lgamma(alpha) + torch.lgamma(beta)
           - torch.lgamma(S)
           - ((alpha-1)*torch.digamma(alpha)
              + (beta-1)*torch.digamma(beta)
              - (S-2)*torch.digamma(S)) ).mean()
    
    return nll + 1e-4*kl


class Bayesian(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.var_max = 1 / 12
        self.w_evid = 1.0
        self.w_rank = 8.0     
        self.w_var  = 1.0     
        self.w_ent  = -0.35    
        self.w_ue   = 2.0     
        self.model = GCN_LSTM_BetaBernoulli_model()
        
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
        total_loss_var = torch.tensor(0.0, device=self.device)
        total_loss_rank = torch.tensor(0.0, device=self.device)


        B, T, C, H, W = front_imgs.shape

        for i in range(B):
            pred_scores = outputs["score_H8"][i]   # (num_preds, 8)
            alpha_H8    = outputs["alpha"][i]      # (num_preds, 8)
            beta_H8     = outputs["beta"][i]       # (num_preds, 8)
            var_H8      = outputs["var_H8"][i]     # (num_preds, 8)
            

            gt_ids      = label_risk_ids[i]        # list[int]
            gt_mask8    = label_risk_interval_H8[i]  # (num_gt, 8)
            all_ids     = all_objs_ids[i][-1]      # list[int] or length=N
            N = len(all_ids) 
            
            if N == 0:
                L_evid_i = torch.sum(pred_scores) * 0.0
                L_var_i   = torch.sum(pred_scores) * 0.0
                L_rank_i  = torch.sum(pred_scores) * 0.0

            elif len(gt_ids) == 0:

                easy_neg_mask = (pred_scores[:N] < 0.3)                   # 明顯 safe
                hard_mask     = ((pred_scores[:N] >= 0.3) & (pred_scores[:N] <= 0.7)) 
                pos_mask = (pred_scores[:N] > 0.7) 
                    
                L_easy = var_H8[:N][easy_neg_mask].mean() if easy_neg_mask.any()>0 else torch.tensor(0., device=self.device)
                L_hard = - var_H8[:N][hard_mask].mean() if hard_mask.any()>0 else torch.tensor(0., device=self.device)
                L_pos_var  = - var_H8[:N][pos_mask].mean() if pos_mask.any()>0 else torch.tensor(0., device=self.device)

                L_var_i = 1.5 * L_easy + 3.0 * L_hard + 1.5 * L_pos_var

                L_rank_i = torch.sum(pred_scores[:N]) * 0.0

                L_evid_i = torch.sum(pred_scores[:N]) * 0.0

            else:

                pos_mask = torch.tensor([oid in gt_ids for oid in all_ids],
                        device=self.device)
                neg_mask = ~pos_mask


                ps = pred_scores[:N][pos_mask].flatten()
                ns = pred_scores[:N][neg_mask].flatten()
                L_rank_i = ranking_margin_loss(ps, ns, margin=0.2)

                if pos_mask.any():
                    pa = alpha_H8[:N][pos_mask]      # shape (P,8)
                    pb = beta_H8 [:N][pos_mask]
  
                    idxs = [gt_ids.index(oid) for oid in all_ids if oid in gt_ids]
                    
                    gt_mask8_tensor = torch.stack(gt_mask8).to(self.device)
                    gt8  = gt_mask8_tensor[idxs]         # shape (P,8)

                    L_evid_i = evidential_BB_loss(pa, pb, gt8)
                    L_pos_var  = - var_H8[:N][pos_mask].mean() if pos_mask.any()>0 else torch.tensor(0., device=self.device)
                else:
       
                    L_evid_i = alpha_H8.sum() * 0.0
                    pos_mask = (pred_scores[:N] > 0.7) 
                    L_pos_var  = - var_H8[:N][pos_mask].mean() if pos_mask.any()>0 else torch.tensor(0., device=self.device)


                easy_neg_mask = (pred_scores[:N] < 0.3)                  
                hard_mask     = ((pred_scores[:N] >= 0.3) & (pred_scores[:N] <= 0.7)) 
                L_easy = var_H8[:N][easy_neg_mask].mean() if easy_neg_mask.any()>0 else torch.tensor(0., device=self.device)
                L_hard = - var_H8[:N][hard_mask].mean() if hard_mask.any()>0 else torch.tensor(0., device=self.device)
                L_var_i = 1.5 * L_easy + 3.0 * L_hard + 1.5 * L_pos_var


            loss_i = (
                self.w_evid * L_evid_i
                + self.w_var  * L_var_i
                + self.w_rank * L_rank_i

            )
            total_loss += loss_i

            total_loss_score += self.w_evid * L_evid_i + self.w_var  * L_var_i
            total_loss_var += self.w_var  * L_var_i
            total_loss_rank += self.w_rank * L_rank_i

            
        total_loss = total_loss / B
        total_loss_score = total_loss_score / B
        total_loss_var = total_loss_var / B
        total_loss_rank = total_loss_rank / B


        self.log("total_loss", total_loss, prog_bar=True)
        self.log("total_loss_score", total_loss_score, prog_bar=True)
        self.log("total_loss_var", total_loss_var, prog_bar=True)
        self.log("total_loss_rank", total_loss_rank, prog_bar=True)


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
        val_total_loss_var = torch.tensor(0.0, device=self.device)
        val_total_loss_rank = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape

        for i in range(B):
            pred_scores = outputs["score_H8"][i]   # (num_preds, 8)
            alpha_H8    = outputs["alpha"][i]      # (num_preds, 8)
            beta_H8     = outputs["beta"][i]       # (num_preds, 8)
            var_H8      = outputs["var_H8"][i]     # (num_preds, 8)
            

            gt_ids      = label_risk_ids[i]        # list[int]
            gt_mask8    = label_risk_interval_H8[i]  # (num_gt, 8)
            all_ids     = all_objs_ids[i][-1]      # list[int]
            N = len(all_ids) 
            
            if N == 0:
  
                L_evid_i = torch.sum(pred_scores) * 0.0
                L_var_i   = torch.sum(pred_scores) * 0.0
                L_rank_i  = torch.sum(pred_scores) * 0.0


            elif len(gt_ids) == 0:

                easy_neg_mask = (pred_scores[:N] < 0.3)                   
                hard_mask     = ((pred_scores[:N] >= 0.3) & (pred_scores[:N] <= 0.7)) 
                pos_mask = (pred_scores[:N] > 0.7) 
                    
                L_easy = var_H8[:N][easy_neg_mask].mean() if easy_neg_mask.any()>0 else torch.tensor(0., device=self.device)
                
                L_hard = - var_H8[:N][hard_mask].mean() if hard_mask.any()>0 else torch.tensor(0., device=self.device)
                L_pos_var  = - var_H8[:N][pos_mask].mean() if pos_mask.any()>0 else torch.tensor(0., device=self.device)

                L_var_i = 1.5 * L_easy + 3.0 * L_hard + 1.5 * L_pos_var

               
                L_rank_i = torch.sum(pred_scores[:N]) * 0.0

                
                L_evid_i = torch.sum(pred_scores[:N]) * 0.0

            else:

                pos_mask = torch.tensor([oid in gt_ids for oid in all_ids],
                        device=self.device)
                neg_mask = ~pos_mask


                ps = pred_scores[:N][pos_mask].flatten()
                ns = pred_scores[:N][neg_mask].flatten()
                L_rank_i = ranking_margin_loss(ps, ns, margin=0.2)

                if pos_mask.any():
                    pa = alpha_H8[:N][pos_mask]      # shape (P,8)
                    pb = beta_H8 [:N][pos_mask]

                    idxs = [gt_ids.index(oid) for oid in all_ids if oid in gt_ids]
                    
                    gt_mask8_tensor = torch.stack(gt_mask8).to(self.device)
                    gt8  = gt_mask8_tensor[idxs]            # shape (P,8)

                    L_evid_i = evidential_BB_loss(pa, pb, gt8)
                    L_pos_var  = - var_H8[:N][pos_mask].mean() if pos_mask.any()>0 else torch.tensor(0., device=self.device)
                else:
                    L_evid_i = alpha_H8.sum() * 0.0
                    pos_mask = (pred_scores[:N] > 0.7) 
                    L_pos_var  = - var_H8[:N][pos_mask].mean() if pos_mask.any()>0 else torch.tensor(0., device=self.device)


                easy_neg_mask = (pred_scores[:N] < 0.3)                
                hard_mask     = ((pred_scores[:N] >= 0.3) & (pred_scores[:N] <= 0.7)) 
                L_easy = var_H8[:N][easy_neg_mask].mean() if easy_neg_mask.any()>0 else torch.tensor(0., device=self.device)

                L_hard = - var_H8[:N][hard_mask].mean() if hard_mask.any()>0 else torch.tensor(0., device=self.device)
                L_var_i = 1.5 * L_easy + 3.0 * L_hard + 1.5 * L_pos_var
            
            loss_i = (
                self.w_evid * L_evid_i
                + self.w_var  * L_var_i
                + self.w_rank * L_rank_i
            )
            val_total_loss += loss_i

            val_total_loss_score += self.w_evid * L_evid_i + self.w_var  * L_var_i
            val_total_loss_var += self.w_var  * L_var_i
            val_total_loss_rank += self.w_rank * L_rank_i
     
        val_total_loss = val_total_loss / B
        val_total_loss_score = val_total_loss_score / B
        val_total_loss_var = val_total_loss_var / B
        val_total_loss_rank = val_total_loss_rank / B


        self.log("val_total_loss", val_total_loss, prog_bar=True)
        self.log("val_total_loss_score", val_total_loss_score, prog_bar=True)
        self.log("val_total_loss_var", val_total_loss_var, prog_bar=True)
        self.log("val_total_loss_rank", val_total_loss_rank, prog_bar=True)
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='Bayesian', help='Unique experiment identifier.')
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

    Bayesian_Model = Bayesian(args.lr)
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

    trainer.fit(Bayesian_Model, dataloader_train, dataloader_val)



