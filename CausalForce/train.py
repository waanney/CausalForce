

import argparse
import os
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from data import MultipleRisksDataset, custom_collate_fn
from model import GCN_model
from torch.utils.data import DataLoader
from online_conformal.saocp import SAOCP

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

def htsc_loss(
    hx_seq,           # Tensor, shape (T, N, H), LSTM hidden vectors
    pred_risk_type,   # Tensor, shape (N, C), class logits for each object at the last frame
    all_objs_id,      # list of length T, each element is a list of N object IDs (may be empty)
    sim_thr=0.0       # float, optional spatial similarity threshold
):
    """
    Hierarchical Temporal-Spatial Consistency Loss (HTSC Loss):
      - For each adjacent frame pair t and t+1, the feature of the same object
        should maintain consistent similarity with objects of the same class in the same frame.
      - Achieved by minimizing (cos_spatial - cos_temporal)^2

    Args:
      hx_seq         : (T, N, H) Tensor, T frames, N objects, H-dimensional hidden vectors
      pred_risk_type : (N, C) Tensor, N objects, C-class risk logits
      all_objs_id    : list of length T, each element is a list of N object IDs (can be empty)
      sim_thr        : spatial similarity threshold; skip pairs if cos_spatial < sim_thr

    Returns:
      A scalar Tensor; returns 0 if no valid pairs are found.
    """
    T, N, _ = hx_seq.shape
    if T < 2 or len(all_objs_id) < T:
        return hx_seq.sum() * 0.0

    # Category logits correspond to object slots in the final input frame.
    final_ids = list(all_objs_id[-1])[:N]
    risk_labels = pred_risk_type[:len(final_ids)].argmax(dim=1).tolist()
    label_by_id = {
        object_id: risk_labels[index]
        for index, object_id in enumerate(final_ids)
    }

    losses = []
    for t in range(T - 1):
        prev_ids = list(all_objs_id[t])[:N]
        next_ids = list(all_objs_id[t + 1])[:N]
        prev_index = {object_id: index for index, object_id in enumerate(prev_ids)}
        next_index = {object_id: index for index, object_id in enumerate(next_ids)}

        # Both objects in a pair must be trackable across t and t+1, and their
        # category must be known from the final-frame classifier output.
        common_ids = [
            object_id for object_id in prev_ids
            if object_id in next_index and object_id in label_by_id
        ]
        for first in range(len(common_ids)):
            object_i = common_ids[first]
            for second in range(first + 1, len(common_ids)):
                object_k = common_ids[second]
                if label_by_id[object_i] != label_by_id[object_k]:
                    continue

                i_t, i_next = prev_index[object_i], next_index[object_i]
                k_t, k_next = prev_index[object_k], next_index[object_k]
                cos_spatial = F.cosine_similarity(
                    hx_seq[t, i_t].unsqueeze(0),
                    hx_seq[t, k_t].unsqueeze(0), dim=1,
                )[0]
                if sim_thr > 0 and cos_spatial.detach().item() < sim_thr:
                    continue

                cos_temporal_i = F.cosine_similarity(
                    hx_seq[t, i_t].unsqueeze(0),
                    hx_seq[t + 1, i_next].unsqueeze(0), dim=1,
                )[0]
                cos_temporal_k = F.cosine_similarity(
                    hx_seq[t, k_t].unsqueeze(0),
                    hx_seq[t + 1, k_next].unsqueeze(0), dim=1,
                )[0]
                delta_temporal = cos_temporal_i - cos_temporal_k
                losses.append((cos_spatial - delta_temporal).pow(2))

    if not losses:
        return hx_seq.sum() * 0.0
    return torch.stack(losses).mean()

 
class CausalForce_CACC_STFA(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.coverage = 0.7
        self.nc_method = "absolute"
        self.model = GCN_model()
        self.class_cps = { 'OBS':SAOCP(model=None, train_data=None, max_scale=1.0, coverage=self.coverage, horizon=8),
                            'OCC':SAOCP(model=None, train_data=None, max_scale=1.0, coverage=self.coverage, horizon=8),
                            'I':SAOCP(model=None, train_data=None, max_scale=1.0, coverage=self.coverage, horizon=8),
                            'C':SAOCP(model=None, train_data=None, max_scale=1.0, coverage=self.coverage, horizon=8),
                            }
        self.index_to_class = {0:'OBS', 1:'OCC', 2:'I', 3:'C'}
        
        self.w_score = 1.0
        self.w_htsc = 10.0

        
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
        total_loss_htsc = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape

        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            pred_risk_type = outputs["risk_type"][i]       # (num_preds, 4)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)

            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_scoreH8_i = pred_risk_score_H8.sum() * 0.0
                loss_htsc_i = pred_risk_score_H8.sum() * 0.0 

            elif len(gt_risk_ids) == 0:
                gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N])  
                loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')
                loss_htsc_i = htsc_loss(
                            hx_seq=outputs["hx_seq"][i],
                            pred_risk_type=outputs["risk_type"][i],
                            all_objs_id=all_objs_ids[i],
                            sim_thr=0.0
                        )    
                
                for type_pred, obj_pred, obj_gt in zip(pred_risk_type[:N], pred_risk_score_H8[:N], gt_risk_score_H8):
                    pred_cls = self.index_to_class[type_pred.argmax().item()]
                    nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                    for t in range(8):
                        nc_t = pd.Series([nc[t].item()], dtype=float)  
                        self.class_cps[pred_cls].update(
                            ground_truth=nc_t,
                            forecast=pd.Series([0], dtype=float),
                            horizon=t+1
                        )
                    
            else:
                # determine matched pairs on obj_id, risk_type, and risk score
                matched_pred, matched_gt = [], []
                match_pred_type = []
                non_risk_pred =[]
                non_risk_pred_type = []
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_score_H8[j])          # (8,)
                        match_pred_type.append(pred_risk_type[j])          # (4,)
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_score_H8[idx])            # (8,)
                    else:
                        non_risk_pred.append(pred_risk_score_H8[j])
                        non_risk_pred_type.append(pred_risk_type[j])          # (4,)
                        
                if len(matched_pred) == 0:
                    gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N])  
                    loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')

                    # calculate nonconformity scores and calibrate risk objects
                    for type_pred, obj_pred, obj_gt in zip(pred_risk_type[:N], pred_risk_score_H8[:N], gt_risk_score_H8):
                        pred_cls = self.index_to_class[type_pred.argmax().item()]
                        nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                        for t in range(8):
                            nc_t = pd.Series([nc[t].item()], dtype=float)  
                            self.class_cps[pred_cls].update(
                                ground_truth=nc_t,
                                forecast=pd.Series([0], dtype=float),
                                horizon=t+1
                            )
                else:
                    if len(non_risk_pred) == 0:
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean')

                        for type_pred, obj_pred, obj_gt in zip(match_pred_type, matched_pred, matched_gt):
                            pred_cls = self.index_to_class[type_pred.argmax().item()]
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float) 
                                self.class_cps[pred_cls].update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )
                    else:
                        # gt risk pairs
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        
                        # non risk pairs
                        non_risk_preds = torch.stack(non_risk_pred)
                        non_risk_gt = torch.zeros_like(non_risk_preds) 

                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean') + \
                                         F.binary_cross_entropy(non_risk_preds, non_risk_gt, reduction='mean')
                        
                        # calculate nonconformity scores and calibrate risk object
                        for type_pred, obj_pred, obj_gt in zip(match_pred_type, matched_pred, matched_gt):
                            pred_cls = self.index_to_class[type_pred.argmax().item()]
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)  
                                self.class_cps[pred_cls].update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

                        # calculate nonconformity scores and calibrate non-risk object
                        for type_pred, obj_pred, obj_gt in zip(non_risk_pred_type, non_risk_preds, non_risk_gt):
                            pred_cls = self.index_to_class[type_pred.argmax().item()]
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)  
                                self.class_cps[pred_cls].update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

                    # compute spatio-temporal feature alignment loss
                    loss_htsc_i = htsc_loss(
                            hx_seq=outputs["hx_seq"][i],
                            pred_risk_type=outputs["risk_type"][i],
                            all_objs_id=all_objs_ids[i],
                            sim_thr=0.0
                        )       


            total_loss += (self.w_score*loss_scoreH8_i + self.w_htsc*loss_htsc_i)
            total_loss_score += self.w_score*loss_scoreH8_i
            total_loss_htsc += self.w_htsc*loss_htsc_i

  
        total_loss = total_loss / B
        total_loss_score = total_loss_score / B
        total_loss_htsc = total_loss_htsc / B

        self.log("total_loss", total_loss, prog_bar=True)
        self.log("total_loss_score", total_loss_score, prog_bar=True)
        self.log("total_loss_htsc", total_loss_htsc, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        trainable = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable, lr=self.lr, weight_decay=1e-7)
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
        val_total_loss_htsc = torch.tensor(0.0, device=self.device)

        B, T, C, H, W = front_imgs.shape
        for i in range(B):
            pred_risk_score_H8 = outputs["score_H8"][i]        # (num_preds, 8)
            pred_risk_type = outputs["risk_type"][i]       # (num_preds, 4)
            gt_risk_ids        = label_risk_ids[i]             # list[int]
            gt_risk_score_H8   = label_risk_interval_H8[i]  # (num_gt, 8)
            all_objs_id        = all_objs_ids[i][-1]           # list[int] or (num_objs,)
                              
            N = len(all_objs_id)

            if len(all_objs_id) == 0:
                loss_scoreH8_i = pred_risk_score_H8.sum() * 0.0
                loss_htsc_i = pred_risk_score_H8.sum() * 0.0 

            elif len(gt_risk_ids) == 0:
                gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N]) 
                loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')
                for type_pred, obj_pred, obj_gt in zip(pred_risk_type[:N], pred_risk_score_H8[:N], gt_risk_score_H8):
                    pred_cls = self.index_to_class[type_pred.argmax().item()]
                    nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                    for t in range(8):
                        nc_t = pd.Series([nc[t].item()], dtype=float)  
                        self.class_cps[pred_cls].update(
                            ground_truth=nc_t,
                            forecast=pd.Series([0], dtype=float),
                            horizon=t+1
                        )
                loss_htsc_i = htsc_loss(
                            hx_seq=outputs["hx_seq"][i],
                            pred_risk_type=outputs["risk_type"][i],
                            all_objs_id=all_objs_ids[i],
                            sim_thr=0.0
                        ) 
                    
            else:
                matched_pred, matched_gt = [], []
                match_pred_type = []
                non_risk_pred =[]
                non_risk_pred_type = []
                for j, obj_id in enumerate(all_objs_id):
                    if obj_id in gt_risk_ids:
                        matched_pred.append(pred_risk_score_H8[j])          # (8,)
                        match_pred_type.append(pred_risk_type[j])          # (4,)
                        idx = gt_risk_ids.index(obj_id)
                        matched_gt.append(gt_risk_score_H8[idx])            # (8,)
                    else:
                        non_risk_pred.append(pred_risk_score_H8[j])
                        non_risk_pred_type.append(pred_risk_type[j])          # (4,)
                        
                if len(matched_pred) == 0:
                    gt_risk_score_H8 = torch.zeros_like(pred_risk_score_H8[:N]) 
                    loss_scoreH8_i = F.binary_cross_entropy(pred_risk_score_H8[:N], gt_risk_score_H8, reduction='mean')
                    
                    for type_pred, obj_pred, obj_gt in zip(pred_risk_type[:N], pred_risk_score_H8[:N], gt_risk_score_H8):
                        pred_cls = self.index_to_class[type_pred.argmax().item()]
                        nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                        for t in range(8):
                            nc_t = pd.Series([nc[t].item()], dtype=float)  
                            self.class_cps[pred_cls].update(
                                ground_truth=nc_t,
                                forecast=pd.Series([0], dtype=float),
                                horizon=t+1
                            )
                else:
                    if len(non_risk_pred) == 0:
                        preds = torch.stack(matched_pred)                      # (P, 8)
                        gts   = torch.stack(matched_gt)                        # (P, 8)
                        loss_scoreH8_i = F.binary_cross_entropy(preds, gts, reduction='mean')

                        for type_pred, obj_pred, obj_gt in zip(match_pred_type, matched_pred, matched_gt):
                            pred_cls = self.index_to_class[type_pred.argmax().item()]
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)  
                                self.class_cps[pred_cls].update(
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
                        
                        for type_pred, obj_pred, obj_gt in zip(match_pred_type, matched_pred, matched_gt):
                            pred_cls = self.index_to_class[type_pred.argmax().item()]
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)  
                                self.class_cps[pred_cls].update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )

                        for type_pred, obj_pred, obj_gt in zip(non_risk_pred_type, non_risk_preds, non_risk_gt):
                            pred_cls = self.index_to_class[type_pred.argmax().item()]
                            nc = compute_nonconformity(obj_pred, obj_gt, method=self.nc_method)
                            for t in range(8):
                                nc_t = pd.Series([nc[t].item()], dtype=float)  
                                self.class_cps[pred_cls].update(
                                    ground_truth=nc_t,
                                    forecast=pd.Series([0], dtype=float),
                                    horizon=t+1
                                )
                                
                    loss_htsc_i = htsc_loss(
                            hx_seq=outputs["hx_seq"][i],
                            pred_risk_type=outputs["risk_type"][i],
                            all_objs_id=all_objs_ids[i],
                            sim_thr=0.0
                        ) 
                    
            val_total_loss += (self.w_score*loss_scoreH8_i + self.w_htsc*loss_htsc_i)
            val_total_loss_score += self.w_score*loss_scoreH8_i
            val_total_loss_htsc += loss_htsc_i

  
        val_total_loss = val_total_loss / B
        val_total_loss_score = val_total_loss_score / B
        val_total_loss_htsc = val_total_loss_htsc / B

        self.log("val_total_loss", val_total_loss, prog_bar=True)
        self.log("val_total_loss_score", val_total_loss_score, prog_bar=True)
        self.log("val_total_loss_htsc", val_total_loss_htsc, prog_bar=True)
        
    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint['saocp_class'] = self.class_cps
        
        return checkpoint
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='Conformal_RiskTube_Prediction_CACC_STFA', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=12, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--pretrain_ckpt', type=str, default='/path/to/your/pretrained_risk_category_cls.ckpt', help='Path to pre-trained risk category classifier checkpoint.')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    train_set = MultipleRisksDataset(data_root='/path/to/your/training_data/')
    print(len(train_set))
    val_set = MultipleRisksDataset(data_root='/path/to/your/validation_data/')
    print(len(val_set))

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    CausalForce_CACC_STFA_Model = CausalForce_CACC_STFA(args.lr)

    checkpoint = torch.load(args.pretrain_ckpt)
    state_dict = checkpoint["state_dict"]
    
    CausalForce_CACC_STFA_Model.load_state_dict(state_dict, strict=False)

    for name, param in CausalForce_CACC_STFA_Model.named_parameters():    
        if 'risk_type_head' in name:
            param.requires_grad = False
            print(f"Freeze parameter: {name}")
        else:
            param.requires_grad = True
            print(f"Trainable parameter: {name}")
            
    CausalForce_CACC_STFA_Model.cuda()
    
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

    trainer.fit(CausalForce_CACC_STFA_Model, dataloader_train, dataloader_val)


