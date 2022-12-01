# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import itertools
import logging
import os
import re
from pathlib import Path
from typing import Iterable
import json

import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
import torchmetrics
from lightning_classes.GRASSP_classes import GRASSPDataModule
from coral_pytorch.losses import corn_loss, coral_loss
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch, proba_to_label


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args=None):
        self.args = args
        super().__init__()
        self.train_accuracy = torchmetrics.Accuracy()
        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_MAE = torchmetrics.MeanAbsoluteError()
        self.val_MSE = torchmetrics.MeanSquaredError(squared=True)
        self.val_RMSE = torchmetrics.MeanSquaredError(squared=False)
        self.val_microPrecision = torchmetrics.Precision(average='micro', num_classes=self.args.num_classes+1)
        self.val_macroPrecision = torchmetrics.Precision(average='macro', num_classes=self.args.num_classes+1)
        self.val_microRecall = torchmetrics.Recall(average='micro', num_classes=self.args.num_classes+1)
        self.val_macroRecall = torchmetrics.Recall(average='macro', num_classes=self.args.num_classes+1)
        self.val_microF1 = torchmetrics.F1Score(average='micro', num_classes=self.args.num_classes+1)
        self.val_macroF1 = torchmetrics.F1Score(average='macro', num_classes=self.args.num_classes+1)

        self.sanity_check = False
        self.val_preds = []
        self.val_target = []
        self.val_loss = []
        self.val_tasks = {}

        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                model_num_class=self.args.num_classes,
            )
            self.batch_key = "video"
        elif self.args.arch == 'slowfast':
            self.model = pytorchvideo.models.slowfast.create_slowfast(
                input_channels=(3,3),
                model_num_class=self.args.num_classes,
                slowfast_channel_reduction_ratio=int(1/self.args.slowfast_beta),
                slowfast_conv_channel_fusion_ratio=self.args.slowfast_fusion_conv_channel_ratio,
                slowfast_fusion_conv_kernel_size=(self.args.slowfast_fusion_kernel_size, 1, 1),
                slowfast_fusion_conv_stride=(self.args.slowfast_alpha, 1, 1), 
            )
            self.batch_key = "video"
            ## Transfer Learning: freeze all layers except final one(s)
            if self.args.transfer_learning:
                # Load pre-trained state dict
                state_dict = torch.load(self.args.pretrained_state_dict)
                self.model.load_state_dict(state_dict, strict=False)
                # Save pointers to layers to unfreeze
                block4_pathway0_res_block2 = self.model.blocks[4].multipathway_blocks[0].res_blocks[2]
                block4_pathway1_res_block2 = self.model.blocks[4].multipathway_blocks[1].res_blocks[2]
                # Freeze params
                for param in self.model.parameters():
                    param.requires_grad = False
                # Unfreeze saved layers
                for param in block4_pathway0_res_block2.parameters(): param.requires_grad = True
                for param in block4_pathway1_res_block2.parameters(): param.requires_grad = True
                # Construct last fc layer
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = torch.nn.Linear(num_features, self.args.num_classes)
            
            ## If using CORN Ordinal regression, replace final layer with -1 output nodes.
            if self.args.ordinal and self.args.ordinal_strat == 'CORN':
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = torch.nn.Linear(num_features, self.args.num_classes-1)
            ## If using CORAL Ordinal regression, replace final layer with a CORAL weight-sharing layer.
            if self.args.ordinal and self.args.ordinal_strat == 'CORAL':
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = CoralLayer(size_in=num_features, num_classes=self.args.num_classes)

        elif self.args.arch == 'mvit':
            if self.args.transfer_learning:
                self.model = torch.hub.load('facebookresearch/pytorchvideo', model='mvit_base_16x4', pretrained=True)
                # Freeze model
                for param in self.model.parameters():
                    param.requires_grad = False
                num_features = self.model.head.proj.in_features
                self.model.head.proj = torch.nn.Linear(num_features, self.args.num_classes)
            else:
                self.model = pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(
                spatial_size=self.args.video_crop_size,
                temporal_size=self.args.num_frames,
                embed_dim_mul           = self.args.mvit_embed_dim_mul,
                atten_head_mul          = self.args.mvit_atten_head_mul,
                pool_q_stride_size      = self.args.mvit_pool_q_stride_size,
                pool_kv_stride_adaptive = self.args.mvit_pool_kv_stride_adaptive,
                pool_kvq_kernel         = self.args.mvit_pool_kvq_kernel,
                head_num_classes        = self.args.num_classes,
                )

            ## If using CORN Ordinal regression, replace final layer with -1 output nodes.
            if self.args.ordinal and self.args.ordinal_strat == 'CORN':
                num_features = self.model.head.proj.in_features
                self.model.head.proj = torch.nn.Linear(num_features, self.args.num_classes-1)
            if self.args.ordinal and self.args.ordinal_strat == 'CORAL':
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = CoralLayer(size_in=num_features, num_classes=self.args.num_classes)

            self.batch_key = "video"
        else:
            raise Exception(f"{self.args.arch} not supported")

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.trainer._accelerator_connector.is_distributed:
           self.trainer.datamodule.train_sampler.set_epoch(epoch)
        self.train_losses = []
        self.train_accs = []
        self.train_maes = []
        # Display val_sub on progress bar
        val_subnum = int(self.args.val_sub.lower().split('sub')[-1])
        self.log("val_sub", val_subnum, prog_bar=True)
    
    def on_train_epoch_end(self) -> None:
        # Log Histogram of model weights
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
        return super().on_train_epoch_end()

    def forward(self, x):
        return self.model(x)
    
    def ordinal_prediction(self, preds):
        # Convert ordinal encoding back to class labels
        ooms = [torch.floor(torch.log10(x)) for x in preds[:,0]]
        cross_thresh = torch.zeros_like(preds)
        for i, oom in enumerate(ooms):
            cross_thresh[i,:] = torch.floor(torch.log10(preds[i,:])) >= oom
        return (cross_thresh.cumprod(axis=1).sum(axis=1) - 1).int()

    def ordinal_loss(self, preds, targets):
        preds = F.softmax(preds, dim=-1)
        # Modify target with ordinal encoding
        ordinal_target = torch.zeros_like(preds)
        for i, target in enumerate(targets):
            ordinal_target[i, 0:target+1] = 1
        return F.mse_loss(preds, ordinal_target, reduction='sum')
        
    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        # import pdb; pdb.set_trace()
        y_hat = self.model(x)
        # == LOSS == 
        if self.args.ordinal:   
            if self.args.ordinal_strat == 'CORN':
                loss = corn_loss(y_hat, batch['label'], num_classes=self.args.num_classes)
            elif self.args.ordinal_strat == 'CORAL':
                levels = levels_from_labelbatch(batch['label'], num_classes=self.args.num_classes)
                loss = coral_loss(y_hat, levels)
            else:
                loss = self.ordinal_loss(y_hat, batch["label"])
        else:                   
            loss = F.cross_entropy(y_hat, batch["label"])
        # Manual logging
        if ((batch_idx % self.args.log_every_n_steps) == 0):
            preds = F.softmax(y_hat, dim=-1)
            if self.args.ordinal:
                if self.args.ordinal_strat == 'CORN':
                    pred_labels = corn_label_from_logits(y_hat)
                elif self.args.ordinal_strat == 'CORAL':
                    pred_labels = proba_to_label(torch.sigmoid(y_hat))
                else:
                    pred_labels = self.ordinal_prediction(preds)
            else:
                pred_labels = torch.argmax(preds, dim=1)
            acc = self.train_accuracy(pred_labels, batch["label"])
            mae = self.train_MAE(pred_labels, batch["label"])
            self.train_losses.append(loss)
            self.train_accs.append(acc)
            self.train_maes.append(mae)
            avg_loss = torch.mean(torch.stack(self.train_losses))
            avg_acc = torch.mean(torch.stack(self.train_accs))
            avg_mae = torch.mean(torch.stack(self.train_maes))
            self.logger.experiment.add_scalar("Loss/Train", avg_loss, batch_idx)
            self.logger.experiment.add_scalar("Accuracy/Train", avg_acc, batch_idx)
            self.logger.experiment.add_scalar("MAE/Train", avg_mae, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        y_hat = self.model(x)
        preds = F.softmax(y_hat, dim=-1)
        if self.args.ordinal:   
            if self.args.ordinal_strat == 'CORN':
                loss = corn_loss(y_hat, batch['label'], num_classes=self.args.num_classes)
                pred_labels = corn_label_from_logits(y_hat)
            elif self.args.ordinal_strat == 'CORAL':
                levels = levels_from_labelbatch(batch['label'], num_classes=self.args.num_classes)
                loss = coral_loss(y_hat, levels)
                pred_labels = proba_to_label(torch.sigmoid(y_hat))
            else:
                loss = self.ordinal_loss(y_hat, batch["label"])
                pred_labels = self.ordinal_prediction(preds)
        else:                  
            loss = F.cross_entropy(y_hat, batch["label"])
            pred_labels = torch.argmax(preds, dim=1)
        target = batch["label"]
        # Calculate metrics
        acc = self.val_accuracy(pred_labels, target)
        MAE = self.val_MAE(pred_labels, target)
        MSE = self.val_MSE(pred_labels, target)
        RMSE = self.val_RMSE(pred_labels, target)
        microPrecision = self.val_microPrecision(pred_labels, target)
        macroPrecision = self.val_macroPrecision(pred_labels, target)
        microRecall = self.val_microRecall(pred_labels, target)
        macroRecall = self.val_macroRecall(pred_labels, target)
        microF1 = self.val_microF1(pred_labels, target)
        macroF1 = self.val_macroF1(pred_labels, target)
        # Log metrics
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, sync_dist=True
        )
        metrics = {
            'val_MAE': MAE,
            'val_MSE': MSE,
            'val_RMSE': RMSE,
            "val_microPrecision": microPrecision,
            "val_macroPrecision": macroPrecision,
            "val_microRecall": microRecall,
            "val_macroRecall": macroRecall,
            "val_microF1": microF1,
            "val_macroF1": macroF1,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        # Custom preds, target, tasks Logging
        self.val_filenames = self.record_filenames(self.val_filenames, pred_labels, target.tolist(), batch)
        self.val_tasks = self.record_tasks(self.val_tasks, pred_labels, target.tolist(), batch['video_name'])
        self.val_preds.append(pred_labels.tolist())
        self.val_loss.append(loss.tolist())
        self.val_target.append(target.tolist())
        return loss
    
    def softmax_index(self, preds):
        for i in range (len(preds)):
            preds[i] = preds[i].index(max(preds[i]))
        return preds
    
    def record_tasks(self, tasks, preds, target, video_names):
        for i in range(len(target)): # Iterate through batch
            video_name = re.split('\d',video_names[i])[-2][1:-1]
            if video_name in tasks:
                tasks[video_name]['preds'].append(preds[i].tolist())
                tasks[video_name]['target'].append(target[i])
            else:
                tasks[video_name] = {}
                tasks[video_name]['preds'] = [preds[i].tolist()]
                tasks[video_name]['target'] = [target[i]]
        return tasks

    def record_filenames(self, filenames, preds, target, batch):
        video_names = batch['video_name']
        clip_index = batch['clip_index']
        for i in range(len(target)): # Iterate through batch
            video_name = video_names[i]
            if video_name in filenames:
                filenames[video_name]['preds'].append(preds[i].tolist())
                filenames[video_name]['target'].append(target[i])
                filenames[video_name]['clip_index'].append(clip_index[i].item())
            else:
                filenames[video_name] = {}
                filenames[video_name]['preds'] = [preds[i].tolist()]
                filenames[video_name]['target'] = [target[i]]
                filenames[video_name]['clip_index'] = [clip_index[i].item()]
        return filenames
    
    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # Linear warm-up: skip args.warmup number of steps
        if self.trainer.global_step < self.args.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.args.warmup))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.args.lr

    def configure_optimizers(self):
        if self.args.arch == 'mvit':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args.max_epochs, last_epoch=-1
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args.max_epochs, last_epoch=-1
            )
        return [optimizer], [scheduler]
    
    # def ordinal

def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)

