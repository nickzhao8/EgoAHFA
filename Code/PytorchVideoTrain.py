# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import itertools
import logging
import os
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


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args=None):
        self.args = args
        super().__init__()
        self.train_accuracy = torchmetrics.Accuracy()
        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_MAE = torchmetrics.MeanAbsoluteError()
        self.val.MSE = torchmetrics.MeanSquaredError(squared=True)
        self.val.RMSE = torchmetrics.MeanSquaredError(squared=False)
        self.val_microPrecision = torchmetrics.Precision(average='micro', num_classes=self.args.num_classes)
        self.val_macroPrecision = torchmetrics.Precision(average='macro', num_classes=self.args.num_classes)
        self.val_microRecall = torchmetrics.Recall(average='micro', num_classes=self.args.num_classes)
        self.val_macroRecall = torchmetrics.Recall(average='macro', num_classes=self.args.num_classes)
        self.val_microF1 = torchmetrics.F1Score(average='micro', num_classes=self.args.num_classes)
        self.val_macroF1 = torchmetrics.F1Score(average='macro', num_classes=self.args.num_classes)

        self.val_preds = []
        self.val_target = []
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
                slowfast_conv_channel_fusion_ratio=self.args.slowfast_alpha,
            )
            self.batch_key = "video"
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
            self.batch_key = "video"
        else:
            raise Exception("{self.args.arch} not supported")

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.trainer._accelerator_connector.is_distributed:
           self.trainer.datamodule.train_dataset.video_sampler.set_epoch(epoch)
        self.train_losses = []
        self.train_accs = []
        self.train_maes = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        # import pdb; pdb.set_trace()
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        if ((batch_idx % self.args.log_every_n_steps) == 0):
            acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
            mae = self.train_MAE(F.softmax(y_hat, dim=-1), batch["label"])
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
        loss = F.cross_entropy(y_hat, batch["label"])
        preds = F.softmax(y_hat, dim=-1)
        target = batch["label"]
        # Calculate metrics
        acc = self.val_accuracy(preds, target)
        MAE = self.val_MAE(preds, target)
        MSE = self.val_MSE(preds, target)
        RMSE = self.val_RMSE(preds, target)
        microPrecision = self.val_microPrecision(preds, target)
        macroPrecision = self.val_macroPrecision(preds, target)
        microRecall = self.val_microRecall(preds, target)
        macroRecall = self.val_macroRecall(preds, target)
        microF1 = self.val_microF1(preds, target)
        macroF1 = self.val_macroF1(preds, target)
        # Log metrics
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
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
        preds = self.softmax_index(preds.tolist())
        self.val_tasks = self.record_tasks(self.val_tasks, preds, target.tolist(), batch['video_name'])
        self.val_preds.append(preds)
        self.val_loss.append(loss.tolist())
        self.val_target.append(target.tolist())
        return loss
    
    def softmax_index(self, preds):
        for i in range (len(preds)):
            preds[i] = preds[i].index(max(preds[i]))
        return preds
    
    def record_tasks(self, tasks, preds, target, video_names):
        for i in range(len(target)): # Iterate through batch
            video_name = video_names[i].split('_')[1]
            if video_name in tasks:
                tasks[video_name]['preds'].append(preds[i])
                tasks[video_name]['target'].append(target[i])
            else:
                tasks[video_name] = {}
                tasks[video_name]['preds'] = [preds[i]]
                tasks[video_name]['target'] = [target[i]]
        return tasks

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

def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)

