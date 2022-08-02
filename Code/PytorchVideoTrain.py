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


"""
This video classification example demonstrates how PyTorchVideo models, datasets and
transforms can be used with PyTorch Lightning module. Specifically it shows how a
simple pipeline to train a Resnet on the Kinetics video dataset can be built.

Don't worry if you don't have PyTorch Lightning experience. We'll provide an explanation
of how the PyTorch Lightning module works to accompany the example.

The code can be separated into three main components:
1. VideoClassificationLightningModule (pytorch_lightning.LightningModule), this defines:
    - how the model is constructed,
    - the inner train or validation loop (i.e. computing loss/metrics from a minibatch)
    - optimizer configuration

2. KineticsDataModule (pytorch_lightning.LightningDataModule), this defines:
    - how to fetch/prepare the dataset
    - the train and val dataloaders for the associated dataset

3. pytorch_lightning.Trainer, this is a concrete PyTorch Lightning class that provides
  the training pipeline configuration and a fit(<lightning_module>, <data_module>)
  function to start the training/validation loop.

All three components are combined in the train() function. We'll explain the rest of the
details inline.
"""


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args=None):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        self.args = args
        super().__init__()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_microPrecision = torchmetrics.Precision(average='micro', num_classes=self.args.num_classes)
        self.val_macroPrecision = torchmetrics.Precision(average='macro', num_classes=self.args.num_classes)
        self.val_microRecall = torchmetrics.Recall(average='micro', num_classes=self.args.num_classes)
        self.val_macroRecall = torchmetrics.Recall(average='macro', num_classes=self.args.num_classes)
        self.val_microF1 = torchmetrics.F1Score(average='micro', num_classes=self.args.num_classes)
        self.val_macroF1 = torchmetrics.F1Score(average='macro', num_classes=self.args.num_classes)

        self.val_preds = []
        self.val_target = []
        self.val_tasks = {}

        #############
        # PTV Model #
        #############

        # Here we construct the PyTorchVideo model. For this example we're using a
        # ResNet that works with Kinetics (e.g. 400 num_classes). For your application,
        # this could be changed to any other PyTorchVideo model (e.g. for SlowFast use
        # create_slowfast).
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

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch. It must
        return a loss that is used for loss.backwards() internally. The self.log(...)
        function can be used to log any training metrics.

        PyTorchVideo batches are dictionaries containing each modality or metadata of
        the batch collated video clips. Kinetics contains the following notable keys:
           {
               'video': <video_tensor>,
               'audio': <audio_tensor>,
               'label': <action_label>,
           }

        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
        - "label" is a Tensor of shape (batch, 1)

        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping the dict and
        feeding it through the model/loss.
        """
        x = batch[self.batch_key]
        # import pdb; pdb.set_trace()
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle. For this
        simple example it's mostly the same as the training loop but with a different
        metric name.
        """
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        preds = F.softmax(y_hat, dim=-1)
        target = batch["label"]
        # Calculate metrics
        acc = self.val_accuracy(preds, target)
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
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
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

