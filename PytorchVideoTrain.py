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
import pytorchvideo.models.resnet
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
    def __init__(self, args):
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


    


def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules above,
    and pass them to the fit function of a pytorch_lightning.Trainer.

    This example can be run either locally (with default parameters) or on a Slurm
    cluster. To run on a Slurm cluster provide the --on_cluster argument.
    """
    setup_logger()

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)

    parser.add_argument("--arch",default="slowfast",type=str,required=True)

    # Data parameters.
    #parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--data_root", default=None, type=str, required=True)
    #parser.add_argument("--val_sub", default=None, type=str, required=True)
    parser.add_argument("--slowfast_alpha", default=4, type=int)
    parser.add_argument("--slowfast_beta", default=1/8, type=float)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--num_classes", default=5, type=int)

    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--clip_duration", default=1, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    #parser.add_argument("--default_root_dir", default="Models", type=str)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)

    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=2,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    for subdir in os.listdir(args.data_root):
        args.val_sub = subdir
        train(args)
    #val(args)


def train(args):
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    classification_module = VideoClassificationLightningModule(args)
    data_module = GRASSPDataModule(args)
    trainer.fit(classification_module, data_module)

    # Save checkpoint
    model_dir = Path('Models', args.arch)
    os.makedirs(model_dir, exist_ok=True)
    model_path = Path(model_dir, f"{args.arch}_{args.val_sub}.ckpt")
    trainer.save_checkpoint(model_path)

    # Validate
    #classification_module = VideoClassificationLightningModule.load_from_checkpoint('slowfasttest.ckpt', args=args)
    metrics = trainer.validate(classification_module, data_module)

    # Save metrics to json
    results_dir = Path('Results', args.arch)
    os.makedirs(results_dir, exist_ok=True)
    savefile = Path(results_dir, args.val_sub + "_metrics.json")
    with open(savefile, 'w') as f:
        json.dump(metrics, f, indent=4)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()
