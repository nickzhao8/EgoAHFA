from numpy import save
from lightning_classes import GRASSP_classes
from PytorchVideoTrain import VideoClassificationLightningModule, setup_logger
import argparse
import gzip
import torch
import pytorch_lightning
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import RandomSampler
import pytorchvideo
from pytorchvideo.data import LabeledVideoDataset
import os
from pathlib import Path
import json
from datetime import datetime

date = datetime.now().strftime("%m_%d_%H%M")
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

pytorch_lightning.trainer.seed_everything()
parser  =  argparse.ArgumentParser()

# Trainer parameters.
parser  =  pytorch_lightning.Trainer.add_argparse_args(parser)

args  =  parser.parse_args()
args.job_name  =  "ptv_video_classification"

# Default Parameters
args.on_cluster                     = False
args.job_name                       = "ptv_video_classification"
args.working_directory              = "."
args.partition                      = "dev"
args.lr                             = float(0.1)
args.momentum                       = float(0.9)
args.weight_decay                   = float(1e-4)
args.slowfast_alpha                 = int(4)
args.slowfast_beta                  = float(1/8)
args.video_path_prefix              = ""
args.data_type                      = "video"
args.video_means                    = tuple((0.45, 0.45, 0.45))
args.video_stds                     = tuple((0.225, 0.225, 0.225))
args.video_crop_size                = int(224)
args.video_min_short_side_scale     = int(256)
args.video_max_short_side_scale     = int(320)
args.video_horizontal_flip_p        = float(0.5)

# Adjustable Parameters
args.workers                        = int(0)
args.batch_size                     = int(8)
args.clip_duration                  = float(1)
args.stride                         = float(0.5)
args.num_frames                     = int(32)
args.num_classes                    = int(6)

# Pytorch Lightning Parameters
args.accelerator                    = 'gpu'
args.devices                        = -1
# args.strategy                       = 'ddp'
args.max_epochs                     = 2
args.callbacks                      = [LearningRateMonitor(), GRASSP_classes.GRASSPValidationCallback()]
# args.replace_sampler_ddp            = False
args.replace_sampler_ddp            = True

# Required Parameters
args.data_root                      = 'D:\\zhaon\\Datasets\\Video Segments'
args.arch                           = "slowfast"
args.val_sub                        = "Sub1"
args.results_path                   = "Results/"+args.arch

# Debugging Parameters
args.fast_dev_run                   = 50
profiler = AdvancedProfiler(dirpath='Debug',filename='profilereport_'+date)
args.profiler                       = profiler
# args.profiler                       = ""

#print(args)

def main():

    datamodule = GRASSP_classes.GRASSPDataModule(args)
    train_transforms = datamodule._make_transforms(mode='train')
    val_trainsforms = datamodule._make_transforms(mode='val')
    video_sampler = RandomSampler

    data_root = Path(args.data_root).resolve()
    save_root = Path('TorchDatasets').resolve()
    subdirs = data_root.glob('*')

    for val_sub in subdirs:
        skipped_val = False
        labeled_video_paths = []
        train_subs = data_root.glob('*')
        for subdir in train_subs:
            if subdir == val_sub:
                skipped_val = True
                val_dataset = pytorchvideo.data.labeled_video_dataset(
                    data_path=subdir.resolve(),
                    clip_sampler=pytorchvideo.data.UniformClipSampler(
					    clip_duration=args.clip_duration,
					    stride=args.stride,
					    backpad_last=True,
                    ),
                    #clip_sampler=pytorchvideo.data.RandomClipSampler(
                    #    clip_duration = self.args.clip_duration
                    #),
                    video_path_prefix=args.video_path_prefix,
                    transform=val_trainsforms,
                    video_sampler=video_sampler,
                    decode_audio=False,
                )
                continue
            for label in subdir.glob('*'):
                labelpath = Path(subdir, label)
                for vid in labelpath.glob('*'):
                    labeled_video_paths.append((str(vid), int(str(label).split('\\')[-1])))
                    #print(labeled_video_paths[-1])
        
        assert skipped_val == True, 'Invalid val_sub; val_sub not found.'

        train_dataset = LabeledVideoDataset(
            labeled_video_paths=labeled_video_paths,
            clip_sampler=pytorchvideo.data.UniformClipSampler(
                clip_duration=args.clip_duration,
                stride=args.stride,
                backpad_last=True,
            ),
            transform=train_transforms,
            video_sampler=video_sampler,
            decode_audio=False,
        )
        
        val_sub = str(val_sub).split('\\')[-1]
        train_savepath = Path(save_root, 'train', f'train_{val_sub}.gz')
        val_savepath = Path(save_root, 'val', f'val_{val_sub}.gz')
        torch.save(train_dataset, gzip.GzipFile(train_savepath,'wb')); 
        torch.save(val_dataset, gzip.GzipFile(val_savepath,'wb')); 


if __name__ == '__main__':
    main()