from lightning_classes import GRASSP_classes
from PytorchVideoTrain import VideoClassificationLightningModule, setup_logger
import argparse
import pytorch_lightning
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pathlib import Path
import json
from datetime import datetime

#os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

# pytorch_lightning.trainer.seed_everything(seed=1)
pytorch_lightning.trainer.seed_everything()
parser  =  argparse.ArgumentParser()
date = datetime.now().strftime("%m_%d_%H")

# Trainer parameters.
parser  =  pytorch_lightning.Trainer.add_argparse_args(parser)

args  =  parser.parse_args()
args.job_name  =  "ptv_video_classification"

# Default Parameters
args.on_cluster                     = False
args.job_name                       = "ptv_video_classification"
args.working_directory              = "."
args.partition                      = "dev"
args.lr                             = float(1.6e-3)
args.momentum                       = float(0.9)
args.weight_decay                   = float(5e-2)
args.video_path_prefix              = ""
args.data_type                      = "video"
args.video_means                    = tuple((0.45, 0.45, 0.45))
args.video_stds                     = tuple((0.225, 0.225, 0.225))
args.video_crop_size                = int(224)
args.video_min_short_side_scale     = int(256)
args.video_max_short_side_scale     = int(320)
args.video_horizontal_flip_p        = float(0.5)

# Trainer Parameters
args.workers                        = int(4)
args.batch_size                     = int(8)

# Data parameters
# args.framerate                      = int(8)
args.num_frames                     = int(32)
# args.clip_duration                  = float(args.num_frames/args.framerate)
# args.clip_duration                  = 1
args.stride                         = int(2)
args.num_classes                    = int(6)
args.shuffle                        = True

# Required Parameters
# args.data_root                      = 'D:\\zhaon\\Datasets\\Video_JPG_Stack'
args.data_root                      = r'C:\Users\zhaon\Documents\Video_JPG_Stack'
# args.data_root                      = 'D:\\zhaon\\Datasets\\Video Segments'
# args.vidclip_root                   = 'D:\\zhaon\\Datasets\\torch_VideoClips'
args.arch                           = "slowfast"
args.annotation_filename            = 'annotation_32x2.txt'

# Pytorch Lightning Parameters
args.accelerator                    = 'gpu'
args.devices                        = -1
# args.strategy                       = 'ddp'
# args.num_nodes                      = 1
args.max_epochs                     = 15
args.replace_sampler_ddp            = False
args.precision                      = 16
args.log_root                       = 'Logs'
# args.logger                         = TensorBoardLogger(args.log_root, name=f"{args.arch}_{date}")
args.log_every_n_steps              = 20

# Model-specific Parameters
args.ordinal                        = True
args.transfer_learning              = True
args.pretrained_state_dict          = 'Models/slowfast/SlowFast_new.pyth'
args.slowfast_alpha                 = int(4)
args.slowfast_beta                  = float(1/8)
args.slowfast_fusion_conv_channel_ratio = int(2)
args.slowfast_fusion_kernel_size        = int(7)
args.mvit_embed_dim_mul                  = [[1, 2.0], [3, 2.0], [14, 2.0]]
args.mvit_atten_head_mul                 = [[1, 2.0], [3, 2.0], [14, 2.0]]
args.mvit_pool_q_stride_size             = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
args.mvit_pool_kv_stride_adaptive        = [1, 8, 8]
args.mvit_pool_kvq_kernel                = [3, 3, 3]

# Debugging Parameters
# args.fast_dev_run                   = 1
# args.limit_train_batches            = 100
# args.limit_val_batches              = 1000
# args.enable_checkpointing           = True
# profiler = AdvancedProfiler(dirpath='Debug',filename='profilereport_'+date)
# profiler = PyTorchProfiler(dirpath='Debug',filename='profilereport_'+date)
# args.profiler                       = profiler

#print(args)

def main():
    setup_logger()

    # for subdir in os.listdir(args.data_root):
    # if True: # Dont want to unindent im lazy
    subdirs = ['Sub3', 'Sub6', 'Sub7', 'Sub12', 'Sub15', 'Sub17']
    # subdirs = ['Sub2', 'Sub3', 'Sub7', 'Sub9', 'Sub13', 'Sub16']
    for subdir in subdirs:
        args.val_sub = subdir
        # args.val_sub = 'Sub8'
        archtype = 'transfer' if args.transfer_learning else 'scratch'
        args.results_path = f'Results/{args.arch}_{archtype}_{date}'
        args.logger                         = TensorBoardLogger(args.log_root, 
                                                                name=f"{args.arch}_{archtype}_{date}",
                                                                version=f"{args.val_sub}_{date}")
        # DEBUG: start at later sub
        # skipsubs = ['Sub1','Sub10','Sub11','Sub12','Sub13','Sub14','Sub15']
        # if subdir in skipsubs: continue

        datamodule = GRASSP_classes.GRASSPFrameDataModule(args)
        # datamodule = GRASSP_classes.GRASSPFrameDataModule(args)
        classification_module = VideoClassificationLightningModule(args)
        trainer = pytorch_lightning.Trainer.from_argparse_args(args)
        # For some reason including callbacks in args causes a multiprocessing error ¯\_(ツ)_/¯
        trainer.callbacks.extend([LearningRateMonitor(), 
                                  GRASSP_classes.GRASSPValidationCallback(),
                                  EarlyStopping(monitor='val_MAE', mode='min', min_delta=0.01, patience=5)])

        # trainer.fit(classification_module, datamodule)
        # == Resume from checkpoint ==
        ckpt_root = Path('Models','slowfast_transfer_09_23_16')
        ckpt_path = Path(ckpt_root, next(x for x in os.listdir(ckpt_root) if f'{subdir}.ckpt' in x))
        trainer.fit(classification_module, datamodule, ckpt_path=ckpt_path)

        # Save checkpoint
        model_dir = Path('Models', f'{args.arch}_{archtype}_{date}')
        os.makedirs(model_dir, exist_ok=True)
        model_path = Path(model_dir, f"{args.arch}_{args.val_sub}.ckpt")
        trainer.save_checkpoint(model_path)

        metrics = trainer.validate(classification_module, datamodule)

        # Save metrics to json
        results_dir = Path(args.results_path, 'Metrics')
        os.makedirs(results_dir, exist_ok=True)
        savefile = Path(results_dir, args.val_sub + "_metrics.json")
        with open(savefile, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f'Saved metrics to {str(savefile)}')

if __name__ == '__main__':
    main()