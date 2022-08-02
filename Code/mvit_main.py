from lightning_classes import GRASSP_classes
from PytorchVideoTrain import VideoClassificationLightningModule, setup_logger
import argparse
import pytorch_lightning
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from pathlib import Path
import json
from datetime import datetime

date = datetime.now().strftime("%m_%d_%H%M")
#os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

pytorch_lightning.trainer.seed_everything(seed=1)
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

# Adjustable Parameters
args.workers                        = int(4)
args.batch_size                     = int(1)
args.clip_duration                  = float(1)
args.framerate                      = int(30)
args.num_frames                     = int(30)
args.stride                         = int(args.num_frames/2)
args.num_classes                    = int(6)

# Pytorch Lightning Parameters
args.accelerator                    = 'gpu'
args.devices                        = 1
# args.strategy                       = 'ddp'
args.max_epochs                     = 20
args.callbacks                      = [LearningRateMonitor(), GRASSP_classes.GRASSPValidationCallback()]
args.replace_sampler_ddp            = False
# args.replace_sampler_ddp            = True

# Required Parameters
args.data_root                      = 'D:\\zhaon\\Datasets\\Video Segments'
args.vidclip_root                   = 'D:\\zhaon\\Datasets\\torch_VideoClips'
args.arch                           = "mvit"

# Model-specific Parameters
args.slowfast_alpha                 = int(4)
args.slowfast_beta                  = float(1/8)
args.mvit_embed_dim_mul                  = [[1, 2.0], [3, 2.0], [14, 2.0]]
args.mvit_atten_head_mul                 = [[1, 2.0], [3, 2.0], [14, 2.0]]
args.mvit_pool_q_stride_size             = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
args.mvit_pool_kv_stride_adaptive        = [1, 8, 8]
args.mvit_pool_kvq_kernel                = [3, 3, 3]

# Debugging Parameters
# args.fast_dev_run                   = 30
# profiler = AdvancedProfiler(dirpath='Debug',filename='profilereport_'+date)
#args.profiler                       = profiler

#print(args)

def main():
    setup_logger()

    # for subdir in os.listdir(args.data_root):
    if True: # Dont want to unindent im lazy
        args.val_sub = 'Sub3'
        args.results_path = f'../Results/{args.arch}'

        # DEBUG: start at later sub
        # skipsubs = ['Sub1','Sub10','Sub11','Sub12','Sub13','Sub14','Sub15']
        # if subdir in skipsubs: continue

        datamodule = GRASSP_classes.GRASSPFastDataModule(args)
        classification_module = VideoClassificationLightningModule(args)
        trainer = pytorch_lightning.Trainer.from_argparse_args(args)

        trainer.fit(classification_module, datamodule)

        # Save checkpoint
        model_dir = Path('..', 'Models', args.arch)
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