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
args.batch_size                     = int(1)
args.clip_duration                  = float(1)
args.num_frames                     = int(32)
args.num_classes                    = int(6)

# Pytorch Lightning Parameters
args.accelerator                    = 'gpu'
args.devices                        = -1
# args.strategy                       = 'ddp'
args.max_epochs                     = 2
args.callbacks                      = [LearningRateMonitor(), GRASSP_classes.GRASSPValidationCallback()]
args.replace_sampler_ddp            = False
# args.replace_sampler_ddp            = True

# Required Parameters
args.data_root                      = 'M:\\Wearable Hand Monitoring\\CODE AND DOCUMENTATION\\Nick Z\\Code\\GRASSP Annotation\\Video Segments'
args.arch                           = "slowfast"
args.val_sub                        = "Sub1"
args.results_path                   = "Results/"+args.arch

# Debugging Parameters
# args.fast_dev_run                   = 10
# profiler = AdvancedProfiler(dirpath='Debug',filename='profilereport_'+date)
#args.profiler                       = profiler

#print(args)

def main():
    setup_logger()

    args.val_sub = "Sub1"

    datamodule = GRASSP_classes.GRASSPDataModule(args)
    classification_module = VideoClassificationLightningModule(args)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    
    #trainer.fit(classification_module, datamodule)

    # Load from checkpoint
    modelpath = Path('Models', args.arch, 'slowfast_Sub1.ckpt')
    model = VideoClassificationLightningModule.load_from_checkpoint(modelpath)

    import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()