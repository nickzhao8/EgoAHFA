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
from grad_cam_slowfast import grad_cam, to_RGB, superimpose
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob

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
args.data_root                      = 'D:\\zhaon\\Datasets\\Video Segments'
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

    for k in range(1,18):
        args.val_sub = f"Sub{k}"

        datamodule = GRASSP_classes.GRASSPDataModule(args)
        val_dataloader = datamodule.val_dataloader(shuffle=True)
        classification_module = VideoClassificationLightningModule(args)
    
        # Load from checkpoint
        modelroot = Path('..','..','..','GRASSP Annotation','Models')
        modelpath = Path(modelroot, args.arch, f'slowfast_Sub{k}.ckpt')
        model = classification_module.load_from_checkpoint(modelpath, args=args)

        # pathways = ['slow', 'fast']
        # target_layers = [model.model.blocks[4].multipathway_blocks[0].res_blocks[1],
        #                 model.model.blocks[4].multipathway_blocks[1].res_blocks[1]]
        pathways = ['fast']
        target_layers = [model.model.blocks[4].multipathway_blocks[1].res_blocks[1]]
        for l, layer in enumerate(target_layers):
            for j in range(10):
                vid_dict = next(iter(val_dataloader)) 
                vid_tensors = vid_dict['video']

                input_vid = vid_tensors[0]

                heatmap = grad_cam(model.model, vid_tensors, layer)

                # Convert video to list of image tensors and superimpose heatmap
                imgs = []
                saveroot = Path('..','Results', args.arch, 'Grad-CAM', pathways[l], f"Sub{k}", str(j))
                os.makedirs(saveroot, exist_ok=True)
                for i in range(input_vid.shape[2]):
                    img = input_vid.index_select(2, torch.tensor(i)).squeeze()
                    img = to_RGB(img)
                    superimposed_img = superimpose(img, heatmap[i])
                    imgs.append(superimposed_img)

                    plt.imshow(imgs[i])
                    plt.savefig(Path(saveroot, f'slowfast_Sub{k}_{j}_{i}'))

                # Convert to .gif
                frames = []
                imgs = glob.glob(str(Path(saveroot, '*.png')))
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)
 
                # Save into a GIF file that loops forever
                frames[0].save(Path(saveroot, f'slowfast_Sub{k}_{j}.gif'), 
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=300, loop=0)

if __name__ == '__main__':
    main()