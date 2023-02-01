from Data import GRASSP_classes
from Tools.tools import none_int_or_str
from GRASSPClassificationModule import GRASSPClassificationModule, setup_logger
import argparse
import pytorch_lightning
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.accelerators import CUDAAccelerator
import os
from pathlib import Path
import json
from datetime import datetime

# Static seed for reproducible results
pytorch_lightning.trainer.seed_everything(seed=1)
parser  =  argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve', prefix_chars='--')
date = datetime.now().strftime("%m_%d_%H")

# Default trainer parameters.
parser  =  pytorch_lightning.Trainer.add_argparse_args(parser)

# === System Parameters ===
parser.add_argument("--arch", default=None, required=True, type=str)
parser.add_argument("--ordinal", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--ordinal_strat", default=None, type=str)
parser.add_argument("--transfer_learning", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--finetune", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--pretrained_state_dict", default='Models/slowfast/slowfast_5class.pyth', type=str)
parser.add_argument("--sparse_temporal_sampling", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--results_path", default=None, type=str)

# LOSO-CV Parameters
parser.add_argument("--start_sub", default=1, type=int)
parser.add_argument("--end_sub", default=9, type=int)
parser.add_argument("--only_sub", default=None, action='append', type=int)

# Learning Rate Parameters
parser.add_argument("--lr"          , default=float(1.6e-3) , type=float) 
parser.add_argument("--momentum"    , default=float(0.9)    , type=float) 
parser.add_argument("--weight_decay", default=float(5e-2)   , type=float) 
parser.add_argument("--warmup"      , default=0             , type=float) 

# Trainer Parameters
parser.add_argument("--workers"     , default= int(4)       , type=int)
parser.add_argument("--batch_size"  , default= int(8)       , type=int)
parser.add_argument("--accumulate_grad_batches", default=int(1), type=int)

### DATASET parameters ###
parser.add_argument("--num_frames"         , default= int(32)                                       , type=int)
parser.add_argument("--stride"             , default= int(2)                                        , type=int)
parser.add_argument("--num_classes"        , default= int(6)                                        , type=int)
parser.add_argument("--shuffle"            , default= True                                          , action=argparse.BooleanOptionalAction)
parser.add_argument("--data_root"          , default= r'C:\Users\zhaon\Documents\GRASSP_JPG_FRAMES' , type=str)
parser.add_argument("--num_segments"       , default= 4                                             , type=int)
parser.add_argument("--frames_per_segment" , default= 8                                             , type=int)
parser.add_argument("--norm"               , default=True                                           , action=argparse.BooleanOptionalAction)

# Epoch Parameters
parser.add_argument("--max_epochs"          , default = 20   , type=int)  
parser.add_argument("--patience"            , default = 10   , type=int)

# Debugging Parameters
parser.add_argument("--fast_dev_run"         , default = False  )
parser.add_argument("--limit_train_batches"  , default = None   , type=none_int_or_str)
parser.add_argument("--limit_val_batches"    , default = None   , type=none_int_or_str )
parser.add_argument("--enable_checkpointing" , default = False , action=argparse.BooleanOptionalAction)
parser.add_argument("--profiler_type"    , default = None       , type=none_int_or_str )

args, _  =  parser.parse_known_args()

# Default Parameters
args.on_cluster                     = False
args.job_name                       = "GRASSP_Classification"
args.working_directory              = "."
args.partition                      = "gpu"
args.video_path_prefix              = ""
args.data_type                      = "video"

# Video Transform Parameters
args.video_means                    = tuple((0.45, 0.45, 0.45))
args.video_stds                     = tuple((0.225, 0.225, 0.225))
args.video_crop_size                = int(224)
args.video_min_short_side_scale     = int(256)
args.video_max_short_side_scale     = int(320)
args.video_horizontal_flip_p        = float(0.5)

# Dataset Parameters
args.num_frames                     = args.num_segments * args.frames_per_segment
if args.sparse_temporal_sampling:
    args.annotation_filename            = f'annotation_sparse_{args.num_segments}x{args.frames_per_segment}.txt'
    args.annotation_source              = 'annotation_32x2.txt'
else:
    args.annotation_filename            = 'annotation_32x2.txt'

# Pytorch Lightning Parameters
args.accelerator                    = 'gpu'
args.devices                        = -1
# args.strategy                       = 'ddp'
# args.num_nodes                      = 1
args.replace_sampler_ddp            = False
args.precision                      = 16
args.log_root                       = 'Logs'
# args.logger                         = TensorBoardLogger(args.log_root, name=f"{args.arch}_{date}")
args.log_every_n_steps              = 20

# Model-specific Parameters
args.slowfast_alpha                 = int(4)
args.slowfast_beta                  = float(1/8)
args.slowfast_fusion_conv_channel_ratio = int(2)
args.slowfast_fusion_kernel_size        = int(7)
args.mvit_embed_dim_mul                  = [[1, 2.0], [3, 2.0], [14, 2.0]]
args.mvit_atten_head_mul                 = [[1, 2.0], [3, 2.0], [14, 2.0]]
args.mvit_pool_q_stride_size             = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
args.mvit_pool_kv_stride_adaptive        = [1, 8, 8]
args.mvit_pool_kvq_kernel                = [3, 3, 3]

# Debug Parameters
if args.profiler_type == 'advanced': profiler = AdvancedProfiler(dirpath='Debug',filename='profilereport_'+date)
elif args.profiler_type == 'pytorch': profiler = PyTorchProfiler(dirpath='Debug',filename='profilereport_'+date)
else: profiler = None
args.profiler                       = profiler

def main():
    setup_logger()

    # Leave-one-subject-out cross validation
    for i in range(args.start_sub, args.end_sub + 1): # +1 because end_sub is inclusive
        subdir = f'Sub{i}'
        args.val_sub = subdir

        # Set up experiment name and result output location
        archtype = 'transfer' if args.transfer_learning else 'scratch'
        if args.ordinal: archtype = archtype + '_ordinal'
        if args.results_path is None:
            args.results_path = f'Results/{args.arch}_{archtype}_{date}'
        exp_name = args.results_path.split('/')[-1]

        args.logger                         = TensorBoardLogger(args.log_root, 
                                                                name=exp_name,
                                                                version=f"{args.val_sub}_{date}")
        # Only use these subs; skip everything else (subset training)
        if args.only_sub is not None: 
            subset = [f"Sub{x}" for x in args.only_sub]
            if subdir not in subset: continue

        datamodule = GRASSP_classes.GRASSPFrameDataModule(args)
        classification_module = GRASSPClassificationModule(args)
        trainer = pytorch_lightning.Trainer.from_argparse_args(args)
        trainer.callbacks[0] = TQDMProgressBar(refresh_rate=50) # overwriting progress bar
        # Including callbacks in args causes a multiprocessing error; append them here.
        trainer.callbacks.extend([LearningRateMonitor(), 
                                  GRASSP_classes.GRASSPValidationCallback(),
                                  EarlyStopping(monitor='val_MAE', mode='min', min_delta=0.01, patience=args.patience)])

        print(f"=== TRAINING. Start: {args.start_sub} End: {args.end_sub} Exp.Name: {exp_name} \
sparse: {args.sparse_temporal_sampling}  ===")
        # == RUN TRAINING LOOP == 
        trainer.fit(classification_module, datamodule)
        # == Resume from checkpoint ==
        # ckpt_root = Path('Models','slowfast_transfer_09_23_16')
        # ckpt_path = Path(ckpt_root, next(x for x in os.listdir(ckpt_root) if f'{subdir}.ckpt' in x))
        # trainer.fit(classification_module, datamodule, ckpt_path=ckpt_path)

        # Save checkpoint
        model_dir = Path('Models', exp_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = Path(model_dir, f"{args.arch}_{args.val_sub}.ckpt")
        trainer.save_checkpoint(model_path)

        # == RUN VALIDATION LOOP ==
        metrics = trainer.validate(classification_module, datamodule)

        # Save metrics to json
        results_dir = Path(args.results_path, 'Metrics')
        os.makedirs(results_dir, exist_ok=True)
        savefile = Path(results_dir, args.val_sub + "_metrics.json")
        with open(savefile, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f'Saved metrics to {str(savefile)}')

        # Save hyperparams
        hp_file = Path(results_dir, 'args.txt')
        with open(hp_file, 'w') as f:
            json.dump({'args':vars(args)}, f, default=GRASSP_classes.dumper, indent=4)

if __name__ == '__main__':
    main()