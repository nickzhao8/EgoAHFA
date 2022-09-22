import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_classes.GRASSP_classes import GRASSPFrameDataModule, GRASSPValidationCallback
from lightning_classes.lightningslowfast_debug import SlowfastModule
from PytorchVideoTrain import VideoClassificationLightningModule
from argparse import Namespace

def main():
    args = Namespace()

    args.data_type                      = "video"
    args.video_means                    = tuple((0.45, 0.45, 0.45))
    args.video_stds                     = tuple((0.225, 0.225, 0.225))
    args.video_crop_size                = int(224)
    args.video_min_short_side_scale     = int(256)
    args.video_max_short_side_scale     = int(320)
    args.video_horizontal_flip_p        = float(0.5)

    # Trainer Parameters
    args.workers                        = int(4)
    args.batch_size                     = int(2)

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
    args.lr = 1e-5
    args.momentum = 0.9
    args.weight_decay = 1e-5
    args.accelerator                    = 'gpu'
    args.devices                        = -1
    # args.strategy                       = 'ddp'
    args.max_epochs = 1
    # args.callbacks   = [LearningRateMonitor()]

    args.transfer_learning = False
    args.slowfast_alpha                 = int(4)
    args.slowfast_beta                  = float(1/8)
    args.slowfast_fusion_conv_channel_ratio = int(2)
    args.slowfast_fusion_kernel_size        = int(7)

    args.log_every_n_steps = 50
    args.fast_dev_run                   = 10
    # args.limit_train_batches            = 100
    # args.limit_val_batches              = 1000
    args.enable_checkpointing           = False

    args.val_sub = 'Sub3'
    
    datamodule = GRASSPFrameDataModule(args)
    vidclassmodule = VideoClassificationLightningModule(args)
    vidclassmodule.val_preds = []
    vidclassmodule.val_target = []
    vidclassmodule.val_tasks = {}
    vidclassmodule.val_filenames = {}
    vidclassmodule.val_loss = []
    lightningmodule = SlowfastModule(args)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    trainer.callbacks.insert(0,LearningRateMonitor())

    args.callbacks = [LearningRateMonitor()]
    trainer_fail = pytorch_lightning.Trainer.from_argparse_args(args)

    trainer.fit(vidclassmodule, datamodule)
    metrics = trainer.validate(vidclassmodule, datamodule)


if __name__ == '__main__':
    main()