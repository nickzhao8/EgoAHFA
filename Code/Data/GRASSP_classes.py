from .video_dataset import VideoFrameDataset, ImglistToTensor, VideoRecord
from .gen_annotations import gen_annotations, gen_sparse_annotations
import pytorch_lightning
from pytorch_lightning.callbacks import Callback
import pytorchvideo.data
import torch

from torch.utils.data import DistributedSampler, RandomSampler, ChainDataset, ConcatDataset
from .transform_classes import PackPathway, MaskPatches, ApplyTransformToFast, ApplyTransformToSlow
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import make_dataset, find_classes
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandAugment,
    ConvertImageDtype,
)
from torchvision.ops import Permute
from torchvideo.transforms import NormalizeVideo
from typing import Any, Callable, Iterable, Optional, Type, Dict, Tuple, Union, List
from pathlib import Path
import json
import os, gzip

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return str(obj)

class GRASSPValidationCallback(Callback):
    '''
    Helper callback for manual logging during validation.
    '''
    def __init__(self) -> None:
        super().__init__()
    # def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    #    import pdb; pdb.set_trace()
    #    print("Validation batch is starting")
    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.sanity_check = True
    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.sanity_check = False

    def on_validation_start(self, trainer, pl_module):
        #import pdb; pdb.set_trace()
        pl_module.val_preds = []
        pl_module.val_target = []
        pl_module.val_tasks = {}
        pl_module.val_filenames = {}
        pl_module.val_loss = []

        print("STARTING VALIDATION, RESETTING METRICS")
    
    def on_validation_end(self, trainer, pl_module):
        if not pl_module.sanity_check:
            args = pl_module.args
            epoch = pl_module.trainer.current_epoch
            savepath = Path(args.results_path, "Raw")
            os.makedirs(savepath, exist_ok=True)
            savefile = Path(savepath, f"{args.val_sub}_epoch{epoch}_raw.json")
            metrics = {
                "preds":pl_module.val_preds,
                "target":pl_module.val_target,
                "loss":pl_module.val_loss,
                "tasks":pl_module.val_tasks,
                "filenames":pl_module.val_filenames,
                'args':vars(pl_module.args),
            }
            with open(savefile, 'a') as f:
                json.dump(metrics, f, default=dumper, indent=4)
            print(f'Saved raw results to {str(savefile)}')
    
    def on_fit_end(self, trainer, pl_module):
        pl_module.logger.save()
        
class GRASSPDataModule(pytorch_lightning.LightningDataModule):
    """
    Implements LightningDataModule for Adapted GRASSP-annotated ANS-SCI (HomeLab) dataset. 
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self, mode: str):
        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        else:
            raise Exception(f"{self.args.data_type} not supported")

        return Compose(transform)

    def _video_transform(self, mode: str):
        """
        Define video transforms for data augmentation. 
        """
        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    #UniformTemporalSubsample(args.num_frames),
                ]
                +(
                    [
                        RandAugment(),
                        ConvertImageDtype(torch.float32),
                        Permute([1,0,2,3]),
                    ]
                    if self.args.randaug
                    else []
                )
                +(
                    [
                        NormalizeVideo(args.video_means, args.video_stds),
                    ]
                    if self.args.norm
                    else []
                )
                + (
                    [
                        RandomShortSideScale(
                            min_size=args.video_min_short_side_scale,
                            max_size=args.video_max_short_side_scale,
                        ),
                        RandomCrop(args.video_crop_size),
                        RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(args.video_min_short_side_scale),
                        CenterCrop(args.video_crop_size),
                    ]
                )
                +(
                    [
                        MaskPatches(args.patch_size, args.mask_ratio, args.maskmode),
                    ]
                    if self.args.maskpathway == "full" and mode == "train"
                    else []
                )
                +(
                    [
                        PackPathway(self.args),

                    ] 
                    if self.args.arch == "slowfast"
                    else []
                )
                +(
                    [
                        ApplyTransformToSlow(MaskPatches(args.patch_size, args.mask_ratio, args.maskmode)),
                    ] if self.args.maskpathway == "slow" and self.args.arch == "slowfast" and mode == "train"
                    else []
                )
            ),
        )

    def train_dataloader(self, **kwargs):
        """
        Define train dataloader for LOSO-CV. 
        """
        video_sampler = RandomSampler
        train_transform = self._make_transforms(mode="train")
        skipped_val = False
        subdirs = Path(self.args.data_root).glob('*')
        datasets = []
        for subdir in subdirs:
            if subdir.name.lower() == self.args.val_sub.lower():
                skipped_val = True
                continue
            subdir = Path(self.args.data_root,subdir).resolve()
            dataset = pytorchvideo.data.labeled_video_dataset(
                data_path = subdir,
                clip_sampler=pytorchvideo.data.UniformClipSampler(
                    clip_duration=self.args.clip_duration,
                    stride=0.5,
                    backpad_last=True,
                ),
                transform=train_transform,
                decode_audio=False,
            )
            datasets.append(dataset) 

        assert skipped_val == True, "Invalid val_sub; val_sub not found."
        self.train_dataset = ChainDataset(datasets)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )

    def val_dataloader(self, **kwargs):
        """
        Define val dataloader for LOSO-CV.
        """
        video_sampler = RandomSampler
        # video_sampler = DistributedSampler if self.trainer._accelerator_connector.is_distributed else RandomSampler
        val_transform = self._make_transforms(mode="val")
        made_val = False
        subdirs = Path(self.args.data_root).glob('*')
        #print(f"dataroot = {Path(self.args.data_root)}, valsub = {self.args.val_sub}")
        for subdir in subdirs:
            if subdir.name.lower() == self.args.val_sub.lower():
                made_val = True
                self.val_dataset = pytorchvideo.data.labeled_video_dataset(
                    data_path=subdir.resolve(),
                    clip_sampler=pytorchvideo.data.UniformClipSampler(
					    clip_duration=self.args.clip_duration,
					    stride=0.5,
					    backpad_last=True,
                    ),
                    #clip_sampler=pytorchvideo.data.RandomClipSampler(
                    #    clip_duration = self.args.clip_duration
                    #),
                    video_path_prefix=self.args.video_path_prefix,
                    transform=val_transform,
                    # video_sampler=video_sampler,
                    decode_audio=False,
                )
        assert made_val == True, "Invalid val_sub; val_sub not found."

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )

class GRASSPFrameDataModule(GRASSPDataModule):
    def __init__(self, args):
        self.args = args
        subdir = next(iter(Path(self.args.data_root).glob('*')))
        # If absent, generate annotations that are used for VideoFrameDataset loading. 
        if not os.path.exists(Path(subdir, self.args.annotation_filename)):
            if args.sparse_temporal_sampling:
                gen_sparse_annotations(dataset_root = self.args.data_root,
                                       annotation_source_filename = self.args.annotation_source,
                                       annotation_filename  = self.args.annotation_filename)
            else:
                gen_annotations(dataset_root = self.args.data_root,
                                num_frames = self.args.num_frames,
                                temporal_stride = self.args.stride,
                                annotation_filename = self.args.annotation_filename)
        super().__init__(args)

    def train_dataloader(self, **kwargs):
        """
        Build train dataset and dataloader for LOSO-CV.
        """
        train_transform = self._make_transforms(mode="train")
        skipped_val = False
        subdirs = Path(self.args.data_root).glob('*')
        # Load all train subjects. Skip val subjects.
        datasets = []
        for subdir in subdirs:
            subname = subdir.name
            annotation_file = Path(subdir, self.args.annotation_filename)
            if subname.lower() == self.args.val_sub.lower():
                skipped_val = True
                continue
            subdir = Path(self.args.data_root,subdir).resolve()
            # Build dataset and append to datasets list. 
            if self.args.sparse_temporal_sampling:
                dataset = GRASSPFrameDataset(
                    root_path           = subdir,
                    annotationfile_path = annotation_file,
                    num_segments        = self.args.num_segments,
                    frames_per_segment  = self.args.frames_per_segment,
                    imagefile_template  = '{:04d}.jpg',
                    transform           = train_transform,
                    randaug             = self.args.randaug,
                )
            else:
                dataset = GRASSPFrameDataset(
                    root_path           = subdir,
                    annotationfile_path = annotation_file,
                    num_segments        = 1,
                    frames_per_segment  = self.args.num_frames,
                    imagefile_template  = '{:04d}.jpg',
                    transform           = train_transform,
                    randaug             = self.args.randaug,
                )
            datasets.append(dataset) 

        assert skipped_val == True, "Invalid val_sub; val_sub not found."
        # Build ConcatDataset from datasets list
        train_dataset = ConcatDataset(datasets)
        # Video sampler (for distributed only)
        if self.trainer._accelerator_connector.is_distributed : self.train_sampler = DistributedSampler(train_dataset)
        elif self.args.shuffle: self.train_sampler = RandomSampler(train_dataset)
        else:                   self.train_sampler = None
        return torch.utils.data.DataLoader(
            train_dataset,
            sampler=self.train_sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )

    def val_dataloader(self, **kwargs):
        """
        Build val dataset and dataloader for LOSO-CV.
        """
        val_transform = self._make_transforms(mode="val")
        made_val = False
        subdirs = Path(self.args.data_root).glob('*')
        # Skip train subjects. Load val subject.
        for subdir in subdirs:
            subname = subdir.name
            annotation_file = Path(subdir, self.args.annotation_filename)
            if subname.lower() == self.args.val_sub.lower():
                made_val = True
                # Build dataset
                if self.args.sparse_temporal_sampling:
                    ## FIXME: val dataset has random start index selection for sparse sampling
                    val_dataset = GRASSPFrameDataset(
                        root_path           = subdir,
                        annotationfile_path = annotation_file,
                        num_segments        = self.args.num_segments,
                        frames_per_segment  = self.args.frames_per_segment,
                        imagefile_template  = '{:04d}.jpg',
                        transform           = val_transform,
                        randaug             = False,
                    )
                else:
                    val_dataset = GRASSPFrameDataset(
                        root_path           = subdir,
                        annotationfile_path = annotation_file,
                        num_segments        = 1,
                        frames_per_segment  = self.args.num_frames,
                        imagefile_template  = '{:04d}.jpg',
                        transform           = val_transform,
                        randaug             = False,
                    )
        assert made_val == True, "Invalid val_sub; val_sub not found."

        val_sampler = DistributedSampler(val_dataset) if self.trainer._accelerator_connector.is_distributed else None
        return torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )

class GRASSPFrameDataset (VideoFrameDataset):
    """
    Implementation of VideoFrameDataset to work with ANS-SCI dataset and GRASSPClassificationModule. 
    Main change is to always output a clip dict with the following keys:
    1. video        -->     tensor containing video frames
    2. video name   -->     name of video, including task name
    3. clip index  
    4. label        -->     ground truth label
    """
    def __init__(self,
                root_path: str,
                annotationfile_path: str,
                num_segments: int = 1,
                frames_per_segment: int = 16,
                imagefile_template: str = '{:04d}.jpg',
                transform=None,
                randaug=False,
                test_mode: bool = False):
        self.randaug=randaug
        super().__init__(root_path, annotationfile_path, num_segments, frames_per_segment, imagefile_template, transform, test_mode)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.
        """
        record: VideoRecord = self.video_list[idx]

        frame_start_indices: 'np.ndarray[int]' = self._get_start_indices(record)

        return self._get(record, frame_start_indices, idx)

    def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]', idx: int) -> Dict[str, Any]:
        """
        Loads the frames of a video at the corresponding indices. 
        Returns clip dict with video, video_name, clip_index, and label. 
        """

        frame_start_indices = frame_start_indices + record.start_frame
        images = list()

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path, frame_index)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1
        
        # If using RandAugment, convert PIL image to uint8 tensor, else float32 tensor.
        image_type = "uint8" if self.randaug else "float32"
        # By default, transform PIL Images to Tensor
        images = ImglistToTensor.forward(images, image_type)
        if not self.randaug:
            images = images.permute(1,0,2,3)

        clip_dict = {
            'video':images,
            'video_name':record.path,
            'clip_index':idx,
            'label':record.label,
        }

        if self.transform is not None:
            clip_dict = self.transform(clip_dict)

        return clip_dict
