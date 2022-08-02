from json import decoder
from tokenize import String
import pytorch_lightning
from pytorch_lightning.callbacks import Callback
import pytorchvideo.data
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
import torch
from torch import Tensor

from torch.utils.data import DistributedSampler, RandomSampler, ChainDataset, ConcatDataset
from .transform_classes import PackPathway
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
)
from torchvideo.transforms import NormalizeVideo
from typing import Any, Callable, Iterable, Optional, Type, Dict, Tuple
from pathlib import Path
import json
import os, gzip

class GRASSPValidationCallback(Callback):
    #def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    #    import pdb; pdb.set_trace()
    #    print("Validation batch is starting")

    def on_validation_start(self, trainer, pl_module):
        #import pdb; pdb.set_trace()
        pl_module.val_preds = []
        pl_module.val_target = []
        pl_module.val_tasks = {}

        print("STARTING VALIDATION, RESETTING METRICS")
    
    def on_validation_end(self, trainer, pl_module):
        args = pl_module.args
        epoch = pl_module.trainer.current_epoch
        savepath = Path(args.results_path, "Raw")
        os.makedirs(savepath, exist_ok=True)
        savefile = Path(savepath, f"{args.val_sub}_epoch{epoch}_raw.json")
        metrics = {
            "preds":pl_module.val_preds,
            "target":pl_module.val_target,
            "tasks":pl_module.val_tasks
        }
        with open(savefile, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f'Saved raw results to {str(savefile)}')


class GRASSPDataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self, mode: str):
        """
        ##################
        # PTV Transforms #
        ##################

        # Each PyTorchVideo dataset has a "transform" arg. This arg takes a
        # Callable[[Dict], Any], and is used on the output Dict of the dataset to
        # define any application specific processing or augmentation. Transforms can
        # either be implemented by the user application or reused from any library
        # that's domain specific to the modality. E.g. for video we recommend using
        # TorchVision, for audio we recommend TorchAudio.
        #
        # To improve interoperation between domain transform libraries, PyTorchVideo
        # provides a dictionary transform API that provides:
        #   - ApplyTransformToKey(key, transform) - applies a transform to specific modality
        #   - RemoveKey(key) - remove a specific modality from the clip
        #
        # In the case that the recommended libraries don't provide transforms that
        # are common enough for PyTorchVideo use cases, PyTorchVideo will provide them in
        # the same structure as the recommended library. E.g. TorchVision didn't
        # have a RandomShortSideScale video transform so it's been added to PyTorchVideo.
        """
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
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(args.num_frames),
                    NormalizeVideo(args.video_means, args.video_stds),
                ]
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
                        PackPathway(self.args),
                    ] 
                    if self.args.arch == "slowfast"
                    else []
                )
            ),
        )

    def train_dataloader(self, **kwargs):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        # video_sampler = DistributedSampler if self.trainer._accelerator_connector.is_distributed else RandomSampler
        video_sampler = RandomSampler
        train_transform = self._make_transforms(mode="train")
        skipped_val = False
        subdirs = Path(self.args.data_root).glob('*')
        # labeled_video_paths = []
        datasets = []
        for subdir in subdirs:
            if str(subdir).split('\\')[-1].lower() == self.args.val_sub.lower():
                skipped_val = True
                continue
            subdir = Path(self.args.data_root,subdir).resolve()
            # for label in subdir.glob("*"):
            #     labelpath = Path(subdir, label)
            #     for vid in labelpath.glob('*'):
            #         vidpath = Path(labelpath,vid)
            #         labeled_video_paths.append((str(vidpath), int(str(label).split('\\')[-1])))
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
        )

    def val_dataloader(self, **kwargs):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        video_sampler = RandomSampler
        # video_sampler = DistributedSampler if self.trainer._accelerator_connector.is_distributed else RandomSampler
        val_transform = self._make_transforms(mode="val")
        made_val = False
        subdirs = Path(self.args.data_root).glob('*')
        #print(f"dataroot = {Path(self.args.data_root)}, valsub = {self.args.val_sub}")
        for subdir in subdirs:
            if str(subdir).split('\\')[-1].lower() == self.args.val_sub.lower():
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
        )

class GRASSPFastDataModule(GRASSPDataModule):
    def train_dataloader(self, **kwargs):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        # video_sampler = DistributedSampler if self.trainer._accelerator_connector.is_distributed else RandomSampler
        video_sampler = RandomSampler
        train_transform = self._make_transforms(mode="train")
        skipped_val = False
        subdirs = Path(self.args.data_root).glob('*')
        # labeled_video_paths = []
        datasets = []
        for subdir in subdirs:
            subname = str(subdir).split('\\')[-1]
            videoclip_file = Path(self.args.vidclip_root, f'{subname}_VideoClip.gz')
            if subname.lower() == self.args.val_sub.lower():
                skipped_val = True
                continue
            subdir = Path(self.args.data_root,subdir).resolve()
            dataset = GRASSPDataset(
                root                = subdir,
                frames_per_clip     = self.args.num_frames,
                frame_rate          = self.args.framerate,
                step_between_clips  = self.args.stride,
                transform           = train_transform,
                num_workers         = self.args.workers,
                videoclip_file      = videoclip_file,
            )
            datasets.append(dataset) 

        assert skipped_val == True, "Invalid val_sub; val_sub not found."
        self.train_dataset = ConcatDataset(datasets)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self, **kwargs):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        video_sampler = RandomSampler
        # video_sampler = DistributedSampler if self.trainer._accelerator_connector.is_distributed else RandomSampler
        val_transform = self._make_transforms(mode="val")
        made_val = False
        subdirs = Path(self.args.data_root).glob('*')
        #print(f"dataroot = {Path(self.args.data_root)}, valsub = {self.args.val_sub}")
        for subdir in subdirs:
            subname = str(subdir).split('\\')[-1]
            videoclip_file = Path(self.args.vidclip_root, f'{subname}_VideoClip.gz')
            if subname.lower() == self.args.val_sub.lower():
                made_val = True
                val_dataset = GRASSPDataset(
                    root                = subdir,
                    frames_per_clip     = self.args.num_frames,
                    frame_rate          = self.args.framerate,
                    step_between_clips  = self.args.stride,
                    transform           = val_transform,
                    num_workers         = self.args.workers,
                    videoclip_file      = videoclip_file,
                )
        assert made_val == True, "Invalid val_sub; val_sub not found."

        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class GRASSPDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        frames_per_clip: int,
        frame_rate: Optional[int] = None,
        step_between_clips: int = 1,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ("mp4"),
        num_workers: int = 1,
        videoclip_file: str = None,
        _video_width: int = 0,
        _video_height: int = 0,
    ) -> None:

        self.extensions = extensions

        self.root = root

        super().__init__(self.root)

        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        if videoclip_file is not None:
            self.video_clips = torch.load(gzip.GzipFile(videoclip_file))
        else:
            self.video_clips = VideoClips(
                video_list,
                frames_per_clip,
                step_between_clips,
                frame_rate,
                num_workers=num_workers,
                _video_width=_video_width,
                _video_height=_video_height,
            )
        self.transform = transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.video_clips.metadata

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video = video.permute(3,0,1,2)
        label = self.samples[video_idx][1]
        clip_dict = {
            'video':video.float(),
            'video_name':self.samples[video_idx][0],
            'video_index':video_idx,
            'clip_index':idx,
            'label':label,
        }

        if self.transform is not None:
            clip_dict = self.transform(clip_dict)

        return clip_dict
