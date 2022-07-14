from json import decoder
from tokenize import String
import pytorch_lightning
from pytorch_lightning.callbacks import Callback
import pytorchvideo.data
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
import torch

from torch.utils.data import DistributedSampler, RandomSampler, ChainDataset
from .transform_classes import PackPathway
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchvideo.transforms import NormalizeVideo
from typing import Any, Callable, Iterable, Optional, Type
from pathlib import Path
import json
import os

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

class ChainLabeledVideoDataset(LabeledVideoDataset):
    r"""Dataset for chaining multiple LabeledVideoDatasets.
        This class is useful to assemble different existing dataset streams. The
        chaining operation is done on-the-fly, so concatenating large-scale
        datasets with this class will be efficient.
    """

    def __init__(
        self,
        datasets: Iterable[LabeledVideoDataset],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                LabeledVideoDatasets.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._datasets = datasets
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    def __iter__(self):
        for d in self._datasets:
            assert isinstance(d, LabeledVideoDataset), "ChainLabeledVideoDataset only supports LabeledVideoDataset"
            for x in d:
                yield x

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        for d in self._datasets:
            assert isinstance(d, LabeledVideoDataset), "ChainLabeledVideoDataset only supports LabeledVideoDataset"
            for x in d:
                yield x
        #return self

    def __len__(self):
        total = 0
        for d in self._datasets:
            assert isinstance(d, LabeledVideoDataset), "ChainLabeledVideoDataset only supports LabeledVideoDataset"
            total += len(d)
        return total