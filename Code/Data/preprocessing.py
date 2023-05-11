import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import Permute
from kornia.augmentation import VideoSequential, Normalize, CenterCrop, Resize, RandomResizedCrop, RandomHorizontalFlip
from kornia.utils import image_list_to_tensor
from pytorchvideo.transforms import ApplyTransformToKey
from .transform_classes import MaskPatches, ApplyTransformToFast, ApplyTransformToSlow, PackPathway
from .video_dataset import ImglistToTensor

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, mode: str, args) -> None:
        super().__init__()
        self.args = args
        self.transforms = self._make_transforms(mode)

    def _make_transforms(self, mode: str):
        transforms = VideoSequential(data_format="BCTHW", same_on_frame=True)
        if mode == "train":
            if self.args.randaug:
                raise TypeError("RandAug is not supported.")
            if self.args.norm:
                transforms.append(Normalize(self.args.video_means, self.args.video_stds))
            # Random resizing, cropping, and flipping
            transforms.append(RandomResizedCrop(
                size=(self.args.video_crop_size,self.args.video_crop_size),
                scale=(0.2761, 0.4313),
                ratio=(1.0, 1.0),
            ))
            transforms.append(RandomHorizontalFlip(p=self.args.video_horizontal_flip_p))
            # Masking and SlowFast Pathway split
            # These transforms are done on video (outside of VideoSequential)
            transforms = nn.Sequential(transforms)
            if self.args.maskpathway == "full":
                transforms.append(MaskPatches(self.args.patch_size,self.args.mask_ratio,self.args.maskmode))
            if self.args.arch == "slowfast":
                transforms.append(PackPathway(self.args))
                if self.args.maskpathway == "slow":
                    transforms.append(ApplyTransformToSlow(
                        MaskPatches(self.args.patch_size,self.args.mask_ratio,self.args.maskmode)
                    ))
                elif self.args.maskpathway == "fast":
                    transforms.append(ApplyTransformToFast(
                        MaskPatches(self.args.patch_size,self.args.mask_ratio,self.args.maskmode)
                    ))
        elif mode == "val":
            transforms.append(Resize(size=self.args.video_min_short_side_scale,side="short"))
            transforms.append(CenterCrop(self.args.video_crop_size))
            # These transforms are done on video (outside of VideoSequential)
            transforms = nn.Sequential(transforms)
            if self.args.arch == "slowfast":
                transforms.append(PackPathway(self.args))
        return (transforms)


    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  
        return x_out

class PreProcess(nn.Module):
    """Module to perform pre-process using Kornia on video dict."""
    def __init__(self) -> None:
        super().__init__()
        self.preprocess = ApplyTransformToKey(
            key="video",
            transform=nn.Sequential(
                ImglistToTensor(),
                Permute([1,0,2,3])
            )
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> dict:
        x_out: dict = self.preprocess(x)
        return x_out