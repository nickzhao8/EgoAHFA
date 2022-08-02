import torch
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips
from pathlib import Path
import os, glob, gzip


data_root = 'D:\\zhaon\\Datasets\\Video Segments'
subdirs = os.listdir(data_root)

for subdir in subdirs:
    subdir_path = Path(data_root, subdir)
    extensions = ('mp4')
    classes, class_to_idx = find_classes(subdir_path)
    samples = make_dataset(subdir_path, class_to_idx, extensions, is_valid_file=None)
    video_list = [x[0] for x in samples]
    video_clips = VideoClips(
        video_list,
        30, # Frames per clip
        15, # Step between clips
        30, # Framerate
    )

    savepath = 'D:\\zhaon\\Datasets\\torch_VideoClips'
    os.makedirs(savepath, exist_ok=True)

    torch.save(video_clips, gzip.GzipFile(Path(savepath, subdir + '_VideoClip.gz'),'wb'))