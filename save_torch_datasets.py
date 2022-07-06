from lightning_classes import GRASSP_classes
from pathlib import Path
import os
from torch.utils.data import RandomSampler

data_root = Path('Video Segments')
subdirs = os.listdir(data_root)

for val_sub in subdirs:
    for subdir in subdirs:
        video_sampler = RandomSampler

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        # video_sampler = RandomSampler
        video_sampler = DistributedSampler if self.trainer._accelerator_connector.is_distributed else RandomSampler
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