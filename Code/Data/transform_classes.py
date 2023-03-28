import torch

####################
# SlowFast transform
####################

class PackPathway(torch.nn.Module):
    """
    Transform for splitting data into slow and fast streams for SlowFast. 
    """
    def __init__(self, args):
        self.args = args
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.args.slowfast_alpha
            ).long(),
        )
        #print(frames.shape[1], self.args.slowfast_alpha)
        frame_list = [slow_pathway, fast_pathway]
        #print(frame_list[0].shape, frame_list[1].shape)
        return frame_list

class ApplyTransformToSlow:
    """
    Apply transform to slow pathway, after data has been split into slow and fast pathways. 
    """
    def __init__(self, transform):
        self.transform = transform
        super().__init__()
    def __call__(self, frame_list: list):
        return [self.transform(frame_list[0]), frame_list[1]]

class ApplyTransformToFast:
    """
    Apply transform to fast pathway, after data has been split into slow and fast pathways. 
    """
    def __init__(self, transform):
        self.transform = transform
        super().__init__()
    def __call__(self, frame_list: list):
        return [frame_list[0], self.transform(frame_list[1])]

class MaskPatches(torch.nn.Module):
    """
    Transform for masking patches in video tensor. 'frame' mode = each frame has a different random masking. 
    'video' mode = all frames in the video have the same masking pattern. 
    """
    def __init__(self, patch_size: int = 14, mask_ratio: float = 0.50, mode: str = 'frame'):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mode = mode
        assert mode in ['frame', 'video'], "Invalid mode given. mode must be in [\'frame\', \'video\']"
        super().__init__()
    
    def forward(self, frames: torch.Tensor):
        # Assume frames is a tensor of [...,T,H,W], and H=W (square). 
        assert frames.shape[-1] == frames.shape[-2], "Invalid shape. Expected a tensor of shape [C,T,H,W], where H=W."
        sidelen = frames.shape[-1]//self.patch_size
        num_patches = sidelen*sidelen
        num_masked_patches = int(num_patches * self.mask_ratio)
        if self.mode == 'frame':
            for i in range(frames.shape[-3]):
                random_indices = torch.randperm(num_patches)[:num_masked_patches]
                for idx in random_indices:
                    h_idx = idx % sidelen
                    w_idx = idx // sidelen
                    # Set masked patch to 0
                    frames[:,i,h_idx*self.patch_size:h_idx*self.patch_size+self.patch_size,
                        w_idx*self.patch_size:w_idx*self.patch_size+self.patch_size] = 0.0
        elif self.mode == 'video':
            random_indices = torch.randperm(num_patches)[:num_masked_patches]
            for i in range(frames.shape[-3]):
                for idx in random_indices:
                    h_idx = idx % sidelen
                    w_idx = idx // sidelen
                    # Set masked patch to 0
                    frames[:,i,h_idx*self.patch_size:h_idx*self.patch_size+self.patch_size,
                        w_idx*self.patch_size:w_idx*self.patch_size+self.patch_size] = 0.0
        return frames
