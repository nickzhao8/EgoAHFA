import torch

####################
# SlowFast transform
####################

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
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