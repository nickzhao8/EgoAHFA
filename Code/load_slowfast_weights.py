import torch
import pytorchvideo.models

def change_key(self, old, new):
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v

def load_slowfast_weights(  weight_path: str,
):
    f = torch.load(weight_path)
    state_dict = f['model_state']
    opt_state = f['optimizer_state']
    cfg = f['cfg']

    model = pytorchvideo.models.create_slowfast(
        input_channels=(3,3),
        model_num_class=6,
        slowfast_channel_reduction_ratio=8,
        slowfast_conv_channel_fusion_ratio=2,
    )
    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = ''
        substr = key.split('.')
        if substr[0] == 'head': continue
        blocknum = int(substr[0][1]) - 1
        if 'fuse' not in substr[0]:
            new_key = 'blocks.' + str(blocknum) + '.multipathway_blocks.'
            pathnum = int(substr[1][7])
            new_key = new_key + str(pathnum)  + '.'
            if blocknum == 0:
                layer_index = 2
            elif blocknum > 0:
                resnum = int(substr[1][-1])
                if '_' in substr[2] or substr[3] == 'weight': 
                    new_key = new_key + 'res_blocks.' + str(resnum) + '.'
                    layer_index = 2
                else: 
                    new_key = new_key + 'res_blocks.' + str(resnum) + '.' + substr[2] + '.'
                    layer_index = 3
            if substr[layer_index] == 'conv':
                new_key = new_key + 'conv.' + substr[layer_index+1]
            elif substr[layer_index] == 'bn':
                new_key = new_key + 'norm.' + substr[layer_index+1]
            elif substr[layer_index] == 'a_bn':
                new_key = new_key + 'norm_a.' + substr[layer_index+1]
            elif substr[layer_index] == 'b_bn':
                new_key = new_key + 'norm_b.' + substr[layer_index+1]
            elif substr[layer_index] == 'c_bn':
                new_key = new_key + 'norm_c.' + substr[layer_index+1]
            elif substr[layer_index] == 'a':
                new_key = new_key + 'conv_a.' + substr[layer_index+1]
            elif substr[layer_index] == 'b':
                new_key = new_key + 'conv_b.' + substr[layer_index+1]
            elif substr[layer_index] == 'c':
                new_key = new_key + 'conv_c.' + substr[layer_index+1]
            elif substr[layer_index] == 'branch1':
                new_key = new_key + 'branch1_conv.' + substr[layer_index+1]
            elif substr[layer_index] == 'branch1_bn':
                new_key = new_key + 'branch1_norm.' + substr[layer_index+1]
        
        else:
            new_key = new_key + 'blocks.' + str(blocknum) + '.multipathway_fusion.'
            if substr[1] == 'conv_f2s':
                new_key = new_key + 'conv_fast_to_slow.' + substr[2]
            elif substr[1] == 'bn':
                new_key = new_key + 'norm.' + substr[2]
    
        change_key(state_dict, key, new_key)

    model.load_state_dict(state_dict, strict=False)
    torch.save(model.state_dict(), 'Models/SlowFast_new.pyth')