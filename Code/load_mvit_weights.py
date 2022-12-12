import torch
from torchvision.models.video.mvit import mvit_v2_s

def change_key(self, old, new):
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v

def load_mvit_weights(  weight_path: str,
):
    f = torch.load(weight_path, map_location=torch.device('cpu'))
    state_dict = f['model_state']
    opt_state = f['optimizer_state']
    cfg = f['cfg']

    model = mvit_v2_s()
    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = ''
        substr = key.split('.')

        if key == 'cls_token': new_key = 'pos_encoding.class_token'
        elif 'patch_embed' in key: new_key = key.replace('patch_embed.proj', 'conv_proj')
        elif 'head.projection' in key: new_key = key.replace('head.projection', 'head.1')
        elif 'attn.proj' in key: new_key = key.replace('attn.proj', 'attn.project.0')
        elif 'attn.pool_q' in key: new_key = key.replace('attn.pool_q', 'attn.pool_q.pool')
        elif 'attn.norm_q' in key: new_key = key.replace('attn.norm_q', 'attn.pool_q.norm_act.0')
        elif 'attn.pool_k' in key: new_key = key.replace('attn.pool_k', 'attn.pool_k.pool')
        elif 'attn.norm_k' in key: new_key = key.replace('attn.norm_k', 'attn.pool_k.norm_act.0')
        elif 'attn.pool_v' in key: new_key = key.replace('attn.pool_v', 'attn.pool_v.pool')
        elif 'attn.norm_v' in key: new_key = key.replace('attn.norm_v', 'attn.pool_v.norm_act.0')
        elif 'mlp.fc1' in key: new_key = key.replace('mlp.fc1', 'mlp.0')
        elif 'mlp.fc2' in key: new_key = key.replace('mlp.fc2', 'mlp.3')
        elif 'proj' in key: new_key = key.replace('proj', 'project')

        else: continue

    
        change_key(state_dict, key, new_key)

    # Fix mismatched tensor shapes
    state_dict['pos_encoding.class_token'] = torch.squeeze(state_dict['pos_encoding.class_token'])
    # Remove final output layers
    state_dict.pop('head.1.weight', None)
    state_dict.pop('head.1.bias', None)
    model.load_state_dict(state_dict, strict=False)
    torch.save(model.state_dict(), '../Models/mvit_SSv2/mvitv2_s_ssv2_new.pyth')