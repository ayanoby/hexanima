import json, struct, urllib.request, torch, numpy as np
from safetensors import safe_open
from huggingface_hub import get_token
TOK = get_token()
URL = 'https://huggingface.co/black-forest-labs/flux-3-image-klein/resolve/main/model.safetensors'
def rng(a, b):
    r = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{b}', 'Authorization': 'Bearer ' + TOK}); return urllib.request.urlopen(r, timeout=120).read()
n = struct.unpack('<Q', rng(0, 7))[0]; hdr = json.loads(rng(8, 8 + n - 1)); base = 8 + n
def root(k):
    m = hdr[k]; a, b = m['data_offsets']; raw = rng(base + a, base + b - 1)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(m['shape'])
D = 'klein/transformer/'; idx = json.load(open(D + 'diffusion_pytorch_model.safetensors.index.json'))['weight_map']
def dif(k):
    with safe_open(D + idx[k], 'pt') as f: return f.get_tensor(k)
pairs = [('final_layer.image.adaLN_modulation.1.weight', 'norm_out.linear.weight'),
         ('early_stream_modulations.image.lin.weight', 'early_stream_modulations.image.linear.weight'),
         ('single_stream_modulations.txt.lin.weight', 'single_stream_modulations.txt.linear.weight'),
         ('single_blocks.0.mlp_in.weight', 'single_transformer_blocks.0.ff.linear_in.weight'),
         ('single_blocks.0.norm.query_norm.scale', 'single_transformer_blocks.0.attn.norm_q.weight')]
for rk, dk in pairs:
    a = root(rk).float(); b = dif(dk).float(); h = a.shape[0] // 2
    same = (a - b).abs().max().item(); swp = (a - torch.cat([b[h:], b[:h]])).abs().max().item()
    print(f'{rk}: same={same:.4g} halves-swapped={swp:.4g}')
