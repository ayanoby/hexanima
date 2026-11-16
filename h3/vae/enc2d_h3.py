"""H3 video-VAE image ENCODER for ONE frame as a rank-4 network (QNN HTP tile): with a single frame every causal Conv3d (zeros in front of the frame) is a Conv2d with the LAST temporal tap, spatial padding is reflect, the stride-2 downsample pads (0,1,0,1) reflect and uses stride 2 without padding.
input [1,3,H,W] ImageNet-normalised pixels -> moments [1,48,H/16,W/16] (mean | logvar, after quant_conv).  python enc2d_h3.py check  -> compares a 256x256 tile with the official 3D encoder (CPU fp32)
python enc2d_h3.py export NAME H W -> NAME.onnx + NAME.ref.pt"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, json, gc, torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
P = (os.environ.get('H3R_DIR', 'h3-ref2va'))
def rpad(x, p):   # reflect pad of 0/1 on each side as slices + concat (the QNN HTP Pad op does not reflect correctly at the borders)
    l, r, t, b = p; assert max(p) <= 1
    if l or r: x = torch.cat(([x[:, :, :, 1:2]] if l else []) + [x] + ([x[:, :, :, -2:-1]] if r else []), 3)
    if t or b: x = torch.cat(([x[:, :, 1:2, :]] if t else []) + [x] + ([x[:, :, -2:-1, :]] if b else []), 2)
    return x
class C3(nn.Module):   # 3x3 conv, reflect pad 1, last temporal tap
    def __init__(s, w, b): super().__init__(); s.w = nn.Parameter(w[:, :, -1].contiguous(), requires_grad=False); s.b = nn.Parameter(b, requires_grad=False)
    def forward(s, x): return F.conv2d(rpad(x, (1, 1, 1, 1)), s.w, s.b)
class C1(nn.Module):   # 1x1 conv (kernel 1 along time: tap 0)
    def __init__(s, w, b): super().__init__(); s.w = nn.Parameter(w[:, :, 0].contiguous(), requires_grad=False); s.b = nn.Parameter(b, requires_grad=False)
    def forward(s, x): return F.conv2d(x, s.w, s.b)
class DS(nn.Module):   # Downsample3D: pad (0,1,0,1) reflect, 3x3 stride 2, no padding, last temporal tap
    def __init__(s, w, b): super().__init__(); s.w = nn.Parameter(w[:, :, -1].contiguous(), requires_grad=False); s.b = nn.Parameter(b, requires_grad=False)
    def forward(s, x): return F.conv2d(rpad(x, (0, 1, 0, 1)), s.w, s.b, stride=2)
class GN(nn.Module):
    def __init__(s, w, b): super().__init__(); s.w = nn.Parameter(w, requires_grad=False); s.b = nn.Parameter(b, requires_grad=False)
    def forward(s, x): return F.group_norm(x, 32, s.w, s.b, 1e-6)
class Res(nn.Module):
    def __init__(s, g, p):
        super().__init__(); s.n1 = GN(g(p + 'norm1.weight'), g(p + 'norm1.bias')); s.c1 = C3(g(p + 'conv1.weight'), g(p + 'conv1.bias')); s.n2 = GN(g(p + 'norm2.weight'), g(p + 'norm2.bias')); s.c2 = C3(g(p + 'conv2.weight'), g(p + 'conv2.bias'))
        s.sc = C1(g(p + 'nin_shortcut.weight'), g(p + 'nin_shortcut.bias')) if (p + 'nin_shortcut.weight') in g.keys else None
    def forward(s, x): h = s.c2(F.silu(s.n2(s.c1(F.silu(s.n1(x)))))); return (s.sc(x) if s.sc is not None else x) + h
class Enc(nn.Module):
    def __init__(s, st):
        super().__init__()
        class G:
            keys = set(st.keys())
            def __call__(self, k): return st.get_tensor('encoder.' + k if not k.startswith('quant_conv') else k).float()
        g = G(); g.keys = {k[len('encoder.'):] for k in st.keys() if k.startswith('encoder.')}
        s.cin = C3(g('conv_in.weight'), g('conv_in.bias')); blocks = []
        for lv in range(6):
            for i in range(2): blocks.append(Res(g, f'down.{lv}.block.{i}.'))
            if f'down.{lv}.downsample.conv.weight' in g.keys: blocks.append(DS(g(f'down.{lv}.downsample.conv.weight'), g(f'down.{lv}.downsample.conv.bias')))
        s.blocks = nn.ModuleList(blocks); s.nout = GN(g('norm_out.weight'), g('norm_out.bias')); s.cout = C3(g('conv_out.weight'), g('conv_out.bias'))
        qw = st.get_tensor('quant_conv.weight').float(); qb = st.get_tensor('quant_conv.bias').float(); s.q = C1(qw, qb)
    def forward(s, x):
        x = s.cin(x)
        for b in s.blocks: x = b(x)
        return s.q(s.cout(F.silu(s.nout(x))))
def load(): return Enc(safe_open(f'{P}/vae/minimax_h3_video_vae_fp16.safetensors', 'pt')).eval()
if __name__ == '__main__':
    if sys.argv[1] == 'check':
        sys.path.insert(0, f'{P}/refcode'); import importlib, h3vae_load; klvae = importlib.import_module('h3vae.klvae'); scfg = json.load(open(f'{P}/refcode/h3vae/source_config.json'))
        kw = dict(clip_length=17, token_drop=3, encoder_tiling=0, decoder_tiling=0, parallel_tiling=0, tile_size=256, tile_overlap_min=64, encoder_parallel=0, decoder_parallel=0, chunk_dim=-1)
        with torch.device('meta'): model, _ = klvae.AutoencoderKLLegacy.from_config(scfg, return_unused_kwargs=True, **kw)
        st = safe_open(f'{P}/vae/minimax_h3_video_vae_fp16.safetensors', 'pt')
        for name in ('encoder', 'quant_conv'):
            m = getattr(model, name); m.to_empty(device='cpu'); m.load_state_dict({k[len(name) + 1:]: st.get_tensor(k).float() for k in st.keys() if k.startswith(name + '.')}, strict=True)
        model.eval(); mine = load(); torch.manual_seed(0); x = torch.randn(1, 3, 256, 256) * 0.8
        with torch.no_grad(): ref = model.encode(x[:, :, None]); out = mine(x)
        ref = ref[:, :, 0]; print('moments', tuple(ref.shape), 'rel err %.2e' % float((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), 'ref rms %.3f' % float(ref.pow(2).mean().sqrt()))
    elif sys.argv[1] == 'cut':   # cut NAME H W N: output after the stem (N = 0) / after N blocks (N > 0), fp32 reference in NAME.ref.pt
        name, H, W, N = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); m = load()
        class Cut(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x):
                x = s.m.cin(x)
                for b in s.m.blocks[:N]: x = b(x)
                return x
        cm = Cut(m).eval(); x = torch.randn(1, 3, H, W) * 0.8
        with torch.no_grad(): y = cm(x)
        torch.onnx.export(cm, (x,), f'{name}.onnx', input_names=['img'], output_names=['mom'], opset_version=17, dynamo=False); torch.save({'x': x, 'y': y}, f'{name}.ref.pt'); print('exported', name, tuple(y.shape))
    else:
        name, H, W = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); m = load(); x = torch.randn(1, 3, H, W) * 0.8
        with torch.no_grad(): y = m(x)
        torch.onnx.export(m, (x,), f'{name}.onnx', input_names=['img'], output_names=['mom'], opset_version=17, dynamo=False); torch.save({'x': x, 'y': y}, f'{name}.ref.pt'); print('exported', name, tuple(y.shape))
