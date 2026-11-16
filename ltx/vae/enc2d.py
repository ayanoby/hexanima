"""LTX conv video ENCODER for ONE image (1 frame) as a rank-4 network for QNN HTP.
With a single frame every causal Conv3d is a Conv2d on the temporal sum of its kernel (the causal pad repeats the first frame); the temporal stride-2 blocks first duplicate the frame.
SpaceToDepthDownsample = conv + pixel_unshuffle (+ channel Gather for the duplicated time phase) + skip path as one constant 1x1 conv (rearrange + group mean).
python enc2d.py check  -> compares with the ltx-core VideoEncoder on a small image"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, json, gc, types, os, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, (ROOT+'/ltx'))
import ltxstub
for name in ('ltx_core.guidance',):
    m = types.ModuleType(name); m.__path__ = [os.path.join((os.environ.get('LTX_C', 'ltxc')), *name.split('.'))]; sys.modules[name] = m
from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator
from safetensors import safe_open
P = (os.environ.get('LTX_W', 'ltx_weights') + '/vae/ltx-2.5-video-vae-conv-bf16.safetensors')
def load_ref():
    st = safe_open(P, 'pt'); cfg = json.loads(st.metadata()['config']); enc = VideoEncoderConfigurator.from_config(cfg); sd = {}
    for k in st.keys():
        if k.startswith('encoder.'): sd[k[len('encoder.'):]] = st.get_tensor(k).float()
        elif k.startswith('per_channel_statistics.'): sd[k] = st.get_tensor(k).float()
    print(enc.load_state_dict(sd, strict=False)); return enc.float().eval()
class PN(nn.Module):
    def __init__(s, eps=1e-8): super().__init__(); s.eps = eps
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + s.eps)
def conv1f(conv):   # CausalConv3d -> Conv2d over one frame: sum the temporal taps
    w = conv.conv.weight.detach().sum(2); b = conv.conv.bias.detach(); c = nn.Conv2d(w.shape[1], w.shape[0], 3, padding=1); c.weight.data = w.contiguous(); c.bias.data = b.clone(); return c
class Res(nn.Module):
    def __init__(s, rb): super().__init__(); s.n1 = PN(getattr(rb.norm1, 'eps', 1e-8)); s.c1 = conv1f(rb.conv1.conv if hasattr(rb.conv1, 'conv') and hasattr(rb.conv1.conv, 'conv') else rb.conv1); s.n2 = PN(getattr(rb.norm2, 'eps', 1e-8)); s.c2 = conv1f(rb.conv2.conv if hasattr(rb.conv2, 'conv') and hasattr(rb.conv2.conv, 'conv') else rb.conv2)
    def forward(s, x): return x + s.c2(F.silu(s.n2(s.c1(F.silu(s.n1(x))))))
class Down(nn.Module):   # SpaceToDepthDownsample for one frame
    def __init__(s, d):
        super().__init__(); st = d.stride; s.st = st; C = d.conv.conv.in_channels; co = d.conv.conv.out_channels; S = st[0] * st[1] * st[2]; s.cc = co
        s.c = conv1f(d.conv); g = d.group_size; ncol = C * st[1] * st[2]      # pixel_unshuffle channels of the input: c*(p2*p3) + k
        # input after (time dup) rearrange: channel (c p1 p2 p3) = c*S + p1*P + k (P = p2*p3); every p1 copy equals the single frame -> source channel c*P + k
        P = st[1] * st[2]; src = [ (ch // S) * P + (ch % P) for ch in range(C * S) ]
        M = torch.zeros(C * S // g, ncol)
        for ch in range(C * S): M[ch // g, src[ch]] += 1.0 / g
        s.m = nn.Conv2d(ncol, C * S // g, 1, bias=False); s.m.weight.data = M.view(C * S // g, ncol, 1, 1)
        s.P = P; s.rep = st[0]
        # output: conv channels co (per phase) -> after rearrange (c p1 p2 p3): index c*S + p1*P + k with source conv-unshuffle channel c*P + k
        s.register_buffer('oidx', torch.tensor([ (o // S) * P + (o % P) for o in range(co * S)]))
    def forward(s, x):
        r = s.st[1]; y = s.c(x); y = F.pixel_unshuffle(y, r) if r > 1 else y
        y = y.index_select(1, s.oidx)
        return y + s.m(F.pixel_unshuffle(x, r))
class Enc(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.mean = enc.per_channel_statistics.get_buffer('mean-of-means').clone().view(1, -1, 1, 1); s.std = enc.per_channel_statistics.get_buffer('std-of-means').clone().view(1, -1, 1, 1)
        s.cin = conv1f(enc.conv_in); L = []
        for b in enc.down_blocks:
            if type(b).__name__ == 'UNetMidBlock3D': L += [Res(r) for r in b.res_blocks]
            else: L.append(Down(b))
        s.blocks = nn.ModuleList(L); s.nout = PN(); s.cout = conv1f(enc.conv_out)
    def forward(s, img):   # img [1,3,H,W] in [-1,1] -> normalised latent [1,128,H/32,W/32]
        x = s.cin(s.patch(img))
        for b in s.blocks: x = b(x)
        m = s.cout(F.silu(s.nout(x)))[:, :128]
        return (m - s.mean) / s.std
    def patch(s, img):   # "b c (h q) (w r) -> b (c r q) h w": channel = c*16 + r*4 + q (r = width phase, q = height phase); pixel_unshuffle gives c*16 + q*4 + r -> swap
        u = F.pixel_unshuffle(img, 4); n, ch, h, w = u.shape
        perm = torch.arange(ch).view(3, 4, 4).permute(0, 2, 1).reshape(-1)
        return u.index_select(1, perm.to(u.device))
def calib_images():
    import numpy as np
    from PIL import Image
    out = []
    for f in ('fox_f16', 'cat_f00', 'fox_f00', 'cat_f16'):
        im = Image.open(f(ROOT+'/ltx' + '/samples/{f}.png')).convert('RGB').resize((256, 256))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].contiguous())
    out.append(torch.rand(1, 3, 256, 256) * 2 - 1)
    return out
@torch.no_grad()
def rescale(m, imgs=None):
    """fp16 has max 65504 and the stream of the deep blocks reaches 2e5: PixelNorm is scale-invariant, so keep the residual stream divided by a power of two k per stage (conv_in, conv2 of the Res blocks and the Down blocks scaled to match)."""
    imgs = imgs or calib_images(); rms = {}
    for img in imgs:
        h = m.cin(m.patch(img)); rms.setdefault('cin', []).append(float(h.pow(2).mean().sqrt()))
        for i, b in enumerate(m.blocks):
            h = b(h)
            if isinstance(b, Down): rms.setdefault(i, []).append(float(h.pow(2).mean().sqrt()))
    p2 = lambda v: float(2 ** round(torch.log2(torch.tensor(max(v, 1.0))).item()))
    k = p2(max(rms['cin'])); m.cin.weight.data /= k; m.cin.bias.data /= k; kc = k; ks = [('cin', k)]
    for i, b in enumerate(m.blocks):
        if isinstance(b, Res): b.c2.weight.data /= kc; b.c2.bias.data /= kc
        else:
            kn = p2(max(rms[i])); f = kc / kn
            b.c.weight.data *= f; b.c.bias.data /= kn; b.m.weight.data *= f; kc = kn; ks.append((i, kn))
    print('stream scales', ks); return ks
if __name__ == '__main__':
    enc = load_ref(); e2 = Enc(enc).eval(); torch.manual_seed(0)
    x = torch.rand(1, 3, 1, 128, 160) * 2 - 1
    if 'scaled' in sys.argv: rescale(e2)
    with torch.no_grad(): ref = enc(x)[:, :, 0]; mine = e2(x[:, :, 0])
    print('ref', tuple(ref.shape), 'mine', tuple(mine.shape), 'rel diff %.3e' % float((ref - mine).norm() / ref.norm()))
