"""LTX conv video decoder rewritten for QNN HTP: video = [T, C, H, W] (frames in the batch axis), every op rank 4 (QNN's elementwise / norm ops reject rank 5).
Conv3d(3x3x3) with replicate padding in time = one Conv2d(3C -> Cout) over the concatenation of frames t-1, t, t+1; DepthToSpace = pixel_shuffle + frame interleave (Gather) + the dropped first frame.
python tiledec.py check  -> compares with the ltx-core decoder on a small latent"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, (ROOT+'/ltx'))
import ref_vae as V
class PN(nn.Module):
    def __init__(s, eps): super().__init__(); s.eps = eps
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + s.eps)
class C3(nn.Module):   # Conv3d 3x3x3 (replicate temporal pad, zero spatial pad) as one Conv2d on [t-1 | t | t+1] channels; perm: output channel permutation
    def __init__(s, conv, perm=None):
        super().__init__(); w, b = conv.weight.detach(), conv.bias.detach()
        if perm is not None: w, b = w[perm], b[perm]
        co, ci = w.shape[:2]; s.conv = nn.Conv2d(3 * ci, co, 3, padding=1); s.conv.weight.data = w.permute(0, 2, 1, 3, 4).reshape(co, 3 * ci, 3, 3).contiguous(); s.conv.bias.data = b.clone()
    def forward(s, x):
        xp = torch.cat([x[:1], x, x[-1:]], 0); T = x.shape[0]
        return s.conv(torch.cat([xp[0:T], xp[1:T + 1], xp[2:T + 2]], 1))
class Res(nn.Module):
    def __init__(s, rb): super().__init__(); s.n1 = PN(rb.norm1.eps); s.c1 = C3(rb.conv1.conv); s.n2 = PN(rb.norm2.eps); s.c2 = C3(rb.conv2.conv)
    def forward(s, x): return x + s.c2(F.silu(s.n2(s.c1(F.silu(s.n1(x))))))
class D2S(nn.Module):
    def __init__(s, up):
        super().__init__(); st = up.stride; s.st = st; cout = up.conv.conv.out_channels; S = st[0] * st[1] * st[2]; cc = cout // S
        # original channel = c*S + p1*(st1*st2) + p2*st2 + p3 ; new order: p1 major, then (c, p2, p3) = what pixel_shuffle expects inside one temporal phase
        idx = torch.arange(cout).view(cc, st[0], st[1] * st[2]).permute(1, 0, 2).reshape(-1); s.c = C3(up.conv.conv, idx); s.cc = cc
    def forward(s, x):
        y = s.c(x); st = s.st; half = y.shape[1] // st[0]; T = y.shape[0]
        parts = [y[:, i * half:(i + 1) * half] for i in range(st[0])]
        if st[1] * st[2] > 1:
            parts = [F.pixel_shuffle(p, st[1]) for p in parts] if st[1] == st[2] else parts
        if st[0] == 1: return parts[0]
        z = torch.cat(parts, 0); idx = [T]; 
        for t in range(1, T): idx += [t, T + t]
        return z[torch.tensor(idx)]          # [y1(0), y0(1), y1(1), ...]: temporal x2, first frame dropped
class TileDecoder(nn.Module):
    def __init__(s, dec):
        super().__init__(); s.mean = dec.per_channel_statistics.get_buffer('mean-of-means').clone().view(1, -1, 1, 1); s.std = dec.per_channel_statistics.get_buffer('std-of-means').clone().view(1, -1, 1, 1)
        s.cin = C3(dec.conv_in.conv); L = []
        for b in dec.up_blocks:
            if type(b).__name__ == 'UNetMidBlock3D': L += [Res(r) for r in b.res_blocks]
            else: L.append(D2S(b))
        s.blocks = nn.ModuleList(L); s.nout = PN(dec.conv_norm_out.eps)
        # conv_out: channel = c*16 + r*4 + q  (r -> width, q -> height); pixel_shuffle wants c*16 + i*4 + j with i = height: swap r and q
        co = dec.conv_out.conv.out_channels; perm = torch.arange(co).view(3, 4, 4).permute(0, 2, 1).reshape(-1); s.cout = C3(dec.conv_out.conv, perm)
    def forward(s, z):   # z [T, 128, h, w] normalised latent
        x = s.cin(z * s.std + s.mean)
        for b in s.blocks: x = b(x)
        return F.pixel_shuffle(s.cout(F.silu(s.nout(x))), 4)    # [8(T-1)+1, 3, 32h, 32w]
if __name__ == '__main__':
    dec = V.dec; td = TileDecoder(dec).eval(); torch.manual_seed(0)
    z = torch.randn(1, 128, 3, 4, 4)
    with torch.no_grad(): ref = dec(z)[0].permute(1, 0, 2, 3); mine = td(z[0].permute(1, 0, 2, 3).contiguous())
    print('ref', tuple(ref.shape), 'mine', tuple(mine.shape), 'rel diff %.3e' % float((ref - mine).norm() / ref.norm()))
