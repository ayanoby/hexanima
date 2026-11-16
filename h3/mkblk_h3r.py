"""(Ref2VA variant: random text/image rows and a random 2688-d time embedding through the real adaln_proj; weights blocks.B.*) One-block test case for the phone (MODE 50) from the real INT8-ConvRot weights, block B. usage: python mkblk_h3.py OUTDIR PACK_BLK.bin [BLOCK] [LH LW]
Rows: image rows first (2x2 patches of the LH x LW latent), then text rows (14 prompt tokens from te_out.pt, refined), padded to 32. Writes wts.bin x32.bin mod.bin cos.bin sin.bin hn.bin ref.npz"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, shutil, numpy as np, torch
sys.path.insert(0, (os.environ.get('H3_W', 'h3_weights')))
from h3lib import *
out, pack = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 0
lh, lw = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (16, 16)
os.makedirs(out, exist_ok=True); torch.manual_seed(5)
import h3lib
if os.environ.get('H3LORA'): h3lib.LORA = load_lora((os.environ.get('H3_W', 'h3_weights') + '/lora/lora_model_0.safetensors'))
st = ST((os.environ.get('H3R_DIR', 'h3-ref2va') + '/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors')); nt = 14; ni = (lh // 2) * (lw // 2)
nimg_pad = (ni + 31) // 32 * 32; assert nimg_pad == ni, 'image rows must fill whole slabs'
ntok = (ni + nt + 31) // 32 * 32
with torch.no_grad():
    txt = torch.randn(nt, 5376) * 0.7
    img = torch.randn(ni, 5376) * 0.7
    x = torch.cat([img, txt])                                   # image rows first, then text
    pos = positions(nt, lh, lw); pos = torch.cat([pos[nt:], pos[:nt]])
    cos96, sin96 = dit_rope(pos)
    p = f'blocks.{B}.'
    temb = torch.randn(2688) * 0.05
    m = (torch.nn.functional.linear(temb, st.t(p + 'adaln_proj.linear.weight').float() * st.t(p + 'adaln_proj.linear.weight_scale').float()) + st.t(p + 'adaln_proj.linear.bias').float()).view(3, 6, 5376)
    ms = lambda c: torch.cat([m[0, c].expand(ni, -1), m[1, c].expand(nt, -1)])
    sh1, sc1, g1, sh2, sc2, g2 = [ms(c) for c in range(6)]
    n1 = rms(x, st.t(p + 'norm1.weight'), 1e-5) * (1 + sc1) + sh1
    n = n1.shape[0]; qkv = lin(st, p + 'attn.qkv_proj', n1); q = qkv[:, :7168].reshape(n, 56, 128); k = qkv[:, 7168:14336].reshape(n, 56, 128); v = qkv[:, 14336:].reshape(n, 56, 128)
    q = rms(q, st.t(p + 'attn.q_norm.weight'), 1e-5); k = rms(k, st.t(p + 'attn.k_norm.weight'), 1e-5); q = apply_rope96(q, cos96, sin96); k = apply_rope96(k, cos96, sin96)
    a = torch.softmax(torch.einsum('qhd,khd->hqk', q, k) / 128 ** 0.5, -1); o = torch.einsum('hqk,khd->qhd', a, v).reshape(n, 7168)
    xa = x + g1 * lin(st, p + 'attn.out_proj', o)
    n2 = rms(xa, st.t(p + 'norm2.weight'), 1e-5) * (1 + sc2) + sh2
    y = lin(st, p + 'mlp.fc1', n2); g, u = y.chunk(2, -1); mid = u * torch.nn.functional.silu(g)
    xo = xa + g2 * lin(st, p + 'mlp.fc2', mid)
def pad(t, rows): r = torch.zeros(rows, t.shape[1]); r[:t.shape[0]] = t; return r
def x32(t): t = pad(t, ntok); return np.ascontiguousarray(t.numpy().reshape(ntok // 32, 32, -1, 32).transpose(0, 2, 1, 3)).astype(np.float32)
def tab(t): t = pad(t, ntok); return np.ascontiguousarray(t.numpy().reshape(ntok // 32, 32, 2, 32).transpose(0, 2, 1, 3)).astype(np.float32)
# rope tables in the engine's permuted-channel form: i < 48 real angle, 48..63 identity
ang = torch.atan2(sin96[:, :48], cos96[:, :48]) if False else None
cs = torch.cat([cos96[:, :48], torch.ones(n, 16)], 1); sn = torch.cat([sin96[:, :48], torch.zeros(n, 16)], 1)
x32(x).tofile(f'{out}/x32.bin'); tab(cs).tofile(f'{out}/cos.bin'); tab(sn).tofile(f'{out}/sin.bin')
mf = np.zeros(2 * 6 * 5376, np.float32)
for kd in range(2):
    for c in range(6): mf[(kd * 6 + c) * 5376:(kd * 6 + c + 1) * 5376] = m[kd, c].numpy()
mf.tofile(f'{out}/mod.bin')
Hn = (torch.load((os.environ.get('H3_W', 'h3_weights') + '/H256.pt'), weights_only=True).float() / 16.0).numpy()   # symmetric: W = Hn, out = in @ Hn
Wt = np.ascontiguousarray(Hn.astype(np.float16).reshape(8, 16, 2, 8, 32).transpose(3, 0, 1, 4, 2)); Wt.tofile(f'{out}/hn.bin')   # [nb][kb][16][32][2]
if not os.path.exists(f'{out}/wts.bin'): os.symlink(os.path.abspath(pack), f'{out}/wts.bin')
np.savez(f'{out}/ref.npz', ntok=ntok, ni=ni, nt=nt, n1=n1.numpy(), q=q.numpy(), k=k.numpy(), o=o.numpy(), xa=xa.numpy(), mid=mid.numpy(), xo=xo.numpy(), x=x.numpy())
print('ntok', ntok, 'nimg slabs', ni // 32, 'nvalid', ni + nt, 'x rms', x.pow(2).mean().sqrt().item(), 'xo rms', xo.pow(2).mean().sqrt().item())
