"""One-block test case for the phone (MODE 60): Klein single block B from the real weights (q8 dequantised, fp32 reference).
usage: python mkblk_kl.py OUTDIR [PACK.bin] [BLOCK] [LH LW] [NT]   rows: image tokens (LH x LW latent grid) then NT text tokens, padded to 32"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, math, numpy as np, torch
sys.path.insert(0, (ROOT+'/flux3'))
import export_flux3 as E
out = sys.argv[1]; pack = sys.argv[2] if len(sys.argv) > 2 else 'pack/sgl00.bin'; B = int(sys.argv[3]) if len(sys.argv) > 3 else 0
lh, lw = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (16, 16); NT = int(sys.argv[6]) if len(sys.argv) > 6 else 80
os.makedirs(out, exist_ok=True); torch.manual_seed(5); T = E.T
ni = lh * lw; assert ni % 32 == 0; ntok = (ni + NT + 31) // 32 * 32; ns = ntok // 32
f = lambda k: T(k).float()
te = torch.load('out/te.pt')['tokens'][:NT].float()
with torch.no_grad():
    img = torch.randn(ni, 128) @ f('x_embedders.image.weight').T
    txt = te @ f('context_embedder.weight').T
    x = torch.cat([img, txt])
    ids = torch.zeros(ntok, 4)
    for r in range(lh):
        for c in range(lw): ids[r * lw + c, 1] = r * 0.5; ids[r * lw + c, 2] = c * 0.5
    ids[ni:ni + NT, 3] = torch.arange(NT).float()
    def temb(t):
        half = 128; fr = torch.exp(-math.log(10000) * torch.arange(half).float() / half); a = 1000.0 * t * fr
        return torch.cat([torch.cos(a), torch.sin(a)])
    def mlp(pfx, v): return f(pfx + '.linear_2.weight') @ torch.nn.functional.silu(f(pfx + '.linear_1.weight') @ v)
    t_img, t_tgt = 0.9412, 0.8421
    vec = {'image': mlp('timestep_embedder', temb(t_img)) + mlp('target_timestep_embedder', temb(t_tgt)),
           'txt': mlp('timestep_embedder', temb(0.0)) + mlp('target_timestep_embedder', temb(t_tgt))}
    mod = {k: (f(f'single_stream_modulations.{k}.linear.weight') @ torch.nn.functional.silu(v)).view(3, 3072) for k, v in vec.items()}   # shift, scale, gate
    kind = torch.cat([torch.zeros(ni, dtype=torch.long), torch.ones(NT, dtype=torch.long)])
    M = {c: torch.stack([mod['image'][c], mod['txt'][c]])[kind] for c in range(3)}
    p = f'single_transformer_blocks.{B}'
    def dq(k):
        q, s = E.quant(T(k)); return torch.from_numpy(q.astype(np.float32)) * torch.from_numpy(s)[:, None]
    Wq, Wk, Wv, Wo, Win, Wd = [dq(f'{p}.{k}') for k in ('attn.to_q.weight', 'attn.to_k.weight', 'attn.to_v.weight', 'attn.to_out.0.weight', 'ff.linear_in.weight', 'ff.linear_out.weight')]
    nq, nk = f(f'{p}.attn.norm_q.weight'), f(f'{p}.attn.norm_k.weight')
    n = ni + NT
    xm = torch.nn.functional.layer_norm(x, (3072,), eps=1e-6) * (1 + M[1]) + M[0]
    q = (xm @ Wq.T).view(n, 24, 128); k = (xm @ Wk.T).view(n, 24, 128); v = (xm @ Wv.T).view(n, 24, 128)
    rn = lambda t, w: t * t.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt() * w
    q, k = rn(q, nq), rn(k, nk)
    ang = torch.zeros(n, 64)
    for i in range(64): ang[:, i] = ids[:n, i // 16] * 10000.0 ** (-(i % 16) / 16.0)
    cs, sn = torch.cos(ang), torch.sin(ang)
    def rope(t):   # adjacent pairs (2i, 2i+1)
        a, b = t[..., 0::2], t[..., 1::2]; c, s = cs[:, None], sn[:, None]
        return torch.stack([c * a - s * b, s * a + c * b], -1).reshape(t.shape)
    q, k = rope(q), rope(k)
    att = torch.softmax(torch.einsum('qhd,khd->hqk', q, k) / 128 ** 0.5, -1); o = torch.einsum('hqk,khd->qhd', att, v).reshape(n, 3072)
    g, u = (xm @ Win.T).chunk(2, -1); mid = torch.nn.functional.silu(g) * u
    xo = x + M[2] * (o @ Wo.T + mid @ Wd.T)
def pad(t, rows): r = torch.zeros(rows, t.shape[1]); r[:t.shape[0]] = t; return r
def x32(t): t = pad(t, ntok); return np.ascontiguousarray(t.numpy().reshape(ns, 32, 96, 32).transpose(0, 2, 1, 3)).astype(np.float32)
def tab(t): t = pad(t, ntok); return np.ascontiguousarray(t.numpy().reshape(ns, 32, 2, 32).transpose(0, 2, 1, 3)).astype(np.float32)
x32(x).tofile(f'{out}/x32.bin'); tab(cs).tofile(f'{out}/cos.bin'); tab(sn).tofile(f'{out}/sin.bin')
np.zeros(131072, np.uint8).tofile(f'{out}/hn.bin')
mf = np.zeros(258048 // 4, np.float32)
for kd, name in enumerate(('image', 'txt')):
    for c in range(3): mf[(kd * 3 + c) * 3072:(kd * 3 + c + 1) * 3072] = mod[name][c].numpy()
mf.tofile(f'{out}/mod.bin')
if os.path.lexists(f'{out}/wts.bin'): os.remove(f'{out}/wts.bin')
os.symlink(os.path.abspath(pack), f'{out}/wts.bin')
np.savez(f'{out}/ref.npz', ntok=ntok, ni=ni, nt=NT, xo=xo.numpy(), x=x.numpy(), q=q.numpy(), o=o.numpy(), mid=mid.numpy())
print('ntok', ntok, 'nimg slabs', ni // 32, 'nvalid', n, '| x rms img/txt', x[:ni].pow(2).mean().sqrt().item(), x[ni:].pow(2).mean().sqrt().item(), '| xo rms', xo.pow(2).mean().sqrt().item(), 'max|x|', x.abs().max().item())
