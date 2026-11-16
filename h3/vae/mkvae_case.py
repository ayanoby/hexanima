"""One-block test case for MODE 82: python mkvae_case.py OUTDIR PACK.bin LAYER [LAT.f32 T LH LW]  (reference = official fp16 weights in fp32, input = embedded latents after LAYER blocks)
phone: NCB=64 H3MODE=82 WTSIZE=67355136 klblk_v DIR NTOK 0 NVALID ; then python mkvae_case.py --cmp DIR"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, os, json, numpy as np, torch
import torch.nn.functional as F
P = (os.environ.get('H3R_DIR', 'h3-ref2va')); D = 2048
def rms(x, w=None, eps=1e-5): y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps); return y * w if w is not None else y
def ids3(T, H, W):
    c = [(torch.arange(0.5, n) / n) * 2 - 1 for n in (T, H, W)]; return torch.stack(torch.meshgrid(*c, indexing='ij'), -1).reshape(-1, 3)
def rope_cs(ids):   # official: angles [N, 24] -> cos/sin over 48 dims (tile 2)
    inv = 1 / 100.0 ** torch.arange(0, 1, 6 / 48); a = 2 * np.pi * ids[:, :, None] * inv[None, None, :]; a = a.flatten(1); return torch.cos(a), torch.sin(a)   # [N, 24]
def block(x, st, i, cs):
    p = f'decoder.transformer_blocks.{i}.'; g = lambda k: st.get_tensor(p + k).float(); N = x.shape[0]
    n1 = rms(x, g('norm1.weight')); qkv = (n1 @ g('attn.to_qkv.weight').T + g('attn.to_qkv.bias')).view(N, 32, 192); q, k, v = qkv[..., :64], qkv[..., 64:128], qkv[..., 128:]
    q, k = rms(q), rms(k); c, s = cs
    def rot(t):
        a, b, rest = t[..., :24], t[..., 24:48], t[..., 48:]; c_, s_ = c[:, None], s[:, None]; return torch.cat([a * c_ - b * s_, b * c_ + a * s_, rest], -1)
    q, k = rot(q), rot(k); o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1).reshape(N, D)
    x = x + (o @ g('attn.to_out.weight').T + g('attn.to_out.bias')) * g('scale1')
    h = rms(x, g('norm2.weight')) @ g('ff.w1.weight').T + g('ff.w1.bias'); a, b = h.chunk(2, -1)
    return x + ((F.silu(a) * b) @ g('ff.w2.weight').T + g('ff.w2.bias')) * g('scale2')
def embed(st, lat, T, LH, LW):
    cfg = json.load(open(f'{P}/refcode/h3vae/config.json')); mean = torch.tensor(cfg['latents_mean']).view(24, 1, 1, 1); std = torch.tensor(cfg['latents_std']).view(24, 1, 1, 1)
    z = (torch.from_numpy(np.fromfile(lat, np.float32)).reshape(24, T, LH, LW) * std + mean); toks = z.permute(1, 2, 3, 0).reshape(-1, 24)
    g = lambda k: st.get_tensor('decoder.' + k).float(); toks = toks @ st.get_tensor('post_quant_conv.weight').float().reshape(24, 24).T + st.get_tensor('post_quant_conv.bias').float(); x = toks @ g('x_embedder.weight').T + g('x_embedder.bias')
    return torch.cat([x, g('register_tokens').reshape(4, D), torch.zeros(1, D)], 0)
def tables(ids, ns, nvalid):
    N = ids.shape[0]; cs = rope_cs(ids); c = torch.ones(ns * 32, 32); s = torch.zeros(ns * 32, 32); c[:N, :24] = cs[0]; s[:N, :24] = cs[1]
    return c, s
if __name__ == '__main__':
    if sys.argv[1] == '--cmp':
        d = sys.argv[2]; ref = np.load(f'{d}/ref.npz'); nv = int(ref['nv']); ns = ref['xo'].shape[0] // 32
        o = np.fromfile(f'{d}/out_x.bin', np.float32).reshape(ns, 64, 32, 32).transpose(0, 2, 1, 3).reshape(ns * 32, D)[:nv]; r = ref['xo'][:nv]; x0 = ref['xin'][:nv]
        e = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean())); print('rel err out vs ref %.4f | delta (out-in) vs (ref-in) %.4f | ref rms %.3f' % (e(o, r), e(o - x0, r - x0), np.sqrt((r ** 2).mean()))); sys.exit()
    out, pack, L = sys.argv[1], sys.argv[2], int(sys.argv[3]); lat = sys.argv[4] if len(sys.argv) > 4 else f'{P}/out/lat_fox50.f32'; T, LH, LW = (int(a) for a in sys.argv[5:8]) if len(sys.argv) > 7 else (2, 16, 16)
    os.makedirs(out, exist_ok=True); from safetensors import safe_open; st = safe_open(f'{P}/vae/minimax_h3_video_vae_fp16.safetensors', 'pt')
    ids = torch.cat([ids3(T, LH, LW), torch.zeros(5, 3)]); N = ids.shape[0]; ns = (N + 31) // 32; ntok = ns * 32; cs = rope_cs(ids)
    with torch.no_grad():
        x = embed(st, lat, T, LH, LW)
        for i in range(L): x = block(x, st, i, cs)
        xo = block(x, st, L, cs)
    xin = torch.zeros(ntok, D); xin[:N] = x; xo_p = torch.zeros(ntok, D); xo_p[:N] = xo
    np.ascontiguousarray(xin.numpy().reshape(ns, 32, 64, 32).transpose(0, 2, 1, 3)).astype(np.float32).tofile(f'{out}/x32.bin')
    c, s = tables(ids, ns, N); c.numpy().reshape(ns, 32, 32).astype(np.float32).tofile(f'{out}/cos.bin'); s.numpy().reshape(ns, 32, 32).astype(np.float32).tofile(f'{out}/sin.bin')
    Hn = (torch.load((ROOT+'/ltx' + '/H256.pt'), weights_only=True).float() / 16.0).numpy()
    np.ascontiguousarray(Hn.astype(np.float16).reshape(8, 16, 2, 8, 32).transpose(3, 0, 1, 4, 2)).tofile(f'{out}/hn.bin'); np.zeros(258048 // 4, np.float32).tofile(f'{out}/mod.bin')
    if os.path.lexists(f'{out}/wts.bin'): os.remove(f'{out}/wts.bin')
    os.symlink(os.path.abspath(pack), f'{out}/wts.bin'); np.savez(f'{out}/ref.npz', xin=xin.numpy(), xo=xo_p.numpy(), nv=N)
    print('layer', L, 'N', N, 'ntok', ntok, 'xin rms %.3f xo rms %.3f' % (float(x.pow(2).mean().sqrt()), float(xo.pow(2).mean().sqrt())))
