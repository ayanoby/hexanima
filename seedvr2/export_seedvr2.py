"""Pack SeedVR2-3B (one step, t = 1000) into HMX q8 tile files for the phone NPU.
Per block file blkNN.bin: [4096 B header][sections]. Header int32: [0] magic 0x53565232, [1] nsets (2 = vid + txt weights, 1 = shared), [2..] offsets of
set 0 (vid or 'all') and set 1 (txt; for shared blocks the same offsets as set 0): 8 offsets per set: Q K V O GU DN VEC, total size.
Matrices: compact q8 (tilepack3 from hexanima/h3/export_h3.py), K x N = in x out:
  Q, K [2560][2560]  output channels permuted inside every head (0,2,..,126, 1,3,..,127): the engine's split-half RoPE (pair (i, i+64)) rotates the original adjacent pairs (2i, 2i+1)
  V, O [2560][2560]
  GU   [2560][13824] n-blocks of 32 interleaved (gate_j, up_j): gate = proj_in_gate, up = proj_in      DN [6912][2560]
VEC (f32): [0:6*2560] constant modulation of this block (t = 1000 is fixed): attn shift, scale, gate, mlp shift, scale, gate = global emb part + the learned vectors;
           [15360:+128] q gain (perm, /sqrt(128)), [15488:+128] k gain (perm), [15616:+2560] proj_out bias.
Glue (glue/): f32 raw files: vid_in_w [2560][132], vid_in_b, txt0 [58][2560] (= txt_in(pos_emb)), out_w [64][2560], out_b, out_norm [2560], out_shift, out_scale (final modulation, constant), meta.json
usage: python export_seedvr2.py OUTDIR [--blocks 0,1] [--check]"""
import os, sys, json, math, numpy as np, torch
sys.path.insert(0, '/home/artem/Documents/Claude/anima/hexanima/h3')
from export_h3 import tilepack3
CK = '/run/media/artem/32363EC6363E8ABD/seedvr2/3b/seedvr2_ema_3b.pth'; TXT = '/run/media/artem/32363EC6363E8ABD/seedvr2/3b/pos_emb.pt'
D, H, HD, FF, NL, MM = 2560, 20, 128, 6912, 32, 10
sd = torch.load(CK, map_location='cpu', mmap=True, weights_only=True)
T = lambda k: sd[k].float()
PERM = np.concatenate([np.arange(0, HD, 2), np.arange(1, HD, 2)])

def quant(w):
    f = w.float(); s = f.abs().amax(1).clamp_min(1e-8) / 127
    return torch.round(f / s[:, None]).clamp(-127, 127).to(torch.int8).numpy(), s.numpy()
def pack(q, s): return tilepack3(np.ascontiguousarray(q.T), s)
def hperm(q, s): return q.reshape(H, HD, -1)[:, PERM].reshape(q.shape), s.reshape(H, HD)[:, PERM].reshape(-1)

def emb():
    half = 128; fr = torch.exp(-math.log(10000) * torch.arange(half).float() / half); a = torch.tensor([1000.0])[:, None] * fr[None]
    e = torch.cat([a.sin(), a.cos()], -1)
    for n in ('proj_in', 'proj_hid', 'proj_out'):
        e = torch.nn.functional.linear(e, T(f'emb_in.{n}.weight'), T(f'emb_in.{n}.bias'))
        if n != 'proj_out': e = torch.nn.functional.silu(e)
    return e[0].view(D, 2, 3)      # (d, layer, g): g = shift, scale, gate
E = None

def mats(p, w):   # p = 'blocks.i.', w in ('vid', 'txt', 'all'); returns list of packed matrices + vec
    q, s = quant(T(f'{p}attn.proj_qkv.{w}.weight'))
    out = []
    for j, n in enumerate(('q', 'k', 'v')):
        qq, ss = q[j * D:(j + 1) * D], s[j * D:(j + 1) * D]
        if n != 'v': qq, ss = hperm(qq, ss)
        out.append(pack(qq, ss))
    qo, so = quant(T(f'{p}attn.proj_out.{w}.weight')); out.append(pack(qo, so))
    qg, sg = quant(T(f'{p}mlp.{w}.proj_in_gate.weight')); qu, su = quant(T(f'{p}mlp.{w}.proj_in.weight'))
    qi = np.stack((qg.reshape(FF // 32, 32, D), qu.reshape(FF // 32, 32, D)), 1).reshape(2 * FF, D)
    si = np.stack((sg.reshape(FF // 32, 32), su.reshape(FF // 32, 32)), 1).reshape(-1)
    out.append(pack(qi, si))
    qd, sd_ = quant(T(f'{p}mlp.{w}.proj_out.weight')); out.append(pack(qd, sd_))
    a = lambda n: T(f'{p}ada.{w}.{n}')
    vec = np.zeros(15616 + 2560, np.float32)
    vec[0 * D:1 * D] = (E[:, 0, 0] + a('attn_shift')).numpy(); vec[1 * D:2 * D] = (E[:, 0, 1] + a('attn_scale')).numpy(); vec[2 * D:3 * D] = (E[:, 0, 2] + a('attn_gate')).numpy()
    if f'{p}ada.{w}.mlp_shift' in sd:
        vec[3 * D:4 * D] = (E[:, 1, 0] + a('mlp_shift')).numpy(); vec[4 * D:5 * D] = (E[:, 1, 1] + a('mlp_scale')).numpy(); vec[5 * D:6 * D] = (E[:, 1, 2] + a('mlp_gate')).numpy()
    vec[15360:15488] = (T(f'{p}attn.norm_q.{w}.weight').numpy()[PERM] / np.sqrt(128.0)); vec[15488:15616] = T(f'{p}attn.norm_k.{w}.weight').numpy()[PERM]
    vec[15616:] = T(f'{p}attn.proj_out.{w}.bias').numpy()
    out.append(vec.tobytes())
    return out

def pack_block(i):
    p = f'blocks.{i}.'; mm = i < MM
    sets = [mats(p, 'vid'), mats(p, 'txt')] if mm else [mats(p, 'all')]
    hdr = np.zeros(1024, np.int32); hdr[0] = 0x53565232; hdr[1] = len(sets)
    body = []; off = 4096
    offs = []
    for st in sets:
        o = []
        for b in st: o.append(off); body.append(b); off += len(b)
        offs.append(o)
    if not mm: offs.append(offs[0])
    for si, o in enumerate(offs):
        for k, v in enumerate(o): hdr[2 + si * 8 + k] = v
    hdr[2 + 16] = off
    return hdr.tobytes() + b''.join(body), offs

def glue(out):
    os.makedirs(out, exist_ok=True)
    w = lambda n, a: np.ascontiguousarray(a, dtype=np.float32).tofile(f'{out}/{n}')
    w('vid_in_w', T('vid_in.proj.weight').numpy()); w('vid_in_b', T('vid_in.proj.bias').numpy())
    txt = torch.load(TXT, map_location='cpu', weights_only=True).float()
    w('txt0', torch.nn.functional.linear(txt, T('txt_in.weight'), T('txt_in.bias')).numpy())
    w('out_w', T('vid_out.proj.weight').numpy()); w('out_b', T('vid_out.proj.bias').numpy()); w('out_norm', T('vid_out_norm.weight').numpy())
    w('out_shift', (E[:, 0, 0] + T('vid_out_ada.out_shift')).numpy()); w('out_scale', (E[:, 0, 1] + T('vid_out_ada.out_scale')).numpy())
    json.dump({'dim': D, 'heads': H, 'head_dim': HD, 'mlp': FF, 'layers': NL, 'mm_layers': MM, 'txt_tokens': 58, 'patch': [1, 2, 2], 'in_channels': 33, 'latent_scale': 0.9152,
               'window_div': 3, 'window_area720': 3600, 'rope_axes': [21, 21, 21], 'eps': 1e-5}, open(f'{out}/meta.json', 'w'), indent=1)

def unpack_matrix(b, off, K, N):
    nbs = 128 + (K // 32) * 1024; rec = np.frombuffer(b, dtype=np.uint8, count=(N // 32) * nbs, offset=off).reshape(N // 32, nbs)
    s = rec[:, :128].copy().view(np.float16).astype(np.float32).reshape(N // 32, 64)[:, ::2].reshape(-1)
    t = (rec[:, 128:].astype(np.int16) - 128).reshape(N // 32, K // 32, 16, 32, 2)
    q = t.transpose(1, 2, 4, 0, 3).reshape(K, N)
    return np.ascontiguousarray(q.T).astype(np.int8), s

if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); E = emb()
    blocks = [int(x) for x in sys.argv[sys.argv.index('--blocks') + 1].split(',')] if '--blocks' in sys.argv else list(range(NL))
    for i in blocks:
        b, offs = pack_block(i); open(f'{out}/blk{i:02d}.bin', 'wb').write(b)
        msg = ''
        if '--check' in sys.argv:
            q, s = unpack_matrix(b, offs[0][3], D, D); q0, s0 = quant(T(f'blocks.{i}.attn.proj_out.{"vid" if i < MM else "all"}.weight'))
            msg = f'O roundtrip max diff {int(np.abs(q.astype(np.int16) - q0).max())} scale {float(np.abs(s - s0).max() / s0.max()):.1e}'
        print('blk', i, len(b), msg, flush=True)
    if '--blocks' not in sys.argv or '--glue' in sys.argv: glue(out + '/glue'); print('glue done')
