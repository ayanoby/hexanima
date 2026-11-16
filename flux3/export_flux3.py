"""Pack the FLUX.3 Klein (distilled, flow map) DiT into HMX q8 tile files for the phone NPU.

Source: klein/transformer (diffusers layout, bf16). Weights are quantised per output row (symmetric int8, scale = absmax/127) and packed
with tilepack3 from hexanima/h3/export_h3.py (compact q8: per n-block [128 B fp16 scale vector][kb x 1024 B offset-binary int8 tiles]).

One file per block, K = 3072 (hidden), 24 heads x 128:
  Q, K  [3072][3072]  output channels permuted inside every head: (0,2,4,...,126, 1,3,...,127), so the engine's split-half RoPE
                      (pair (i, i+64)) rotates the original adjacent pairs (2i, 2i+1)
  V     [3072][3072]   O [3072][3072]
  GU    [3072][18432]  n-blocks of 32 interleaved (gate_j, up_j); gate = rows 0..9215 of ff.linear_in, up = rows 9216..  (silu(gate) * up)
  DN    [9216][3072]
  gain  f32 x 256: q_norm(perm) / sqrt(128) [128], k_norm(perm) [128]
The block is "parallel": x += gate * (O(attn(LN_mod(x))) + DN(silu(g) * u)) with ONE modulated LN of x feeding Q, K, V and GU.

Block files (T2I needs txt00-04, img00-04, sgl00-27; cond00-04 only for editing with reference images; --cond to write them):
  txt{d}.bin  txt_mode_blocks.d     img{d}.bin  mode_blocks.image.d     cond{d}.bin  mode_blocks.image_cond.d     sgl{d}.bin  single_transformer_blocks.d
Glue (glue/): raw little-endian files, see GLUE below.  NOTE norm_out.linear: diffusers stores (scale, shift), the reference wants (shift, scale): the halves are swapped here.

usage: python export_flux3.py OUTDIR [--cond] [--blocks txt,img,sgl] [--check]"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os, sys, json, numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, (ROOT+'/h3'))
from export_h3 import tilepack3

D = (os.environ.get('KLEIN_DIR', 'klein') + '/transformer/')
IDX = json.load(open(D + 'diffusion_pytorch_model.safetensors.index.json'))['weight_map']
H, NH, HD, FF = 3072, 24, 128, 9216
_files = {}
def T(k):
    f = IDX[k]
    if f not in _files: _files[f] = safe_open(D + f, 'pt')
    return _files[f].get_tensor(k)
PERM = np.concatenate([np.arange(0, HD, 2), np.arange(1, HD, 2)])

def quant(w):   # w bf16/f32 [N,K] -> int8 [N,K], scale f32 [N]
    f = w.float(); s = f.abs().amax(1).clamp_min(1e-8) / 127
    return torch.round(f / s[:, None]).clamp(-127, 127).to(torch.int8).numpy(), s.numpy()

def pack(q, s):   # q int8 [N,K] (output x input), s [N] -> tile bytes
    return tilepack3(np.ascontiguousarray(q.T), s)

def hperm(q, s):  # rows are NH heads x HD: permute inside each head
    return q.reshape(NH, HD, -1)[:, PERM].reshape(q.shape), s.reshape(NH, HD)[:, PERM].reshape(-1)

def pack_block(p):
    out, sz = [], {}
    def add(n, b): sz[n] = len(b); out.append(b)
    for n, k in (('q', 'to_q'), ('k', 'to_k'), ('v', 'to_v'), ('o', 'to_out.0')):
        q, s = quant(T(f'{p}.attn.{k}.weight'))
        if n in ('q', 'k'): q, s = hperm(q, s)
        add(n, pack(q, s))
    q, s = quant(T(f'{p}.ff.linear_in.weight'))
    gate, up = (q[:FF], s[:FF]), (q[FF:], s[FF:])
    qi = np.stack((gate[0].reshape(FF // 32, 32, H), up[0].reshape(FF // 32, 32, H)), 1).reshape(2 * FF, H)
    si = np.stack((gate[1].reshape(FF // 32, 32), up[1].reshape(FF // 32, 32)), 1).reshape(-1)
    add('gate_up', pack(qi, si))
    q, s = quant(T(f'{p}.ff.linear_out.weight')); add('down', pack(q, s))
    qn = T(f'{p}.attn.norm_q.weight').float().numpy()[PERM] / np.sqrt(128.0); kn = T(f'{p}.attn.norm_k.weight').float().numpy()[PERM]
    add('gain', np.concatenate([qn, kn]).astype(np.float32).tobytes())
    return b''.join(out), sz

def unpack_matrix(b, off, K, N):   # inverse of tilepack3 -> (int8 [N,K], scale [N])
    nbs = 128 + (K // 32) * 1024; rec = np.frombuffer(b, dtype=np.uint8, count=(N // 32) * nbs, offset=off).reshape(N // 32, nbs)
    s = rec[:, :128].copy().view(np.float16).astype(np.float32).reshape(N // 32, 64)[:, ::2].reshape(-1)
    t = (rec[:, 128:].astype(np.int16) - 128).reshape(N // 32, K // 32, 16, 32, 2)
    q = t.transpose(1, 2, 4, 0, 3).reshape(K, N)
    return np.ascontiguousarray(q.T).astype(np.int8), s

def check(p, b):   # round trip of Q (with head perm), V and the interleaved GU against the source weights
    nbs3 = 96 * (128 + 96 * 1024)   # bytes of one 3072x3072 matrix
    err = {}
    q0, s0 = quant(T(f'{p}.attn.to_q.weight')); q0, s0 = hperm(q0, s0)
    q, s = unpack_matrix(b, 0, H, H); err['q'] = (int(np.abs(q.astype(np.int16) - q0).max()), float(np.abs(s - s0).max() / s0.max()))
    qv, sv = quant(T(f'{p}.attn.to_v.weight')); q, s = unpack_matrix(b, 2 * nbs3, H, H); err['v'] = (int(np.abs(q.astype(np.int16) - qv).max()), float(np.abs(s - sv).max() / sv.max()))
    qg, sg = quant(T(f'{p}.ff.linear_in.weight')); q, s = unpack_matrix(b, 4 * nbs3, H, 2 * FF)
    q = q.reshape(FF // 32, 2, 32, H); err['gate'] = int(np.abs(q[:, 0].reshape(FF, H).astype(np.int16) - qg[:FF]).max()); err['up'] = int(np.abs(q[:, 1].reshape(FF, H).astype(np.int16) - qg[FF:]).max())
    return err

def glue(out):
    os.makedirs(out, exist_ok=True)
    def w(name, a): np.ascontiguousarray(a).tofile(f'{out}/{name}')
    f32 = lambda k: T(k).float().numpy(); f16 = lambda k: T(k).float().numpy().astype(np.float16)
    w('x_img.f32', f32('x_embedders.image.weight')); w('x_cond.f32', f32('x_embedders.image_cond.weight'))   # [3072,128]
    w('ctx.f16', f16('context_embedder.weight'))                                                              # [3072,20480]
    for a, b in (('time', 'timestep_embedder'), ('ref', 'target_timestep_embedder')):
        w(f'{a}_1.f32', f32(f'{b}.linear_1.weight')); w(f'{a}_2.f32', f32(f'{b}.linear_2.weight'))            # [3072,256], [3072,3072]
    for e in ('early', 'single'):
        for m in ('image', 'image_cond', 'txt'): w(f'{e}_{m}.f16', f16(f'{e}_stream_modulations.{m}.linear.weight'))   # [9216,3072] = shift, scale, gate
    n = f32('norm_out.linear.weight'); h = n.shape[0] // 2; w('nout.f32', np.concatenate([n[h:], n[:h]]))     # [6144,3072] (shift, scale) after the swap
    w('proj_out.f32', f32('proj_out.weight'))                                                                 # [128,3072]
    json.dump({'hidden': H, 'heads': NH, 'head_dim': HD, 'mlp': FF, 'txt_in': 20480, 'latent': 128, 'theta': 10000, 'axes': [32, 32, 32, 32],
               'rope_scale_xy': 0.5, 'norm_out_swapped': True, 'te_layers': [4, 8, 12, 16, 20, 24, 28, 32], 'pad_multiple': 80}, open(f'{out}/meta.json', 'w'), indent=1)

if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    kinds = ['txt', 'img', 'sgl'] + (['cond'] if '--cond' in sys.argv else [])
    if '--blocks' in sys.argv: kinds = sys.argv[sys.argv.index('--blocks') + 1].split(',')
    src = {'txt': ('txt_mode_blocks.{}', 5), 'img': ('mode_blocks.image.{}', 5), 'cond': ('mode_blocks.image_cond.{}', 5), 'sgl': ('single_transformer_blocks.{}', 28)}
    for kd in kinds:
        pat, n = src[kd]
        for d in range(n):
            p = pat.format(d); b, sz = pack_block(p); open(f'{out}/{kd}{d:02d}.bin', 'wb').write(b)
            print(kd, d, len(b), check(p, b) if ('--check' in sys.argv and d == 0) else '', flush=True)
    if 'glue' in kinds or '--glue' in sys.argv or kinds == ['txt', 'img', 'sgl']: glue(out + '/glue'); print('glue done')
