"""Pack MiniMax-H3 Ref2VA DiT (Comfy-Org minimax_h3_ref2va_int8_convrot.safetensors, int8 ConvRot, per-row scale) into the same HMX q8 tile block files as export_h3.py (blkNN.bin) and the per-block
modulation layers into adalnNN.bin (the Ref2VA blocks have a real Linear 2688 -> 96768 instead of the small table of the image model):
  blkNN.bin: identical layout to the H3-Image pack (Q,K,V,O,GU,DN,gain). Differences in the source: qkv_proj = [q;k;v] stacked (7168 rows each), fc1 = [gate | up] (the image model has [up | gate]).
  adalnNN.bin: int8 [96768][2688] row-major, f32 scale [96768], f32 bias [96768]
usage: python export_h3_ref2va.py DIT.safetensors OUTDIR [first last]"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_h3 import ST, PERM, K0, QN, FF, tilepack3, hperm
def lin(st, base): return st.t(base + '.weight'), st.t(base + '.weight_scale').reshape(-1)
def pack_block(st, i):
    p = f'blocks.{i}.'; out = []; sz = {}
    def add(n, b): sz[n] = len(b); out.append(b)
    q, s = lin(st, p + 'attn.qkv_proj')
    for j, n in enumerate(('to_q', 'to_k', 'to_v')):
        qq, ss = q[j * QN:(j + 1) * QN], s[j * QN:(j + 1) * QN]
        if n in ('to_q', 'to_k'): qq, ss = hperm(qq, ss)
        add(n, tilepack3(np.ascontiguousarray(qq.T), ss))
    qq, ss = lin(st, p + 'attn.out_proj'); add('to_out', tilepack3(np.ascontiguousarray(qq.T), ss))
    q, s = lin(st, p + 'mlp.fc1'); gate, up = (q[:FF], s[:FF]), (q[FF:], s[FF:])
    qi = np.stack((gate[0].reshape(FF // 32, 32, K0), up[0].reshape(FF // 32, 32, K0)), 1).reshape(2 * FF, K0)
    si = np.stack((gate[1].reshape(FF // 32, 32), up[1].reshape(FF // 32, 32)), 1).reshape(-1)
    add('gate_up', tilepack3(np.ascontiguousarray(qi.T), si))
    q, s = lin(st, p + 'mlp.fc2'); add('down', tilepack3(np.ascontiguousarray(q.T), s))
    qn = st.t(p + 'attn.q_norm.weight').astype(np.float32)[PERM] / np.sqrt(128.0); kn = st.t(p + 'attn.k_norm.weight').astype(np.float32)[PERM]
    g = np.concatenate([qn, kn, st.t(p + 'norm1.weight'), st.t(p + 'norm2.weight')]).astype(np.float32)
    add('gain', g.tobytes() + bytes(16384 * 4 - g.nbytes))
    return b''.join(out), sz
def pack_adaln(st, i):
    p = f'blocks.{i}.adaln_proj.linear.'
    return np.ascontiguousarray(st.t(p + 'weight')).tobytes() + st.t(p + 'weight_scale').reshape(-1).astype(np.float32).tobytes() + st.t(p + 'bias').astype(np.float32).tobytes()
if __name__ == '__main__':
    st = ST(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    a = int(sys.argv[3]) if len(sys.argv) > 3 else 0; b = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    for i in range(a, b + 1):
        d, sz = pack_block(st, i); open(f'{out}/blk{i:02d}.bin', 'wb').write(d); m = pack_adaln(st, i); open(f'{out}/adaln{i:02d}.bin', 'wb').write(m); print(i, len(d), len(m), flush=True)
    print(sz)
