"""Pack the H3 text encoder (Qwen3-VL text part, INT8-ConvRot, layers 0..49) into HMX q8 tile files, one per layer, lossless like export_h3.py.
layNN.bin (K = input, N = output): Q [5120][8192] (no head permutation: plain half-split RoPE over 128), K [5120][1024], V [5120][1024], O [8192][5120], GU [5120][51200] (n-blocks (gate_j, up_j)), DN [25600][5120],
gain f32 x 16384: q_norm/sqrt(128) [128], k_norm [128], input_layernorm [5120], post_attention_layernorm [5120]
usage: python export_h3_te.py TE.safetensors OUTDIR [first last]    (also writes embed.bf16 = embed_tokens [151936][5120] when first == 0)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_h3 import ST, tilepack3
def lin(st, base): return st.t(base + '.weight'), st.t(base + '.weight_scale').reshape(-1)
def pack_layer(st, i):
    p = f'model.layers.{i}.'; out = []; sz = {}
    def add(n, b): sz[n] = len(b); out.append(b)
    for n in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'):
        q, s = lin(st, p + n); add(n, tilepack3(np.ascontiguousarray(q.T), s))
    g, gs = lin(st, p + 'mlp.gate_proj'); u, us = lin(st, p + 'mlp.up_proj'); FF = g.shape[0]; K = g.shape[1]
    qi = np.stack((g.reshape(FF // 32, 32, K), u.reshape(FF // 32, 32, K)), 1).reshape(2 * FF, K); si = np.stack((gs.reshape(FF // 32, 32), us.reshape(FF // 32, 32)), 1).reshape(-1)
    add('gate_up', tilepack3(np.ascontiguousarray(qi.T), si))
    q, s = lin(st, p + 'mlp.down_proj'); add('down', tilepack3(np.ascontiguousarray(q.T), s))
    f = lambda k: st.t(p + k).astype(np.float32)
    gn = np.concatenate([f('self_attn.q_norm.weight') / np.sqrt(128.0), f('self_attn.k_norm.weight'), f('input_layernorm.weight'), f('post_attention_layernorm.weight')]).astype(np.float32)
    add('gain', gn.tobytes() + bytes(16384 * 4 - gn.nbytes))
    return b''.join(out), sz
if __name__ == '__main__':
    st = ST(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    a = int(sys.argv[3]) if len(sys.argv) > 3 else 0; b = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    if a == 0:
        m = st.h['model.embed_tokens.weight']; o0, o1 = m['data_offsets']; st.f.seek(st.base + o0); open(f'{out}/embed.bf16', 'wb').write(st.f.read(o1 - o0)); print('embed', o1 - o0)
    for i in range(a, b + 1):
        d, sz = pack_layer(st, i); open(f'{out}/lay{i:02d}.bin', 'wb').write(d); print(i, len(d), flush=True)
    print(sz)
