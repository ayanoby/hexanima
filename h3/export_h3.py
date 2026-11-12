"""Pack MiniMax-H3-Image DiT (unsloth INT8-ConvRot safetensors) into HMX q8 tile files, one per block, LOSSLESS: the int8 values are kept as they are and the per-output-row scale
(fp16) is replicated in the 128 B scale vector of every tile of that output column (the ConvRot weights are stored rotated; the activations get rotated at run time).
blkNN.bin (K = input, N = output, tile records [nb][kb] 1152 B as in moss/export_all_q8.q8tiles):
  Q,K  [5376][7168] (output channels permuted per head: orig 0..47,96..111,48..95,112..127 so the engine's split-half RoPE rotates pairs (i,i+64), i<48 real, 48..63 identity)
  V    [5376][7168]   O [7168][5376]   GU [5376][28672] n-blocks of 32 interleaved (gate_j, up_j) (gate = rows 14336.. of ff.net.0.proj, up = rows ..14336)   DN [14336][5376]
  gain f32 x 16384: q_norm(perm)/sqrt(128) [128], k_norm(perm) [128], norm1 w [5376], norm2 w [5376]
usage: python export_h3.py DIT.safetensors OUTDIR [first last]"""
import os, sys, json, struct, numpy as np
PERM = np.array(list(range(0, 48)) + list(range(96, 112)) + list(range(48, 96)) + list(range(112, 128)))
K0, QN, FF = 5376, 7168, 14336
class ST:
    def __init__(s, p):
        s.f = open(p, 'rb'); n = struct.unpack('<Q', s.f.read(8))[0]; s.h = json.loads(s.f.read(n)); s.base = 8 + n
    def t(s, k):
        m = s.h[k]; a, b = m['data_offsets']; s.f.seek(s.base + a)
        dt = {'I8': np.int8, 'F32': np.float32, 'BF16': np.uint16}[m['dtype']]
        x = np.frombuffer(s.f.read(b - a), dtype=dt).reshape(m['shape'])
        return (x.astype(np.uint32) << 16).view(np.float32) if m['dtype'] == 'BF16' else x
def tilepack(q, s):
    """q int8 [K,N] (input x output), s fp32 [N] per-output scale -> bytes of [nb][kb] records (128 B fp16 scale vector (halfword l = column l//2) + 1024 B offset-binary int8 in wt16 order)"""
    K, N = q.shape; assert K % 32 == 0 and N % 32 == 0
    t = np.ascontiguousarray(q.reshape(K // 32, 16, 2, N // 32, 32).transpose(3, 0, 1, 4, 2))      # [nb][kb][16][32][2]
    nb, kb = t.shape[:2]
    s16 = np.maximum(s.astype(np.float32), 6.2e-5).astype(np.float16).reshape(nb, 32)                # [nb][32]
    sv = np.repeat(s16, 2, axis=1)                                                                    # [nb][64]
    svb = np.broadcast_to(sv.view(np.uint8).reshape(nb, 1, 128), (nb, kb, 128))
    u = (t.astype(np.int16) + 128).astype(np.uint8).reshape(nb, kb, 1024)
    return np.ascontiguousarray(np.concatenate([svb, u], axis=2)).tobytes()
def tilepack3(q, s):
    """COMPACT q8 (fmt 3): q int8 [K,N], s fp32 [N] per-output scale -> per n-block [128 B fp16 scale vector (halfword l = column l//2)][kb x 1024 B offset-binary int8 tiles in wt16 order]. 12.5 % smaller than tilepack, the same values"""
    K, N = q.shape; assert K % 32 == 0 and N % 32 == 0
    t = np.ascontiguousarray(q.reshape(K // 32, 16, 2, N // 32, 32).transpose(3, 0, 1, 4, 2))      # [nb][kb][16][32][2]
    nb, kb = t.shape[:2]
    s16 = np.maximum(s.astype(np.float32), 6.2e-5).astype(np.float16).reshape(nb, 32)
    sv = np.repeat(s16, 2, axis=1).view(np.uint8).reshape(nb, 128)
    u = (t.astype(np.int16) + 128).astype(np.uint8).reshape(nb, kb * 1024)
    return np.ascontiguousarray(np.concatenate([sv, u], axis=1)).tobytes()
def lin(st, base):   # -> int8 [N,K], scale [N]
    return st.t(base + '._weight_qdata'), st.t(base + '._weight_scale').reshape(-1)
def hperm(q, s):     # rows of 56 heads x 128: permute inside each head
    return q.reshape(56, 128, -1)[:, PERM].reshape(q.shape), s.reshape(56, 128)[:, PERM].reshape(-1)
def pack_block(st, i, prefix='transformer_blocks.'):
    p = f'{prefix}{i}.'; out = []; sz = {}
    def add(n, b): sz[n] = len(b); out.append(b)
    for n in ('to_q', 'to_k', 'to_v', 'to_out.0'):
        q, s = lin(st, p + 'attn.' + n)
        if n in ('to_q', 'to_k'): q, s = hperm(q, s)
        add(n, tilepack3(np.ascontiguousarray(q.T), s))
    q, s = lin(st, p + 'ff.net.0.proj'); up, gate = (q[:FF], s[:FF]), (q[FF:], s[FF:])
    qi = np.stack((gate[0].reshape(FF // 32, 32, K0), up[0].reshape(FF // 32, 32, K0)), 1).reshape(2 * FF, K0)       # n-blocks (gate_j, up_j)
    si = np.stack((gate[1].reshape(FF // 32, 32), up[1].reshape(FF // 32, 32)), 1).reshape(-1)
    add('gate_up', tilepack3(np.ascontiguousarray(qi.T), si))
    q, s = lin(st, p + 'ff.net.2'); add('down', tilepack3(np.ascontiguousarray(q.T), s))
    qn = st.t(p + 'attn.norm_q.weight').astype(np.float32)[PERM] / np.sqrt(128.0); kn = st.t(p + 'attn.norm_k.weight').astype(np.float32)[PERM]
    g = np.concatenate([qn, kn, st.t(p + 'norm1.weight'), st.t(p + 'norm2.weight')]).astype(np.float32)
    add('gain', g.tobytes() + bytes(16384 * 4 - g.nbytes))
    return b''.join(out), sz
def unpack_check(st, i, b):   # round-trip: dequantise a few tiles of the pack and compare with the source int8*scale
    p = f'transformer_blocks.{i}.'; q, s = lin(st, p + 'attn.to_v'); nbs = 128 + 168 * 1024; rec = np.frombuffer(b, dtype=np.uint8)[2 * 224 * nbs: 3 * 224 * nbs].reshape(224, nbs)   # third matrix = to_v (no head permutation)
    err = 0.0
    for nb, kb in ((0, 0), (3, 7), (100, 90), (223, 167)):
        sc = rec[nb, :128].view(np.float16).astype(np.float32)[::2]; err = max(err, float(np.abs(sc - s[nb * 32:(nb + 1) * 32]).max() / np.abs(s[nb * 32:(nb + 1) * 32]).max()))
        r = rec[nb, 128 + kb * 1024:128 + (kb + 1) * 1024]; u = r.astype(np.int16) - 128
        t = u.reshape(16, 32, 2).transpose(0, 2, 1).reshape(32, 32)             # [k][col] after undoing (r,c) = (16,32,2) paired rows: row = 2*a + b
        w = q.T[kb * 32:(kb + 1) * 32, nb * 32:(nb + 1) * 32].astype(np.int16)
        err = max(err, float(np.abs(t - w).max()))
    return err   # largest of: int8 difference, relative scale difference (fp16 rounding)
if __name__ == '__main__':
    st = ST(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    a = int(sys.argv[3]) if len(sys.argv) > 3 else 0; b = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    for i in range(a, b + 1):
        d, sz = pack_block(st, i); open(f'{out}/blk{i:02d}.bin', 'wb').write(d); print(i, len(d), 'rt err (int8 steps)', unpack_check(st, i, d), flush=True)
    print(sz)
