"""H3 LoRA (pDMD 2-step, rank 128) -> per-block fp16 tile packs for the phone. The base weights are stored ConvRot-rotated, the DiT sees rotated activations hr = x.Hn, so
A x = A (Hn hr) = (A.Hn) hr: A' = A rotated per 256-column block, and the engine adds D = (A' hr) . B^T before the epilogue of each GEMM (delta = B (A x), scale alpha/rank folded into B).
loraNN.bin (fp16 tiles, wt16 layout [nb][kb][16][32][2], 2048 B per tile): A'q A'k A'v [K=5376][N=128], A'o [7168][128], A'gu [5376][128], A'dn [14336][128],
 Bq Bk Bv [128][7168] (q, k output channels permuted per head like the base pack), Bo [128][5376], Bgu [128][28672] (n-blocks (gate_j, up_j), gate = rows 14336.. of ff.net.0.proj, up = rows ..14336), Bdn [128][5376]
usage: python export_h3_lora.py LORA.safetensors OUTDIR     -> lora00..49.bin and lorar0.bin lorar1.bin (refiner blocks)"""
import os, sys, numpy as np
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_h3 import PERM
Hn = (np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H256.npy')).astype(np.float32) / 16.0)
def wt16(w):   # w [K,N] fp32 -> fp16 tiles [nb][kb][16][32][2]
    K, N = w.shape; return np.ascontiguousarray(w.astype(np.float16).reshape(K // 32, 16, 2, N // 32, 32).transpose(3, 0, 1, 4, 2)).tobytes()
def rotA(A): r, K = A.shape; return (A.reshape(r, K // 256, 256) @ Hn).reshape(r, K)
def pack(f, p):
    g = lambda n, ab: f.get_tensor(f'transformer.{p}{n}.lora_{ab}.weight').float().numpy()
    out = []; sz = {}
    def add(n, b): sz[n] = len(b); out.append(b)
    for n in ('attn.to_q', 'attn.to_k', 'attn.to_v'): add('A' + n[-1], wt16(rotA(g(n, 'A')).T))
    add('Ao', wt16(rotA(g('attn.to_out.0', 'A')).T)); add('Agu', wt16(rotA(g('ff.net.0.proj', 'A')).T)); add('Adn', wt16(rotA(g('ff.net.2', 'A')).T))
    for n in ('attn.to_q', 'attn.to_k', 'attn.to_v'):
        B = g(n, 'B')
        if n != 'attn.to_v': B = B.reshape(56, 128, -1)[:, PERM].reshape(B.shape)
        add('B' + n[-1], wt16(B.T))
    add('Bo', wt16(g('attn.to_out.0', 'B').T))
    B = g('ff.net.0.proj', 'B'); FF = B.shape[0] // 2; up, gate = B[:FF], B[FF:]
    Bi = np.stack((gate.reshape(FF // 32, 32, -1), up.reshape(FF // 32, 32, -1)), 1).reshape(2 * FF, -1)
    add('Bgu', wt16(Bi.T)); add('Bdn', wt16(g('ff.net.2', 'B').T))
    return b''.join(out), sz
if __name__ == '__main__':
    f = safe_open(sys.argv[1], 'pt'); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    for i in range(50):
        d, sz = pack(f, f'transformer_blocks.{i}.'); open(f'{out}/lora{i:02d}.bin', 'wb').write(d); print(i, len(d), flush=True)
    for i in range(2):
        d, sz = pack(f, f'token_refiner.refiner_blocks.{i}.'); open(f'{out}/lorar{i}.bin', 'wb').write(d); print('r', i, len(d), flush=True)
    off = 0
    for k, v in sz.items(): print(k, off, v); off += v
    print('total', off)
