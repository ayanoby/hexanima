"""compare hexagon-sim block dumps with ref.npz: python cmp_sim.py CASE_DIR(ref.npz) SIM_DIR NTOK NVALID"""
import sys, numpy as np
ref = np.load(sys.argv[1] + '/ref.npz'); d = sys.argv[2]; ntok = int(sys.argv[3]); n = int(sys.argv[4]); ns = ntok // 32
def tiles(name, ncols):   # fp16 tiles [slab][col/32][1024]: (r/2)*64 + (c%32)*2 + (r&1)
    a = np.fromfile(f'{d}/{name}', dtype=np.float16).astype(np.float32).reshape(ns, ncols // 32, 16, 32, 2)   # [s][cb][r/2][c][r&1]
    return a.transpose(0, 2, 4, 1, 3).reshape(ns * 32, ncols)[:n]
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
perm = np.concatenate([np.arange(0, 128, 2), np.arange(1, 128, 2)])
mid = tiles('out_mid.bin', 9216); print('mid rel', rel(mid, ref['mid'][:n]), 'corr', float((mid * ref['mid'][:n]).sum() / np.sqrt((mid ** 2).sum() * (ref['mid'][:n] ** 2).sum())))
o = tiles('out_o.bin', 3072); print('o   rel', rel(o, ref['o'][:n]))
q = tiles('out_q.bin', 3072).reshape(n, 24, 128); rq = ref['q'][:n]
for nm, rr in (('q vs ref q permuted', rq[:, :, perm]), ('q vs ref q', rq)):
    print(nm, rel(q, rr), 'corr', float((q * rr).sum() / np.sqrt((q ** 2).sum() * (rr ** 2).sum())))
