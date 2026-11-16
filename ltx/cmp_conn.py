"""compare a connector block output (out_x.bin fp32 tiles [slab][D/32][32][32]) with ref.npz: python cmp_conn.py CASE_DIR OUT_X.bin"""
import sys, numpy as np
r = np.load(sys.argv[1] + '/ref.npz'); D = int(r['D']); ns = 32
a = np.fromfile(sys.argv[2], dtype=np.float32).reshape(ns, D // 32, 32, 32).transpose(0, 2, 1, 3).reshape(ns * 32, D)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
print('out vs ref rel %.3e  nan %d  | out vs input %.3f, ref vs input %.3f' % (rel(a, r['xo']), int(np.isnan(a).sum()), rel(a, r['xin']), rel(r['xo'], r['xin'])))
for s0 in range(0, ns, 8): print('  slabs %d-%d rel %.3e' % (s0, s0 + 7, rel(a[s0 * 32:(s0 + 8) * 32], r['xo'][s0 * 32:(s0 + 8) * 32])))
# dumped fp16 buffers (klblk writes out_q/k/v/o/mid): tiles [slab][cols/32][1024]
import os
d = os.path.dirname(sys.argv[2])
def tiles(name, cols):
    p = f'{d}/{name}'
    if not os.path.exists(p): return None
    a = np.fromfile(p, dtype=np.float16, count=ns * cols // 32 * 1024).astype(np.float32).reshape(ns, cols // 32, 16, 32, 2); return a.transpose(0, 2, 4, 1, 3).reshape(ns * 32, cols)
for nm, key, cols in (('out_q.bin', 'q', D), ('out_k.bin', 'k', D), ('out_v.bin', 'v', D), ('out_o.bin', 'gated', D), ('out_mid.bin', 'mid', 4 * D)):
    a = tiles(nm, cols)
    if a is not None: print(nm, 'vs ref', key, 'rel %.3e' % rel(a, r[key]) + ('  (reference q divided by sqrt(hd): the kernel folds it into the q gain)' if key == 'q' else ''))
