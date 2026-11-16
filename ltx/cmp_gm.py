"""compare a Gemma layer output (out_x.bin, fp32 tiles [slab][120][32][32]) with ref.npz: python cmp_gm.py CASE_DIR OUT_X.bin"""
import sys, numpy as np
r = np.load(sys.argv[1] + '/ref.npz'); n = int(r['n']); ntok = int(r['ntok']); xo = r['xo'][:n]; xin = r['xin'][:n]
a = np.fromfile(sys.argv[2], dtype=np.float32).reshape(ntok // 32, 120, 32, 32).transpose(0, 2, 1, 3).reshape(ntok, 3840)[:n]
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
print('out vs ref: rel %.3e  nan %d  | out vs input %.3f, ref vs input %.3f' % (rel(a, xo), int(np.isnan(a).sum()), rel(a, xin), rel(xo, xin)))
for i0 in range(0, n, 16): print('  rows %d-%d rel %.3e' % (i0, min(n, i0 + 16) - 1, rel(a[i0:i0 + 16], xo[i0:i0 + 16])))
