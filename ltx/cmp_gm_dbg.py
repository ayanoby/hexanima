"""compare dumped q/k/v/o of the Gemma layer kernel with ref.npz: python cmp_gm_dbg.py CASE_DIR DUMP_DIR  (fp16 tiles [slab][cols/32][1024], tile = (r/2)*64 + c*2 + (r&1))"""
import sys, numpy as np
r = np.load(sys.argv[1] + '/ref.npz'); d = sys.argv[2]; n = int(r['n']); ntok = int(r['ntok']); ns = ntok // 32
def tiles(name, cols): a = np.fromfile(f'{d}/{name}', dtype=np.float16, count=ns * cols // 32 * 1024).astype(np.float32).reshape(ns, cols // 32, 16, 32, 2); return a.transpose(0, 2, 4, 1, 3).reshape(ns * 32, cols)[:n]
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
hd = r['q'].shape[-1]; nkv = r['k'].shape[1]
for nm, cols, ref in (('q', 16 * hd, r['q'].reshape(n, -1)), ('k', nkv * hd, r['k'].reshape(n, -1)), ('v', nkv * hd, r['v'].reshape(n, -1)), ('o', 16 * hd, r['o'])):
    a = tiles(f'out_{nm}.bin', cols); print(nm, 'rel all %.3e' % rel(a, ref), ' slab0 %.3e' % rel(a[:32], ref[:32]), ' slab1 %.3e' % rel(a[32:], ref[32:]))
