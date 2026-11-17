"""full 32-layer reference velocity for the phone test inputs: python ref_full.py DIR H8 W8  (reads xt.raw, cond.raw; writes v_ref.raw; Q8=1 default)"""
import sys, re, os, numpy as np, torch
sys.path.insert(0, '/home/artem/Documents/Claude/anima/seedvr2')
from ref_dit import DiT
torch.set_grad_enabled(False)
d, h8, w8 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if os.environ.get('Q8', '1') == '1':
    _g = DiT.g
    def g(self, k):
        w = _g(self, k)
        if w.ndim == 2 and w.shape[0] > 128 and re.search(r'(proj_qkv|proj_out|proj_in|proj_in_gate)\.[a-z]+\.weight$', k) and 'emb_in' not in k:
            sc = (w.abs().amax(1).clamp_min(1e-8) / 127).clamp_min(6.2e-5).half().float()
            return torch.round(w / sc[:, None]).clamp(-127, 127) * sc[:, None]
        return w
    DiT.g = g
xt = torch.from_numpy(np.fromfile(f'{d}/xt.raw', np.float32)).view(16, h8, w8); cond = torch.from_numpy(np.fromfile(f'{d}/cond.raw', np.float32)).view(16, h8, w8)
v = DiT().forward(xt, cond, 1000.0, log=True); v.numpy().astype(np.float32).tofile(f'{d}/v_ref.raw'); print('std', float(v.std()))
