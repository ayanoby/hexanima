"""Reference for the phone test of single layers: python ref_layers.py OUTDIR H8 W8 L0 L1 [Q8=1 env]
writes xt.raw / cond.raw (random latents, seed 1), x_in.f32 (token-order tokens + 58 text rows entering layer L0) and x_<l>.f32 after every layer L0..L1-1
(f32 [T+58][2560]). Layers before L0 are run from the start (so the input of layer L0 is exact)."""
import sys, os, re, torch, numpy as np
sys.path.insert(0, '/home/artem/Documents/Claude/anima/seedvr2')
import ref_dit
from ref_dit import DiT, windows, rope_freqs, D
torch.set_grad_enabled(False)
out, h8, w8, L0, L1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out, exist_ok=True)
if os.environ.get('Q8', '1') == '1':
    _g = DiT.g
    def g(self, k):
        w = _g(self, k)
        if w.ndim == 2 and w.shape[0] > 128 and re.search(r'(proj_qkv|proj_out|proj_in|proj_in_gate)\.[a-z]+\.weight$', k) and 'emb_in' not in k:
            sc = (w.abs().amax(1).clamp_min(1e-8) / 127).clamp_min(6.2e-5).half().float()
            return torch.round(w / sc[:, None]).clamp(-127, 127) * sc[:, None]
        return w
    DiT.g = g
gen = torch.Generator().manual_seed(1)
xt = torch.randn(16, h8, w8, generator=gen); cond = torch.randn(16, h8, w8, generator=gen) * 0.8
xt.numpy().astype(np.float32).tofile(f'{out}/xt.raw'); cond.numpy().astype(np.float32).tofile(f'{out}/cond.raw')
m = DiT(); h, w = h8 // 2, w8 // 2
inp = torch.cat([xt, cond, torch.ones(1, h8, w8)], 0)
tok = inp.view(33, h, 2, w, 2).permute(1, 3, 2, 4, 0).reshape(h * w, 132)
vid = m.lin(tok, 'vid_in.proj'); txt = m.lin(m.txt0, 'txt_in'); emb = m.emb(1000.0); L = txt.shape[0]
grid = torch.arange(h * w).view(h, w); W = {}
for sh in (False, True):
    ws = windows(h, w, sh); W[sh] = {'idx': [grid[a, b].reshape(-1) for a, b in ws], 'rope': [rope_freqs(1, a.stop - a.start, b.stop - b.start, L) for a, b in ws]}
    print('windows', sh, [(len(i)) for i in W[sh]['idx']])
for i in range(L1):
    if i == L0: torch.cat([vid, txt]).numpy().astype(np.float32).tofile(f'{out}/x_in.f32')
    vid, txt = m.block(i, vid, txt, W[i % 2 == 1], None, emb)
    if i >= L0: torch.cat([vid, txt]).numpy().astype(np.float32).tofile(f'{out}/x_{i:02d}.f32'); print('layer', i, float(vid.std()), flush=True)
