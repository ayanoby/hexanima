"""SeedVR2-3B one-step SR on the laptop (CPU, reference): python ref_sr.py in.png out.png [scale] [seed]
the input is resized (bicubic) by `scale` (default 2, sides multiples of 16), VAE-encoded, one DiT pass, VAE-decoded."""
import sys, time, torch, numpy as np
from PIL import Image
sys.path.insert(0, '/home/artem/Documents/Claude/anima/seedvr2')
from ref_vae import VAE2D, SCALE
from ref_dit import DiT
import os, re
if os.environ.get('Q8'):
    _g = DiT.g
    def g(self, k):
        w = _g(self, k)
        if w.ndim == 2 and w.shape[0] > 128 and re.search(r'(proj_qkv|proj_out|proj_in|proj_in_gate)\.[a-z]+\.weight$', k) and 'emb_in' not in k:
            sc = (w.abs().amax(1).clamp_min(1e-8) / 127); sc = sc.clamp_min(6.2e-5).half().float()
            return torch.round(w / sc[:, None]).clamp(-127, 127) * sc[:, None]
        return w
    DiT.g = g
torch.set_grad_enabled(False)
src, dst = sys.argv[1], sys.argv[2]; sc = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 666
im = Image.open(src).convert('RGB'); W, H = int(im.width * sc) // 16 * 16, int(im.height * sc) // 16 * 16
im = im.resize((W, H), Image.BICUBIC)
x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 127.5 - 1
vae = VAE2D(); t0 = time.time(); z = vae.encode(x)[0] * SCALE; print('encode', time.time() - t0, tuple(z.shape), flush=True)
g = torch.Generator().manual_seed(seed); noise = torch.randn(z.shape, generator=g)
dit = DiT(); t0 = time.time(); v = dit.forward(noise, z, 1000.0, log=True); print('dit', time.time() - t0, flush=True)
x0 = (noise - v) / SCALE
y = vae.decode(x0[None]); o = ((y.clamp(-1, 1) + 1) * 127.5)[0].permute(1, 2, 0).round().byte().numpy()
Image.fromarray(o).save(dst); print('done', dst, o.shape)
