"""SeedVR2 VAE tiles for QNN (one frame, 2D): python export_vae_svr.py OUTDIR [T=64]
enc: img [1,3,8T,8T] in [-1,1] -> mean latent [1,16,T,T] (un-scaled); dec: z [1,16,T,T] (un-scaled) -> img [1,3,8T,8T]"""
import sys, os, torch
sys.path.insert(0, '/home/artem/Documents/Claude/anima/seedvr2')
from ref_vae import VAE2D
out = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 64
os.makedirs(out, exist_ok=True); v = VAE2D()
class E(torch.nn.Module):
    def forward(s, x): return v.encode(x)
class Dc(torch.nn.Module):
    def forward(s, z): return v.decode(z)
torch.manual_seed(1)
for name, m, inp in (('svr_enc', E(), torch.randn(1, 3, 8 * T, 8 * T).clamp(-1, 1)), ('svr_dec', Dc(), torch.randn(1, 16, T, T))):
    with torch.no_grad(): ref = m(inp)
    print(name, tuple(inp.shape), '->', tuple(ref.shape), 'max', float(ref.abs().max()), flush=True)
    with torch.no_grad(): torch.onnx.export(m, inp, f'{out}/{name}_t{T}.onnx', input_names=['x'], output_names=['y'], opset_version=17, dynamo=False)
    inp.numpy().tofile(f'{out}/{name}_t{T}.onnx.x.bin'); ref.numpy().tofile(f'{out}/{name}_t{T}.onnx.ref.bin')
print('saved')
