"""FLUX.3 Klein VAE decoder tile for QNN: z [1,32,2T,2T] (de-normalised, un-patchified) -> img [1,3,16T,16T] in [-1,1]. usage: python export_vae_klein.py OUT.onnx T"""
import sys, torch
sys.path.insert(0, 'ref/src')
from flux3.autoencoder import AutoEncoder
from flux3.checkpoint import load_checkpoint
out, T = sys.argv[1], int(sys.argv[2])
ae = load_checkpoint(AutoEncoder(), 'klein/ae.safetensors', 'cpu', torch.float32).eval()
class D(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, z): return s.d(z)
m = D(ae.decoder).eval()
for p in m.parameters(): p.requires_grad_(False)
torch.manual_seed(1); z = torch.randn(1, 32, 2 * T, 2 * T)
with torch.no_grad(): ref = m(z)
print('out', tuple(ref.shape), 'max', float(ref.abs().max()))
with torch.no_grad(): torch.onnx.export(m, z, out, input_names=['z'], output_names=['img'], opset_version=17, dynamo=False)
z.numpy().tofile(out + '.z.bin'); ref.numpy().tofile(out + '.ref.bin')
# BatchNorm stats for the host side: mean, sqrt(var + eps)
import numpy as np
np.concatenate([ae.bn.running_mean.numpy(), torch.sqrt(ae.bn.running_var + ae.bn_eps).numpy()]).astype(np.float32).tofile('pack/glue/vae_bn.f32')
print('saved', out)
