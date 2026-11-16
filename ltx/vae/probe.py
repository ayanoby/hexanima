"""QNN op support probe: export a piece of the LTX conv VAE decoder to ONNX (fp32, opset 17): python probe.py NAME  (res128: 4 res blocks of 128 ch + conv_out + pixel_norm/silu; d2s: the (1,2,2) DepthToSpaceUpsample (256->512) )"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ref_vae as V
dec = V.dec; name = sys.argv[1]
class PN4(torch.nn.Module):   # PixelNorm on a 4D view: QNN's RMSNorm op does not take rank 5
    def __init__(s, eps=1e-8): super().__init__(); s.eps = eps
    def forward(s, x):
        n, c, t, h, w = x.shape; y = x.reshape(n, c, t * h, w); y = y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + s.eps); return y.reshape(n, c, t, h, w)
for mod in dec.modules():
    for k, v in list(mod._modules.items()):
        if type(v).__name__ == 'PixelNorm': mod._modules[k] = PN4(getattr(v, 'eps', 1e-8))
if name == 'res128':
    class M(torch.nn.Module):
        def __init__(s): super().__init__(); s.b = dec.up_blocks[8]; s.n = dec.conv_norm_out; s.a = dec.conv_act; s.o = dec.conv_out
        def forward(s, x): return s.o(s.a(s.n(s.b(x, causal=False)))) if False else s.o(s.a(s.n(s.b(x))), causal=False)
    m = M().eval(); x = torch.randn(1, 128, 6, 16, 16)
else:
    class M(torch.nn.Module):
        def __init__(s): super().__init__(); s.u = dec.up_blocks[7]
        def forward(s, x): return s.u(x, causal=False)
    m = M().eval(); x = torch.randn(1, 256, 6, 8, 8)
with torch.no_grad(): y = m(x); print(name, tuple(x.shape), '->', tuple(y.shape))
torch.onnx.export(m, (x,), f'{name}.onnx', input_names=['x'], output_names=['y'], opset_version=17); print('exported')
