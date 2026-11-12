"""H3 VAE ViT decoder tile for QNN: z [1,24,16,16] (already latents*std+mean) -> image [1,3,256,256] (ImageNet-normalised RGB). usage: python export_vae_h3.py OUT.onnx [VAE.safetensors]"""
import sys, os, math, struct, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/home/artem/anima-npu/work/h3')
from h3lib import ST
class Dec(nn.Module):
    def __init__(s, path, H=16, W=16):
        super().__init__(); st = ST(path); T = lambda k: st.t('decoder.' + k).float().clone(); s.H, s.W = H, W
        s.pq_w = nn.Parameter(st.t('post_quant_conv.weight').float().reshape(24, 24).clone(), requires_grad=False); s.pq_b = nn.Parameter(st.t('post_quant_conv.bias').float().clone(), requires_grad=False)
        s.xe_w = nn.Parameter(T('x_embedder.weight'), requires_grad=False); s.xe_b = nn.Parameter(T('x_embedder.bias'), requires_grad=False)
        s.reg = nn.Parameter(T('register_tokens')[0], requires_grad=False)
        s.blocks = nn.ModuleList()
        for b in range(24):
            p = f'transformer_blocks.{b}.'; m = nn.Module()
            for n, k in (('n1', 'norm1.weight'), ('n2', 'norm2.weight'), ('s1', 'scale1'), ('s2', 'scale2'), ('qkv_w', 'attn.to_qkv.weight'), ('qkv_b', 'attn.to_qkv.bias'), ('o_w', 'attn.to_out.weight'), ('o_b', 'attn.to_out.bias'),
                         ('w1', 'ff.w1.weight'), ('b1', 'ff.w1.bias'), ('w2', 'ff.w2.weight'), ('b2', 'ff.w2.bias')):
                m.register_parameter(n, nn.Parameter(T(p + k), requires_grad=False))
            s.blocks.append(m)
        s.no_w = nn.Parameter(T('norm_out.weight'), requires_grad=False); s.no_b = nn.Parameter(T('norm_out.bias'), requires_grad=False); s.po_w = nn.Parameter(T('proj_out.weight'), requires_grad=False); s.po_b = nn.Parameter(T('proj_out.bias'), requires_grad=False)
        gh = 2.0 * (torch.arange(0.5, H) / H) - 1.0; gw = 2.0 * (torch.arange(0.5, W) / W) - 1.0
        pos = torch.stack(torch.meshgrid(gh, gw, indexing='ij'), -1).reshape(-1, 2); pos = torch.cat([pos, torch.zeros(5, 2)])
        inv = 1.0 / 100.0 ** torch.arange(0, 1, 4 / 32, dtype=torch.float32); ang = (2 * math.pi * pos[:, :, None] * inv[None, None]).flatten(1, 2).tile(2)
        s.register_buffer('cos', ang.cos()[:, None, :]); s.register_buffer('sin', ang.sin()[:, None, :])
    def rope(s, t):
        r, p = t[..., :32], t[..., 32:]; a, b = r[..., :16], r[..., 16:]; r = r * s.cos + torch.cat([-b, a], -1) * s.sin; return torch.cat([r, p], -1)
    @staticmethod
    def rn(t, w=None, eps=1e-5):
        y = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps); return y if w is None else y * w
    def forward(s, z):
        H, W = s.H, s.W; z = F.conv2d(z, s.pq_w.view(24, 24, 1, 1), s.pq_b)
        x = z[0].permute(1, 2, 0).reshape(H * W, 24); x = F.linear(x, s.xe_w, s.xe_b); x = torch.cat([x, s.reg, torch.zeros(1, 1024)]); n = x.shape[0]
        for m in s.blocks:
            h = s.rn(x, m.n1); qkv = F.linear(h, m.qkv_w, m.qkv_b).view(n, 16, 3, 64); q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = s.rope(s.rn(q)); k = s.rope(s.rn(k))
            a = torch.softmax(torch.matmul(q.transpose(0, 1), k.permute(1, 2, 0)) / 8.0, -1); o = torch.matmul(a, v.transpose(0, 1)).transpose(0, 1).reshape(n, 1024)
            x = x + F.linear(o, m.o_w, m.o_b) * m.s1
            h = s.rn(x, m.n2); y = F.linear(h, m.w1, m.b1); g, u = y.chunk(2, -1); x = x + F.linear(u * F.silu(g), m.w2, m.b2) * m.s2   # native w1 halves are (gate, value): diffusers has them swapped
        x = F.layer_norm(x, (1024,), s.no_w, s.no_b, 1e-5); x = F.linear(x, s.po_w, s.po_b)[:H * W]
        return x.view(H, W, 3, 16, 16).permute(2, 0, 3, 1, 4).reshape(1, 3, H * 16, W * 16)
if __name__ == '__main__':
    out = sys.argv[1]; path = sys.argv[2] if len(sys.argv) > 2 else '/home/artem/anima-npu/work/h3/ref/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors'
    m = Dec(path).eval(); z = torch.randn(1, 24, 16, 16) * 1.2
    with torch.no_grad(): ref = m(z)
    from h3lib import vae_decode
    with torch.no_grad(): r2 = vae_decode(z, log=lambda s: None)
    print('module vs h3lib.vae_decode rel diff', ((ref - r2).norm() / r2.norm()).item(), 'out', tuple(ref.shape), float(ref.abs().max()))
    torch.onnx.export(m, z, out, input_names=['z'], output_names=['img'], opset_version=17, dynamo=False)
    z.numpy().tofile(out + '.z.bin'); ref.numpy().tofile(out + '.ref.bin'); print('exported', out)
