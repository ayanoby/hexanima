"""SeedVR2 VAE for ONE frame, plain 2D PyTorch (causal Conv3d with a replicated first frame == Conv2d with the weight summed over time).
usage: python ref_vae.py image.png [out.png]  -> encode (posterior mean) + decode, prints PSNR"""
import sys, torch, torch.nn.functional as F, numpy as np
from PIL import Image
VAE = '/run/media/artem/32363EC6363E8ABD/seedvr2/3b/ema_vae.pth'
SCALE = 0.9152

class VAE2D:
    def __init__(self, path=VAE):
        sd = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        self.w = {}
        for k, v in sd.items():
            if v.ndim == 5: v = v.sum(2)          # (O,I,kt,kh,kw) -> (O,I,kh,kw)
            self.w[k] = v.float()
    def conv(self, x, p, stride=1, pad=1):
        return F.conv2d(x, self.w[p + '.weight'], self.w[p + '.bias'], stride=stride, padding=pad)
    def gn(self, x, p):
        return F.group_norm(x, 32, self.w[p + '.weight'], self.w[p + '.bias'], 1e-6)
    def res(self, x, p):
        h = self.conv(F.silu(self.gn(x, p + '.norm1')), p + '.conv1')
        h = self.conv(F.silu(self.gn(h, p + '.norm2')), p + '.conv2')
        if p + '.conv_shortcut.weight' in self.w: x = self.conv(x, p + '.conv_shortcut', pad=0)
        return x + h
    def attn(self, x, p):
        B, C, H, W = x.shape
        h = self.gn(x, p + '.group_norm').flatten(2).transpose(1, 2)
        lin = lambda t, n: F.linear(t, self.w[p + '.' + n + '.weight'], self.w[p + '.' + n + '.bias'])
        q, k, v = lin(h, 'to_q'), lin(h, 'to_k'), lin(h, 'to_v')
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        o = F.linear(o, self.w[p + '.to_out.0.weight'], self.w[p + '.to_out.0.bias'])
        return x + o.transpose(1, 2).reshape(B, C, H, W)
    def mid(self, x, p):
        x = self.res(x, p + '.resnets.0'); x = self.attn(x, p + '.attentions.0'); return self.res(x, p + '.resnets.1')
    def encode(self, x):                          # x in [-1,1], (1,3,H,W) -> latent (1,16,H/8,W/8), unscaled mean
        e = 'encoder'
        x = self.conv(x, e + '.conv_in')
        for i in range(4):
            for j in range(2): x = self.res(x, f'{e}.down_blocks.{i}.resnets.{j}')
            if i < 3: x = self.conv(F.pad(x, (0, 1, 0, 1)), f'{e}.down_blocks.{i}.downsamplers.0.conv', stride=2, pad=0)
        x = self.mid(x, e + '.mid_block')
        x = self.conv(F.silu(self.gn(x, e + '.conv_norm_out')), e + '.conv_out')
        return x[:, :16]
    def decode(self, z):
        d = 'decoder'
        x = self.conv(z, d + '.conv_in')
        x = self.mid(x, d + '.mid_block')
        for i in range(4):
            for j in range(3): x = self.res(x, f'{d}.up_blocks.{i}.resnets.{j}')
            if i < 3:
                p = f'{d}.up_blocks.{i}.upsamplers.0'
                C = x.shape[1]; w = self.w[p + '.upscale_conv.weight']; b = self.w[p + '.upscale_conv.bias']
                Z = w.shape[0] // (4 * C)
                # channel ((a*2+b2)*Z + z)*C + c  ->  keep the last temporal copy (z = Z-1) and order the rows as c*4 + a*2 + b2 so that PixelShuffle(2) does the rest (7-D reshapes do not survive QNN)
                idx = torch.tensor([((a * 2 + b2) * Z + (Z - 1)) * C + c for c in range(C) for a in range(2) for b2 in range(2)])
                x = F.pixel_shuffle(F.conv2d(x, w[idx], b[idx]), 2)
                x = self.conv(x, p + '.conv')
        return self.conv(F.silu(self.gn(x, d + '.conv_norm_out')), d + '.conv_out')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    im = Image.open(sys.argv[1]).convert('RGB'); W, H = im.size; im = im.crop((0, 0, W // 16 * 16, H // 16 * 16))
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float() / 127.5 - 1
    v = VAE2D(); z = v.encode(x); print('latent', tuple(z.shape), 'std', z.std().item(), 'mean', z.mean().item())
    y = v.decode(z)
    o = ((y.clamp(-1, 1) + 1) * 127.5)[0].permute(1, 2, 0).round().byte().numpy()
    mse = ((o.astype(float) - np.asarray(im).astype(float)) ** 2).mean(); print('PSNR', 10 * np.log10(255 ** 2 / mse))
    if len(sys.argv) > 2: Image.fromarray(o).save(sys.argv[2])
