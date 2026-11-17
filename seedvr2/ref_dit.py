"""SeedVR2-3B DiT, one image, plain PyTorch (own implementation of the official NaDiT, written from models/dit_v2/*).
sr(x_cond_latent, seed) -> x0 latent after ONE step (t = 1000, cfg 1, fixed text embedding)."""
import math, torch, torch.nn.functional as F
CK = '/run/media/artem/32363EC6363E8ABD/seedvr2/3b/seedvr2_ema_3b.pth'
TXT = '/run/media/artem/32363EC6363E8ABD/seedvr2/3b/pos_emb.pt'
D, H, HD, NL, MM, EPS = 2560, 20, 128, 32, 10, 1e-5
SCALE = 0.9152

def windows(h, w, shifted):   # official make_720Pwindows_bysize / shifted, t = 1; returns list of (slice_h, slice_w)
    scale = math.sqrt((45 * 80) / (h * w)); rh, rw = round(h * scale), round(w * scale)
    wh, ww = math.ceil(rh / 3), math.ceil(rw / 3)
    if not shifted:
        nh, nw = math.ceil(h / wh), math.ceil(w / ww)
        return [(slice(ih * wh, min((ih + 1) * wh, h)), slice(iw * ww, min((iw + 1) * ww, w))) for iw in range(nw) for ih in range(nh) if min((iw + 1) * ww, w) > iw * ww and min((ih + 1) * wh, h) > ih * wh]
    sh, sw = (0.5 if wh < h else 0), (0.5 if ww < w else 0)
    nh, nw = math.ceil((h - sh) / wh), math.ceil((w - sw) / ww); nh, nw = (nh + 1 if sh > 0 else 1), (nw + 1 if sw > 0 else 1)
    out = []
    for iw in range(nw):
        w0, w1 = max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)
        if w1 <= w0: continue
        for ih in range(nh):
            h0, h1 = max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)
            if h1 > h0: out.append((slice(h0, h1), slice(w0, w1)))
    return out

def rope_freqs(nt, wh, ww, txt_len):
    """cos/sin for the window tokens (T index = txt_len, H, W window-local) and for the text tokens (position i on all 3 axes). 126 rotated dims (3 x 42), the last 2 stay."""
    f = 1.0 / (10000 ** (torch.arange(0, 42, 2).float() / 42))            # 21
    def ax(pos): return torch.repeat_interleave(pos[:, None].float() * f[None], 2, dim=-1)   # (n, 42) interleaved pairs
    hh, ww_ = torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing='ij')
    vid = torch.cat([ax(torch.full((wh * ww,), txt_len)), ax(hh.reshape(-1)), ax(ww_.reshape(-1))], -1)    # (n, 126)
    txt = ax(torch.arange(txt_len)).repeat(1, 3)
    return vid, txt

def rot(x, ang):   # x (L, h, 128); ang (L, 126) interleaved pairs
    c, s = ang.cos()[:, None], ang.sin()[:, None]
    xr, xp = x[..., :126], x[..., 126:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    rh = torch.stack([-x2, x1], -1).flatten(-2)
    return torch.cat([xr * c + rh * s, xp], -1)

def rms(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if w is None else y * w

class DiT:
    def __init__(self, ck=CK):
        self.sd = torch.load(ck, map_location='cpu', mmap=True, weights_only=True)
        self.txt0 = torch.load(TXT, map_location='cpu', weights_only=True).float()
    def g(self, k): return self.sd[k].float()
    def lin(self, x, k, bias=True): return F.linear(x, self.g(k + '.weight'), self.g(k + '.bias') if bias and (k + '.bias') in self.sd else None)
    def emb(self, t):
        half = 128; fr = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        a = torch.tensor([t]).float()[:, None] * fr[None]; e = torch.cat([a.sin(), a.cos()], -1)
        for n in ('proj_in', 'proj_hid', 'proj_out'):
            e = self.lin(e, 'emb_in.' + n)
            if n != 'proj_out': e = F.silu(e)
        return e[0]                                    # (6*D)
    def block(self, i, vid, txt, wins, rope, emb):
        mm = i < MM; last = i == NL - 1
        br = lambda s: ('vid', 'txt') if mm else ('all', 'all')
        vb, tb = br(0)
        P = f'blocks.{i}.'
        e = emb.view(D, 2, 3)                          # (d, layer, g) layer 0 = attn, 1 = mlp; g = shift, scale, gate
        def ada_in(x, which, site):
            return x * (e[:, site, 1] + self.g(P + f'ada.{which}.{["attn","mlp"][site]}_scale')) + (e[:, site, 0] + self.g(P + f'ada.{which}.{["attn","mlp"][site]}_shift'))
        def ada_out(x, which, site): return x * (e[:, site, 2] + self.g(P + f'ada.{which}.{["attn","mlp"][site]}_gate'))
        # attention
        v = ada_in(rms(vid), vb, 0); t = ada_in(rms(txt), tb, 0)
        qkv_v = self.lin(v, P + f'attn.proj_qkv.{vb}', False).view(-1, 3, H, HD); qkv_t = self.lin(t, P + f'attn.proj_qkv.{tb}', False).view(-1, 3, H, HD)
        qt, kt, vt = qkv_t.unbind(1)
        qt = rms(qt, self.g(P + f'attn.norm_q.{tb}.weight')); kt = rms(kt, self.g(P + f'attn.norm_k.{tb}.weight'))
        out_v = torch.empty(vid.shape[0], H, HD); out_t = torch.zeros(txt.shape[0], H, HD)
        qv_all, kv_all, vv_all = qkv_v.unbind(1)
        qv_all = rms(qv_all, self.g(P + f'attn.norm_q.{vb}.weight')); kv_all = rms(kv_all, self.g(P + f'attn.norm_k.{vb}.weight'))
        for idx, (ang_v, ang_t) in zip(wins['idx'], wins['rope']):
            qv, kv_, vv = qv_all[idx], kv_all[idx], vv_all[idx]
            qv, kv_ = rot(qv, ang_v), rot(kv_, ang_v)
            q = torch.cat([qv, rot(qt, ang_t)]); k = torch.cat([kv_, rot(kt, ang_t)]); vvv = torch.cat([vv, vt])
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], vvv.transpose(0, 1)[None])[0].transpose(0, 1)   # (n+L, h, d)
            out_v[idx] = o[:len(idx)]; out_t += o[len(idx):]
        out_t /= len(wins['idx'])                      # text pooling = mean over the windows
        ov = self.lin(out_v.reshape(-1, D), P + f'attn.proj_out.{vb}'); ot = self.lin(out_t.reshape(-1, D), P + f'attn.proj_out.{tb}')
        vid = vid + ada_out(ov, vb, 0); txt = txt + ada_out(ot, tb, 0)
        # mlp
        def mlp(x, which):
            return self.lin(F.silu(self.lin(x, P + f'mlp.{which}.proj_in_gate', False)) * self.lin(x, P + f'mlp.{which}.proj_in', False), P + f'mlp.{which}.proj_out', False)
        vid = vid + ada_out(mlp(ada_in(rms(vid), vb, 1), vb), vb, 1)
        if not last: txt = txt + ada_out(mlp(ada_in(rms(txt), tb, 1), tb), tb, 1)
        return vid, txt
    @torch.no_grad()
    def forward(self, x_t, cond, t=1000.0, log=False):
        """x_t, cond: (16, h8, w8) latents (cond = scaled cond latent); returns v (16, h8, w8)"""
        C, h8, w8 = x_t.shape; h, w = h8 // 2, w8 // 2
        inp = torch.cat([x_t, cond, torch.ones(1, h8, w8)], 0)                       # (33, h8, w8)
        tok = inp.view(33, h, 2, w, 2).permute(1, 3, 2, 4, 0).reshape(h * w, 2 * 2 * 33)   # feature order (t=1) h w c
        vid = self.lin(tok, 'vid_in.proj')
        txt = self.lin(self.txt0, 'txt_in')
        emb = self.emb(t)
        L = txt.shape[0]; tokgrid = torch.arange(h * w).view(h, w)
        W = {}
        for sh in (False, True):
            ws = windows(h, w, sh); idxs = [tokgrid[a, b].reshape(-1) for a, b in ws]
            ropes = []
            for (a, b) in ws:
                wh_, ww_ = a.stop - a.start, b.stop - b.start; ropes.append(rope_freqs(1, wh_, ww_, L))
            W[sh] = {'idx': idxs, 'rope': ropes}
        for i in range(NL):
            vid, txt = self.block(i, vid, txt, W[i % 2 == 1], None, emb)
            if log and i % 4 == 0: print('block', i, float(vid.std()), flush=True)
        e = emb.view(D, 2, 3)
        v = rms(vid, self.g('vid_out_norm.weight'))
        v = v * (e[:, 0, 1] + self.g('vid_out_ada.out_scale')) + (e[:, 0, 0] + self.g('vid_out_ada.out_shift'))
        o = self.lin(v, 'vid_out.proj')                                              # (h*w, 64) = (t h w c) with t = 1
        return o.view(h, w, 2, 2, 16).permute(4, 0, 2, 1, 3).reshape(16, h8, w8)
