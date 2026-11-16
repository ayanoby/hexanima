"""Torch (CPU) reference of the MiniMax-H3 Ref2VA DiT for text-to-video, read off vllm-omni (minimax_h3_transformer.py / minimax_h3_blocks.py / packed_sequence.py) and run on the int8 ConvRot file.
Sequence: [text refined rows | video rows], tags text=1 video=0, ONE timestep t for all rows (text inherits the video timestep), full bidirectional attention, no padding here.
usage: python ref_r2v.py OUTDIR [N_BLOCKS=2] [LT=24] [T=2] [LH=16] [LW=16] [T_VAL=0.3] [SEED=1]   -> OUTDIR/{txt.f32 z.f32 v.f32 x_blocks.f32 meta.json}"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, json, math, numpy as np, torch
sys.path.insert(0, (os.environ.get('H3_W', 'h3_weights')))
from h3lib import ST, Hn, rms, apply_rope96
torch.set_num_threads(16)
W = (os.environ.get('H3R_DIR', 'h3-ref2va') + '/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors')
H64 = torch.from_numpy(np.load((ROOT+'/h3' + '/H64.npy'))).float()
HID, NH, HD, QW, FF = 5376, 56, 128, 7168, 14336
def rotg(x, g): H = Hn if g == 256 else H64; K = x.shape[-1]; return (x.reshape(*x.shape[:-1], K // g, g) @ H).reshape(x.shape)
def qlin(st, base, x, g=256): w = st.t(base + '.weight').float() * st.t(base + '.weight_scale').float(); return rotg(x, g) @ w.T
def blin(st, base, x): y = x @ st.t(base + '.weight').float().T; return y + st.t(base + '.bias').float() if (base + '.bias') in st.h else y
def sdpa(q, k, v): a = torch.softmax(torch.einsum('qhd,khd->hqk', q, k) / HD ** 0.5, -1); return torch.einsum('hqk,khd->qhd', a, v).reshape(q.shape[0], QW)
def swiglu(y): g, u = y.chunk(2, -1); return torch.nn.functional.silu(g) * u        # fc1 = [gate | up]
def refiner(st, i, x):
    p = f'token_refiner.blocks.{i}.'; n = x.shape[0]
    h = rms(x, st.t(p + 'norm1.weight'), 1e-5); qkv = blin(st, p + 'attn.qkv_proj', h); q, k, v = [qkv[:, j * QW:(j + 1) * QW].reshape(n, NH, HD) for j in range(3)]
    q = rms(q, st.t(p + 'attn.q_norm.weight'), 1e-5); k = rms(k, st.t(p + 'attn.k_norm.weight'), 1e-5)
    x = x + blin(st, p + 'attn.out_proj', sdpa(q, k, v))
    h = rms(x, st.t(p + 'norm2.weight'), 1e-5); return x + blin(st, p + 'mlp.fc2', swiglu(blin(st, p + 'mlp.fc1', h)))
def t_embed(st, t):
    half = 128; fr = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half); a = t * fr; tf = torch.cat([a.cos(), a.sin()])
    h = torch.nn.functional.silu(blin(st, 'time_embedder.proj_in', tf[None])); return blin(st, 'time_embedder.proj_out', h)[0]       # [2688]
def positions(lt, T, lh, lw):
    F5 = (1, 4, 4, 4, 4); spans = torch.tensor([5.0 / 3.0 * F5[k % 5] for k in range(T)], dtype=torch.float64); tg = float(lt) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
    sq = math.sqrt(lh * lw)
    def axis(dim): r = dim / sq; left = (1 - r) / 2; return torch.from_numpy(np.linspace(left, left + r, dim // 2, endpoint=False) * 32)
    hg, wg = axis(lh), axis(lw); th, tw = len(hg), len(wg)
    pos = torch.zeros(lt + T * th * tw, 3, dtype=torch.float64); pos[:lt, 0] = torch.arange(lt, dtype=torch.float64)
    v = pos[lt:].view(T, th, tw, 3); v[..., 0] = tg[:, None, None]; v[..., 1] = hg[None, :, None]; v[..., 2] = wg[None, None, :]
    return pos
def rope(pos, inv):
    f = pos.float()[:, :, None] * inv[None, None]; f = torch.cat([f[:, 0], f[:, 1], f[:, 2]], -1); f = torch.cat([f, f], -1); return f.cos(), f.sin()
def run(out, nblk=2, lt=24, T=2, lh=16, lw=16, tval=0.3, seed=1):
    os.makedirs(out, exist_ok=True); torch.manual_seed(seed); st = ST(W)
    txt = torch.randn(lt, 5120); z = torch.randn(24, T, lh, lw); txt.numpy().tofile(f'{out}/txt.f32'); z.numpy().tofile(f'{out}/z.f32')
    with torch.no_grad():
        # text: condition_proj -> token refiner (2 blocks) -> final norm
        x_t = blin(st, 'condition_proj', txt)
        for i in range(2): x_t = refiner(st, i, x_t)
        x_t = rms(x_t, st.t('token_refiner.final_norm.weight'), 1e-5)
        # video rows: patch (c, r, p, q) as in packed_tokens
        th, tw = lh // 2, lw // 2; pz = z.reshape(24, T, 1, th, 2, tw, 2); pz = torch.einsum('ctrhpwq->thwcrpq', pz).reshape(T * th * tw, 96)
        x_v = blin(st, 'video_patch_proj', pz)
        x = torch.cat([x_t, x_v]); n = x.shape[0]; tags = torch.cat([torch.ones(lt, dtype=torch.long), torch.zeros(T * th * tw, dtype=torch.long)])
        cos, sin = rope(positions(lt, T, lh, lw), st.t('rope.inv_freq').float())
        te = t_embed(st, tval); se = torch.nn.functional.silu(te)[None]
        for b in range(nblk):
            p = f'blocks.{b}.'
            m = (qlin(st, p + 'adaln_proj.linear', se, 64) + st.t(p + 'adaln_proj.linear.bias').float()).view(3, 6, HID)[tags]       # [n, 6, H]
            sh1, sc1, g1, sh2, sc2, g2 = [m[:, c] for c in range(6)]
            h = rms(x, st.t(p + 'norm1.weight'), 1e-5) * (1 + sc1) + sh1
            qkv = qlin(st, p + 'attn.qkv_proj', h); q, k, v = [qkv[:, j * QW:(j + 1) * QW].reshape(n, NH, HD) for j in range(3)]
            q = apply_rope96(rms(q, st.t(p + 'attn.q_norm.weight'), 1e-5), cos, sin); k = apply_rope96(rms(k, st.t(p + 'attn.k_norm.weight'), 1e-5), cos, sin)
            x = x + g1 * qlin(st, p + 'attn.out_proj', sdpa(q, k, v))
            h = rms(x, st.t(p + 'norm2.weight'), 1e-5) * (1 + sc2) + sh2
            x = x + g2 * qlin(st, p + 'mlp.fc2', swiglu(qlin(st, p + 'mlp.fc1', h)))
            print('block', b, 'x rms %.3f' % float(x.pow(2).mean().sqrt()), flush=True)
        fm = (blin(st, 'final_layer.adaln_proj.linear', se))[0]; shf, scf = fm[:HID], fm[HID:]
        hf = rms(x, st.t('final_layer.norm.weight'), 1e-5) * (1 + scf) + shf
        v = blin(st, 'final_layer.video_out', hf)[lt:]
    x.numpy().tofile(f'{out}/x_blocks.f32'); v.numpy().tofile(f'{out}/v.f32')
    json.dump({'nblk': nblk, 'lt': lt, 'T': T, 'lh': lh, 'lw': lw, 't': tval, 'seed': seed, 'v_rms': float(v.pow(2).mean().sqrt())}, open(f'{out}/meta.json', 'w'))
    print('v rms %.4f' % float(v.pow(2).mean().sqrt()), 'x rms', float(x.pow(2).mean().sqrt()))
def positions_ref(lt, T, lh, lw, refs):
    """rows [text lt | ref images | video]: text (r,0,0); image k: t = lt + k, own sqrt-area grid; video t origin = lt + n_images then the 5/3 x (1,4,4,4,4) spans"""
    F5 = (1, 4, 4, 4, 4); spans = torch.tensor([5.0 / 3.0 * F5[k % 5] for k in range(T)], dtype=torch.float64); org = float(lt + len(refs)); tg = org + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
    def axis(dim, sq): r = dim / sq; left = (1 - r) / 2; return torch.from_numpy(np.linspace(left, left + r, dim // 2, endpoint=False) * 32)
    parts = [torch.stack([torch.arange(lt, dtype=torch.float64), torch.zeros(lt, dtype=torch.float64), torch.zeros(lt, dtype=torch.float64)], 1)]
    for k, (rh, rw) in enumerate(refs):
        sq = math.sqrt(rh * rw); hg, wg = axis(rh, sq), axis(rw, sq); hh, ww = torch.meshgrid(hg, wg, indexing='ij'); parts.append(torch.stack([torch.full((hh.numel(),), float(lt + k), dtype=torch.float64), hh.reshape(-1), ww.reshape(-1)], 1))
    sq = math.sqrt(lh * lw); hg, wg = axis(lh, sq), axis(lw, sq); hh, ww = torch.meshgrid(hg, wg, indexing='ij'); fr = torch.stack([hh.reshape(-1), ww.reshape(-1)], 1)
    vt = torch.empty(T, fr.shape[0], 3, dtype=torch.float64); vt[:, :, 0] = tg[:, None]; vt[:, :, 1:] = fr[None]; parts.append(vt.reshape(-1, 3)); return torch.cat(parts)
def run_ref(out, ref_files, ref_hw, nblk=50, lt=24, T=2, lh=16, lw=16, tval=0.3, seed=1, tcond=0.999):
    """Ref2VA forward with reference images: ref_files = normalised latents (24,1,rh,rw) .f32, ref_hw = [(rh, rw)]; condition rows carry timestep tcond (clean), text and target rows tval"""
    os.makedirs(out, exist_ok=True); torch.manual_seed(seed); st = ST(W)
    txt = torch.randn(lt, 5120); z = torch.randn(24, T, lh, lw); txt.numpy().tofile(f'{out}/txt.f32'); z.numpy().tofile(f'{out}/z.f32')
    def patch(zz, TT, hh_, ww_): th, tw = hh_ // 2, ww_ // 2; p = zz.reshape(24, TT, 1, th, 2, tw, 2); return torch.einsum('ctrhpwq->thwcrpq', p).reshape(TT * th * tw, 96)
    with torch.no_grad():
        x_t = blin(st, 'condition_proj', txt)
        for i in range(2): x_t = refiner(st, i, x_t)
        x_t = rms(x_t, st.t('token_refiner.final_norm.weight'), 1e-5)
        rrows = []
        for f, (rh, rw) in zip(ref_files, ref_hw): rrows.append(patch(torch.from_numpy(np.fromfile(f, np.float32)).reshape(24, 1, rh, rw), 1, rh, rw))
        x_r = blin(st, 'video_patch_proj', torch.cat(rrows)); x_v = blin(st, 'video_patch_proj', patch(z, T, lh, lw))
        nr, nv = x_r.shape[0], x_v.shape[0]
        x = torch.cat([x_t, x_r, x_v]); n = x.shape[0]
        tags = torch.cat([torch.ones(lt, dtype=torch.long), torch.zeros(nr + nv, dtype=torch.long)]); tsel = torch.cat([torch.zeros(lt, dtype=torch.long), torch.ones(nr, dtype=torch.long), torch.zeros(nv, dtype=torch.long)])   # 0: tval, 1: tcond
        cos, sin = rope(positions_ref(lt, T, lh, lw, ref_hw), st.t('rope.inv_freq').float())
        ses = [torch.nn.functional.silu(t_embed(st, tv))[None] for tv in (tval, tcond)]
        for b in range(nblk):
            p = f'blocks.{b}.'
            ms = [(qlin(st, p + 'adaln_proj.linear', se, 64) + st.t(p + 'adaln_proj.linear.bias').float()).view(3, 6, HID) for se in ses]
            m = torch.stack([ms[0], ms[1]])[tsel, tags]       # [n, 6, H]
            sh1, sc1, g1, sh2, sc2, g2 = [m[:, c] for c in range(6)]
            h = rms(x, st.t(p + 'norm1.weight'), 1e-5) * (1 + sc1) + sh1
            qkv = qlin(st, p + 'attn.qkv_proj', h); q, k, v = [qkv[:, j * QW:(j + 1) * QW].reshape(n, NH, HD) for j in range(3)]
            q = apply_rope96(rms(q, st.t(p + 'attn.q_norm.weight'), 1e-5), cos, sin); k = apply_rope96(rms(k, st.t(p + 'attn.k_norm.weight'), 1e-5), cos, sin)
            x = x + g1 * qlin(st, p + 'attn.out_proj', sdpa(q, k, v))
            h = rms(x, st.t(p + 'norm2.weight'), 1e-5) * (1 + sc2) + sh2
            x = x + g2 * qlin(st, p + 'mlp.fc2', swiglu(qlin(st, p + 'mlp.fc1', h)))
            if b % 10 == 0 or b == nblk - 1: print('block', b, 'x rms %.3f' % float(x.pow(2).mean().sqrt()), flush=True)
        fm = (blin(st, 'final_layer.adaln_proj.linear', ses[0]))[0]; shf, scf = fm[:HID], fm[HID:]
        hf = rms(x, st.t('final_layer.norm.weight'), 1e-5) * (1 + scf) + shf
        v = blin(st, 'final_layer.video_out', hf)[lt + nr:]
    v.numpy().tofile(f'{out}/v.f32'); print('v rms %.4f' % float(v.pow(2).mean().sqrt()))
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--ref':
    a = sys.argv; run_ref(a[2], [a[3]], [(int(a[4]), int(a[5]))], 50, int(a[6]), int(a[7]), int(a[8]), int(a[9]), float(a[10]), int(a[11]))
elif __name__ == '__main__':
    a = sys.argv; run(a[1], *(int(v) for v in a[2:6]), *(int(v) for v in a[6:7]) if False else [], ) if False else run(a[1], int(a[2]) if len(a) > 2 else 2, int(a[3]) if len(a) > 3 else 24, int(a[4]) if len(a) > 4 else 2, int(a[5]) if len(a) > 5 else 16, int(a[6]) if len(a) > 6 else 16, float(a[7]) if len(a) > 7 else 0.3, int(a[8]) if len(a) > 8 else 1)
