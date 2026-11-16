"""Torch reference of the LTX-2.5 DiT, VIDEO ONLY (audio=None), real ltx-core block code with the dequantised int8 ConvRot weights, streamed block by block.
usage: python ref_dit.py FRAMES H W [SIGMA] [NBLOCKS] -> /tmp/ltx_dit_ref.npz (z, ctx, per-block x after blocks 0,1,2,..,N-1 (every block), final velocity)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, math, json, time, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ltxstub
import types
for name in ('ltx_core.guidance',):
    import os; m = types.ModuleType(name); m.__path__ = [os.path.join((os.environ.get('LTX_C', 'ltxc')), *name.split('.'))]; sys.modules[name] = m
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.rope import precompute_freqs_cis, generate_freq_grid_np, LTXRopeType
from safetensors import safe_open
P = (os.environ.get('LTX_W', 'ltx_weights') + '/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors')
st = safe_open(P, 'pt'); cfg = json.loads(st.metadata()['config'])['transformer']
Hn = torch.load((ROOT+'/ltx' + '/H256.pt'), weights_only=True).float() / 16.0
PRE = 'model.diffusion_model.'
def deq(b):
    W = st.get_tensor(b + '.weight').float() * st.get_tensor(b + '.weight_scale').float(); return (W.reshape(W.shape[0], -1, 256) @ Hn).reshape(W.shape)
def T(k): return st.get_tensor(PRE + k).float()
def make_block(i):
    vc = TransformerConfig(dim=4096, heads=32, d_head=128, context_dim=4096, apply_gated_attention=True, cross_attention_adaln=True)
    b = BasicAVTransformerBlock(idx=i, video=vc, audio=None, rope_type=LTXRopeType(cfg['rope_type']), norm_eps=1e-6)
    sd = {}; p = f'transformer_blocks.{i}.'
    for k in b.state_dict():
        full = PRE + p + k
        if k.endswith('.weight') and full[:-7] + '.weight_scale' in st.keys(): sd[k] = deq(PRE + p + k[:-7])
        elif full in st.keys(): sd[k] = st.get_tensor(full).float()
        elif k.startswith('ff.') and k.endswith('.bias'): sd[k] = torch.zeros(b.state_dict()[k].shape)   # ff_bias = false in the checkpoint config
        else: raise KeyError(full)
    b.load_state_dict(sd, strict=True); return b.float().eval()
def positions(F, H, W, fps=24.0):
    g = torch.stack(torch.meshgrid(torch.arange(F), torch.arange(H), torch.arange(W), indexing='ij'), 0)      # [3, F, H, W] starts
    lc = torch.stack((g, g + 1), -1).reshape(3, -1, 2)[None].float()                                        # [1, 3, T, 2]
    sc = torch.tensor([8, 32, 32]).view(1, 3, 1, 1).float(); pc = lc * sc
    pc[:, 0] = (pc[:, 0] + 1 - 8).clamp(min=0); pc[:, 0] = pc[:, 0] / fps; return pc
def adaln(prefix, coef, t):   # AdaLayerNormSingle with the real class
    a = AdaLayerNormSingle(4096, embedding_coefficient=coef)
    sd = {}
    for k in a.state_dict(): sd[k] = T(prefix + '.' + k)
    a.load_state_dict(sd); a = a.float().eval()
    with torch.no_grad(): return a(t, hidden_dtype=torch.float32)
if __name__ == '__main__':
    F, Hh, Ww = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0; NB = int(sys.argv[5]) if len(sys.argv) > 5 else 48
    Fl, Hl, Wl = (F - 1) // 8 + 1, Hh // 32, Ww // 32; Nv = Fl * Hl * Wl; print('latent grid', Fl, Hl, Wl, 'tokens', Nv)
    g = torch.Generator().manual_seed(1); z = torch.randn(1, Nv, 128, generator=g)
    ctx = torch.from_numpy(np.load('/tmp/ltx_conn_video.npz')['out']).float()[None]                           # [1, 1024, 4096] video connector output
    pos = positions(Fl, Hl, Wl)
    x = z @ T('patchify_proj.weight').T + T('patchify_proj.bias')
    ts, emb_ts = adaln('adaln_single', 9, torch.tensor([sigma * 1000.0])); ts = ts.view(1, -1, ts.shape[-1]); emb_ts = emb_ts.view(1, -1, emb_ts.shape[-1])
    pts, _ = adaln('prompt_adaln_single', 2, torch.tensor([sigma * 1000.0])); pts = pts.view(1, -1, pts.shape[-1])
    pe = precompute_freqs_cis(pos, dim=4096, out_dtype=torch.float32, theta=cfg['positional_embedding_theta'], max_pos=cfg['positional_embedding_max_pos'], use_middle_indices_grid=True,
                              num_attention_heads=32, rope_type=LTXRopeType(cfg['rope_type']), freq_grid_generator=generate_freq_grid_np)
    args = TransformerArgs(x=x, context=ctx, context_mask=None, timesteps=ts, embedded_timestep=emb_ts, positional_embeddings=pe, cross_positional_embeddings=None, cross_scale_shift_timestep=None,
                           cross_gate_timestep=None, enabled=True, prompt_timestep=pts, self_attention_mask=None)
    outs = [x[0].numpy()]; t0 = time.time()
    with torch.no_grad():
        for i in range(NB):
            blk = make_block(i); args, _ = blk(video=args, audio=None); outs.append(args.x[0].numpy())
            if i % 8 == 7: print('block', i, '%.0fs' % (time.time() - t0), 'rms', float(args.x.pow(2).mean().sqrt()), flush=True)
        sst = T('scale_shift_table'); sv = sst[None, None] + emb_ts[:, :, None]; shift, scale = sv[:, :, 0], sv[:, :, 1]
        h = torch.nn.functional.layer_norm(args.x, (4096,), eps=1e-6) * (1 + scale) + shift; vel = h @ T('proj_out.weight').T + T('proj_out.bias')
    np.savez('/tmp/ltx_dit_ref.npz', z=z[0].numpy(), ctx=ctx[0].numpy(), xs=np.stack(outs), vel=vel[0].numpy(), sigma=sigma, grid=np.array([Fl, Hl, Wl]), ts=ts[0, 0].numpy(), pts=pts[0, 0].numpy(), emb=emb_ts[0, 0].numpy())
    print('velocity rms', float(vel.pow(2).mean().sqrt()), 'out', tuple(vel.shape))
