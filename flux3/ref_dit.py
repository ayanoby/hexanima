# reference FLUX.3 Klein (distilled, flow map) on CPU, bf16, weights streamed block by block from the diffusers files
import sys, json, time, torch
sys.path.insert(0, 'ref/src')
from safetensors import safe_open
from flux3.dit import DiT, Packed, Block, IN_CHANNELS, MODES
from flux3.sampling import timesteps
from flux3.autoencoder import AutoEncoder, latent_ids, latent_shape
from flux3.checkpoint import load_checkpoint
from flux3.pipeline import scaled_ids, to_pil
import os
Q8 = os.environ.get('Q8') == '1'
torch.set_num_threads(8)
D = 'klein/transformer/'
idx = json.load(open(D + 'diffusion_pytorch_model.safetensors.index.json'))['weight_map']
files = {}
def T(k):
    f = idx[k]
    if f not in files: files[f] = safe_open(D + f, 'pt')
    return files[f].get_tensor(k)
def blk_map(prefix):  # our Block param -> diffusers key
    return {'q_proj.weight': prefix + '.attn.to_q.weight', 'k_proj.weight': prefix + '.attn.to_k.weight', 'v_proj.weight': prefix + '.attn.to_v.weight',
            'attn_out.weight': prefix + '.attn.to_out.0.weight', 'mlp_in.weight': prefix + '.ff.linear_in.weight', 'mlp_out.weight': prefix + '.ff.linear_out.weight',
            'norm.query_norm.scale': prefix + '.attn.norm_q.weight', 'norm.key_norm.scale': prefix + '.attn.norm_k.weight'}
scratch = Block(3072, 24, 3.0).to(torch.bfloat16)
class SB(torch.nn.Module):
    def __init__(s, prefix): super().__init__(); s.m = blk_map(prefix)
    def forward(s, x, freqs, bounds, mod):
        sd = dict(scratch.named_parameters())
        with torch.no_grad():
            for n, k in s.m.items():
                w = T(k)
                if Q8 and n.endswith('proj.weight') or Q8 and n in ('attn_out.weight','mlp_in.weight','mlp_out.weight'):
                    f = w.float(); sc = f.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
                    w = (torch.round(f / sc).clamp(-127, 127) * sc).to(torch.bfloat16)
                sd[n].copy_(w)
        return scratch(x, freqs, bounds, mod)
with torch.device('meta'): dit = DiT(True)
for m in ('image', 'image_cond'):
    for d in range(5): dit.content_mode_blocks[m][d] = SB(f'mode_blocks.{m}.{d}')
for d in range(5): dit.txt_mode_blocks[d] = SB(f'txt_mode_blocks.{d}')
for d in range(28): dit.single_blocks[d] = SB(f'single_transformer_blocks.{d}')
small = {'emb_in.image.weight': 'x_embedders.image.weight', 'emb_in.image_cond.weight': 'x_embedders.image_cond.weight', 'txt_in.weight': 'context_embedder.weight',
         'time_in.in_layer.weight': 'timestep_embedder.linear_1.weight', 'time_in.out_layer.weight': 'timestep_embedder.linear_2.weight',
         'ref_time_in.in_layer.weight': 'target_timestep_embedder.linear_1.weight', 'ref_time_in.out_layer.weight': 'target_timestep_embedder.linear_2.weight',
         'final_layer.image.linear.weight': 'proj_out.weight', 'final_layer.image.adaLN_modulation.1.weight': 'norm_out.linear.weight'}
for e in ('early', 'single'):
    for k in ('image', 'image_cond', 'txt'): small[f'{e}_stream_modulations.{k}.lin.weight'] = f'{e}_stream_modulations.{k}.linear.weight'
sd = {n: T(k).to(torch.bfloat16) for n, k in small.items()}
w = sd['final_layer.image.adaLN_modulation.1.weight']; h = w.shape[0] // 2
sd['final_layer.image.adaLN_modulation.1.weight'] = torch.cat([w[h:], w[:h]])  # diffusers stores (scale, shift), the reference wants (shift, scale)
for n in list(dit.state_dict()):
    if n not in sd and not n.startswith(('content_mode_blocks', 'txt_mode_blocks', 'single_blocks')): print('MISSING', n)
dit.load_state_dict(sd, strict=False, assign=True); dit.eval()

W = H = int(sys.argv[1]); steps = int(sys.argv[2]); alpha = float(sys.argv[3]); seed = int(sys.argv[4]); outp = sys.argv[5]
te = torch.load('out/te.pt'); tokens = te['tokens']
if os.environ.get('NOPAD'): tokens = tokens[:te['real']]
L = tokens.shape[0]; print('text tokens', L, flush=True)
h, w = latent_shape(H, W)
noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).float()
ids = latent_ids(h, w); iid = scaled_ids(ids)
tid = torch.stack([torch.zeros(L)] * 3 + [torch.arange(L).float()], -1)
def mk(x):
    return Packed(data={'image': x.to(torch.bfloat16), 'image_cond': torch.zeros(0, 128, dtype=torch.bfloat16), 'text': tokens},
                  ids={'image': iid, 'image_cond': torch.zeros(0, 4), 'text': tid},
                  timesteps={'image': torch.zeros(h * w, dtype=torch.bfloat16), 'image_cond': torch.zeros(0, dtype=torch.bfloat16), 'text': torch.zeros(L, dtype=torch.bfloat16)},
                  mode_indices={'image': [(0, h * w)], 'image_cond': [(0, 0)], 'text': [(0, L)]})
sch = timesteps(steps, alpha); x = noise.clone(); t0 = time.time()
torch.save({'noise': noise, 'ids': ids}, 'out/noise.pt')
with torch.inference_mode():
    for s in range(steps):
        t, dt = sch[s].item(), (sch[s + 1] - sch[s]).item()
        p = mk(x); p.timesteps['image'].fill_(t)
        v = dit(p, timesteps_ref=sch[s + 1].reshape(1))
        x = x + dt * v.float()
        print(f'step {s} t={t:.4f} |v|={v.float().std():.3f} |x|={x.std():.3f}  {time.time()-t0:.0f}s', flush=True)
torch.save({'latent': x}, outp + '.lat.pt')
ae = load_checkpoint(AutoEncoder(), 'klein/ae.safetensors', 'cpu', torch.float32)
to_pil(ae.decode(x, ids)).save(outp); print('wrote', outp)
