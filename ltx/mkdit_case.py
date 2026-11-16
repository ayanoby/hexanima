"""One DiT block test case for ltxrun `dittest` (video only): python mkdit_case.py OUTDIR BLOCK_PACK.bin BLOCK  (needs /tmp/ltx_dit_ref.npz from ref_dit.py)
files: x0.f32 [Nv][4096] (input of the block), ctx.f32 [1024][4096], cos.bin sin.bin (rope tables [slab][head][tile][32][32] f32), mods.f32 (12 vectors of 4096: M1 N1 g1 M2 N2 g2 M3 N3 g3 shift_kv scale_kv, in that order, M = 1 + scale), blk.bin (link), ref.f32 (expected output rows)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ref_dit as R
from ltx_core.model.transformer.rope import precompute_freqs_cis, generate_freq_grid_np, LTXRopeType
out, pack, b = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
d = np.load('/tmp/ltx_dit_ref.npz'); Fl, Hl, Wl = [int(v) for v in d['grid']]; Nv = Fl * Hl * Wl; ns = Nv // 32
d['xs'][b].astype(np.float32).tofile(f'{out}/x0.f32'); d['ctx'].astype(np.float32).tofile(f'{out}/ctx.f32'); d['xs'][b + 1].astype(np.float32).tofile(f'{out}/ref.f32')
pos = R.positions(Fl, Hl, Wl)
cos, sin = precompute_freqs_cis(pos, dim=4096, out_dtype=torch.float32, theta=R.cfg['positional_embedding_theta'], max_pos=R.cfg['positional_embedding_max_pos'], use_middle_indices_grid=True,
                                num_attention_heads=32, rope_type=LTXRopeType(R.cfg['rope_type']), freq_grid_generator=generate_freq_grid_np)   # [1, 32, T, 64]
def tab(t): return np.ascontiguousarray(t[0].numpy().reshape(32, ns, 32, 2, 32).transpose(1, 0, 3, 2, 4)).astype(np.float32)
tab(cos).tofile(f'{out}/cos.bin'); tab(sin).tofile(f'{out}/sin.bin')
sst = R.T(f'transformer_blocks.{b}.scale_shift_table'); psst = R.T(f'transformer_blocks.{b}.prompt_scale_shift_table')
v = sst + torch.from_numpy(d['ts']).view(9, 4096); pv = psst + torch.from_numpy(d['pts']).view(2, 4096)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_q, scale_q, gate_q = [v[i] for i in range(9)]; shift_kv, scale_kv = pv[0], pv[1]
mods = torch.stack([1 + scale_msa, shift_msa, gate_msa, 1 + scale_q, shift_q, gate_q, 1 + scale_mlp, shift_mlp, gate_mlp, shift_kv, scale_kv]); mods.numpy().astype(np.float32).tofile(f'{out}/mods.f32')
if os.path.lexists(f'{out}/blk.bin'): os.remove(f'{out}/blk.bin')
os.symlink(os.path.abspath(pack), f'{out}/blk.bin'); print('tokens', Nv, 'block', b, 'x rms', float(np.sqrt((d['xs'][b] ** 2).mean())), 'out rms', float(np.sqrt((d['xs'][b + 1] ** 2).mean())))
