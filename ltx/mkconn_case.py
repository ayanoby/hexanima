"""One-block test case for the connector kernel (MODE 81): python mkconn_case.py OUTDIR PACK.bin KIND(video|audio) LAYER   (needs /tmp/ltx_conn_KIND.npz from ref_conn.py)
phone: H3MODE=81 FMT=0|1 WTSIZE=bytes klblk DIR 1024 0 1024"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ltxstub, ref_conn
from ltx_core.text_encoders.gemma.embeddings_connector import generate_freq_grid_np, generate_freq_grid_pytorch
from ltx_core.model.transformer.rope import precompute_freqs_cis
out, pack, kind, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); os.makedirs(out, exist_ok=True)
conn, st, cfg, heads, hd = ref_conn.build(kind); conn = conn.float().eval()
d = np.load(f'/tmp/ltx_conn_{kind}.npz'); inp = torch.from_numpy(d['inp']).float(); T, D = inp.shape; LL = 1024
hid = torch.zeros(1, LL, D); hid[0, LL - T:] = inp; mask = torch.zeros(1, LL); mask[0, LL - T:] = 1
add = (mask.to(torch.int64) - 1).float().reshape(1, 1, -1, LL) * torch.finfo(torch.float32).max
with torch.no_grad():
    x, m0 = conn._replace_padded_with_learnable_registers(hid, add)
    idx = torch.arange(LL, dtype=torch.float32)[None, None, :]
    gen = generate_freq_grid_np if conn.double_precision_rope else generate_freq_grid_pytorch
    cos, sin = precompute_freqs_cis(indices_grid=idx, dim=conn.inner_dim, out_dtype=x.dtype, theta=conn.positional_embedding_theta, max_pos=conn.positional_embedding_max_pos,
                                    num_attention_heads=conn.num_attention_heads, rope_type=conn.rope_type, freq_grid_generator=gen)    # [1, H, T, hd/2]
    for i in range(L): x = conn.transformer_1d_blocks[i](x, attention_mask=m0, pe=(cos, sin))
    xin = x.clone(); xo = conn.transformer_1d_blocks[L](xin, attention_mask=m0, pe=(cos, sin))
    from ltx_core.model.transformer.rope import apply_rotary_emb
    from ltx_core.model.transformer.attention import PytorchAttention
    blk = conn.transformer_1d_blocks[L]; a1 = blk.attn1; rn = lambda t: torch.nn.functional.rms_norm(t, (t.shape[-1],), eps=1e-6)
    nrm = rn(xin); q = a1.to_q(nrm); k = a1.to_k(nrm); v = a1.to_v(nrm); qr = apply_rotary_emb(a1.q_norm(q), (cos, sin), a1.rope_type); kr = apply_rotary_emb(a1.k_norm(k), (cos, sin), a1.rope_type)
    att = PytorchAttention()(qr, kr, v, a1.heads, None); gt = 2.0 * torch.sigmoid(a1.to_gate_logits(nrm)); gated = (att.view(1, LL, a1.heads, a1.dim_head) * gt.unsqueeze(-1)).view(1, LL, -1)
    xa = a1.to_out(gated) + xin; mid = torch.nn.functional.gelu(blk.ff.net[0].proj(rn(xa)), approximate='tanh')
    dbg = dict(q=qr[0].numpy() / (a1.dim_head ** 0.5), k=kr[0].numpy(), v=v[0].numpy(), att=att[0].numpy(), gated=gated[0].numpy(), xa=xa[0].numpy(), mid=mid[0].numpy())
ns = LL // 32; hh = hd // 64; H = conn.num_attention_heads
x32 = np.ascontiguousarray(xin[0].numpy().reshape(ns, 32, D // 32, 32).transpose(0, 2, 1, 3)).astype(np.float32); x32.tofile(f'{out}/x32.bin')
def tab(t):   # t [H, T, hd/2] -> [slab][head][tile][32 rows][32]
    a = t.numpy().reshape(H, ns, 32, hh, 32).transpose(1, 0, 3, 2, 4); return np.ascontiguousarray(a).astype(np.float32)
tab(cos[0]).tofile(f'{out}/cos.bin'); tab(sin[0]).tofile(f'{out}/sin.bin')
Hn = (torch.load((ROOT+'/ltx' + '/H256.pt'), weights_only=True).float() / 16.0).numpy()
np.ascontiguousarray(Hn.astype(np.float16).reshape(8, 16, 2, 8, 32).transpose(3, 0, 1, 4, 2)).tofile(f'{out}/hn.bin'); np.zeros(258048 // 4, np.float32).tofile(f'{out}/mod.bin')
if os.path.lexists(f'{out}/wts.bin'): os.remove(f'{out}/wts.bin')
os.symlink(os.path.abspath(pack), f'{out}/wts.bin'); np.savez(f'{out}/ref.npz', xin=xin[0].numpy(), xo=xo[0].numpy(), D=D, **dbg)
print(kind, 'layer', L, 'D', D, 'xin rms', float(xin.pow(2).mean().sqrt()), 'xo rms', float(xo.pow(2).mean().sqrt()), 'cos tbl bytes/slab', H * hh * 4096)
