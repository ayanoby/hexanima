"""Glue of the MiniMax-H3 Ref2VA DiT for the phone runner (r2vrun): plain f32 files for the CPU part and the 2 token-refiner blocks as q8 block packs (rN.bin).
The refiner weights are bf16 in the checkpoint (no int8): they are ConvRot-rotated (W.Hn per 256 input columns) and quantised to int8 per output row here, then packed like a DiT block.
usage: python export_r2v_glue.py DIT.safetensors OUTDIR"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_h3 import ST, PERM, K0, QN, FF, tilepack3, hperm
HN = (np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H256.npy')).astype(np.float32))   # symmetric, entries +-1/16 after /16 below if stored as +-1
HN = HN / 16.0   # H256.npy is +-1: Hn = H / 16 (orthonormal, symmetric)
H64 = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H64.npy')).astype(np.float32)
def quant_rot(w):   # w fp32 [N,K] -> int8 [N,K] (rotated), scale [N]
    N, K = w.shape; wr = np.empty_like(w)
    for r0 in range(0, N, 2048): wr[r0:r0 + 2048] = (w[r0:r0 + 2048].reshape(-1, K // 256, 256) @ HN).reshape(-1, K)
    s = np.maximum(np.abs(wr).max(1), 1e-8) / 127.0; q = np.clip(np.rint(wr / s[:, None]), -127, 127).astype(np.int8); return q, s.astype(np.float32)
def pack_refiner(st, i):
    p = f'token_refiner.blocks.{i}.'; out = []
    qkv = st.t(p + 'attn.qkv_proj.weight').astype(np.float32)
    for j, n in enumerate(('q', 'k', 'v')):
        q, s = quant_rot(np.ascontiguousarray(qkv[j * QN:(j + 1) * QN]))
        if n in ('q', 'k'): q, s = hperm(q, s)
        out.append(tilepack3(np.ascontiguousarray(q.T), s))
    q, s = quant_rot(st.t(p + 'attn.out_proj.weight').astype(np.float32)); out.append(tilepack3(np.ascontiguousarray(q.T), s))
    w = st.t(p + 'mlp.fc1.weight').astype(np.float32); q, s = quant_rot(w); gate, up = (q[:FF], s[:FF]), (q[FF:], s[FF:])
    qi = np.stack((gate[0].reshape(FF // 32, 32, K0), up[0].reshape(FF // 32, 32, K0)), 1).reshape(2 * FF, K0); si = np.stack((gate[1].reshape(FF // 32, 32), up[1].reshape(FF // 32, 32)), 1).reshape(-1)
    out.append(tilepack3(np.ascontiguousarray(qi.T), si))
    q, s = quant_rot(st.t(p + 'mlp.fc2.weight').astype(np.float32)); out.append(tilepack3(np.ascontiguousarray(q.T), s))
    qn = st.t(p + 'attn.q_norm.weight').astype(np.float32)[PERM] / np.sqrt(128.0); kn = st.t(p + 'attn.k_norm.weight').astype(np.float32)[PERM]
    g = np.concatenate([qn, kn, st.t(p + 'norm1.weight'), st.t(p + 'norm2.weight')]).astype(np.float32); out.append(g.tobytes() + bytes(16384 * 4 - g.nbytes))
    return b''.join(out)
if __name__ == '__main__':
    st = ST(sys.argv[1]); o = sys.argv[2]; os.makedirs(o, exist_ok=True)
    def f32(name, key, tr=False):
        a = st.t(key).astype(np.float32); (a.T if tr else a).copy().tofile(f'{o}/{name}')
    f32('vpp_w.f32', 'video_patch_proj.weight'); f32('vpp_b.f32', 'video_patch_proj.bias'); f32('cond_w.f32', 'condition_proj.weight'); f32('cond_b.f32', 'condition_proj.bias')
    f32('te_in_w.f32', 'time_embedder.proj_in.weight'); f32('te_in_b.f32', 'time_embedder.proj_in.bias'); f32('te_out_w.f32', 'time_embedder.proj_out.weight'); f32('te_out_b.f32', 'time_embedder.proj_out.bias')
    f32('fin_ada_w.f32', 'final_layer.adaln_proj.linear.weight'); f32('fin_ada_b.f32', 'final_layer.adaln_proj.linear.bias'); f32('fin_norm.f32', 'final_layer.norm.weight')
    f32('vout_w.f32', 'final_layer.video_out.weight'); f32('vout_b.f32', 'final_layer.video_out.bias'); f32('rfin.f32', 'token_refiner.final_norm.weight'); f32('rope_inv.f32', 'rope.inv_freq')
    H64.copy().tofile(f'{o}/H64.f32'); (HN * 16.0).astype(np.float32).tofile(f'{o}/H256.f32')   # H256.f32 = +-1 as in the H3 pack (the runner divides by 16)
    for i in range(2): open(f'{o}/r{i}.bin', 'wb').write(pack_refiner(st, i)); print('refiner', i, flush=True)
    print('glue done')
