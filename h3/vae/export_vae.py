"""Pack the MiniMax-H3 video VAE decoder (ViT, 36 blocks, D 2048) for the phone NPU (MODE 82): python export_vae.py OUTDIR [first last]
Source: Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_int8_convrot.safetensors (int8 ConvRot group 256, per-row f32 scale).
vbNN.bin = Q K V O [2048 x 2048], W1 [16384 x 2048] (n-blocks interleaved gate_j, up_j), W2 [2048 x 8192] (tilepack3), f32 g1 g2 s1 s2 [2048 each], gq (1/8 x64) gk (1 x64), fp16 bias vectors bq bk bv bo bf1(interleaved) bf2.
q/k head dims are permuted so that the kernel's split-half rope over 64 dims equals the model's rope over the first 48 dims (pairs (i, i+24)): new = old[0:24] | old[48:56] | old[24:48] | old[56:64].
glue/: xe_w.f32 [2048][24], xe_b, reg.f32 [4][2048], no_w, no_b, po_w.f16 [3072][2048], po_b.f32"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import os, sys, numpy as np
sys.path.insert(0, (ROOT+'/h3'))
from export_h3 import tilepack3
from safetensors import safe_open
P = (os.environ.get('H3R_DIR', 'h3-ref2va') + '/vae_dl/vae/minimax_h3_video_vae_int8_convrot.safetensors')
PERM = np.concatenate([np.arange(0, 24), np.arange(48, 56), np.arange(24, 48), np.arange(56, 64)])
def bvec(b): return np.repeat(b.astype(np.float16), 2).tobytes()
def pack_block(st, i):
    p = f'decoder.transformer_blocks.{i}.'; g = lambda k: st.get_tensor(p + k)
    def mat(b): return g(b + '.weight').numpy(), g(b + '.weight_scale').float().numpy().reshape(-1)
    qkv, qs = mat('attn.to_qkv'); qb = g('attn.to_qkv.bias').float().numpy()
    qkv = qkv.reshape(32, 3, 64, 2048); qs = qs.reshape(32, 3, 64); qb = qb.reshape(32, 3, 64)
    out = []; biases = []
    for j in range(3):
        m, s, b = qkv[:, j], qs[:, j], qb[:, j]
        if j < 2: m, s, b = m[:, PERM], s[:, PERM], b[:, PERM]
        out.append(tilepack3(np.ascontiguousarray(m.reshape(2048, 2048).T), s.reshape(-1))); biases.append(b.reshape(-1))
    m, s = mat('attn.to_out'); out.append(tilepack3(np.ascontiguousarray(m.T), s)); biases.append(g('attn.to_out.bias').float().numpy())
    m, s = mat('ff.w1'); b = g('ff.w1.bias').float().numpy(); F = 8192
    mi = np.stack((m[:F].reshape(F // 32, 32, 2048), m[F:].reshape(F // 32, 32, 2048)), 1).reshape(2 * F, 2048)
    si = np.stack((s[:F].reshape(F // 32, 32), s[F:].reshape(F // 32, 32)), 1).reshape(-1); bi = np.stack((b[:F].reshape(F // 32, 32), b[F:].reshape(F // 32, 32)), 1).reshape(-1)
    out.append(tilepack3(np.ascontiguousarray(mi.T), si)); biases.append(bi)   # order of the bias list below: bq bk bv bo bf1 bf2
    m, s = mat('ff.w2'); out.append(tilepack3(np.ascontiguousarray(m.T), s)); biases.append(g('ff.w2.bias').float().numpy())
    vec = np.concatenate([g('norm1.weight').float().numpy(), g('norm2.weight').float().numpy(), g('scale1').float().numpy(), g('scale2').float().numpy(), np.full(64, 0.125, np.float32), np.ones(64, np.float32)]).astype(np.float32)
    out.append(vec.tobytes()); out += [bvec(b) for b in biases]
    return b''.join(out)
def glue(st, od):
    os.makedirs(od, exist_ok=True); g = lambda k: st.get_tensor('decoder.' + k).float().numpy()
    Wp = st.get_tensor('post_quant_conv.weight').float().numpy().reshape(24, 24); bp = st.get_tensor('post_quant_conv.bias').float().numpy(); We, be = g('x_embedder.weight'), g('x_embedder.bias')
    (We @ Wp).astype(np.float32).tofile(f'{od}/xe_w.f32'); (We @ bp + be).astype(np.float32).tofile(f'{od}/xe_b.f32');   # post_quant_conv folded into the embedder g('register_tokens').reshape(4, 2048).tofile(f'{od}/reg.f32')
    g('norm_out.weight').tofile(f'{od}/no_w.f32'); g('norm_out.bias').tofile(f'{od}/no_b.f32'); g('proj_out.weight').astype(np.float16).tofile(f'{od}/po_w.f16'); g('proj_out.bias').tofile(f'{od}/po_b.f32')
    import json, torch; md = json.loads(json.loads(st.metadata()['minimax_h3_video_vae'])['source_config'] and st.metadata()['minimax_h3_video_vae']); np.array(md['latents_mean'], np.float32).tofile(f'{od}/lat_mean.f32'); np.array(md['latents_std'], np.float32).tofile(f'{od}/lat_std.f32')
    Hn = (torch.load((ROOT+'/ltx' + '/H256.pt'), weights_only=True).float() / 16.0).numpy()
    np.ascontiguousarray(Hn.astype(np.float16).reshape(8, 16, 2, 8, 32).transpose(3, 0, 1, 4, 2)).tofile(f'{od}/hn.bin')
if __name__ == '__main__':
    st = safe_open(P, 'pt'); od = sys.argv[1]; os.makedirs(od, exist_ok=True)
    a = int(sys.argv[2]) if len(sys.argv) > 2 else 0; b = int(sys.argv[3]) if len(sys.argv) > 3 else 35
    if a == 0: glue(st, od + '/glue')
    for i in range(a, b + 1): d = pack_block(st, i); open(f'{od}/vb{i:02d}.bin', 'wb').write(d); print(i, len(d), flush=True)
