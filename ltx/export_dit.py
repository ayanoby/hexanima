"""Pack the LTX-2.5 DiT blocks (VIDEO stream only: attn1 self, attn2 text cross, ff; the audio parts are not exported) for the phone NPU (MODE 90 / 92): python export_dit.py OUTDIR [--layers 0,1]
Block file: [attn1 pack][attn2 pack][ff pack]. Attention pack (compact q8 tiles fmt 3, see blk.c MODE 90): Q [I x Dq] K [I x Dk] V [I x Dk] O [Dq x I] GATE [64 x Dq] (rows 0..31 real, int8 per row WITHOUT ConvRot), f32 q gain (q_norm / sqrt(head_dim)) [I], f32 k gain [I],
fp16 bias vectors bq bk bv [I/32] bo [Dq/32] bg [2] (one 128 B vector per 32 columns, each column twice). FF pack: W1 [16384 x 4096] W2 [4096 x 16384], no biases."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os, sys, numpy as np
sys.path.insert(0, (ROOT+'/ltx'))
import export_conn as C
st = C.st
def pack_attn(pre, hd=128):
    out = []
    for k in ('to_q', 'to_k', 'to_v', 'to_out.0'): out.append(C.pack(*C.mat(pre + k)))
    gw = st.get_tensor(pre + 'to_gate_logits.weight').float().numpy(); gw = np.concatenate([gw, np.zeros_like(gw)], 0)
    s = np.maximum(np.abs(gw).max(1), 1e-8) / 127; out.append(C.pack(np.round(gw / s[:, None]).clip(-127, 127).astype(np.int8), s.astype(np.float32)))
    g = lambda k: st.get_tensor(pre + k).float().numpy()
    out.append((g('q_norm.weight') / np.sqrt(hd)).astype(np.float32).tobytes()); out.append(g('k_norm.weight').astype(np.float32).tobytes())
    for k in ('to_q', 'to_k', 'to_v', 'to_out.0'): out.append(C.bvec(g(k + '.bias')))
    out.append(C.bvec(g('to_gate_logits.bias'), 32)); return b''.join(out)
def pack_block(i):
    p = f'model.diffusion_model.transformer_blocks.{i}.'
    return pack_attn(p + 'attn1.') + pack_attn(p + 'attn2.') + C.pack(*C.mat(p + 'ff.net.0.proj')) + C.pack(*C.mat(p + 'ff.net.2'))
if __name__ == '__main__':
    od = sys.argv[1]; os.makedirs(od, exist_ok=True); layers = [int(x) for x in sys.argv[sys.argv.index('--layers') + 1].split(',')] if '--layers' in sys.argv else range(48)
    for i in layers: b = pack_block(i); open(f'{od}/vb{i:02d}.bin', 'wb').write(b); print(i, len(b), flush=True)
