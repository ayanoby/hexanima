"""Pack the LTX-2.5 text-embeddings connector blocks (video D=4096 / audio D=2048, 8 blocks each) for the phone NPU (MODE 81, M_FMT 0 video / 1 audio): python export_conn.py OUTDIR [--kinds video,audio] [--layers 0,1]
Layer file: compact q8 tiles (fmt 3) Q, K, V, O [D x D], FF1 [4D x D], FF2 [D x 4D], GATE [64 x D] (rows 0..31 = to_gate_logits per-row int8 WITHOUT ConvRot, rows 32..63 zero; its input is the un-rotated rms(x)),
then f32 q gain (q_norm / sqrt(head_dim)) [D], f32 k gain [D], then fp16 bias vectors (one 128 B vector per 32 columns, each column twice): bq bk bv bo [D/32 vectors], bg (2 vectors), bf1 [4D/32], bf2 [D/32]."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os, sys, numpy as np, torch
sys.path.insert(0, (ROOT+'/h3'))
from export_h3 import tilepack3
from safetensors import safe_open
P = (os.environ.get('LTX_W', 'ltx_weights') + '/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors')
st = safe_open(P, 'pt')
def mat(b): return st.get_tensor(b + '.weight').numpy(), st.get_tensor(b + '.weight_scale').float().numpy().reshape(-1)
def pack(q, s): return tilepack3(np.ascontiguousarray(q.T), s)
def bvec(b, pad=0):
    b = b.astype(np.float16)
    if pad: b = np.concatenate([b, np.zeros(pad, np.float16)])
    return np.repeat(b, 2).tobytes()
def pack_layer(kind, i):
    pre = f'model.diffusion_model.{kind}_embeddings_connector.transformer_1d_blocks.{i}.'; hd = 128 if kind == 'video' else 64; out = []
    for k in ('attn1.to_q', 'attn1.to_k', 'attn1.to_v', 'attn1.to_out.0', 'ff.net.0.proj', 'ff.net.2'): out.append(pack(*mat(pre + k)))
    gw = st.get_tensor(pre + 'attn1.to_gate_logits.weight').float().numpy(); gw = np.concatenate([gw, np.zeros_like(gw)], 0)   # [64, D]
    s = np.maximum(np.abs(gw).max(1), 1e-8) / 127; out.append(pack(np.round(gw / s[:, None]).clip(-127, 127).astype(np.int8), s.astype(np.float32)))
    g = lambda k: st.get_tensor(pre + k).float().numpy()
    out.append((g('attn1.q_norm.weight') / np.sqrt(hd)).astype(np.float32).tobytes()); out.append(g('attn1.k_norm.weight').astype(np.float32).tobytes())
    for k in ('attn1.to_q', 'attn1.to_k', 'attn1.to_v', 'attn1.to_out.0'): out.append(bvec(g(k + '.bias')))
    out.append(bvec(g('attn1.to_gate_logits.bias'), 32)); out.append(bvec(g('ff.net.0.proj.bias'))); out.append(bvec(g('ff.net.2.bias')))
    return b''.join(out)
if __name__ == '__main__':
    od = sys.argv[1]; os.makedirs(od, exist_ok=True)
    kinds = sys.argv[sys.argv.index('--kinds') + 1].split(',') if '--kinds' in sys.argv else ['video', 'audio']
    layers = [int(x) for x in sys.argv[sys.argv.index('--layers') + 1].split(',')] if '--layers' in sys.argv else range(8)
    for kd in kinds:
        for i in layers: b = pack_layer(kd, i); open(f'{od}/{kd[0]}c{i}.bin', 'wb').write(b); print(kd, i, len(b), flush=True)
