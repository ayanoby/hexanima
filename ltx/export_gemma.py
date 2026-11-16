"""Pack the Gemma 4 text-encoder layers of the LTX-2.5 int8 ConvRot checkpoint for the phone NPU (MODE 80): lossless (int8 + per-row scale are re-tiled, no requantisation).
usage: python export_gemma.py OUTDIR [--layers 0,5] [--check]    -> gmNN.bin per layer (sliding 0..: 224 MB, full: 190 MB) + glue/ (see GLUE)
Layer file (compact q8 tiles, per n-block [128 B fp16 scale vector][kb x 1024 B offset-binary int8 tiles], fmt 3, K = 3840 unless noted):
  Q [QN x 3840] (QN = 4096 sliding / 8192 full)  K [KN x 3840] (2048 / 512)  V [2048 x 3840] (sliding only)  O [3840 x QN]  GU [30720 x 3840] (n-blocks of 32 interleaved gate_j, up_j)  DN [3840 x 15360]
  gains f32: input_ln[3840] post_attn_ln[3840] pre_ffn_ln[3840] post_ffn_ln[3840] q_norm[HD] k_norm[HD] layer_scalar x 32   (HD = 256 / 512)
No RoPE channel permutation: Gemma rotates (i, i + HD/2) = the engine's rotate-half."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os, sys, numpy as np, torch
sys.path.insert(0, (ROOT+'/h3'))
from export_h3 import tilepack3
from safetensors import safe_open
P = (os.environ.get('LTX_W', 'ltx_weights') + '/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors')
TYPES = ['full_attention' if i % 6 == 5 else 'sliding_attention' for i in range(48)]
st = safe_open(P, 'pt')
def mat(base):   # int8 [N,K] (W_rot rows), f32 scale [N]
    return st.get_tensor(base + '.weight').numpy(), st.get_tensor(base + '.weight_scale').float().numpy().reshape(-1)
def pack(q, s): return tilepack3(np.ascontiguousarray(q.T), s)
def pack_layer(i):
    p = f'model.layers.{i}.'; full = TYPES[i] == 'full_attention'; out = []
    for k in (['q_proj', 'k_proj'] if full else ['q_proj', 'k_proj', 'v_proj']): out.append(pack(*mat(p + 'self_attn.' + k)))
    out.append(pack(*mat(p + 'self_attn.o_proj')))
    (qg, sg), (qu, su) = mat(p + 'mlp.gate_proj'), mat(p + 'mlp.up_proj'); F = qg.shape[0]
    qi = np.stack((qg.reshape(F // 32, 32, -1), qu.reshape(F // 32, 32, -1)), 1).reshape(2 * F, -1); si = np.stack((sg.reshape(F // 32, 32), su.reshape(F // 32, 32)), 1).reshape(-1)
    out.append(pack(qi, si)); out.append(pack(*mat(p + 'mlp.down_proj')))
    g = lambda k: st.get_tensor(p + k).float().numpy()
    gains = [g('input_layernorm.weight'), g('post_attention_layernorm.weight'), g('pre_feedforward_layernorm.weight'), g('post_feedforward_layernorm.weight'), g('self_attn.q_norm.weight'), g('self_attn.k_norm.weight'), np.full(32, float(g('layer_scalar')[0]), np.float32)]
    out.append(np.concatenate(gains).astype(np.float32).tobytes())
    return b''.join(out)
if __name__ == '__main__':
    od = sys.argv[1]; os.makedirs(od, exist_ok=True)
    layers = [int(x) for x in sys.argv[sys.argv.index('--layers') + 1].split(',')] if '--layers' in sys.argv else range(48)
    for i in layers:
        b = pack_layer(i); open(f'{od}/gm{i:02d}.bin', 'wb').write(b); print(i, TYPES[i], len(b), flush=True)
