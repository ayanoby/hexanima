"""Torch reference of the LTX-2.5 text embeddings connectors (video 4096 / audio 2048) with the real ltx-core module and the dequantised int8 ConvRot weights of the DiT file.
usage: python ref_conn.py [video|audio] -> /tmp/ltx_conn_{video,audio}.npz (inp [T, D] real tokens, out [1024, D])"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, json, math, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ltxstub
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core.model.transformer.rope import LTXRopeType
from safetensors import safe_open
def build(kind):
    P = (os.environ.get('LTX_W', 'ltx_weights') + '/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors')
    st = safe_open(P, 'pt'); cfg = json.loads(st.metadata()['config'])['transformer']
    Hn = torch.load((ROOT+'/ltx' + '/H256.pt'), weights_only=True).float() / 16.0
    pre = f'model.diffusion_model.{kind}_embeddings_connector.'
    heads = cfg['connector_num_attention_heads'] if kind == 'video' else cfg.get('audio_connector_num_attention_heads', 32)
    hd = cfg['connector_attention_head_dim'] if kind == 'video' else cfg.get('audio_connector_attention_head_dim', 64)
    conn = Embeddings1DConnector(attention_head_dim=hd, num_attention_heads=heads, num_layers=cfg['connector_num_layers'], positional_embedding_max_pos=cfg['connector_positional_embedding_max_pos'],
        rope_type=LTXRopeType(cfg['rope_type']), double_precision_rope=cfg['frequencies_precision'] == 'float64', apply_gated_attention=cfg['connector_apply_gated_attention'])
    def deq(b):
        W = st.get_tensor(b + '.weight').float() * st.get_tensor(b + '.weight_scale').float(); return (W.reshape(W.shape[0], -1, 256) @ Hn).reshape(W.shape)
    sd = {}
    for k in conn.state_dict():
        full = pre + k
        if k == 'learnable_registers': sd[k] = st.get_tensor(full).float()
        elif k.endswith('.weight') and (full[:-7] + '.weight_scale') in st.keys(): sd[k] = deq(full[:-7])
        else: sd[k] = st.get_tensor(full).float()
    print(conn.load_state_dict(sd, strict=True)); conn = conn.float().eval()
    return conn, st, cfg, heads, hd

if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'video'
    conn, st, cfg, heads, hd = build(kind)
    ref = np.load('/tmp/ltx_ref.npz'); inp = torch.from_numpy(ref['v' if kind == 'video' else 'a']).float(); T = inp.shape[0]; L = 1024; D = inp.shape[1]
    hid = torch.zeros(1, L, D); hid[0, L - T:] = inp                     # left padded like the tokenizer
    mask = torch.zeros(1, L); mask[0, L - T:] = 1
    add = (mask.to(torch.int64) - 1).float().reshape(1, 1, -1, L) * torch.finfo(torch.float32).max
    with torch.no_grad(): out, m = conn(hid, add)
    np.savez(f'/tmp/ltx_conn_{kind}.npz', inp=inp.numpy(), out=out[0].numpy())
    print(kind, 'out', tuple(out.shape), 'rms', float(out.pow(2).mean().sqrt()), 'mask all zero', bool((m == 0).all()))
