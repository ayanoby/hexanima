"""Torch reference of the LTX-2.5 text encoder (Gemma 4 unified 12B text model, ComfyUI int8 ConvRot weights) on the CPU, layer by layer.
usage: python ref_gemma.py "prompt" [--logits] [--out states.pt]
Spec read from transformers 5.20 gemma4_unified (modeling_gemma4_unified.py) and ltx-core feature_extractor.py:
 embeddings * sqrt(3840); layer: x += post_attn_norm(attn(input_norm(x))); x += post_ffn_norm(mlp(pre_ffn_norm(x))); x *= layer_scalar
 attn: q/k RMSNorm (weight) per head, v RMSNorm (no weight), scaling 1.0, causal; sliding layers: head_dim 256, 16 q / 8 kv heads, rope theta 1e4 rotate_half;
 full layers (every 6th): head_dim 512, 16 q / 1 kv head, v = raw k_proj output (k_eq_v), rope theta 1e6 'proportional' (only the first 25 % of head_dim/2 angles rotate)
 mlp: gelu_tanh(gate) * up -> down; hidden_states[0] = scaled embeddings, [1..47] = layer outputs, [48] = final norm output (HF ties the last one to last_hidden_state)
ConvRot: weight_int8 * scale = W_rot (W . Hn per 256 input block), so y = W_rot @ rot(x), rot = x_block @ Hn (H256.pt / 16)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, json, struct, math, time, torch, numpy as np
V = os.environ.get('V', '')   # debug variants: norot, noscalar, scale
from safetensors import safe_open
from tokenizers import Tokenizer
P = (os.environ.get('LTX_W', 'ltx_weights') + '/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors')
CFG = json.loads(json.loads(open(os.environ.get('LTX_W', 'ltx_weights') + '/hdr_gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.json').read())['__metadata__']['gemma_config'])['text_config']
HID, NL, NH = CFG['hidden_size'], CFG['num_hidden_layers'], CFG['num_attention_heads']; TYPES = CFG['layer_types']
Hn = torch.load((ROOT+'/ltx' + '/H256.pt'), weights_only=True).float() / 16.0
st = safe_open(P, 'pt')
def rot(x):
    if 'norot' in V: return x
    return (x.reshape(*x.shape[:-1], x.shape[-1] // 256, 256) @ Hn).reshape(x.shape)
def lin(base, xr):   # xr already rotated
    return xr @ (st.get_tensor(base + '.weight').float() * st.get_tensor(base + '.weight_scale').float()).T
def rms(x, w=None, eps=1e-6):
    y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5); return y if w is None else y * w.float()
def rotate_half(x): a, b = x.chunk(2, -1); return torch.cat([-b, a], -1)
def rope_cs(n, hd, theta, frac=1.0):
    ang = int(frac * hd // 2); inv = 1.0 / theta ** (torch.arange(0, 2 * ang, 2, dtype=torch.float) / hd)
    inv = torch.cat([inv, torch.zeros(hd // 2 - ang)]); f = torch.arange(n, dtype=torch.float)[:, None] * inv[None]; e = torch.cat([f, f], -1)
    return e.cos(), e.sin()   # [n, hd]
DBG = {}
def layer(i, x, cs):
    p = f'model.layers.{i}.'; full = TYPES[i] == 'full_attention'; hd = 512 if full else 256; nkv = 1 if full else 8; n = x.shape[0]
    g = lambda k: st.get_tensor(p + k).float()
    h = rms(x, g('input_layernorm.weight')); hr = rot(h)
    q = lin(p + 'self_attn.q_proj', hr).view(n, NH, hd); q = rms(q, g('self_attn.q_norm.weight'))
    k0 = lin(p + 'self_attn.k_proj', hr).view(n, nkv, hd); v0 = k0 if full else lin(p + 'self_attn.v_proj', hr).view(n, nkv, hd)
    k = rms(k0, g('self_attn.k_norm.weight')); v = rms(v0)
    cos, sin = cs[TYPES[i]]; cos, sin = cos[:, None], sin[:, None]
    q = q * cos + rotate_half(q) * sin; k = k * cos + rotate_half(k) * sin
    DBG['q'] = q.clone(); DBG['k'] = k.clone(); DBG['v'] = v.clone()
    k = k.repeat_interleave(NH // nkv, 1); v = v.repeat_interleave(NH // nkv, 1)
    a = torch.einsum('nhd,mhd->hnm', q, k) * (hd ** -0.5 if 'scale' in V else 1.0); a = a.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float('-inf')).softmax(-1)
    o = torch.einsum('hnm,mhd->nhd', a, v).reshape(n, NH * hd); DBG['o'] = o.clone(); o = lin(p + 'self_attn.o_proj', rot(o))
    x = x + rms(o, g('post_attention_layernorm.weight'))
    h = rot(rms(x, g('pre_feedforward_layernorm.weight')))
    m = torch.nn.functional.gelu(lin(p + 'mlp.gate_proj', h), approximate='tanh') * lin(p + 'mlp.up_proj', h)
    x = x + rms(lin(p + 'mlp.down_proj', rot(m)), g('post_feedforward_layernorm.weight'))
    return x if 'noscalar' in V else x * g('layer_scalar')
def encode(ids, log=print):
    emb = st.get_slice('model.embed_tokens.weight'); E = torch.stack([emb[int(t):int(t) + 1][0].float() for t in ids]) if False else st.get_tensor('model.embed_tokens.weight')[ids].float()
    x = E * math.sqrt(HID); n = len(ids); cs = {'sliding_attention': rope_cs(n, 256, 1e4), 'full_attention': rope_cs(n, 512, 1e6, 0.25)}
    states = [x]; t0 = time.time()
    for i in range(NL):
        x = layer(i, x, cs); states.append(x)
        if i % 8 == 7: log(f'layer {i} done {time.time() - t0:.0f}s rms {x.pow(2).mean().sqrt():.1f}')
    states[-1] = rms(x, st.get_tensor('model.norm.weight').float()); return states
if __name__ == '__main__':
    tok = Tokenizer.from_str(bytes(st.get_tensor('tokenizer_json').numpy()).decode())
    text = sys.argv[1]; ids = tok.encode(text).ids; ids = ids if ids[0] == 2 else [2] + ids; print('ids', ids)
    S = encode(ids)
    if '--logits' in sys.argv:
        Wt = st.get_tensor('model.embed_tokens.weight'); h = S[-1][-1].float()
        lg = torch.cat([(Wt[i:i + 32768].float() @ h) for i in range(0, Wt.shape[0], 32768)]); lg = 30 * torch.tanh(lg / 30)
        top = lg.topk(8); print('next token:', [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
    if '--out' in sys.argv: torch.save({'ids': ids, 'states': torch.stack(S)}, sys.argv[sys.argv.index('--out') + 1])
