"""check ref_gemma.layer() against the real HF Gemma4UnifiedTextDecoderLayer (transformers 5.x, venv tf5env) with the same dequantised weights"""
import sys, math, torch
sys.path.insert(0, '.')
import ref_gemma as R
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer, Gemma4UnifiedTextRotaryEmbedding
cfg = Gemma4UnifiedTextConfig(**{k: v for k, v in R.CFG.items() if k not in ('dtype',)}); cfg._attn_implementation = 'eager'
rope = Gemma4UnifiedTextRotaryEmbedding(cfg)
n = 7; torch.manual_seed(0); ids = torch.randint(10, 200000, (n,))
x = R.st.get_tensor('model.embed_tokens.weight')[ids].float() * math.sqrt(R.HID)
for i in (0, 5, 11):
    L = Gemma4UnifiedTextDecoderLayer(cfg, i).float().eval(); p = f'model.layers.{i}.'
    sd = {}
    def deq(b):
        W = R.st.get_tensor(b + '.weight').float() * R.st.get_tensor(b + '.weight_scale').float()   # W_rot
        return (W.reshape(W.shape[0], -1, 256) @ R.Hn).reshape(W.shape)    # rotate back to the original basis: W = W_rot @ H
    for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        if p + f'self_attn.{k}.weight' in R.st.keys(): sd[f'self_attn.{k}.weight'] = deq(p + f'self_attn.{k}')
    for k in ('gate_proj', 'up_proj', 'down_proj'): sd[f'mlp.{k}.weight'] = deq(p + f'mlp.{k}')
    for k in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm'): sd[k + '.weight'] = R.st.get_tensor(p + k + '.weight').float()
    for k in ('q_norm', 'k_norm'): sd[f'self_attn.{k}.weight'] = R.st.get_tensor(p + f'self_attn.{k}.weight').float()
    sd['layer_scalar'] = R.st.get_tensor(p + 'layer_scalar').float()
    missing = L.load_state_dict(sd, strict=False); print(i, 'missing', [m for m in missing.missing_keys], 'unexpected', missing.unexpected_keys)
    pos = torch.arange(n)[None]; pe = rope(x[None], pos, cfg.layer_types[i])
    mask = torch.full((n, n), float('-inf')).triu(1)[None, None]
    with torch.no_grad(): y_hf = L(x[None], shared_kv_states={}, position_embeddings=pe, attention_mask=mask, position_ids=pos)[0]
    cs = {'sliding_attention': R.rope_cs(n, 256, 1e4), 'full_attention': R.rope_cs(n, 512, 1e6, 0.25)}
    y_me = R.layer(i, x, cs)
    print(f'layer {i} ({cfg.layer_types[i]}): rel diff mine vs HF = {(y_me - y_hf).norm() / y_hf.norm():.3e}, |y| rms {y_hf.pow(2).mean().sqrt():.3f}')
