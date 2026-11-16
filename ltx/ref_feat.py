"""Reference of the LTX text features for a prompt: Gemma states -> per-token RMS norm -> concat -> video/audio aggregate embeds. python ref_feat.py OUT_PREFIX ["prompt"]  -> OUT_PREFIX.i32 (ids), OUT_PREFIX.npz (states, v, a)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, math, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ref_gemma as R
from tokenizers import Tokenizer
out = sys.argv[1]
text = sys.argv[2] if len(sys.argv) > 2 else 'A cinematic shot of a red fox running through fresh snow at sunrise, the camera follows low to the ground, soft golden light, steam rising from its breath, shallow depth of field, gentle ambient wind and the crunch of snow under its paws'
tok = Tokenizer.from_str(bytes(R.st.get_tensor('tokenizer_json').numpy()).decode()); ids = tok.encode(text).ids; ids = ids if ids[0] == 2 else [2] + ids
np.array(ids, dtype=np.int32).tofile(out + '.i32')
with torch.no_grad():
    S = torch.stack(R.encode(ids)); T = len(ids)                       # [49, T, 3840]
    normed = (S * torch.rsqrt(S.pow(2).mean(-1, keepdim=True) + 1e-6)).permute(1, 2, 0).reshape(T, -1)   # [T, D*L] with the layer axis fastest (torch.stack(hidden_states, dim=-1) then reshape): index = d * 49 + layer
    res = {}
    for nm, key, od in (('v', 'video', 4096), ('a', 'audio', 2048)):
        W = R.st.get_tensor(f'text_embedding_projection.{key}_aggregate_embed.weight'); b = R.st.get_tensor(f'text_embedding_projection.{key}_aggregate_embed.bias').float()
        y = torch.cat([normed @ W[i:i + 512].float().T for i in range(0, od, 512)], 1) * math.sqrt(od / 3840) + b; res[nm] = y.numpy()
np.savez(out + '.npz', states=S.numpy(), v=res['v'], a=res['a'], ids=np.array(ids))
print('T', T, 'v rms', float(np.sqrt((res['v'] ** 2).mean())), 'a rms', float(np.sqrt((res['a'] ** 2).mean())))
