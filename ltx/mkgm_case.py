"""One-layer test case for the Gemma layer kernel (MODE 80): python mkgm_case.py OUTDIR LAYER_PACK.bin LAYER [NTOK_MAX]
writes x32.bin cos.bin sin.bin hn.bin mod.bin wts.bin(link) ref.npz (xin, xo, ids); sim: sim.sh BLK=1 SIMARGS="DIR NTOK 0 NVALID 80 FMT" (FMT 1 for a full-attention layer)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, os, math, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ref_gemma as R
from tokenizers import Tokenizer
out, pack, L = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
text = 'A cinematic shot of a red fox running through fresh snow at sunrise, the camera follows low to the ground, soft golden light, steam rising from its breath, shallow depth of field, gentle ambient wind and the crunch of snow under its paws'
tok = Tokenizer.from_str(bytes(R.st.get_tensor('tokenizer_json').numpy()).decode()); ids = tok.encode(text).ids; ids = ids if ids[0] == 2 else [2] + ids
n = len(ids); ntok = (n + 31) // 32 * 32; full = R.TYPES[L] == 'full_attention'; hd = 512 if full else 256
cs = {'sliding_attention': R.rope_cs(n, 256, 1e4), 'full_attention': R.rope_cs(n, 512, 1e6, 0.25)}
x = R.st.get_tensor('model.embed_tokens.weight')[torch.tensor(ids)].float() * math.sqrt(R.HID)
with torch.no_grad():
    for i in range(L): x = R.layer(i, x, cs)
    xin = x.clone(); xo = R.layer(L, xin, cs)
c, s = cs[R.TYPES[L]]; c, s = c[:, :hd // 2], s[:, :hd // 2]
def pad(t, rows): r = torch.zeros(rows, t.shape[1]); r[:t.shape[0]] = t; return r
x32 = lambda t: np.ascontiguousarray(pad(t, ntok).numpy().reshape(ntok // 32, 32, -1, 32).transpose(0, 2, 1, 3)).astype(np.float32)
x32(xin).tofile(f'{out}/x32.bin'); x32(c).tofile(f'{out}/cos.bin'); x32(s).tofile(f'{out}/sin.bin')
np.zeros(258048 // 4, np.float32).tofile(f'{out}/mod.bin')
Hn = (R.Hn).numpy(); np.ascontiguousarray(Hn.astype(np.float16).reshape(8, 16, 2, 8, 32).transpose(3, 0, 1, 4, 2)).tofile(f'{out}/hn.bin')
if os.path.lexists(f'{out}/wts.bin'): os.remove(f'{out}/wts.bin')
os.symlink(os.path.abspath(pack), f'{out}/wts.bin')
np.savez(f'{out}/ref.npz', ntok=ntok, n=n, xo=xo.numpy(), xin=xin.numpy(), ids=np.array(ids), q=R.DBG['q'].numpy(), k=R.DBG['k'].numpy(), v=R.DBG['v'].numpy(), o=R.DBG['o'].numpy())
print('layer', L, R.TYPES[L], 'n', n, 'ntok', ntok, 'xin rms', xin.pow(2).mean().sqrt().item(), 'xo rms', xo.pow(2).mean().sqrt().item())
