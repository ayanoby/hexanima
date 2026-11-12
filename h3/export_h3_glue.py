"""Everything of the H3 DiT that is not a block: raw little-endian files in OUTDIR for the phone runner.
usage: python export_h3_glue.py DIT.safetensors OUTDIR     (also writes refiner packs r0.bin r1.bin = pack_block(prefix token_refiner.refiner_blocks.))"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_h3 import ST, pack_block
st = ST(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
def w(name, a): np.ascontiguousarray(a).tofile(f'{out}/{name}')
f = lambda k: st.t(k).astype(np.float32)
w('proj_in_w.f32', f('proj_in.weight')); w('proj_in_b.f32', f('proj_in.bias'))                       # [5376,96], [5376]
w('ctx_q.i8', st.t('context_embedder._weight_qdata')); w('ctx_s.f32', f('context_embedder._weight_scale').reshape(-1)); w('ctx_b.f32', f('context_embedder.bias'))   # [5376,5120] int8 (rotated), [5376], [5376]
w('table.f32', f('time_embedder.table'))                                                               # [1025,8]
w('adaln_w.f32', np.concatenate([f(f'transformer_blocks.{b}.adaln_proj.linear.weight')[:64512] for b in range(50)]))   # [50][64512][8]: kind 0 (image) 6x5376 then kind 1 (text) 6x5376
w('adaln_b.f32', np.concatenate([f(f'transformer_blocks.{b}.adaln_proj.linear.bias')[:64512] for b in range(50)]))
w('nout_w.f32', f('norm_out.linear.weight')); w('nout_b.f32', f('norm_out.linear.bias')); w('nout_norm.f32', f('norm_out.norm.weight'))   # [10752,8], [10752] (shift then scale), [5376]
w('heads_w.f32', np.stack([f(f'pdd_proj_out.{i}.weight') for i in range(6)])); w('heads_b.f32', np.stack([f(f'pdd_proj_out.{i}.bias') for i in range(6)]))   # [6,96,5376], [6,96]
w('rfin.f32', f('token_refiner.final_norm.weight'))
H = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H256.npy')) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H256.npy')) else None
for i in range(2):
    d, _ = pack_block(st, i, 'token_refiner.refiner_blocks.'); open(f'{out}/r{i}.bin', 'wb').write(d)
print('glue done')
