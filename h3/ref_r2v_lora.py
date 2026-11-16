"""ref_r2v.py with the Ref2VA turbo LoRA (comfy format) added to every quantised / refiner linear: python ref_r2v_lora.py OUTDIR [args of ref_r2v]"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, torch, ref_r2v as R
from safetensors import safe_open
LF = safe_open((os.environ.get('H3R_DIR', 'h3-ref2va') + '/lora_dl/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'), 'pt')
def delta(base, x):
    k = f'diffusion_model.{base}'
    if k + '.lora_A.weight' not in LF.keys(): return 0
    A, B = LF.get_tensor(k + '.lora_A.weight').float(), LF.get_tensor(k + '.lora_B.weight').float(); sc = float(LF.get_tensor(k + '.alpha')) / A.shape[0]
    if base.endswith('qkv_proj'): r = 128; return sc * torch.cat([(x @ A[j*r:(j+1)*r].T) @ B[j*7168:(j+1)*7168, j*r:(j+1)*r].T for j in range(3)], -1)
    return sc * (x @ A.T) @ B.T
_q, _b = R.qlin, R.blin
R.qlin = lambda st, base, x, g=256: _q(st, base, x, g) + delta(base, x)
R.blin = lambda st, base, x: _b(st, base, x) + (delta(base, x) if base.startswith('token_refiner.blocks') and 'norm' not in base else 0)
a = sys.argv; R.run(a[1], *(int(v) for v in a[2:7]), float(a[7]), int(a[8]))
