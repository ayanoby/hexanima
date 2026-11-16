"""MiniMax-H3 Ref2VA turbo LoRA (Comfy-Org/MiniMax-H3 loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors: rank 128, scale alpha/rank = 0.0625, fused qkv = concat A [384,5376] + block diagonal B [21504,384],
fc1 = [gate; value]) -> per-block fp16 tile packs in the layout of export_h3_lora.py (loraNN.bin, lorar0/1.bin = token refiner blocks). Same slots as the H3-Image pDMD LoRA, so the engine path (M_LORA) is shared.
usage: python export_r2v_lora.py LORA.safetensors OUTDIR"""
import os, sys, numpy as np
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_h3 import PERM
from export_h3_lora import wt16, rotA
LS = float(os.environ.get('LSCALE', '1')); LM = os.environ.get('LMASK', 'qkv,out,fc1,fc2').split(',')
def pack(f, p):
    def g(n, ab): return f.get_tensor(f'diffusion_model.{p}{n}.lora_{ab}.weight').float().numpy()
    def sc(n, rank):
        k = {'attn.qkv_proj': 'qkv', 'attn.out_proj': 'out', 'mlp.fc1': 'fc1', 'mlp.fc2': 'fc2'}[n]
        return LS * float(f.get_tensor(f'diffusion_model.{p}{n}.alpha')) / rank * (1.0 if k in LM else 0.0)
    out = []; sz = {}
    def add(n, b): sz[n] = len(b); out.append(b)
    Aq, Bqkv = g('attn.qkv_proj', 'A'), g('attn.qkv_proj', 'B'); s = sc('attn.qkv_proj', 384); r = 128; QN = 7168
    A = [Aq[j * r:(j + 1) * r] for j in range(3)]; B = [Bqkv[j * QN:(j + 1) * QN, j * r:(j + 1) * r] * s for j in range(3)]
    assert not np.abs(Bqkv[:QN, r:]).max() > 0, 'qkv B must be block diagonal'
    for j, n in enumerate('qkv'): add('A' + n, wt16(rotA(A[j]).T))
    add('Ao', wt16(rotA(g('attn.out_proj', 'A')).T)); add('Agu', wt16(rotA(g('mlp.fc1', 'A')).T)); add('Adn', wt16(rotA(g('mlp.fc2', 'A')).T))
    for j, n in enumerate('qkv'):
        b = B[j]
        if n != 'v': b = b.reshape(56, 128, -1)[:, PERM].reshape(b.shape)
        add('B' + n, wt16(b.T))
    add('Bo', wt16((g('attn.out_proj', 'B') * sc('attn.out_proj', 128)).T))
    Bf = g('mlp.fc1', 'B') * sc('mlp.fc1', 128); FF = Bf.shape[0] // 2; gate, up = Bf[:FF], Bf[FF:]
    Bi = np.stack((gate.reshape(FF // 32, 32, -1), up.reshape(FF // 32, 32, -1)), 1).reshape(2 * FF, -1)
    add('Bgu', wt16(Bi.T)); add('Bdn', wt16((g('mlp.fc2', 'B') * sc('mlp.fc2', 128)).T))
    return b''.join(out), sz
if __name__ == '__main__':
    f = safe_open(sys.argv[1], 'pt'); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    a0, a1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 49)
    for i in range(a0, a1 + 1): d, sz = pack(f, f'blocks.{i}.'); open(f'{out}/lora{i:02d}.bin', 'wb').write(d); print(i, len(d), flush=True)
    for i in range(2 if len(sys.argv) <= 4 else 0): d, sz = pack(f, f'token_refiner.blocks.{i}.'); open(f'{out}/lorar{i}.bin', 'wb').write(d); print('r', i, len(d), flush=True)
