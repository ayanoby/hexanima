"""MOSS 8-step add-on: the lodestones DMD LoRA (same format as the 4-step turbo LoRA: rank 128 on q,k,v,o,ffn_up_gate,ffn_down of 42 blocks) replaces the turbo
LoRA tiles of the packs through overlay files, plus glue8.bin (time embedder, modulation biases, latent_mix Q, 8 step codes) that overrides tensors of glue.bin.
usage: make_lora8.py LORA.safetensors STRENGTH SRC_DIR(lora_latest.. time_embedding.. dataspace..) GLUE.bin OUTDIR   -> OUTDIR/lora8/blkNN.ovl + OUTDIR/glue8.bin"""
import os, sys, struct, types, numpy as np, torch
sys.modules.setdefault("gguf", types.ModuleType("gguf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_all_q8 import wt16, q8tiles
from make_lora_overlay import MATS, TILE, il
from safetensors import safe_open
lora, strength, src, glue, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
os.makedirs(out + "/lora8", exist_ok=True)
lf = safe_open(lora, "pt"); NAME = {"q": "q", "k": "k", "v": "v", "o": "o", "up_gate": "ffn_up_gate", "down": "ffn_down"}
for i in range(42):
    recs = []; off = 0; Ap = {}
    for name, K, N in MATS:
        nb, kb = N // 32, K // 32; panel = (kb + 4) * TILE; k = f"diffusion_model.layers.{i}.proj.{NAME[name]}."
        A = lf.get_tensor(k + "lora_down.weight").float(); B = lf.get_tensor(k + "lora_up.weight").float() * (float(lf.get_tensor(k + "alpha")) / A.shape[0] * strength)
        assert A.shape == (128, K) and B.shape == (N, 128)
        b = B.T.contiguous()
        if name == "up_gate": b = il(b)
        t = q8tiles(wt16(b.numpy())); assert len(t) == nb * 4 * TILE
        for n in range(nb): recs.append((off + n * panel + kb * TILE, t[n * 4 * TILE:(n + 1) * 4 * TILE]))
        off += nb * panel; Ap[name] = A.T.contiguous()
    for nm, m in (("A_qkv", torch.cat((Ap["q"], Ap["k"], Ap["v"]), 1)), ("A_o", Ap["o"]), ("A_ug", Ap["up_gate"]), ("A_down", Ap["down"])):
        d = q8tiles(wt16(m.numpy())); recs.append((off, d)); off += len(d)
    with open(f"{out}/lora8/blk{i:02d}.ovl", "wb") as f:
        f.write(struct.pack("<I", len(recs)))
        for o, d in recs: f.write(struct.pack("<QI", o, len(d))); f.write(d)
    print(i, flush=True)
# glue8.bin: only the tensors that differ
te = safe_open(src + "/time_embedding.safetensors", "pt"); ds = safe_open(src + "/dataspace.safetensors", "pt")
T = [(n, te.get_tensor(n).float().numpy()) for n in ("time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias", "shared_block_modulation.bias", "video_head_modulation.bias")]
T += [("latent_mix.Q", ds.get_tensor("latent_mix.Q").float().numpy()), ("step_codes", ds.get_tensor("step_codes").float().numpy())]
hdr = 8 + len(T) * 76; off = (hdr + 127) // 128 * 128; ent = b""; data = []
for n, a in T:
    a = np.ascontiguousarray(a, dtype=np.float32); d = list(a.shape) + [1] * (3 - a.ndim)
    ent += struct.pack("<40sii3iqq", n.encode(), 0, a.ndim, *d, off, a.nbytes); data.append((off, a.tobytes())); off = (off + a.nbytes + 127) // 128 * 128
with open(out + "/glue8.bin", "wb") as f:
    f.write(b"MGL1" + struct.pack("<i", len(T)) + ent)
    for o, b in data: f.seek(o); f.write(b)
print("glue8.bin", off)
