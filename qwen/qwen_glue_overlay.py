"""Qwen-Image host-side LoRA layers as a glue override: glue8.bin (plain Turbo) + the LoRA delta of img_in, proj_out, time embedder, modulation, norm_out, txt_in -> a small MGL1 file
holding only the tensors the LoRA touches (qwenrun loads it after glue.bin / glue8.bin: names in the later file win).
usage: qwen_glue_overlay.py LORA.safetensors STRENGTH GLUE8.bin OUT.bin [REFERENCE_GLUE.bin]   (the reference, e.g. the real 4-step glue.bin, is only compared)"""
import sys, struct, numpy as np, torch
from safetensors import safe_open
MAP = {"img_in.weight": "img_in", "proj_out.weight": "proj_out", "time.linear_1.weight": "time_text_embed.timestep_embedder.linear_1", "time.linear_2.weight": "time_text_embed.timestep_embedder.linear_2",
       "modulation.weight": "modulation.1", "norm_out.weight": "norm_out.linear", "txt_in.weight": "txt_in.in_layer", "txt_out.weight": "txt_in.out_layer"}
def parse(p):
    b = open(p, "rb").read(); assert b[:4] == b"MGL1"; n = struct.unpack("<i", b[4:8])[0]; t = {}
    for i in range(n):
        e = b[8 + i * 76:8 + (i + 1) * 76]; name = e[:40].split(b"\0")[0].decode(); dt, nd = struct.unpack("<ii", e[40:48]); dims = struct.unpack("<3i", e[48:60]); off, nb = struct.unpack("<qq", e[60:76])
        a = np.frombuffer(b[off:off + nb], np.float32 if dt == 0 else np.float16).reshape([d for d in dims[:nd]]).copy(); t[name] = (dt, a)
    return t
def write(path, T):
    hdr = 8 + len(T) * 76; off = (hdr + 127) // 128 * 128; ent = b""; data = []
    for name, (dt, a) in T.items():
        d = list(a.shape) + [1] * (3 - a.ndim); ent += struct.pack("<40sii3iqq", name.encode(), dt, a.ndim, *d, off, a.nbytes); data.append((off, a.tobytes())); off = (off + a.nbytes + 127) // 128 * 128
    with open(path, "wb") as f:
        f.write(b"MGL1" + struct.pack("<i", len(T)) + ent)
        for o, b in data: f.seek(o); f.write(b)
if __name__ == "__main__":
    lora, s, g8, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    f = safe_open(lora, "pt"); keys = set(f.keys()); base = parse(g8); res = {}
    for gk, lk in MAP.items():
        for pre, ka, kb in (("transformer.", ".lora_down.weight", ".lora_up.weight"), ("diffusion_model.", ".lora_A.weight", ".lora_B.weight")):
            if pre + lk + ka in keys:
                A = f.get_tensor(pre + lk + ka).float(); B = f.get_tensor(pre + lk + kb).float(); al = f.get_tensor(pre + lk + ".alpha").float() if pre + lk + ".alpha" in keys else torch.tensor(float(A.shape[0]))
                dt, W = base[gk]; W2 = torch.from_numpy(W.astype(np.float32)) + (B * (float(al) / A.shape[0] * s)) @ A
                res[gk] = (dt, W2.numpy().astype(np.float32 if dt == 0 else np.float16)); break
    write(out, res); print("glue override:", {k: v[1].shape for k, v in res.items()})
    if len(sys.argv) > 5:
        ref = parse(sys.argv[5])
        for k, (dt, a) in res.items(): print(f"{k:24s} max|diff| vs reference {np.abs(a.astype(np.float32) - ref[k][1].astype(np.float32)).max():.3e}  (max|w| {np.abs(ref[k][1].astype(np.float32)).max():.2f})")
