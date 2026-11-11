"""glue.bin for a MOSS-Image fine-tune (single fp16 .safetensors from Civitai): take an existing glue.bin (the original pack's: sampler tensors, 4-step time embedding, VAE statistics)
and replace the tensors that come from the DiT weights (video_embedder, shared_block_modulation.weight, video_head_modulation.weight, final_linear, final_norm) by the fine-tune's,
with the fp16-fix scaling undone (export_moss_orig.SW).  usage: make_glue_ft.py FT.safetensors OLD_GLUE.bin OUT_GLUE.bin"""
import os, sys, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_moss_orig import SW
ft, old, out = sys.argv[1:4]
REPL = ("video_embedder.weight", "video_embedder.bias", "shared_block_modulation.weight", "video_head_modulation.weight", "final_linear_video.weight", "final_norm_video.weight")
g = SW(ft); b = open(old, "rb").read(); assert b[:4] == b"MGL1"; n = struct.unpack("<i", b[4:8])[0]
T = []
for i in range(n):
    name, dt, nd, d0, d1, d2, off, nb = struct.unpack("<40sii3iqq", b[8 + i * 76: 8 + (i + 1) * 76]); name = name.split(b"\0")[0].decode()
    dims = [d0, d1, d2][:nd]; dty = np.float32 if dt == 0 else np.float16
    a = np.frombuffer(b[off:off + nb], dtype=dty).reshape(dims).copy()
    if name in REPL:
        w = g(name).numpy().astype(dty); assert w.shape == a.shape, (name, w.shape, a.shape)
        print(name, "rel diff to old", float(np.linalg.norm(w.astype(np.float32) - a.astype(np.float32)) / np.linalg.norm(a.astype(np.float32))))
        a = w
    T.append((name, dt, a))
hdr = 8 + len(T) * 76; off = (hdr + 127) // 128 * 128; ent = b""; data = []
for name, dt, a in T:
    d = list(a.shape) + [1] * (3 - a.ndim)
    ent += struct.pack("<40sii3iqq", name.encode(), dt, a.ndim, *d, off, a.nbytes); data.append((off, a.tobytes())); off = (off + a.nbytes + 127) // 128 * 128
with open(out, "wb") as f:
    f.write(b"MGL1" + struct.pack("<i", len(T)) + ent)
    for o, x in data: f.seek(o); f.write(x)
print(out, off, "bytes,", len(T), "tensors")
