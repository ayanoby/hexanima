"""Edit inputs for `magerun edit` (the same files the app writes, Mage.java prepareEdit): up to 3 pictures.
usage: python mk_edit_in.py OUTDIR W H "instruction" IMG1 [IMG2 [IMG3]]   (W, H = target size in px, multiples of 16; every picture is stretched to it)
env: MAGE_TOK = tokenizer.json of Qwen3-VL-4B (default: text_encoder/tokenizer.json of the Mage weights, dir in MAGE_HF)"""
import os, sys
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
out, W, H, instr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; imgs = sys.argv[5:8]; os.makedirs(out, exist_ok=True)
SYS = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. "
       "Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n")
END = "<|im_end|>\n<|im_start|>assistant\n"
tok = Tokenizer.from_file(os.environ.get("MAGE_TOK", os.path.join(os.environ.get("MAGE_HF", "hf"), "text_encoder/tokenizer.json")))
body = ""; meta = []
for k, p in enumerate(imgs):
    im = Image.open(p).convert("RGB"); w0, h0 = im.size; m = max(w0, h0); s = 384 / m if m > 384 else 1
    vw, vh = max(1, round(w0 * s)), max(1, round(h0 * s)); Wv, Hv = max(32, round(vw / 32) * 32), max(32, round(vh / 32) * 32); gh, gw = Hv // 16, Wv // 16
    a = np.asarray(im.resize((Wv, Hv), Image.BILINEAR), dtype=np.float32) / 255.0; a = (a - 0.5) / 0.5      # [Hv][Wv][3]
    px = np.zeros((gh * gw, 3, 2, 16, 16), np.float32); bw = gw // 2
    for pi in range(gh * gw):   # 16x16 patches in 2x2 merge order, each (c, t = 2 copies, 16, 16)
        blk, ir, ic = pi // 4, (pi // 2) % 2, pi % 2; row, col = (blk // bw) * 2 + ir, (blk % bw) * 2 + ic
        t = a[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16, :].transpose(2, 0, 1); px[pi, :, 0] = t; px[pi, :, 1] = t
    sfx = "" if k == 0 else str(k + 1); px.tofile(f"{out}/px{sfx}.f32")
    ref = np.asarray(im.resize((W, H), Image.BILINEAR), dtype=np.float32) / 127.5 - 1; ref.tofile(f"{out}/ref{sfx}.rgb")
    meta.append(f"{gh} {gw} {H // 16} {W // 16}"); body += f"Image {k + 1}: <|vision_start|>" + "<|image_pad|>" * (gh * gw // 4) + "<|vision_end|>"
ids = tok.encode(SYS + body + instr + END, add_special_tokens=False).ids; drop = len(tok.encode(SYS, add_special_tokens=False).ids)
np.array(ids, np.int32).tofile(f"{out}/ids.i32"); open(f"{out}/meta.txt", "w").write("\n".join(meta) + "\n")
print("tokens", len(ids), "drop", drop, "pads", ids.count(151655))
