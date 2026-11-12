#!/usr/bin/env python3
"""Build the downloadable Qwen-Image-2.1 pack (generation + editing): copies/links the model files into OUT/ and writes manifest.json (size + sha256 per file) and the catalog entry.
usage: make_moss_pack.py MODELDIR OUT   (writes OUT/manifest.json; upload MODELDIR + manifest.json + README.md to the HF repo; MODELDIR = glue.bin embd.q4k tok_*.txt te/ ref/ dit/ vae/vae_WxH.bin, as on the phone under files/qwen: glue.bin embd.q4k tok_*.txt te/ dit/ vit/ vae/{vae_t32.bin,enc_t32.bin}; follow symlinks)"""
import hashlib, json, os, sys
src, out = sys.argv[1], sys.argv[2]; files = []
for root, _, fs in os.walk(src, followlinks=True):
    for f in sorted(fs):
        p = os.path.join(root, f); rel = os.path.relpath(p, src)
        if rel in ("meta.json", "manifest.json", "README.md") or f.startswith("."): continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
        files.append({"path": rel, "size": os.path.getsize(p), "sha256": h.hexdigest()}); print(rel, files[-1]["size"], flush=True)
os.makedirs(out, exist_ok=True)
json.dump({"files": files}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
total = sum(f["size"] for f in files)
print(json.dumps({"id": "qwen-image-21-turbo", "kind": "qwen", "name": "Qwen-Image 2.1 Turbo", "params": "7B + Qwen3-VL-8B", "quant": "Q5_K_S + 4-step LoRA", "size": total, "speed": 50, "manifest": "https://huggingface.co/ayanoby/hexanima-qwen-packs/resolve/main/manifest.json",
                  "desc": "Модель с точными надписями и редактированием фото по описанию, всё на NPU (4 шага). Только некоммерческое использование.", "desc_en": "Accurate text in pictures and photo editing by prompt, everything on the NPU (4 steps). Non-commercial use only.", "author": "Qwen team, LightX2V"}, ensure_ascii=False, indent=1))
