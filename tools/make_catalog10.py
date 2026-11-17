#!/usr/bin/env python3
"""catalog10.json = catalog9.json + MiniMax-H3 Ref2VA Video (family "MiniMax-H3-Video", listed under MiniMax-H3; pack in ayanoby/hexanima-r2v-packs).
The pack's manifest.json lists every file with size and sha256 (the app downloads them raw into files/r2v; the licence texts are part of the pack); the text encoder is the one of MiniMax-H3-Image, which has to be installed first.
Read first by apps >= 1.8.2; older apps keep catalog9.json. usage: make_catalog10.py CATALOG9.json R2V_MANIFEST.json OUT.json"""
import json, sys
d = json.load(open(sys.argv[1])); rm = json.load(open(sys.argv[2]))
d["version"] = 10; d["min_app"] = "1.8.2"
size = sum(f["size"] for f in rm["files"])
d["packs"]["r2v-video"] = {"url": "https://huggingface.co/ayanoby/hexanima-r2v-packs/resolve/main/", "manifest": "manifest.json", "size": size}
new = [{"id": "r2v-video", "family": "MiniMax-H3-Video", "name": "MiniMax H3 Ref2VA Video", "params": "33B + Qwen3-VL", "quant": "INT8", "author": "MiniMax",
  "desc": "Видео по описанию или по фото-образцу (MiniMax-H3 Ref2VA): 4 шага без CFG, 22-90 кадров, размеры от 256 до 1280, без звука. Всё на NPU, файлы читаются с диска. Нужно 38 ГБ места и установленная MiniMax-H3-Image (её текстовый энкодер). Лицензия MiniMax: не для США, ЕС, Великобритании и Южной Кореи.",
  "desc_en": "Video from a prompt or a reference photo (MiniMax-H3 Ref2VA): 4 steps without CFG, 22-90 frames, sizes from 256 to 1280, no sound. Everything runs on the NPU, the weights are streamed from storage. It needs 38 GB of space and MiniMax-H3-Image installed (its text encoder). MiniMax licence: not for the US, EU, UK and South Korea.",
  "modes": [{"id": "r2v-video", "label": "", "pack": "r2v-video", "steps": 4, "speed": 120, "name": "MiniMax H3 Ref2VA Video"}]}]
ids = {m["id"] for m in new}
d["models"] = [m for m in d["models"] if m.get("id") not in ids] + new
json.dump(d, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
