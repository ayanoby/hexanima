#!/usr/bin/env python3
"""catalog9.json = catalog8.json + FLUX.3 Image Klein (family "FLUX.3-Klein", pack in ayanoby/hexanima-klein-packs) + LTX-2.5 Video (family "LTX-2.5", pack in ayanoby/hexanima-ltx-packs).
Each pack's manifest.json lists every file with size and sha256 (the app downloads them raw into files/klein and files/ltx; the licence texts are part of the packs).
Read first by apps >= 1.7.0; older apps keep catalog8.json. usage: make_catalog9.py CATALOG8.json KLEIN_MANIFEST.json LTX_MANIFEST.json OUT.json"""
import json, sys
d = json.load(open(sys.argv[1])); km = json.load(open(sys.argv[2])); lm = json.load(open(sys.argv[3]))
d["version"] = 9; d["min_app"] = "1.7.0"
size = lambda m: sum(f["size"] for f in m["files"])
d["packs"]["flux3-klein"] = {"url": "https://huggingface.co/ayanoby/hexanima-klein-packs/resolve/main/", "manifest": "manifest.json", "size": size(km)}
d["packs"]["ltx-2.5-video"] = {"url": "https://huggingface.co/ayanoby/hexanima-ltx-packs/resolve/main/", "manifest": "manifest.json", "size": size(lm)}
new = [
 {"id": "flux3-klein", "family": "FLUX.3-Klein", "name": "FLUX.3 Image Klein", "params": "5.6B + Qwen3-VL-4B", "quant": "INT8", "author": "Black Forest Labs",
  "desc": "Новая модель Black Forest Labs: фотореализм и надписи на картинках, 16 шагов без CFG, размеры от 256 до 1536. Всё на NPU, файлы читаются с диска. Нужно 9 ГБ места. Только некоммерческое использование (FLUX Kommunity License).",
  "desc_en": "The new Black Forest Labs model: photorealism and legible text in pictures, 16 steps without CFG, sizes from 256 to 1536. Everything runs on the NPU, the weights are streamed from storage. It needs 9 GB of space. Non-commercial use only (FLUX Kommunity License).",
  "modes": [{"id": "flux3-klein", "label": "", "pack": "flux3-klein", "steps": 16, "speed": 60, "name": "FLUX.3 Image Klein"}]},
 {"id": "ltx-2.5-video", "family": "LTX-2.5", "name": "LTX-2.5 Video", "params": "22B + Gemma 4 12B", "quant": "INT8", "author": "Lightricks",
  "desc": "Видео по описанию или по фото (Lightricks): 8 шагов без CFG, 17-97 кадров, размеры от 256 до 1280, без звука. Всё на NPU, файлы читаются с диска. Нужно 31 ГБ места. LTX-2 Community License.",
  "desc_en": "Video from a prompt or a photo (Lightricks): 8 steps without CFG, 17-97 frames, sizes from 256 to 1280, no sound. Everything runs on the NPU, the weights are streamed from storage. It needs 31 GB of space. LTX-2 Community License.",
  "modes": [{"id": "ltx-2.5-video", "label": "", "pack": "ltx-2.5-video", "steps": 8, "speed": 95, "name": "LTX-2.5 Video"}]}]
ids = {m["id"] for m in new}
d["models"] = [m for m in d["models"] if m.get("id") not in ids] + new
json.dump(d, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
