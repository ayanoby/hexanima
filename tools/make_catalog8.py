#!/usr/bin/env python3
"""catalog8.json = catalog7.json + MiniMax-H3-Image (family "MiniMax-H3"; the pack is in the public repo ayanoby/hexanima-h3, its manifest.json lists every file with size and sha256).
Read first by apps >= 1.6.0; older apps keep catalog7.json. usage: make_catalog8.py CATALOG7.json H3_MANIFEST.json OUT.json"""
import json, sys
d = json.load(open(sys.argv[1])); h3 = json.load(open(sys.argv[2]))
d["version"] = 8; d["min_app"] = "1.6.0"
d["packs"]["minimax-h3-image"] = {"url": "https://huggingface.co/ayanoby/hexanima-h3/resolve/main/", "manifest": "manifest.json", "size": sum(f["size"] for f in h3["files"])}
d["models"] = [m for m in d["models"] if m.get("id") != "minimax-h3-image"] + [{
    "id": "minimax-h3-image", "family": "MiniMax-H3", "name": "MiniMax-H3-Image", "params": "33B + Qwen3-VL", "quant": "INT8", "author": "MiniMax",
    "desc": "Крупная модель MiniMax: сцены описываются обычными предложениями, 6 шагов без CFG (2 шага с pDMD LoRA), размеры от 512 до 2048. Всё на NPU, файлы грузятся с диска для каждой картинки. Нужно 48 ГБ места. Лицензия MiniMax: не для США, ЕС, Великобритании и Южной Кореи.",
    "desc_en": "The large MiniMax model: describe scenes in plain sentences, 6 steps without CFG (2 steps with the pDMD LoRA), sizes from 512 to 2048. Everything runs on the NPU and the weights stream from storage for every picture. Needs 48 GB of storage. MiniMax licence: not for the US, EU, UK and South Korea.",
    "modes": [{"id": "minimax-h3-image", "label": "", "pack": "minimax-h3-image", "steps": 6, "speed": 110, "name": "MiniMax-H3-Image"}]}]
json.dump(d, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
