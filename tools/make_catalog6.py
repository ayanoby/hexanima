#!/usr/bin/env python3
"""catalog6.json = catalog5.json + the MiaoMiao MOSS fine-tunes (RealSkin, Harem). Read first by apps >= 1.5.3 (older apps keep reading catalog5.json: they do not know these two
ids and would unpack their files over the MOSS-Image pack).  usage: make_catalog6.py CATALOG5.json OUT.json REALSKIN_MANIFEST.json HAREM_MANIFEST.json PACK_BASE_URL"""
import json, sys
cat5, out, mr, mh, base = sys.argv[1:6]
d = json.load(open(cat5)); size = lambda p: sum(f["size"] for f in json.load(open(p))["files"])
FT = [("moss-realskin-1", "moss/realskin-1.0/", mr, "MiaoMiao RealSkin MOSS 1.0",
       "Дообученная MOSS-Image от MIAOKA: реалистичная кожа и лица. Нужна установленная MOSS-Image 5B (общие файлы берутся из неё), 4 шага, всё на NPU.",
       "A MOSS-Image fine-tune by MIAOKA: realistic skin and faces. Needs the installed MOSS-Image 5B (the shared files come from it), 4 steps, all on the NPU."),
      ("moss-harem-1", "moss/harem-1.1/", mh, "MiaoMiao Harem MOSS 1.1",
       "Дообученная MOSS-Image от MIAOKA: стиль Harem. Нужна установленная MOSS-Image 5B (общие файлы берутся из неё), 4 шага, всё на NPU.",
       "A MOSS-Image fine-tune by MIAOKA: the Harem style. Needs the installed MOSS-Image 5B (the shared files come from it), 4 steps, all on the NPU.")]
for i, path, man, name, ru, en in FT:
    d["packs"][i] = {"url": base + path, "manifest": "manifest.json", "size": size(man), "needs": ["moss-sft-q5"]}
    d["models"] = [m for m in d["models"] if m.get("id") != i] + [{"id": i, "family": "MOSS-Image", "name": name, "params": "5.1B + Qwen3.5-9B", "author": "MIAOKA", "desc": ru, "desc_en": en,
        "modes": [{"id": i, "label": "4 step", "pack": i, "steps": 4, "speed": 35, "name": name}]}]
d["version"] = 6; d["min_app"] = "1.5.3"
json.dump(d, open(out, "w"), ensure_ascii=False, indent=1); print("packs", len(d["packs"]), "models", len(d["models"]))
