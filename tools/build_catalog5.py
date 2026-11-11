#!/usr/bin/env python3
"""catalog5.json for the HexAnima app: packs (downloadable file sets with a manifest) + models (a family card with switchable modes).

usage: build_catalog5.py CATALOG4.json OLD_BASE_URL OUT.json [EXTRA_CATALOG4.json OLD_BASE_URL ...]
Every catalog4 item becomes a pack whose manifest URL is kept as is (so nothing has to be copied to the new repository first);
pairs are merged into one model with modes: Anima id + id+"base" (Turbo / Base), Qwen and MOSS default pack + "-8" add-on (4 step / 8 step).
A pack may later move: change its "url" (absolute dir) or use "path" (relative to catalog5.json) without touching the app.
"""
import json, sys

OLD_COMMON = {"v69": "common/", "v79": "common_v79/"}
ADDON_OF = {"qwen-image-21-turbo-8": "qwen-image-21-turbo", "moss-sft-q5-8": "moss-sft-q5"}   # (the Qwen plain Turbo is a pack of its own since 1.5.0, see STANDALONE)
STANDALONE = {"qwen-image-21-turbo-8": ("manifestbase.json", 14948295009)}   # shared files + dit8 in one manifest; no "needs"   # add-on id -> the pack it needs

def line_of(name):
    return "RealSkin" if "RealSkin" in name else "3D Harem" if "3D Harem" in name else "Harem"

def main():
    args = sys.argv[1:]
    out_path = args[2]
    srcs = [(args[0], args[1])] + list(zip(args[3::2], args[4::2]))
    packs, items = {}, {}
    common_base = None
    for path, base in srcs:
        d = json.load(open(path))
        if common_base is None: common_base = base
        for it in d["models"]:
            man = it["manifest"] if it["manifest"].startswith("http") else base + it["manifest"]
            it = dict(it); it["_url"] = man.rsplit("/", 1)[0] + "/"; it["_man"] = man.rsplit("/", 1)[1]
            items[it["id"]] = it
    for arch, sub in OLD_COMMON.items():
        packs["anima-common@" + arch] = {"url": common_base + sub, "arch": arch}
    for i, it in items.items():
        packs[i] = {"url": it["_url"], "manifest": it["_man"], "size": it["size"]}
        if it.get("kind", "anima") == "anima": packs[i]["needs"] = ["anima-common"]
        if i in ADDON_OF: packs[i]["needs"] = [ADDON_OF[i]]
        if i in STANDALONE: packs[i]["manifest"], packs[i]["size"] = STANDALONE[i]; packs[i].pop("needs", None)
    models, done = [], set()
    for i, it in items.items():
        if i in done: continue
        kind = it.get("kind", "anima")
        m = {"id": i, "family": {"anima": "Anima", "moss": "MOSS-Image", "qwen": "Qwen-Image", "mage": "Mage-Flow"}[kind], "name": it["name"],
             "params": it.get("params", ""), "author": it.get("author", ""), "desc": it.get("desc", ""), "desc_en": it.get("desc_en", it.get("desc", ""))}
        if kind == "anima":
            b = items.get(i + "base"); m["line"] = line_of(it["name"]); m["base"] = i
            m["modes"] = [{"id": i, "label": "Turbo", "pack": i, "steps": 4, "speed": it.get("speed", 0)}]
            if b: m["modes"].append({"id": i + "base", "label": "Base", "pack": i + "base", "steps": 25, "cfg": True, "speed": b.get("speed", 0)}); done.add(i + "base")
            elif i.endswith("base"): m["id"] = i; m["modes"] = [{"id": i, "label": "Base", "pack": i, "steps": 25, "cfg": True, "speed": it.get("speed", 0)}]; m["line"] = line_of(it["name"])
        elif i in ADDON_OF: continue
        else:
            add = next((a for a, b in ADDON_OF.items() if b == i), None)
            m["modes"] = [{"id": i, "label": "4 step", "pack": i, "steps": 4, "speed": it.get("speed", 0)}]
            if add:
                m["modes"].append({"id": add, "label": "8 step", "pack": add, "steps": 8, "speed": items[add].get("speed", 0)}); done.add(add)
            if kind == "mage": m["modes"][0]["label"] = ""
        for x in m["modes"]: x["name"] = items[x["id"]]["name"]
        done.add(i); models.append(m)
    # an Anima base whose turbo twin does not exist is listed as its own model (handled above); drop duplicates created by iteration order
    seen, uniq = set(), []
    for m in models:
        key = tuple(x["id"] for x in m["modes"])
        if key in seen: continue
        seen.add(key); uniq.append(m)
    json.dump({"version": 5, "min_app": "1.5.0", "packs": packs, "models": uniq}, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(len(packs), "packs,", len(uniq), "models")

main()
