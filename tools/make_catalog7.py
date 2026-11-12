#!/usr/bin/env python3
"""catalog7.json = catalog6.json with the common files of v69 moved to common512/ (text encoder for 512 tokens). Read first by apps >= 1.5.4; older apps keep catalog6.json
and the 128 token files in common/ (they feed 128 tokens to the graph). usage: make_catalog7.py CATALOG6.json OUT.json"""
import json, sys
d = json.load(open(sys.argv[1])); d["version"] = 7; d["min_app"] = "1.5.4"
d["packs"]["anima-common@v69"]["url"] = d["packs"]["anima-common@v69"]["url"].replace("/common/", "/common512/")
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
