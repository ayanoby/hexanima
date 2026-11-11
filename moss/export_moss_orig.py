"""Pack the ORIGINAL MOSS-Image-SFT transformer (bf16 safetensors, OpenMOSS-Team/MOSS-Image-SFT/transformer) + the rapid-moss LoRA into the same HMX q8 block files as export_moss.py
(which reads the unsloth UD-Q5_K_XL GGUF). The GGUF holds the weights re-quantised to Q5_K/Q6_K (~4% error vs the original); from the original only the q8 g32 tiles add error (~0.5%).
usage: python export_moss_orig.py TRANSFORMER_DIR LORA.safetensors OUTDIR     -> OUTDIR/dit/blk00..41.bin and OUTDIR/ref/{proj,ref0,ref1}.bin  (same names and sizes as the pack on the phone)"""
import os, sys, json, types, torch
sys.modules.setdefault("gguf", types.ModuleType("gguf"))   # moss_ref imports gguf for its GGUF reader class, which is not used here
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moss_ref import Lora
from export_moss import pack_layer, pack_refiner, pack_proj

class SW:
    """name -> fp32 tensor [out, in] from the sharded bf16 safetensors (same tensor names as the GGUF)"""
    def __init__(self, d):
        idx = json.load(open(os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        self.d, self.m, self.f = d, idx, {}
    def __call__(self, name):
        fn = self.m[name]
        if fn not in self.f: self.f[fn] = safe_open(os.path.join(self.d, fn), "pt")
        return self.f[fn].get_tensor(name).float()

if __name__ == "__main__":
    tdir, lora, out = sys.argv[1:4]
    g = SW(tdir); lo = Lora(lora)
    os.makedirs(out + "/dit", exist_ok=True); os.makedirs(out + "/ref", exist_ok=True)
    open(out + "/ref/proj.bin", "wb").write(pack_proj(g))
    for i in range(2):
        b, _ = pack_refiner(g, i); assert len(b) == 135397376; open(f"{out}/ref/ref{i}.bin", "wb").write(b)
    for i in range(42):
        b, _ = pack_layer(g, lo, i); open(f"{out}/dit/blk{i:02d}.bin", "wb").write(b); print("block", i, len(b), flush=True)
    print("EXPORTED")
