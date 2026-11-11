"""Pack the ORIGINAL MOSS-Image-SFT transformer (bf16 safetensors, OpenMOSS-Team/MOSS-Image-SFT/transformer) + the rapid-moss LoRA into the same HMX q8 block files as export_moss.py
(which reads the unsloth UD-Q5_K_XL GGUF). The GGUF holds the weights re-quantised to Q5_K/Q6_K (~4% error vs the original); from the original only the q8 g32 tiles add error (~0.5%).
usage: python export_moss_orig.py TRANSFORMER_DIR_OR_FILE LORA.safetensors OUTDIR     -> OUTDIR/dit/blk00..41.bin and OUTDIR/ref/{proj,ref0,ref1}.bin  (same names and sizes as the pack on the phone)"""
import os, sys, json, types, torch
sys.modules.setdefault("gguf", types.ModuleType("gguf"))   # moss_ref imports gguf for its GGUF reader class, which is not used here
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moss_ref import Lora
from export_moss import pack_layer, pack_refiner, pack_proj

class SW:
    """name -> fp32 tensor [out, in] from the sharded bf16 safetensors (same tensor names as the GGUF); D may also be a single .safetensors file (Civitai fine-tunes), where q/k/v sit
    fused as layers.N.proj.qkv.weight [3*3072, 3072] (q, k, v stacked) and are split here"""
    def __init__(self, d):
        if os.path.isfile(d):
            self.d, self.f = os.path.dirname(d), {os.path.basename(d): safe_open(d, "pt")}
            self.m = {k: os.path.basename(d) for k in self.f[os.path.basename(d)].keys()}
            md = self.f[os.path.basename(d)].metadata() or {}; fx = md.get("kohakumoss_fp16_fix")
            self.c = float(json.loads(fx)["c"]) if fx else None
        else:
            idx = json.load(open(os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
            self.d, self.m, self.f = d, idx, {}
    def fix(self, name, w):
        """kohakumoss_fp16_fix checkpoints (Civitai fp16 builds): the residual stream is kept at 1/16 for fp16, so the embedders, the text refiner's proj / ffn_down and the two gate chunks of
        shared_block_modulation are stored times c = 1/16; the engine runs the original scale, so multiply them back"""
        c = getattr(self, "c", None)
        if not c: return w
        k = 1.0 / c
        if name.startswith("video_embedder.") or name.startswith("text_embedder.proj.") or (name.startswith("text_embedder.blocks.") and (".proj." in name or ".ffn_down." in name)): return w * k
        if name.startswith("shared_block_modulation."):
            w = w.clone(); r = w.shape[0] // 6
            for i in (2, 5): w[i * r:(i + 1) * r] *= k
        return w
    def __call__(self, name):
        return self.fix(name, self.raw(name))
    def raw(self, name):
        if name not in self.m and name.rsplit(".", 2)[-2] in ("q", "k", "v") and name.replace(".q.", ".qkv.").replace(".k.", ".qkv.").replace(".v.", ".qkv.") in self.m:
            n = name[:name.rindex(".proj.") + 6] + "qkv.weight"; w = self.raw(n); r = w.shape[0] // 3
            return w[{"q": 0, "k": 1, "v": 2}[name.split(".")[-2]] * r:][:r]
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
    for i in range(42 if len(sys.argv) < 5 or sys.argv[4] != "ref" else 0):   # 4th argument "ref": only proj.bin / ref0 / ref1
        b, _ = pack_layer(g, lo, i); open(f"{out}/dit/blk{i:02d}.bin", "wb").write(b); print("block", i, len(b), flush=True)
    print("EXPORTED")
