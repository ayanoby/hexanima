"""Qwen-Image LoRA slots as overlay files (reference for the phone tool): the 4 extra K tiles of every n-panel of the big matrices and the A matrices of a v2 DiT block pack
(export_qwen2.py layout) replaced by a LoRA file, so one plain pack (dit8, empty slots) + an overlay = the Lightning 4-step pack or any other turbo LoRA.
usage: qwen_overlay.py LORA.safetensors STRENGTH DIT8_DIR OUTDIR [first last]    ->  OUTDIR/blkNN.ovl = u32 n, n x (u64 offset, u32 len, data)   (read by qwenrun / apply_ovl)
Slot budget per module: 128 ranks; q/k/v/o/down take rank <= 128; gate and proj (the fused gate_up) 64 each (a fused gate_up LoRA with one shared A and rank > 64 is cut to its best rank 64 by SVD)."""
import os, sys, struct, types, numpy as np, torch
sys.modules.setdefault("gguf", types.ModuleType("gguf"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../moss"))
from export_all_q8 import wt16, q8tiles
from safetensors import safe_open

PERM = None
def head_perm():   # rows of 32 heads x 128 are permuted inside every head (adjacent pairs -> half-split rope), same as qwen_ref.PERM
    global PERM
    if PERM is None:
        h = np.arange(128); PERM = np.concatenate((h[0::2], h[1::2]))
    return np.arange(4096).reshape(32, 128)[:, PERM].reshape(-1)
HP = None
def q4a_tiles(q, S, M):   # copy of export_te.q4a_tiles: q [out, in] uint8; S, M [out, in/32] -> bytes [nb][kb] x 768
    out, inn = q.shape; nb, kb = out // 32, inn // 32
    t = q.reshape(nb, 32, kb, 32).transpose(0, 2, 3, 1)
    t = t.reshape(nb, kb, 16, 2, 32).transpose(0, 1, 2, 4, 3).reshape(nb, kb, 1024)
    t2 = t.reshape(nb, kb, 4, 2, 128)
    nib = (t2[:, :, :, 0] | (t2[:, :, :, 1] << 4)).astype(np.uint8)
    s = S.reshape(nb, 32, kb).transpose(0, 2, 1).astype(np.float16); m = M.reshape(nb, 32, kb).transpose(0, 2, 1).astype(np.float16)
    rec = np.concatenate([np.repeat(s, 2, axis=2).view(np.uint8).reshape(nb, kb, 128), np.repeat(m, 2, axis=2).view(np.uint8).reshape(nb, kb, 128), nib.reshape(nb, kb, 512)], axis=2)
    return np.ascontiguousarray(rec).tobytes()
def quant_b(B):   # B [rows, 128] float -> uint8 [rows,128], S, M [rows,4]: 4-bit per (row, 32 ranks)
    g = B.reshape(B.shape[0], 4, 32); mn = g.min(2); mx = g.max(2); S = (mx - mn) / 15.0
    q = np.where(S[..., None] > 0, np.rint((g - mn[..., None]) / np.where(S[..., None] > 0, S[..., None], 1)), 0).clip(0, 15).astype(np.uint8)
    return q.reshape(B.shape), S.astype(np.float32), mn.astype(np.float32)
def inter(a):   # [24576, ...] gate rows then proj rows -> 32-row blocks of gate and proj interleaved (gate_j, proj_j)
    h = a.shape[0] // 2
    return np.stack((a[:h].reshape(-1, 32, *a.shape[1:]), a[h:].reshape(-1, 32, *a.shape[1:])), 1).reshape(a.shape)

KS = [4096] * 5 + [12288]; NS = [4096, 4096, 4096, 4096, 24576, 4096]; TS = {1: 1152, 2: 768}

class Lora:
    def __init__(self, path, strength):
        self.f = safe_open(path, "pt"); self.keys = set(self.f.keys()); self.s = strength
        self.style = "lightning" if any(k.startswith("transformer.transformer_blocks.") for k in self.keys) else "diffusers"
    def _get(self, i, mod):   # (A [r,in], B [out,r] * scale) or None
        n = f"transformer_blocks.{i}.{mod}"
        if self.style == "lightning":
            b = "transformer." + n; ka, kb, kal = b + ".lora_down.weight", b + ".lora_up.weight", b + ".alpha"
        else:
            b = "diffusion_model." + n; ka, kb, kal = b + ".lora_A.weight", b + ".lora_B.weight", b + ".alpha"
        if ka not in self.keys: return None
        A = self.f.get_tensor(ka).float(); B = self.f.get_tensor(kb).float()
        al = float(self.f.get_tensor(kal)) if kal in self.keys else float(A.shape[0])
        return A, B * (al / A.shape[0] * self.s)
    def block(self, i):
        g = {}
        for k, mod in (("q", "attn.to_q"), ("k", "attn.to_k"), ("v", "attn.to_v"), ("o", "attn.to_out.0"), ("down", "img_mlp.out")): g[k] = self._get(i, mod)
        gg, pp = self._get(i, "img_mlp.gate_layer"), self._get(i, "img_mlp.proj")
        if gg is None and pp is None:   # fused gate_up (ai-toolkit): one A, B rows = gate then proj; rank cut to 64 for each half
            gu = self._get(i, "img_mlp.gate_up")
            if gu is not None:
                A, B = gu; r = A.shape[0]
                if r <= 64: gg, pp = (A, B[:12288]), (A, B[12288:])
                else:   # exact SVD of each half, rank 64 (the two halves get their own A)
                    def cut(Bh):
                        Qb, Rb = torch.linalg.qr(Bh); Qa, Ra = torch.linalg.qr(A.T); U, S, Vh = torch.linalg.svd(Rb @ Ra.T); s = S[:64].sqrt()
                        return (s[:, None] * Vh[:64]) @ Qa.T, (Qb @ U[:, :64]) * s
                    gg, pp = cut(B[:12288]), cut(B[12288:])
        g["gate"], g["proj"] = gg, pp
        return g

def apad(A, c0=0):   # A [r,in] -> [in,128] with cols c0..c0+r
    o = torch.zeros(A.shape[1], 128); o[:, c0:c0 + A.shape[0]] = A.T; return o

def slot_tiles(B, fmt):   # B [N, 128] float (rows already in pack order) -> bytes [nb][4] tiles
    if fmt == 2:
        q, S, M = quant_b(B.numpy()); return q4a_tiles(q, S, M)
    return q8tiles(wt16(np.ascontiguousarray(B.numpy().T)))    # q8: the 128 LoRA rows as [K=128, N] tiles

def overlay_block(L, i, hdr):
    magic, *rest = struct.unpack("<i11i6i", hdr[:72]); offs, fmts = rest[:11], rest[11:]
    assert magic == 0x00325751, hex(magic)
    g = L.block(i); recs = []
    hp = torch.from_numpy(head_perm())
    def B128(x, rows, c0=0):   # B [rows, r] -> [rows, 128]
        o = torch.zeros(rows, 128)
        if x is not None: o[:, c0:c0 + x.shape[1]] = x
        return o
    Bs = []
    for k in ("q", "k", "v", "o"):
        B = B128(g[k][1] if g[k] else None, 4096)
        Bs.append(B[hp] if k in ("q", "k") else B)
    Bgp = torch.zeros(24576, 128)
    if g["gate"]: Bgp[:12288, :g["gate"][1].shape[1]] = g["gate"][1]
    if g["proj"]: Bgp[12288:, 64:64 + g["proj"][1].shape[1]] = g["proj"][1]
    Bs.append(torch.from_numpy(inter(Bgp.numpy())))
    Bs.append(B128(g["down"][1] if g["down"] else None, 4096))
    for m in range(6):
        t = slot_tiles(Bs[m], fmts[m]); ts = TS[fmts[m]]; kb = KS[m] // 32; nb = NS[m] // 32; panel = (kb + 4) * ts
        for n in range(nb): recs.append((offs[m] + n * panel + kb * ts, t[n * 4 * ts:(n + 1) * 4 * ts]))
    Aq = torch.cat([apad(g[k][0]) if g[k] else torch.zeros(4096, 128) for k in ("q", "k", "v")], 1)
    Ao = apad(g["o"][0]) if g["o"] else torch.zeros(4096, 128)
    Agp = (apad(g["gate"][0]) if g["gate"] else torch.zeros(4096, 128)) + (apad(g["proj"][0], 64) if g["proj"] else 0)
    Ad = apad(g["down"][0]) if g["down"] else torch.zeros(12288, 128)
    for m, A in ((6, Aq), (7, Ao), (8, Agp), (9, Ad)): recs.append((offs[m], q8tiles(wt16(A.numpy()))))
    return recs

if __name__ == "__main__":
    lora, strength, dit8, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    a = int(sys.argv[5]) if len(sys.argv) > 5 else 0; b = int(sys.argv[6]) if len(sys.argv) > 6 else 31
    os.makedirs(out, exist_ok=True); L = Lora(lora, strength)
    for i in range(a, b + 1):
        with open(f"{dit8}/blk{i:02d}.bin", "rb") as f: hdr = f.read(4096)
        recs = overlay_block(L, i, hdr)
        with open(f"{out}/blk{i:02d}.ovl", "wb") as f:
            f.write(struct.pack("<I", len(recs)))
            for o, d in recs: f.write(struct.pack("<QI", o, len(d))); f.write(d)
        print(i, len(recs), flush=True)
