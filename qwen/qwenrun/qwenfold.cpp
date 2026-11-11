// qwenfold: a LoRA for Qwen-Image on the phone (Lightning-style turbo LoRAs or any LoRA of the Qwen-Image 2.1 DiT).
//   qwenfold check FILE.safetensors
//       -> one line "OK\tlayers=..\tmods=..\trank=..\thost=..\twarn=.." or "ERR\tCODE\tdetail"
//   qwenfold replace DITDIR FILE.safetensors STRENGTH OUTDIR [GLUE8.bin]
//       DITDIR = the plain pack (dit8: header QW2 + empty LoRA slots). Writes OUTDIR/blkNN.ovl = [u32 n] n x ([u64 offset][u32 len] bytes): the 4 extra K tiles of every n-panel
//       (q4a with 4-bit B rows, or q8) and the A matrices, i.e. the slot content of the 4-step pack (verified byte for byte against the Lightning pack with qwen_overlay.py);
//       with GLUE8.bin also OUTDIR/glueL.bin (MGL1: the host layers the LoRA touches, delta added, f16/f32 like the glue). Prints "progress K 32".
// Slot budget: q/k/v/o/down rank <= 128; gate and proj 64 each (a fused gate_up LoRA with rank <= 64 is split: shared A, B rows halves).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
typedef unsigned char u8; typedef uint16_t u16;

// ---------------------------------------------------------------- tiny JSON (enough for the safetensors header)
struct J { enum T { NUL, NUM, STR, ARR, OBJ } t = NUL; double n = 0; std::string s; std::vector<J> a; std::vector<std::pair<std::string, J>> o;
  const J* get(const std::string& k) const { for (auto& p : o) if (p.first == k) return &p.second; return nullptr; } };
struct JP { const char* p; const char* e; bool bad = false;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++; }
  std::string str() { std::string r; p++; while (p < e && *p != '"') { if (*p == '\\' && p + 1 < e) { p++; if (*p == 'u') { p += 4; r += '?'; } else r += *p == 'n' ? '\n' : *p; } else r += *p; p++; } p++; return r; }
  J val() { J v; ws(); if (p >= e) { bad = true; return v; }
    if (*p == '{') { v.t = J::OBJ; p++; ws(); while (p < e && *p != '}') { ws(); std::string k = str(); ws(); if (*p != ':') { bad = true; return v; } p++; v.o.emplace_back(k, val()); ws(); if (*p == ',') p++; ws(); if (bad) return v; } p++; }
    else if (*p == '[') { v.t = J::ARR; p++; ws(); while (p < e && *p != ']') { v.a.push_back(val()); ws(); if (*p == ',') p++; ws(); if (bad) return v; } p++; }
    else if (*p == '"') { v.t = J::STR; v.s = str(); }
    else { char* q; v.n = strtod(p, &q); if (q == p) { if (!strncmp(p, "null", 4)) { p += 4; return v; } bad = true; return v; } v.t = J::NUM; p = q; }
    return v; } };

// ---------------------------------------------------------------- safetensors
struct TI { int dt = 0; std::vector<long> shape; size_t o0 = 0, o1 = 0; };   // dt: 0 F32, 1 F16, 2 BF16
struct ST {
  u8* map = nullptr; size_t size = 0, base = 0; std::map<std::string, TI> t; std::map<std::string, std::string> meta; std::string err;
  bool open(const char* path) {
    int fd = ::open(path, O_RDONLY); if (fd < 0) { err = "cannot open file"; return false; }
    struct stat sb; fstat(fd, &sb); size = sb.st_size; if (size < 16) { err = "file too small"; ::close(fd); return false; }
    map = (u8*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); ::close(fd); if (map == MAP_FAILED) { map = nullptr; err = "mmap failed"; return false; }
    uint64_t hl; memcpy(&hl, map, 8); if (hl > size - 8 || hl > (64u << 20)) { err = "not a safetensors file"; return false; }
    JP jp{(const char*)map + 8, (const char*)map + 8 + hl}; J h = jp.val(); if (jp.bad || h.t != J::OBJ) { err = "bad header"; return false; }
    base = 8 + hl;
    for (auto& kv : h.o) {
      if (kv.first == "__metadata__") { for (auto& m : kv.second.o) meta[m.first] = m.second.t == J::STR ? m.second.s : std::to_string(m.second.n); continue; }
      const J* dt = kv.second.get("dtype"); const J* sh = kv.second.get("shape"); const J* of = kv.second.get("data_offsets"); if (!dt || !sh || !of || of->a.size() != 2) continue;
      TI ti; const std::string& d = dt->s; ti.dt = d == "F32" ? 0 : d == "F16" ? 1 : d == "BF16" ? 2 : 3;
      for (auto& x : sh->a) ti.shape.push_back((long)x.n); ti.o0 = (size_t)of->a[0].n; ti.o1 = (size_t)of->a[1].n;
      if (base + ti.o1 > size) { err = "truncated file"; return false; }
      t[kv.first] = ti;
    }
    return true;
  }
  float at(const TI& ti, size_t i) const {
    const u8* p = map + base + ti.o0;
    if (ti.dt == 0) { float f; memcpy(&f, p + i * 4, 4); return f; }
    u16 h; memcpy(&h, p + i * 2, 2);
    if (ti.dt == 2) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
    _Float16 hf; memcpy(&hf, &h, 2); return (float)hf;
  }
};

static const int NBLK = 32;
static const long KS[6] = {4096, 4096, 4096, 4096, 4096, 12288}, NS[6] = {4096, 4096, 4096, 4096, 24576, 4096};
static inline u16 f2h(float f) { _Float16 h = (_Float16)f; u16 u; memcpy(&u, &h, 2); return u; }
static inline float rh(float f) { return (float)(_Float16)f; }

struct LT { const TI* A = nullptr; const TI* B = nullptr; int r = 0; float sc = 0; long in = 0, out = 0, brow0 = 0; bool has = false; };
struct LoraSet {
  ST st; std::string pre, ka, kb; float strength = 1;
  bool detect() {
    for (auto& kv : st.t) {
      const std::string& k = kv.first;
      if (k.compare(0, 31, "transformer.transformer_blocks.") == 0 && k.find(".lora_down.weight") != std::string::npos) { pre = "transformer."; ka = ".lora_down.weight"; kb = ".lora_up.weight"; return true; }
      if (k.compare(0, 36, "diffusion_model.transformer_blocks.") == 0 && k.find(".lora_A.weight") != std::string::npos) { pre = "diffusion_model."; ka = ".lora_A.weight"; kb = ".lora_B.weight"; return true; }
    }
    return false;
  }
  // module name like "transformer_blocks.3.attn.to_q" or "img_in"
  LT get(const std::string& n, long brow0 = 0, long outRows = 0) const {
    LT l; auto a = st.t.find(pre + n + ka), b = st.t.find(pre + n + kb); if (a == st.t.end() || b == st.t.end()) return l;
    l.A = &a->second; l.B = &b->second; l.r = (int)l.A->shape[0]; l.in = l.A->shape[1]; l.out = outRows ? outRows : l.B->shape[0]; l.brow0 = brow0;
    float al = (float)l.r; auto al_it = st.t.find(pre + n + ".alpha"); if (al_it != st.t.end()) al = st.at(al_it->second, 0);
    l.sc = al / l.r * strength; l.has = true; return l;
  }
  float b(const LT& l, long row, int rank) const { return st.at(*l.B, (size_t)(l.brow0 + row) * l.r + rank) * l.sc; }
  float a(const LT& l, int rank, long k) const { return st.at(*l.A, (size_t)rank * l.in + k); }
};

struct BlockMods { LT m[6]; LT gate, proj; };   // q k v o gate/proj(fused, see gate/proj) down: m[4] unused
static BlockMods block_mods(const LoraSet& L, int i, std::string* err = nullptr) {
  BlockMods bm; std::string p = "transformer_blocks." + std::to_string(i) + ".";
  bm.m[0] = L.get(p + "attn.to_q"); bm.m[1] = L.get(p + "attn.to_k"); bm.m[2] = L.get(p + "attn.to_v"); bm.m[3] = L.get(p + "attn.to_out.0"); bm.m[5] = L.get(p + "img_mlp.out");
  bm.gate = L.get(p + "img_mlp.gate_layer"); bm.proj = L.get(p + "img_mlp.proj");
  if (!bm.gate.has && !bm.proj.has) { LT gu = L.get(p + "img_mlp.gate_up"); if (gu.has) { bm.gate = gu; bm.proj = gu; bm.gate.out = bm.proj.out = 12288; bm.proj.brow0 = 12288; } }
  return bm;
}

// ---- tiles
static void put_q4a(u8* t, const u8 q[32][32], const float S[32], const float M[32]) {   // q[c][r]
  memset(t, 0, 768);
  for (int c = 0; c < 32; c++) { u16 s = f2h(S[c]), m = f2h(M[c]); memcpy(t + 4 * c, &s, 2); memcpy(t + 4 * c + 2, &s, 2); memcpy(t + 128 + 4 * c, &m, 2); memcpy(t + 128 + 4 * c + 2, &m, 2); }
  for (int c = 0; c < 32; c++) for (int r = 0; r < 32; r++) {
    const int hw = ((r >> 1) * 32 + c) * 2 + (r & 1), mm = hw >> 8, h = (hw >> 7) & 1, bb = hw & 127; u8& d = t[256 + mm * 128 + bb]; d |= (u8)(h ? (q[c][r] << 4) : q[c][r]);
  }
}
static void put_q8(u8* t, const float a[32][32]) {   // a[k][c], values already rounded to fp16
  float sf[32];
  for (int c = 0; c < 32; c++) {
    float m = 0; for (int k = 0; k < 32; k++) m = std::max(m, fabsf(a[k][c])); float s = m / 127.f; if (s < 6.2e-5f) s = 6.2e-5f; u16 h = f2h(s); sf[c] = (float)(_Float16)s;
    memcpy(t + 4 * c, &h, 2); memcpy(t + 4 * c + 2, &h, 2);
  }
  for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) {
    float q = nearbyintf(a[k][c] / sf[c]); q = std::min(127.f, std::max(-127.f, q)); t[128 + (((k >> 1) * 32 + c) << 1) + (k & 1)] = (u8)((int)q + 128);
  }
}
static int perm_row(long p) { long h = p / 128, j = p % 128; return (int)(h * 128 + (j < 64 ? 2 * j : 2 * (j - 64) + 1)); }

typedef std::vector<std::pair<uint64_t, std::vector<u8>>> Recs;
static bool block_ovl(const LoraSet& L, int i, const std::string& ditdir, const std::string& outdir) {
  char path[600]; snprintf(path, sizeof path, "%s/blk%02d.bin", ditdir.c_str(), i);
  FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; } u8 hdr[4096]; if (fread(hdr, 1, 4096, f) != 4096) { fclose(f); return false; } fclose(f);
  int32_t magic, offs[11], fmts[6]; memcpy(&magic, hdr, 4); memcpy(offs, hdr + 4, 44); memcpy(fmts, hdr + 48, 24);
  if (magic != 0x00325751) { fprintf(stderr, "%s: not a v2 Qwen pack\n", path); return false; }
  BlockMods bm = block_mods(L, i); Recs recs;
  for (int m = 0; m < 6; m++) {
    const long N = NS[m], K = KS[m]; const int fmt = fmts[m]; const size_t ts = fmt == 2 ? 768 : 1152; const long NB = N / 32, KB = K / 32; const size_t panel = (KB + 4) * ts;
    // value of the slot matrix: row p (pack order), rank j (0..127)
    auto val = [&](long p, int j) -> float {
      if (m == 4) { long blk = p / 32, w = p % 32, src = (blk >> 1) * 32 + w; const LT& l = (blk & 1) ? bm.proj : bm.gate; int off = (blk & 1) ? 64 : 0; int jj = j - off; if (!l.has || jj < 0 || jj >= l.r) return 0.f; return L.b(l, src, jj); }
      const LT& l = bm.m[m]; if (!l.has || j >= l.r) return 0.f; long src = (m < 2) ? perm_row(p) : p; return L.b(l, src, j);
    };
    for (long nb = 0; nb < NB; nb++) {
      std::vector<u8> out(4 * ts);
      for (int g = 0; g < 4; g++) {
        u8* t = &out[g * ts];
        if (fmt == 2) {
          u8 q[32][32]; float S[32], M[32];
          for (int c = 0; c < 32; c++) {
            float v[32], mn = 1e30f, mx = -1e30f; for (int r = 0; r < 32; r++) { v[r] = val(nb * 32 + c, g * 32 + r); mn = std::min(mn, v[r]); mx = std::max(mx, v[r]); }
            float s = (mx - mn) / 15.0f; S[c] = s; M[c] = mn;
            for (int r = 0; r < 32; r++) { float qq = s > 0 ? nearbyintf((v[r] - mn) / s) : 0.f; qq = std::min(15.f, std::max(0.f, qq)); q[c][r] = (u8)qq; }
          }
          put_q4a(t, q, S, M);
        } else {
          float a[32][32]; for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) a[k][c] = rh(val(nb * 32 + c, g * 32 + k));
          put_q8(t, a);
        }
      }
      recs.push_back({(uint64_t)offs[m] + (uint64_t)nb * panel + (uint64_t)KB * ts, std::move(out)});
    }
  }
  // A matrices (q8 tiles [nb][kb]): value(k = input index, col)
  struct AM { int off; long K, N; };
  const AM ams[4] = {{offs[6], 4096, 384}, {offs[7], 4096, 128}, {offs[8], 4096, 128}, {offs[9], 12288, 128}};
  for (int x = 0; x < 4; x++) {
    const long K = ams[x].K, N = ams[x].N, NB = N / 32, KB = K / 32; std::vector<u8> out((size_t)NB * KB * 1152);
    auto val = [&](long k, long col) -> float {
      const LT* l = nullptr; long rk = col;
      if (x == 0) { l = &bm.m[col / 128]; rk = col % 128; } else if (x == 1) l = &bm.m[3]; else if (x == 3) l = &bm.m[5];
      else { if (col < 64) { l = &bm.gate; rk = col; } else { l = &bm.proj; rk = col - 64; } }
      if (!l->has || rk >= l->r) return 0.f; return rh(L.a(*l, (int)rk, k));
    };
    for (long nb = 0; nb < NB; nb++) for (long kb = 0; kb < KB; kb++) { float a[32][32]; for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) a[k][c] = val(kb * 32 + k, nb * 32 + c); put_q8(&out[((size_t)nb * KB + kb) * 1152], a); }
    recs.push_back({(uint64_t)ams[x].off, std::move(out)});
  }
  snprintf(path, sizeof path, "%s/blk%02d.ovl", outdir.c_str(), i); FILE* o = fopen(path, "wb"); if (!o) return false;
  uint32_t n = (uint32_t)recs.size(); fwrite(&n, 4, 1, o); for (auto& r : recs) { uint32_t l = (uint32_t)r.second.size(); fwrite(&r.first, 8, 1, o); fwrite(&l, 4, 1, o); fwrite(r.second.data(), 1, l, o); }
  fclose(o); return true;
}

// ---- host layers (glue)
struct GMap { const char* gk; const char* lk; };
static const GMap GM[] = {{"img_in.weight", "img_in"}, {"proj_out.weight", "proj_out"}, {"time.linear_1.weight", "time_text_embed.timestep_embedder.linear_1"}, {"time.linear_2.weight", "time_text_embed.timestep_embedder.linear_2"},
  {"modulation.weight", "modulation.1"}, {"norm_out.weight", "norm_out.linear"}, {"txt_in.weight", "txt_in.in_layer"}, {"txt_out.weight", "txt_in.out_layer"}};
static int host_layers(const LoraSet& L) { int n = 0; for (auto& g : GM) if (L.get(g.lk).has) n++; return n; }
static bool glue_override(const LoraSet& L, const std::string& g8, const std::string& out) {
  int fd = open(g8.c_str(), O_RDONLY); if (fd < 0) { fprintf(stderr, "cannot open %s\n", g8.c_str()); return false; } struct stat sb; fstat(fd, &sb); u8* base = (u8*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
  if (base == MAP_FAILED || memcmp(base, "MGL1", 4)) return false; int n; memcpy(&n, base + 4, 4);
  struct Ent { std::string name; int dt, nd, dims[3]; long long off, nb; std::vector<u8> data; }; std::vector<Ent> ents;
  for (int i = 0; i < n; i++) {
    const u8* e = base + 8 + i * 76; char nm[41] = {0}; memcpy(nm, e, 40); const GMap* gm = nullptr; for (auto& g : GM) if (!strcmp(g.gk, nm)) gm = &g; if (!gm) continue;
    LT l = L.get(gm->lk); if (!l.has) continue;
    Ent en; en.name = nm; memcpy(&en.dt, e + 40, 4); memcpy(&en.nd, e + 44, 4); memcpy(en.dims, e + 48, 12); memcpy(&en.off, e + 60, 8); memcpy(&en.nb, e + 68, 8);
    const long O = en.dims[0], I = en.nd >= 2 ? en.dims[1] : 1; if (O != l.out || I != l.in) { fprintf(stderr, "glue %s: shape mismatch\n", nm); munmap(base, sb.st_size); return false; }
    std::vector<float> W((size_t)O * I); for (size_t k = 0; k < W.size(); k++) { if (en.dt == 0) { float v; memcpy(&v, base + en.off + k * 4, 4); W[k] = v; } else { _Float16 h; memcpy(&h, base + en.off + k * 2, 2); W[k] = (float)h; } }
    std::vector<float> Bm((size_t)O * l.r), Am((size_t)l.r * I); for (long o = 0; o < O; o++) for (int r = 0; r < l.r; r++) Bm[(size_t)o * l.r + r] = L.b(l, o, r);
    for (int r = 0; r < l.r; r++) for (long k = 0; k < I; k++) Am[(size_t)r * I + k] = L.a(l, r, k);
    std::atomic<long> next(0); int nth = std::max(1u, std::min(6u, std::thread::hardware_concurrency())); std::vector<std::thread> th;
    for (int t = 0; t < nth; t++) th.emplace_back([&] { for (;;) { long o = next++; if (o >= O) break; float* w = &W[(size_t)o * I]; for (int r = 0; r < l.r; r++) { const float b = Bm[(size_t)o * l.r + r]; const float* a = &Am[(size_t)r * I]; for (long k = 0; k < I; k++) w[k] += b * a[k]; } } });
    for (auto& t : th) t.join();
    en.data.resize(en.nb); if (en.dt == 0) memcpy(en.data.data(), W.data(), en.nb); else { _Float16* d = (_Float16*)en.data.data(); for (size_t k = 0; k < W.size(); k++) d[k] = (_Float16)W[k]; }
    ents.push_back(std::move(en));
  }
  munmap(base, sb.st_size);
  size_t hdr = 8 + ents.size() * 76, off = (hdr + 127) / 128 * 128; FILE* f = fopen(out.c_str(), "wb"); if (!f) return false;
  std::vector<u8> head(off, 0); memcpy(head.data(), "MGL1", 4); int ne = (int)ents.size(); memcpy(head.data() + 4, &ne, 4);
  size_t o = off; for (size_t i = 0; i < ents.size(); i++) { u8* e = head.data() + 8 + i * 76; memcpy(e, ents[i].name.c_str(), std::min<size_t>(39, ents[i].name.size())); memcpy(e + 40, &ents[i].dt, 4); memcpy(e + 44, &ents[i].nd, 4); memcpy(e + 48, ents[i].dims, 12); long long oo = o, nb = ents[i].nb; memcpy(e + 60, &oo, 8); memcpy(e + 68, &nb, 8); o = (o + ents[i].nb + 127) / 128 * 128; }
  fwrite(head.data(), 1, off, f); o = off;
  for (auto& en : ents) { fseek(f, (long)o, SEEK_SET); fwrite(en.data.data(), 1, en.data.size(), f); o = (o + en.nb + 127) / 128 * 128; }
  fclose(f); return true;
}

static int fail(const char* code, const std::string& detail) { printf("ERR\t%s\t%s\n", code, detail.c_str()); return 1; }
static int check(LoraSet& L) {
  if (!L.detect()) { for (auto& kv : L.st.t) if (kv.first.find("lora") != std::string::npos) return fail("OTHERMODEL", kv.first.substr(0, 60)); return fail("NOTLORA", ""); }
  int layers = 0, mods = 0, rank = 0; bool r64 = false;
  for (auto& kv : L.st.t) { const std::string& k = kv.first; size_t p = k.find("transformer_blocks."); if (p == std::string::npos) continue; int idx = atoi(k.c_str() + p + 19); if (idx >= NBLK || idx < 0) return fail("LAYER", std::to_string(idx)); }
  for (int i = 0; i < NBLK; i++) {
    BlockMods bm = block_mods(L, i); bool any = false;
    for (int m = 0; m < 6; m++) { if (m == 4) continue; const LT& l = bm.m[m]; if (!l.has) continue; any = true; mods++; rank = std::max(rank, l.r); if (l.r > 128) return fail("RANK", std::to_string(l.r)); if (l.in != KS[m] || l.out != NS[m] || (long)l.B->shape[0] != NS[m]) return fail("SHAPE", "block " + std::to_string(i) + " module " + std::to_string(m)); }
    for (const LT* l : {&bm.gate, &bm.proj}) if (l->has) { any = true; mods++; rank = std::max(rank, l->r); if (l->r > 64) return fail("RANK64", "gate/proj " + std::to_string(l->r)); if (l->in != 4096 || l->out != 12288) return fail("SHAPE", "block " + std::to_string(i) + " gate/proj"); }
    if (any) layers++;
  }
  if (!layers) return fail("NOTLORA", "no transformer blocks");
  if (rank > 64) r64 = true;
  printf("OK\tlayers=%d\tmods=%d\trank=%d\thost=%d\twarn=%s\n", layers, mods, rank, host_layers(L), r64 ? "rank>64" : ""); return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && !strcmp(argv[1], "check")) { LoraSet L; if (!L.st.open(argv[2])) return fail("FILE", L.st.err); return check(L); }
  if (argc >= 6 && !strcmp(argv[1], "replace")) {
    LoraSet L; if (!L.st.open(argv[3])) return fail("FILE", L.st.err); if (check(L)) return 1; L.strength = (float)atof(argv[4]); std::string dit = argv[2], out = argv[5]; mkdir(out.c_str(), 0755);
    std::atomic<int> next(0), done(0), bad(0); std::mutex mu; int nth = std::max(1u, std::min(6u, std::thread::hardware_concurrency())); std::vector<std::thread> th;
    for (int t = 0; t < nth; t++) th.emplace_back([&] { for (;;) { int i = next++; if (i >= NBLK) break; if (!block_ovl(L, i, dit, out)) bad++; int d = ++done; std::lock_guard<std::mutex> lk(mu); printf("progress %d %d\n", d, NBLK); fflush(stdout); } });
    for (auto& t : th) t.join(); if (bad) return fail("FOLD", "failed on " + std::to_string(bad.load()) + " blocks");
    if (argc >= 7 && host_layers(L) > 0 && !glue_override(L, argv[6], out + "/glueL.bin")) return fail("GLUE", "cannot build the glue override");
    printf("OK\n"); return 0;
  }
  fprintf(stderr, "usage: qwenfold check FILE | replace DITDIR FILE STRENGTH OUTDIR [GLUE8.bin]\n"); return 2;
}
