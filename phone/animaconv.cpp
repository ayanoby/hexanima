// animaconv: converts an Anima (Cosmos-Predict2 + LLM adapter) checkpoint (.safetensors, bf16/fp16/fp32) into the files the NPU runner reads, on the phone.
// Same output as export/export_all.py --q8 and phone/extract_small.py (byte for byte), plus the sd.cpp stub (the checkpoint header + a sparse zero body).
//   animaconv check SRC                      -> "ok blocks=<n> d=<2048>" or "error <text>"
//   animaconv convert SRC OUTDIR STUB        -> OUTDIR/blkNN.bin (q8 tiles), OUTDIR/small.bin, STUB (stub safetensors); prints "progress <done> <total>", then "done"
// One block at a time is in memory (the largest matrix is 67 MB as fp32), the matrices are cut into tile columns and converted by all cores.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

typedef std::vector<uint8_t> Bytes;

// ---- half precision (round to nearest even, like numpy astype(float16)) -------------------------------------------------------------
static uint16_t f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
  if (e == 0xff) return sign | 0x7c00 | (m ? 0x200 | (m >> 13) : 0);   // inf / nan
  int ne = (int)e - 127 + 15;
  if (ne >= 31) return sign | 0x7c00;                                 // overflow
  if (ne <= 0) {                                                       // subnormal half or zero
    if (ne < -10) return sign;
    m |= 0x800000; int sh = 14 - ne; uint32_t h = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return sign | h;
  }
  uint32_t h = ((uint32_t)ne << 10) | (m >> 13), rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;                 // a carry into the exponent is right (up to inf)
  return sign | h;
}
static float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
  if (e == 0) { if (!m) x = sign; else { int s = 0; while (!(m & 0x400)) { m <<= 1; s++; } m &= 0x3ff; x = sign | ((uint32_t)(113 - s) << 23) | (m << 13); } }
  else if (e == 31) x = sign | 0x7f800000 | (m << 13);
  else x = sign | ((e + 112) << 23) | (m << 13);
  float f; memcpy(&f, &x, 4); return f;
}

// ---- safetensors header -------------------------------------------------------------------------------------------------------------
struct Tensor { std::string name, dtype; std::vector<int64_t> shape; uint64_t b = 0, e = 0; };
struct Src {
  int fd = -1; uint64_t base = 0, size = 0, hdrlen = 0; std::vector<Tensor> order; std::map<std::string, size_t> idx; std::string err;
  const Tensor* get(const std::string& k) const { auto i = idx.find(k); return i == idx.end() ? nullptr : &order[i->second]; }
  bool read(const Tensor& t, Bytes& out) const {
    out.resize(t.e - t.b); uint64_t o = 0;
    while (o < out.size()) { ssize_t n = pread(fd, out.data() + o, out.size() - o, base + t.b + o); if (n <= 0) return false; o += n; }
    return true;
  }
};
struct JP {   // just enough JSON for a safetensors header: objects, arrays, strings, integers
  const char* p; const char* end;
  void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++; }
  bool str(std::string& s) {
    ws(); if (p >= end || *p != '"') return false; p++; s.clear();
    while (p < end && *p != '"') { if (*p == '\\' && p + 1 < end) { p++; char c = *p; s += c == 'n' ? '\n' : c == 't' ? '\t' : c; } else s += *p; p++; }
    if (p >= end) return false; p++; return true;
  }
  bool num(int64_t& v) { ws(); char* q; v = strtoll(p, &q, 10); if (q == p) return false; p = q; return true; }
  bool skip() {   // any value
    ws(); if (p >= end) return false;
    if (*p == '"') { std::string s; return str(s); }
    if (*p == '{' || *p == '[') { char open = *p, close = open == '{' ? '}' : ']'; p++; ws(); if (p < end && *p == close) { p++; return true; }
      for (;;) { if (open == '{') { std::string k; if (!str(k)) return false; ws(); if (p >= end || *p != ':') return false; p++; } if (!skip()) return false; ws(); if (p < end && *p == ',') { p++; continue; } if (p < end && *p == close) { p++; return true; } return false; } }
    while (p < end && *p != ',' && *p != '}' && *p != ']') p++; return true;
  }
};
static bool open_src(const char* path, Src& s) {
  s.fd = open(path, O_RDONLY); if (s.fd < 0) { s.err = "cannot open the file"; return false; }
  struct stat st; fstat(s.fd, &st); s.size = st.st_size;
  uint64_t n = 0; if (pread(s.fd, &n, 8, 0) != 8 || n == 0 || n > (1ull << 28) || 8 + n > s.size) { s.err = "not a safetensors file"; return false; }
  std::string h(n, 0); if (pread(s.fd, &h[0], n, 8) != (ssize_t)n) { s.err = "short header"; return false; }
  s.hdrlen = n; s.base = 8 + n;
  JP j{h.data(), h.data() + h.size()}; j.ws(); if (j.p >= j.end || *j.p != '{') { s.err = "bad header"; return false; } j.p++;
  for (;;) {
    j.ws(); if (j.p < j.end && *j.p == '}') break;
    Tensor t; if (!j.str(t.name)) { s.err = "bad header"; return false; } j.ws(); if (j.p >= j.end || *j.p != ':') { s.err = "bad header"; return false; } j.p++;
    if (t.name == "__metadata__") { if (!j.skip()) { s.err = "bad header"; return false; } }
    else {
      j.ws(); if (j.p >= j.end || *j.p != '{') { s.err = "bad header"; return false; } j.p++;
      for (;;) {
        j.ws(); if (j.p < j.end && *j.p == '}') { j.p++; break; }
        std::string k; if (!j.str(k)) { s.err = "bad header"; return false; } j.ws(); if (j.p >= j.end || *j.p != ':') { s.err = "bad header"; return false; } j.p++;
        if (k == "dtype") { if (!j.str(t.dtype)) return false; }
        else if (k == "shape") { j.ws(); if (*j.p != '[') return false; j.p++; for (;;) { j.ws(); if (*j.p == ']') { j.p++; break; } int64_t v; if (!j.num(v)) return false; t.shape.push_back(v); j.ws(); if (*j.p == ',') j.p++; } }
        else if (k == "data_offsets") { j.ws(); if (*j.p != '[') return false; j.p++; int64_t a, b; if (!j.num(a)) return false; j.ws(); if (*j.p == ',') j.p++; if (!j.num(b)) return false; j.ws(); if (*j.p == ']') j.p++; t.b = a; t.e = b; }
        else if (!j.skip()) return false;
        j.ws(); if (j.p < j.end && *j.p == ',') j.p++;
      }
      if (t.e < t.b || s.base + t.e > s.size) { s.err = "the file is truncated (" + t.name + ")"; return false; }
      s.idx[t.name] = s.order.size(); s.order.push_back(t);
    }
    j.ws(); if (j.p < j.end && *j.p == ',') j.p++;
  }
  return true;
}
static int64_t numel(const Tensor& t) { int64_t n = 1; for (int64_t d : t.shape) n *= d; return n; }
// tensor bytes -> float32 (F32, F16, BF16)
static bool to_f32(const Tensor& t, const Bytes& raw, std::vector<float>& out) {
  int64_t n = numel(t); out.resize(n);
  if (t.dtype == "F32" && raw.size() == (size_t)n * 4) { memcpy(out.data(), raw.data(), n * 4); return true; }
  if (t.dtype == "F16" && raw.size() == (size_t)n * 2) { for (int64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, &raw[i * 2], 2); out[i] = h2f(h); } return true; }
  if (t.dtype == "BF16" && raw.size() == (size_t)n * 2) { for (int64_t i = 0; i < n; i++) { uint32_t x = (uint32_t)raw[i * 2] << 16 | (uint32_t)raw[i * 2 + 1] << 24; memcpy(&out[i], &x, 4); } return true; }
  return false;
}

// ---- the model ----------------------------------------------------------------------------------------------------------------------
static const char* ORDER[10] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj", "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj", "mlp.layer1", "mlp.layer2"};
struct Model { std::string prefix; int blocks = 0; std::string err; };
static Model inspect(const Src& s) {
  Model m; const std::string probe = "blocks.0.self_attn.q_proj.weight"; bool found = false;
  for (auto& t : s.order) { size_t p = t.name.find(probe); if (p != std::string::npos && p + probe.size() == t.name.size()) { m.prefix = t.name.substr(0, p); found = true; break; } }
  if (!found) { m.err = "this is not an Anima checkpoint (no blocks.N.self_attn.q_proj)"; return m; }
  while (s.get(m.prefix + "blocks." + std::to_string(m.blocks) + ".self_attn.q_proj.weight")) m.blocks++;
  if (m.blocks != 28 && m.blocks != 40 && !getenv("ANIMACONV_ANY")) {   // ANIMACONV_ANY: tests with a few blocks
     m.err = "unsupported block count " + std::to_string(m.blocks) + " (28 or 40 expected)"; return m; }
  static const int64_t want[10][2] = {{2048, 2048}, {2048, 2048}, {2048, 2048}, {2048, 2048}, {2048, 2048}, {2048, 1024}, {2048, 1024}, {2048, 2048}, {8192, 2048}, {2048, 8192}};   // [out, in]
  for (int b = 0; b < m.blocks; b++) {
    std::string p = m.prefix + "blocks." + std::to_string(b) + ".";
    for (int i = 0; i < 10; i++) {
      const Tensor* t = s.get(p + ORDER[i] + ".weight");
      if (!t) { m.err = "missing " + p + ORDER[i] + ".weight"; return m; }
      if (t->shape.size() != 2 || t->shape[0] != want[i][0] || t->shape[1] != want[i][1]) { m.err = "unexpected shape of " + t->name; return m; }
      if (t->dtype != "F32" && t->dtype != "F16" && t->dtype != "BF16") { m.err = "unsupported weight type " + t->dtype + " (fp8 files are not supported)"; return m; }
    }
    static const char* extra[] = {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight", "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight", "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight", "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight"};
    for (const char* e : extra) if (!s.get(p + e)) { m.err = "missing " + p + e; return m; }
  }
  return m;
}

// W [N out][K in] (row major, as stored) -> q8 records [nb][kb] of 128 B scales + 1024 B tile (see export_all.py wt16 + q8tiles)
static void q8_matrix(const float* W, int N, int K, uint8_t* out, int nthreads) {
  const int NB = N / 32, KB = K / 32; std::atomic<int> next(0);
  auto work = [&]() {
    for (int nb; (nb = next.fetch_add(1)) < NB;) {
      for (int kb = 0; kb < KB; kb++) {
        uint8_t* rec = out + ((size_t)nb * KB + kb) * 1152; float a[16 * 32 * 2];
        for (int kk = 0; kk < 16; kk++) for (int nn = 0; nn < 32; nn++) for (int e = 0; e < 2; e++)
          a[(kk * 32 + nn) * 2 + e] = h2f(f2h(W[(size_t)(nb * 32 + nn) * K + kb * 32 + kk * 2 + e]));
        float sf[32];
        for (int nn = 0; nn < 32; nn++) {
          float mx = 0; for (int kk = 0; kk < 16; kk++) for (int e = 0; e < 2; e++) mx = std::max(mx, std::fabs(a[(kk * 32 + nn) * 2 + e]));
          float s = mx / 127.0f; uint16_t h = f2h(s == 0 ? 1.0f : s); sf[nn] = h2f(h);
          memcpy(rec + (nn * 2) * 2, &h, 2); memcpy(rec + (nn * 2 + 1) * 2, &h, 2);   // halfword l = scale of column l/2
        }
        uint8_t* q = rec + 128;
        for (int kk = 0; kk < 16; kk++) for (int nn = 0; nn < 32; nn++) for (int e = 0; e < 2; e++) {
          float v = nearbyintf(a[(kk * 32 + nn) * 2 + e] / sf[nn]); v = std::min(127.0f, std::max(-127.0f, v));
          q[(kk * 32 + nn) * 2 + e] = (uint8_t)((int)v + 128);
        }
      }
    }
  };
  std::vector<std::thread> th; for (int i = 1; i < nthreads; i++) th.emplace_back(work); work(); for (auto& t : th) t.join();
}

static bool write_all(FILE* f, const void* p, size_t n) { return fwrite(p, 1, n, f) == n; }
static bool mkdirs(const std::string& d) { std::string cur; for (size_t i = 0; i <= d.size(); i++) { if (i == d.size() || d[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0700); } if (i < d.size()) cur += d[i]; } return true; }   // failures show when the files are opened

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: animaconv check SRC | convert SRC OUTDIR STUB\n"); return 2; }
  const std::string cmd = argv[1]; Src s;
  if (!open_src(argv[2], s)) { printf("error %s\n", s.err.c_str()); return 1; }
  Model m = inspect(s); if (!m.err.empty()) { printf("error %s\n", m.err.c_str()); return 1; }
  if (cmd == "check") { printf("ok blocks=%d d=2048\n", m.blocks); return 0; }
  if (cmd != "convert" || argc < 5) return 2;
  const std::string out = argv[3], stub = argv[4];
  if (!mkdirs(out)) { printf("error cannot create %s\n", out.c_str()); return 1; }
  int nth = (int)std::min<long>(8, std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN)));
  // small.bin: the tensors outside the blocks, fp16 (see extract_small.py)
  std::vector<const Tensor*> keep;
  for (auto& t : s.order) { std::string r = t.name; for (size_t p; (p = r.find("llm_adapter.blocks.")) != std::string::npos;) r.erase(p, 19); if (r.find("blocks.") == std::string::npos) keep.push_back(&t); }
  const int total = m.blocks + 1; int done = 0;
  {
    FILE* f = fopen((out + "/small.bin.tmp").c_str(), "wb"); if (!f) { printf("error cannot write %s\n", out.c_str()); return 1; }
    uint32_t cnt = keep.size(); write_all(f, &cnt, 4);
    for (const Tensor* t : keep) {
      Bytes raw; std::vector<float> v; if (!s.read(*t, raw) || !to_f32(*t, raw, v)) { printf("error cannot read %s (%s)\n", t->name.c_str(), t->dtype.c_str()); fclose(f); return 1; }
      std::string name = t->name; if (!m.prefix.empty() && name.compare(0, m.prefix.size(), m.prefix) == 0) name = name.substr(m.prefix.size()); else if (name.compare(0, 4, "net.") == 0) name = name.substr(4);
      uint32_t nl = name.size(), nd = t->shape.size(); write_all(f, &nl, 4); write_all(f, name.data(), nl); write_all(f, &nd, 4); for (int64_t d : t->shape) write_all(f, &d, 8);
      std::vector<uint16_t> h(v.size()); for (size_t i = 0; i < v.size(); i++) h[i] = f2h(v[i]); write_all(f, h.data(), h.size() * 2);
    }
    fclose(f); rename((out + "/small.bin.tmp").c_str(), (out + "/small.bin").c_str());
    printf("progress %d %d\n", ++done, total); fflush(stdout);
  }
  const float sc = (float)(1.0 / std::sqrt(128.0));
  for (int b = 0; b < m.blocks; b++) {
    char fn[600]; snprintf(fn, sizeof fn, "%s/blk%02d.bin", out.c_str(), b); std::string tmp = std::string(fn) + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb"); if (!f) { printf("error cannot write %s\n", fn); return 1; }
    const std::string p = m.prefix + "blocks." + std::to_string(b) + ".";
    for (int i = 0; i < 10; i++) {
      const Tensor* t = s.get(p + ORDER[i] + ".weight"); Bytes raw; std::vector<float> w;
      if (!s.read(*t, raw) || !to_f32(*t, raw, w)) { printf("error cannot read %s\n", t->name.c_str()); fclose(f); return 1; }
      int N = t->shape[0], K = t->shape[1]; Bytes q((size_t)(N / 32) * (K / 32) * 1152); q8_matrix(w.data(), N, K, q.data(), nth);
      write_all(f, q.data(), q.size());
    }
    float gain[512]; bool ok = true; static const char* nm[4] = {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight"};
    for (int i = 0; i < 4; i++) {
      const Tensor* t = s.get(p + nm[i]); Bytes raw; std::vector<float> v; if (!s.read(*t, raw) || !to_f32(*t, raw, v) || v.size() != 128) { ok = false; break; }
      for (int j = 0; j < 128; j++) gain[i * 128 + j] = (i == 0 || i == 2) ? v[j] * sc : v[j];
    }
    if (!ok) { printf("error bad norm weights in block %d\n", b); fclose(f); return 1; }
    write_all(f, gain, sizeof gain); static const uint8_t zero[4096 - 2048] = {0}; write_all(f, zero, sizeof zero);
    static const char* sites[3] = {"self_attn", "cross_attn", "mlp"};
    for (int si = 0; si < 3; si++) for (int k = 1; k <= 2; k++) {
      const Tensor* t = s.get(p + "adaln_modulation_" + sites[si] + "." + std::to_string(k) + ".weight"); Bytes raw; std::vector<float> v;
      if (!t || !s.read(*t, raw) || !to_f32(*t, raw, v)) { printf("error cannot read modulation of block %d\n", b); fclose(f); return 1; }
      std::vector<uint16_t> h(v.size()); for (size_t i = 0; i < v.size(); i++) h[i] = f2h(v[i]); write_all(f, h.data(), h.size() * 2);
    }
    if (fclose(f) != 0) { printf("error write failed (storage full?)\n"); return 1; }
    rename(tmp.c_str(), fn);
    printf("progress %d %d\n", ++done, total); fflush(stdout);
  }
  {   // sd.cpp stub: the checkpoint header, then zeros (a sparse file; the DiT weights are never read, the blocks above are what runs)
    Bytes hdr(s.base); if (pread(s.fd, hdr.data(), s.base, 0) != (ssize_t)s.base) { printf("error cannot read the header\n"); return 1; }
    std::string tmp = stub + ".tmp"; FILE* f = fopen(tmp.c_str(), "wb"); if (!f) { printf("error cannot write the stub\n"); return 1; }
    write_all(f, hdr.data(), hdr.size()); fflush(f); if (ftruncate(fileno(f), s.size) != 0) { printf("error cannot size the stub\n"); fclose(f); return 1; }
    fclose(f); rename(tmp.c_str(), stub.c_str());
  }
  printf("done\n"); return 0;
}
