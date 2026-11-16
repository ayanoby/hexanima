// MiniMax-H3-Image DiT sampler on the phone: 6 steps x 50 blocks on the NPU (MODE 50), glue on the CPU.
// usage: h3run MODELDIR LH LW NOISE.f32 TE.f32 NTEXT OUT_LAT.f32 [WORKDIR]
//   MODELDIR: blk00..49.bin r0.bin r1.bin + glue files (see anima/hexanima/h3/export_h3_glue.py)
//   NOISE.f32: (24, LH, LW) fp32 N(0,1); TE.f32: (NTEXT, 5120) fp32 text-encoder hidden state after layer 50. Output: (24, LH, LW) latents (still normalised).
#include <atomic>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <future>
#include <sys/ioctl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
extern "C" {
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
}
#include "blk_meta.h"
typedef float f32;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const int HID = 5376, QW = 7168, FF = 14336, TXTD = 5120, PD = 96, NBLK = 50;
static const size_t WT = 385660928u, WTA = 487940096u, MODB = 258048;   // WTA: weights buffer size = the larger text-encoder layer pack
static const double SIG[7] = {1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0};
static std::string MD;
static const size_t DITSZ = 385660928u, LORASZ = 26607616u;   // block pack size, LoRA pack size (the LoRA pack sits right behind the block pack in the weights buffer)
static std::string LD;   // LoRA dir (H3_LORA): loraNN.bin, lorar0/1.bin; empty = no LoRA
static std::vector<double> SG; static std::vector<int> HEADS;
template <class T> static std::vector<T> load(const std::string &name, size_t n = 0) {
  FILE *f = fopen((MD + "/" + name).c_str(), "rb"); if (!f) { fprintf(stderr, "missing %s\n", name.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET); std::vector<T> v(sz / sizeof(T)); if (fread(v.data(), 1, sz, f) != sz) { fprintf(stderr, "short %s\n", name.c_str()); exit(1); } fclose(f);
  if (n && v.size() != n) { fprintf(stderr, "%s: %zu elements, want %zu\n", name.c_str(), v.size(), n); exit(1); }
  return v;
}
static void parfor(int n, const std::function<void(int)> &fn, int nth = 8) {
  std::atomic<int> nxt(0); std::vector<std::thread> th;
  for (int t = 0; t < nth; t++) th.emplace_back([&] { for (int i; (i = nxt++) < n;) fn(i); });
  for (auto &t : th) t.join();
}
// read a whole block file into rpcmem with several parallel preads (flash likes 4 readers)
struct dma_sync_ { unsigned long long flags; };
static void sync_fd(int fd, unsigned long long f) { dma_sync_ s = {f}; ioctl(fd, _IOW('b', 0, dma_sync_), &s); }
static void read_file(const std::string &path, uint8_t *dst, size_t size) {
  int fd = open(path.c_str(), O_RDONLY); if (fd < 0) { fprintf(stderr, "open %s\n", path.c_str()); exit(1); }
  const int NT = 4; std::vector<std::thread> th; const size_t chunk = 8u << 20; std::atomic<size_t> nxt(0);
  for (int t = 0; t < NT; t++) th.emplace_back([&] { for (;;) { size_t o = nxt.fetch_add(chunk); if (o >= size) break; size_t len = std::min(chunk, size - o); size_t got = 0; while (got < len) { ssize_t r = pread(fd, dst + o + got, len - got, o + got); if (r <= 0) { fprintf(stderr, "pread fail\n"); exit(1); } got += r; } } });
  for (auto &t : th) t.join(); close(fd);
  int bfd = rpcmem_to_fd(dst); (void)bfd;
}
static size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }
static void rms_row(const f32 *x, const f32 *w, f32 *y, int n, float eps) { double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i]; float r = 1.0f / sqrtf((float)(s / n) + eps); for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i]; }

int main(int argc, char **argv) {
  if (argc < 8) { fprintf(stderr, "usage: h3run MODELDIR LH LW NOISE TE NTEXT OUT [WORKDIR]\n"); return 1; }
  MD = argv[1]; const int lh = atoi(argv[2]), lw = atoi(argv[3]); int nt = atoi(argv[6]); const char *outp = argv[7];
  const std::string a5 = argv[5]; const bool use_ids = nt == 0 || (a5.size() > 4 && a5.substr(a5.size() - 4) == ".ids");   // TE argument: NTEXT 0 or a .ids file = int32 token ids (run the text encoder on the NPU), else fp32 hidden states
  std::vector<int32_t> ids; if (use_ids) { FILE *f = fopen(argv[5], "rb"); if (!f) { fprintf(stderr, "ids\n"); return 1; } fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(sz / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) return 1; fclose(f); nt = (int)ids.size(); }
  const std::string TED = getenv("H3_TE") ? getenv("H3_TE") : MD + "/../te";
  SG.assign(SIG, SIG + 7); HEADS = {0, 1, 2, 3, 4, 5}; if (getenv("H3_LORA")) LD = getenv("H3_LORA");
  auto csv = [](const char *t, auto conv) { std::vector<double> v; std::string x = t; size_t p = 0; while (p < x.size()) { size_t q = x.find(',', p); if (q == std::string::npos) q = x.size(); v.push_back(conv(x.substr(p, q - p).c_str())); p = q + 1; } return v; };
  if (getenv("H3_SIG")) SG = csv(getenv("H3_SIG"), [](const char *z) { return atof(z); });
  if (getenv("H3_HEADS")) { HEADS.clear(); for (double d : csv(getenv("H3_HEADS"), [](const char *z) { return atof(z); })) HEADS.push_back((int)d); } else if (SG.size() != 7) { HEADS.clear(); for (size_t i = 0; i + 1 < SG.size(); i++) HEADS.push_back((int)i); }
  const int stopb = getenv("H3_BLOCKS") ? atoi(getenv("H3_BLOCKS")) : NBLK, nsteps = getenv("H3_STEPS") ? atoi(getenv("H3_STEPS")) : (int)SG.size() - 1;
  const int ni = (lh / 2) * (lw / 2); if (ni % 32) { fprintf(stderr, "image rows %d must be a multiple of 32\n", ni); return 1; }
  const int ntok = (ni + nt + 31) / 32 * 32, ns = ntok / 32, nimgslabs = ni / 32, nvalid = ni + nt;
  setvbuf(stdout, 0, _IONBF, 0); double T0 = now();
  // ---------------- glue weights
  auto proj_w = load<f32>("proj_in_w.f32", (size_t)HID * PD), proj_b = load<f32>("proj_in_b.f32", HID);
  auto ctx_q = load<int8_t>("ctx_q.i8", (size_t)HID * TXTD); auto ctx_s = load<f32>("ctx_s.f32", HID), ctx_b = load<f32>("ctx_b.f32", HID);
  auto table = load<f32>("table.f32", 1025 * 8), adaw = load<f32>("adaln_w.f32", (size_t)NBLK * 64512 * 8), adab = load<f32>("adaln_b.f32", (size_t)NBLK * 64512);
  auto now_w = load<f32>("nout_w.f32", (size_t)10752 * 8), now_b = load<f32>("nout_b.f32", 10752), now_n = load<f32>("nout_norm.f32", HID);
  auto hw = load<f32>("heads_w.f32", (size_t)6 * PD * HID), hb = load<f32>("heads_b.f32", 6 * PD), rfin = load<f32>("rfin.f32", HID);
  auto H = load<f32>("H256.f32", 65536); for (auto &v : H) v /= 16.0f;   // Hn, symmetric
  auto noise = std::vector<f32>(); if (!strncmp(argv[4], "seed:", 5)) { std::mt19937_64 rng(strtoull(argv[4] + 5, 0, 10)); std::normal_distribution<float> nd(0.f, 1.f); noise.resize((size_t)24 * lh * lw); for (auto &v : noise) v = nd(rng); } else { FILE *f = fopen(argv[4], "rb"); noise.resize((size_t)24 * lh * lw); if (!f || fread(noise.data(), 4, noise.size(), f) != noise.size()) { fprintf(stderr, "noise\n"); return 1; } fclose(f); }
  std::vector<f32> te((size_t)nt * TXTD); if (!use_ids) { FILE *f = fopen(argv[5], "rb"); if (!f || fread(te.data(), 4, te.size(), f) != te.size()) { fprintf(stderr, "te\n"); return 1; } fclose(f); }
  printf("glue loaded %.1f s\n", now() - T0);
  // ---------------- DSP buffers
  int meta[M_NMETA]; memset(meta, 0, sizeof meta);
  const bool hasL = !LD.empty(); const int ntokT = (nt + 31) / 32 * 32; const size_t bigFF = std::max((size_t)ntok * FF, (size_t)ntokT * 25600) * 2;
  struct { int id; size_t bytes; } ent[] = {{M_X, (size_t)ntok * HID * 4}, {M_COS, (size_t)ns * 8192}, {M_SIN, (size_t)ns * 8192}, {M_HN, 131072}, {M_H, (size_t)ntok * HID * 2}, {M_HR, bigFF},
    {M_LQ, hasL ? (size_t)(ntok / 32) * 12 * 2048 : 4096}, {M_LO, hasL ? (size_t)(ntok / 32) * 4 * 2048 : 4096}, {M_LU, hasL ? (size_t)(ntok / 32) * 4 * 2048 : 4096}, {M_LD, hasL ? (size_t)(ntok / 32) * 4 * 2048 : 4096}, {M_DLT, hasL ? (size_t)ntok * 28672 * 2 : 4096},
    {M_Q, std::max((size_t)ntok * QW, (size_t)ntokT * 8192) * 2}, {M_K, (size_t)ntok * QW * 2}, {M_V, (size_t)ntok * QW * 2}, {M_O, std::max((size_t)ntok * QW, (size_t)ntokT * 8192) * 2}, {M_T, (size_t)ntok * HID * 2}, {M_MID, bigFF}};
  size_t off = 0; for (auto &e : ent) { meta[e.id] = (int)off; off += up(e.bytes); }
  rpcmem_init();
  uint8_t *wb[2]; for (int i = 0; i < 2; i++) { wb[i] = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WTA); if (!wb[i]) { fprintf(stderr, "rpcmem weights\n"); return 1; } }
  uint8_t *blob = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, off); if (!blob) { fprintf(stderr, "rpcmem blob\n"); return 1; } memset(blob, 0, off);
  auto load_blk = [&](int n, uint8_t *dst) { char nm[32]; snprintf(nm, sizeof nm, "/blk%02d.bin", n); read_file(MD + nm, dst, DITSZ); if (!LD.empty()) { snprintf(nm, sizeof nm, "/lora%02d.bin", n); read_file(LD + nm, dst + DITSZ, LORASZ); } };
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; if (mega_open(mega_URI CDSP_DOMAIN, &h)) { fprintf(stderr, "mega_open failed\n"); return 1; }
  for (int i = 0; i < 2; i++) if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(wb[i]), wb[i], 0, WTA, FASTRPC_MAP_FD)) { fprintf(stderr, "mmap w\n"); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(blob), blob, 0, off, FASTRPC_MAP_FD)) { fprintf(stderr, "mmap blob\n"); return 1; }
  { auto hn = load<uint8_t>("hn.bin", 131072); memcpy(blob + meta[M_HN], hn.data(), 131072); }
  meta[M_MODE] = 50; meta[M_NCB] = 168; meta[M_NW] = 3;
  const int fdb = rpcmem_to_fd(blob);
  // x32 tiles <-> rows
  auto put_rows = [&](const f32 *rows, int n, int hid = HID) { f32 *X = (f32 *)(blob + meta[M_X]); memset(X, 0, ent[0].bytes); const int ncb = hid / 32;
    for (int r = 0; r < n; r++) for (int c = 0; c < hid; c++) X[((size_t)(r / 32) * ncb + c / 32) * 1024 + (r % 32) * 32 + c % 32] = rows[(size_t)r * hid + c]; };
  auto get_rows = [&](f32 *rows, int n, int hid = HID) { sync_fd(fdb, 3); const f32 *X = (const f32 *)(blob + meta[M_X]); const int ncb = hid / 32;
    for (int r = 0; r < n; r++) for (int c = 0; c < hid; c++) rows[(size_t)r * hid + c] = X[((size_t)(r / 32) * ncb + c / 32) * 1024 + (r % 32) * 32 + c % 32]; };
  auto put_tab = [&](int id, const std::vector<f32> &t) { f32 *T = (f32 *)(blob + meta[id]); memset(T, 0, ent[id == M_COS ? 1 : 2].bytes);
    for (int r = 0; r < ns * 32; r++) for (int c = 0; c < 64; c++) T[((size_t)(r / 32) * 2 + c / 32) * 1024 + (r % 32) * 32 + c % 32] = r < (int)(t.size() / 64) ? t[(size_t)r * 64 + c] : (id == M_COS ? 1.0f : 0.0f); };
  // one block call; weights buffer index wi
  auto run_block = [&](int wi, const std::vector<f32> &mod, bool first, bool last, int nvalid_, int nimg_slabs, int ntok_) {
    meta[M_NTOK] = ntok_; meta[M_LORA] = (meta[M_MODE] == 50 && !LD.empty()) ? 1 : 0; meta[M_NVALID] = nvalid_; meta[M_NIMG] = nimg_slabs; meta[M_FIRST] = first; meta[M_REL] = last;
    if (first) sync_fd(fdb, 7);
    sync_fd(rpcmem_to_fd(wb[wi]), 7);
    int times[32]; int rc = 0; int r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wb[wi]), fdb, (const uint8_t *)mod.data(), (int)(mod.size() * 4), times, 32, &rc);
    if (r || rc) { fprintf(stderr, "mega_block r=0x%x rc=%d\n", r, rc); exit(1); } };
  // ---------------- text encoder on the NPU (Qwen3-VL layers 0..49, MODE 51): embeddings -> 50 layers -> hidden state after layer 50
  if (use_ids) {
    double t0 = now(); std::vector<f32> x((size_t)nt * TXTD); { FILE *f = fopen((TED + "/embed.bf16").c_str(), "rb"); if (!f) { fprintf(stderr, "embed.bf16\n"); return 1; } std::vector<uint16_t> row(TXTD);
      for (int r = 0; r < nt; r++) { fseek(f, (long)ids[r] * TXTD * 2, SEEK_SET); if (fread(row.data(), 2, TXTD, f) != (size_t)TXTD) return 1; for (int i = 0; i < TXTD; i++) { uint32_t u = (uint32_t)row[i] << 16; memcpy(&x[(size_t)r * TXTD + i], &u, 4); } } fclose(f); }
    std::vector<f32> cs((size_t)nt * 64), sn((size_t)nt * 64);
    for (int r = 0; r < nt; r++) for (int c = 0; c < 64; c++) { double a = (float)r * (double)(float)pow(5000000.0, -(double)c / 64.0); cs[(size_t)r * 64 + c] = cosf((float)a); sn[(size_t)r * 64 + c] = sinf((float)a); }
    put_tab(M_COS, cs); put_tab(M_SIN, sn); put_rows(x.data(), nt, TXTD); meta[M_MODE] = 51; meta[M_NCB] = 160;
    const int nl = getenv("H3_TE_LAYERS") ? atoi(getenv("H3_TE_LAYERS")) : 50; std::vector<f32> dm(16, 0.f); std::future<void> pre;
    { char nm[32]; snprintf(nm, sizeof nm, "/lay%02d.bin", 0); std::string p = TED + nm; pre = std::async(std::launch::async, [p, &wb] { read_file(p, wb[0], 487940096u); }); }
    for (int l = 0; l < nl; l++) {
      pre.get();
      if (l + 1 < nl) { char nm[32]; snprintf(nm, sizeof nm, "/lay%02d.bin", l + 1); std::string p = TED + nm; uint8_t *dst = wb[(l + 1) & 1]; pre = std::async(std::launch::async, [p, dst] { read_file(p, dst, 487940096u); }); }
      run_block(l & 1, dm, l == 0, l == nl - 1, nt, 0, ntokT);
    }
    get_rows(te.data(), nt, TXTD); meta[M_MODE] = 50; meta[M_NCB] = 168;
    printf("text encoder %.2f s (%d layers)\n", now() - t0, nl);
    if (getenv("H3_TE_OUT")) { FILE *f = fopen(getenv("H3_TE_OUT"), "wb"); fwrite(te.data(), 4, te.size(), f); fclose(f); } }
  // ---------------- text: context embedder (CPU, ConvRot) + 2 refiner blocks (NPU, same kernel, no modulation / rope) + final norm
  std::vector<f32> txt((size_t)nt * HID);
  { double t0 = now(); std::vector<f32> xr((size_t)nt * TXTD);
    for (int r = 0; r < nt; r++) for (int b = 0; b < TXTD / 256; b++) for (int j = 0; j < 256; j++) { float s = 0; for (int i = 0; i < 256; i++) s += te[(size_t)r * TXTD + b * 256 + i] * H[(size_t)i * 256 + j]; xr[(size_t)r * TXTD + b * 256 + j] = s; }
    parfor(HID, [&](int o) { const int8_t *q = &ctx_q[(size_t)o * TXTD]; for (int r = 0; r < nt; r++) { float s = 0; const f32 *x = &xr[(size_t)r * TXTD]; for (int i = 0; i < TXTD; i++) s += x[i] * (float)q[i]; txt[(size_t)r * HID + o] = s * ctx_s[o] + ctx_b[o]; } });
    const int ntk = (nt + 31) / 32 * 32, nsr = ntk / 32; (void)nsr;
    std::vector<f32> mod(MODB / 4, 0.f); for (int i = 0; i < HID; i++) mod[6 * HID + 2 * HID + i] = mod[6 * HID + 5 * HID + i] = 1.0f;   // kind 1: shift 0 scale 0 gate 1
    std::vector<f32> one(0); put_tab(M_COS, one); put_tab(M_SIN, one);   // cos 1 sin 0 everywhere = no rope
    put_rows(txt.data(), nt);
    for (int i = 0; i < 2; i++) { read_file(MD + "/r" + std::to_string(i) + ".bin", wb[i & 1], DITSZ); if (!LD.empty()) read_file(LD + "/lorar" + std::to_string(i) + ".bin", wb[i & 1] + DITSZ, LORASZ); run_block(i & 1, mod, i == 0, i == 1, nt, 0, ntk); }
    get_rows(txt.data(), nt);
    for (int r = 0; r < nt; r++) rms_row(&txt[(size_t)r * HID], rfin.data(), &txt[(size_t)r * HID], HID, 1e-5f);
    printf("text path %.2f s\n", now() - t0); }
  // ---------------- positions -> rope tables (image rows first, then text), engine channel order (i < 48 real, 48..63 identity)
  { std::vector<f32> cs((size_t)(ni + nt) * 64), sn((size_t)(ni + nt) * 64);
    auto grid = [&](int dim, int k) { double sq = sqrt((double)lh * lw), ratio = dim / sq, left = (1.0 - ratio) / 2.0; return (left + k * ratio / (dim / 2)) * 32.0; };
    for (int r = 0; r < ni + nt; r++) {
      double p[3]; if (r < ni) { p[0] = nt; p[1] = grid(lh, r / (lw / 2)); p[2] = grid(lw, r % (lw / 2)); } else { p[0] = r - ni; p[1] = p[2] = 0; }
      for (int c = 0; c < 64; c++) { if (c >= 48) { cs[(size_t)r * 64 + c] = 1; sn[(size_t)r * 64 + c] = 0; continue; }
        double a = (float)p[c / 16] * (double)(float)pow(10000.0, -(double)(c % 16) / 16.0); cs[(size_t)r * 64 + c] = cosf((float)a); sn[(size_t)r * 64 + c] = sinf((float)a); } }
    put_tab(M_COS, cs); put_tab(M_SIN, sn); }
  // ---------------- sampling
  std::vector<f32> x_img((size_t)ni * PD);   // patchified latents
  for (int ph = 0; ph < lh / 2; ph++) for (int pw = 0; pw < lw / 2; pw++) for (int c = 0; c < 24; c++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
    x_img[(size_t)(ph * (lw / 2) + pw) * PD + c * 4 + a * 2 + b] = noise[((size_t)c * lh + ph * 2 + a) * lw + pw * 2 + b];
  std::vector<f32> rows((size_t)(ni + nt) * HID), mod(MODB / 4);
  std::future<void> pre;
  for (int step = 0; step < nsteps; step++) {
    double ts = now(); const double t = 1.0 - SG[step]; f32 F[8];
    { double r = t * 1024.0; int i0 = std::min((int)r, 1023); double fr = r - i0; for (int k = 0; k < 8; k++) F[k] = (float)(table[i0 * 8 + k] * (1 - fr) + table[(i0 + 1) * 8 + k] * fr); }
    parfor(ni, [&](int r) { for (int o = 0; o < HID; o++) { float s = proj_b[o]; const f32 *w = &proj_w[(size_t)o * PD], *x = &x_img[(size_t)r * PD]; for (int i = 0; i < PD; i++) s += w[i] * x[i]; rows[(size_t)r * HID + o] = s; } });
    memcpy(&rows[(size_t)ni * HID], txt.data(), (size_t)nt * HID * 4);
    put_rows(rows.data(), ni + nt);
    if (step == 0) { pre = std::async(std::launch::async, [&] { load_blk(0, wb[0]); }); }
    double tw = 0, tc = 0;
    for (int b = 0; b < stopb; b++) {
      double a0 = now(); pre.get(); tw += now() - a0;
      if (b + 1 < stopb) { uint8_t *dst = wb[(b + 1) & 1]; const int nb = b + 1; pre = std::async(std::launch::async, [&, dst, nb] { load_blk(nb, dst); }); }
      else if (step + 1 < nsteps) { pre = std::async(std::launch::async, [&] { load_blk(0, wb[0]); }); }
      for (int k = 0; k < 2; k++) for (int c = 0; c < 6; c++) for (int i = 0; i < HID; i++) { const size_t row = (size_t)(k * 6 + c) * HID + i; const f32 *w = &adaw[((size_t)b * 64512 + row) * 8]; float s = adab[(size_t)b * 64512 + row]; for (int q = 0; q < 8; q++) s += w[q] * F[q]; mod[row] = s; }
      double c0 = now(); run_block(b & 1, mod, b == 0, b == stopb - 1, nvalid, nimgslabs, ntok); tc += now() - c0;
    }
    get_rows(rows.data(), ni + nt);
    f32 sc[HID], sh[HID]; for (int i = 0; i < HID; i++) { float a = now_b[i], c = now_b[HID + i]; for (int q = 0; q < 8; q++) { a += now_w[(size_t)i * 8 + q] * F[q]; c += now_w[(size_t)(HID + i) * 8 + q] * F[q]; } sh[i] = a; sc[i] = c; }
    parfor(ni, [&](int r) { std::vector<f32> hn(HID); rms_row(&rows[(size_t)r * HID], now_n.data(), hn.data(), HID, 1e-5f); for (int i = 0; i < HID; i++) hn[i] = hn[i] * (1 + sc[i]) + sh[i];
      for (int o = 0; o < PD; o++) { float s = hb[HEADS[step] * PD + o]; const f32 *w = &hw[((size_t)HEADS[step] * PD + o) * HID]; for (int i = 0; i < HID; i++) s += w[i] * hn[i]; x_img[(size_t)r * PD + o] += (float)(SG[step] - SG[step + 1]) * s; } });
    double sd = 0, mn = 0; for (float v : x_img) { mn += v; sd += (double)v * v; } mn /= x_img.size(); sd = sqrt(sd / x_img.size() - mn * mn);
    printf("step %d: %.2f s (weights wait %.2f s, blocks %.2f s) x mean %.3f std %.3f\n", step, now() - ts, tw, tc, mn, sd);
    if (stopb < NBLK) break;
  }
  // unpatchify -> latents (24, lh, lw)
  std::vector<f32> lat((size_t)24 * lh * lw);
  for (int ph = 0; ph < lh / 2; ph++) for (int pw = 0; pw < lw / 2; pw++) for (int c = 0; c < 24; c++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
    lat[((size_t)c * lh + ph * 2 + a) * lw + pw * 2 + b] = x_img[(size_t)(ph * (lw / 2) + pw) * PD + c * 4 + a * 2 + b];
  if (getenv("H3_DENORM")) {   // latents*std+mean = what the VAE decoder takes (vae/config.json)
    static const float LM[24] = {0.858090341f, -0.960659146f, 1.06616402f, -0.509032547f, -0.272758186f, -1.36754143f, -0.255325496f, -0.269075543f, -0.537684083f, -0.0464097299f, 0.665737033f, 0.196901277f, -0.546060801f, -0.403534204f, -0.236830249f, 0.259284526f, -0.301339447f, 0.211341992f, -1.12068486f, 0.358193338f, -0.0422514379f, 0.260482997f, 0.228640929f, 0.705603182f}, LS[24] = {1.22237742f, 1.27672637f, 1.68317747f, 1.75494552f, 1.56362164f, 2.19414353f, 0.965313792f, 1.0569886f, 0.841948926f, 0.772995293f, 1.89559376f, 0.946841836f, 0.799680948f, 0.449889004f, 0.719739974f, 0.693629324f, 2.96109509f, 2.76941991f, 3.04961848f, 2.10880542f, 3.27622628f, 3.1627357f, 2.2816813f, 2.61278439f};
    for (int c = 0; c < 24; c++) for (size_t i = 0; i < (size_t)lh * lw; i++) lat[(size_t)c * lh * lw + i] = lat[(size_t)c * lh * lw + i] * LS[c] + LM[c]; }
  FILE *f = fopen(outp, "wb"); fwrite(lat.data(), 4, lat.size(), f); fclose(f);
  printf("finished %.1f s\n", now() - T0); mega_close(h); return 0;
}
