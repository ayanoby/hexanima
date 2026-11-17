// SeedVR2-3B one-step DiT on the NPU (MODE 100 of blk.c). usage:
//   svrun PACKDIR X_T.raw COND.raw V_OUT.raw H8 W8 [FIRST LAST]   X_T, COND: f32 [16][H8][W8] (noise and scaled cond latent), V_OUT: f32 [16][H8][W8] (the DiT velocity)
// FIRST LAST: run only layers FIRST..LAST-1 (test) and write the token-order x (f32 [T][2560], text rows after) to V_OUT instead.
// env SVDUMP=dir: write x after every layer (token order, f32 [T][2560]).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <sys/ioctl.h>
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
#include "blk_meta.h"

static const int D = 2560, NCBH = 80, MID_T = 216, NL = 32, TXT = 58;
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static std::vector<float> rdf(const std::string& p, size_t n) { std::vector<float> v(n); FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); } size_t g = fread(v.data(), 4, n, f); fclose(f); if (g != n) { fprintf(stderr, "%s: got %zu want %zu\n", p.c_str(), g, n); exit(1); } return v; }
static size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }

struct Win { int h0, h1, w0, w1; };
// official make_720Pwindows_bysize / make_shifted_720Pwindows_bysize for t = 1 (seedvr2-code/models/dit_v2/window.py)
static std::vector<Win> windows(int h, int w, bool shifted) {
  double sc = sqrt((45.0 * 80.0) / ((double)h * w)); int rh = (int)nearbyint(h * sc), rw = (int)nearbyint(w * sc);
  int wh = (int)ceil(rh / 3.0), ww = (int)ceil(rw / 3.0); std::vector<Win> out;
  if (!shifted) {
    int nh = (int)ceil((double)h / wh), nw = (int)ceil((double)w / ww);
    for (int iw = 0; iw < nw; iw++) { int w0 = iw * ww, w1 = std::min((iw + 1) * ww, w); if (w1 <= w0) continue;
      for (int ih = 0; ih < nh; ih++) { int h0 = ih * wh, h1 = std::min((ih + 1) * wh, h); if (h1 > h0) out.push_back({h0, h1, w0, w1}); } }
    return out;
  }
  double sh = wh < h ? 0.5 : 0, sw = ww < w ? 0.5 : 0;
  int nh = (int)ceil((h - sh) / wh), nw = (int)ceil((w - sw) / ww); nh = sh > 0 ? nh + 1 : 1; nw = sw > 0 ? nw + 1 : 1;
  for (int iw = 0; iw < nw; iw++) { int w0 = std::max((int)((iw - sw) * ww), 0), w1 = std::min((int)((iw - sw + 1) * ww), w); if (w1 <= w0) continue;
    for (int ih = 0; ih < nh; ih++) { int h0 = std::max((int)((ih - sh) * wh), 0), h1 = std::min((int)((ih - sh + 1) * wh), h); if (h1 > h0) out.push_back({h0, h1, w0, w1}); } }
  return out;
}

struct Layout {   // one window partition: rows in the DSP order
  std::vector<Win> wins; std::vector<int> row_of_tok;   // token (y*w+x) -> row
  std::vector<int> table;                                // per window: first vid slab, vid slabs, valid rows
  int nv = 0, ns = 0, maxnvw = 0;                        // vid slabs, total slabs (nv + 2)
  std::vector<float> cs, sn;                             // rope tables [ns][2][32][32]
};
static void build(Layout& L, int h, int w, bool shifted) {
  L.wins = windows(h, w, shifted); L.row_of_tok.assign((size_t)h * w, -1); int slab = 0; L.maxnvw = 0;
  for (auto& W : L.wins) {
    int n = (W.h1 - W.h0) * (W.w1 - W.w0), nvw = (n + 31) / 32; L.table.push_back(slab); L.table.push_back(nvw); L.table.push_back(n); L.maxnvw = std::max(L.maxnvw, nvw);
    int r = 0; for (int y = W.h0; y < W.h1; y++) for (int x = W.w0; x < W.w1; x++) L.row_of_tok[(size_t)y * w + x] = slab * 32 + r++;
    slab += nvw;
  }
  L.nv = slab; L.ns = slab + 2;
  std::vector<double> f(21); for (int k = 0; k < 21; k++) f[k] = 1.0 / pow(10000.0, 2.0 * k / 42.0);
  L.cs.assign((size_t)L.ns * 2048, 1.f); L.sn.assign((size_t)L.ns * 2048, 0.f);
  auto set = [&](int row, const double* ang) { int s = row / 32, r = row % 32; for (int d = 0; d < 64; d++) { size_t i = ((size_t)s * 2 + d / 32) * 1024 + r * 32 + d % 32; L.cs[i] = (float)cos(ang[d]); L.sn[i] = (float)sin(ang[d]); } };
  for (auto& W : L.wins) for (int y = W.h0; y < W.h1; y++) for (int x = W.w0; x < W.w1; x++) {
    double a[64]; for (int d = 0; d < 21; d++) a[d] = TXT * f[d]; for (int d = 0; d < 21; d++) a[21 + d] = (y - W.h0) * f[d]; for (int d = 0; d < 21; d++) a[42 + d] = (x - W.w0) * f[d]; a[63] = 0;
    set(L.row_of_tok[(size_t)y * w + x], a); }
  for (int i = 0; i < TXT; i++) { double a[64]; for (int d = 0; d < 63; d++) a[d] = i * f[d % 21]; a[63] = 0; set(L.nv * 32 + i, a); }
}
// x32 tiles [slab][cb][32 rows][32]: element (row, c) at ((row/32*80 + c/32)*32 + row%32)*32 + c%32
static inline float* xrow(float* x, int row) { return x + (size_t)(row / 32) * NCBH * 1024 + (row % 32) * 32; }
static inline const float* xrow(const float* x, int row) { return x + (size_t)(row / 32) * NCBH * 1024 + (row % 32) * 32; }
static void put_row(float* x, int row, const float* v) { float* b = xrow(x, row); for (int cb = 0; cb < NCBH; cb++) memcpy(b + cb * 1024, v + cb * 32, 128); }
static void get_row(const float* x, int row, float* v) { const float* b = xrow(x, row); for (int cb = 0; cb < NCBH; cb++) memcpy(v + cb * 32, b + cb * 1024, 128); }
template <class F> static void par(int n, F f) { int nt = std::min(n, 4); std::vector<std::thread> th; for (int t = 0; t < nt; t++) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); }); for (auto& x : th) x.join(); }

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: svrun PACK XT COND OUT H8 W8 [FIRST LAST]\n"); return 1; }
  const std::string pack = argv[1]; const int h8 = atoi(argv[5]), w8 = atoi(argv[6]), h = h8 / 2, w = w8 / 2, T = h * w;
  const int L0 = argc > 7 ? atoi(argv[7]) : 0, L1 = argc > 8 ? atoi(argv[8]) : NL; const bool partial = argc > 8;
  setvbuf(stdout, 0, _IONBF, 0);
  double t0 = now();
  Layout LY[2]; build(LY[0], h, w, false); build(LY[1], h, w, true);
  printf("tokens %d, windows %zu / %zu, slabs %d / %d, max window slabs %d / %d\n", T, LY[0].wins.size(), LY[1].wins.size(), LY[0].ns, LY[1].ns, LY[0].maxnvw, LY[1].maxnvw);
  const int nsmax = std::max(LY[0].ns, LY[1].ns), mxw = std::max(LY[0].maxnvw, LY[1].maxnvw);
  // ---- glue (CPU): vid_in, text start
  auto vin_w = rdf(pack + "/glue/vid_in_w", (size_t)D * 132), vin_b = rdf(pack + "/glue/vid_in_b", D), txt0 = rdf(pack + "/glue/txt0", (size_t)TXT * D);
  auto xt = rdf(argv[2], (size_t)16 * h8 * w8), cond = rdf(argv[3], (size_t)16 * h8 * w8);
  std::vector<float> tok((size_t)T * D);
  par(h, [&](int ty) { std::vector<float> in(132); for (int tx = 0; tx < w; tx++) {
    for (int hh = 0; hh < 2; hh++) for (int ww = 0; ww < 2; ww++) { int p = (hh * 2 + ww) * 33; size_t o = (size_t)(2 * ty + hh) * w8 + 2 * tx + ww;
      for (int c = 0; c < 16; c++) { in[p + c] = xt[(size_t)c * h8 * w8 + o]; in[p + 16 + c] = cond[(size_t)c * h8 * w8 + o]; } in[p + 32] = 1.f; }
    float* o = &tok[((size_t)ty * w + tx) * D]; for (int n = 0; n < D; n++) { const float* wr = &vin_w[(size_t)n * 132]; float a = vin_b[n]; for (int k = 0; k < 132; k++) a += wr[k] * in[k]; o[n] = a; } } });
  printf("vid_in %.2fs\n", now() - t0);
  // ---- blob
  int meta[M_NMETA]; memset(meta, 0, sizeof meta);
  size_t off = 0; auto alloc = [&](size_t b) { size_t o = off; off += up(b); return o; };
  const size_t oX = alloc((size_t)nsmax * NCBH * 4096), oC[2] = {alloc((size_t)nsmax * 8192), alloc((size_t)nsmax * 8192)}, oS[2] = {alloc((size_t)nsmax * 8192), alloc((size_t)nsmax * 8192)};
  const size_t oH = alloc((size_t)nsmax * NCBH * 2048), oQ = alloc((size_t)nsmax * NCBH * 2048), oK = alloc((size_t)nsmax * NCBH * 2048), oV = alloc((size_t)nsmax * NCBH * 2048), oO = alloc((size_t)nsmax * NCBH * 2048), oT = alloc((size_t)nsmax * NCBH * 2048);
  const size_t oM = alloc((size_t)nsmax * MID_T * 2048), oLQ = alloc((size_t)(mxw + 2) * NCBH * 2048), oLO = alloc((size_t)(mxw + 2) * NCBH * 2048), oLU = alloc(2 * NCBH * 4096 + 2 * NCBH * 2048);
  const size_t oLD[2] = {alloc(LY[0].table.size() * 4 + 256), alloc(LY[1].table.size() * 4 + 256)};
  rpcmem_init();
  uint8_t* blob = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, off);
  const size_t WMAX = 160u << 20; uint8_t* wb[2] = {(uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WMAX), (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WMAX)};
  if (!blob || !wb[0] || !wb[1]) { fprintf(stderr, "rpcmem_alloc failed (blob %zu MB)\n", off >> 20); return 1; }
  memset(blob, 0, off);
  float* X = (float*)(blob + oX);
  for (int p = 0; p < 2; p++) { memcpy(blob + oC[p], LY[p].cs.data(), LY[p].cs.size() * 4); memcpy(blob + oS[p], LY[p].sn.data(), LY[p].sn.size() * 4); memcpy(blob + oLD[p], LY[p].table.data(), LY[p].table.size() * 4); }
  remote_rpc_control_unsigned_module dd = {CDSP_DOMAIN_ID, 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &dd, sizeof dd);
  remote_handle64 hd = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &hd); if (r) { fprintf(stderr, "mega_open 0x%x\n", r); return 1; }
  for (int i = 0; i < 2; i++) if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(wb[i]), wb[i], 0, WMAX, FASTRPC_MAP_FD)) { fprintf(stderr, "wts mmap failed\n"); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(blob), blob, 0, off, FASTRPC_MAP_FD)) { fprintf(stderr, "blob mmap failed (%zu MB)\n", off >> 20); return 1; }
  struct S_ { unsigned long long f; };
  auto sync = [&](uint8_t* p, bool end) { S_ sy = {(end ? (1ull << 2) : 0ull) | 3}; ioctl(rpcmem_to_fd(p), _IOW('b', 0, S_), &sy); };
  // ---- initial rows (layout of layer L0's parity)
  const int par0 = L0 % 2; { const Layout& Y = LY[par0]; memset(X, 0, (size_t)nsmax * NCBH * 4096);
    if (L0 == 0) { par(T, [&](int t) { put_row(X, Y.row_of_tok[t], &tok[(size_t)t * D]); }); for (int i = 0; i < TXT; i++) put_row(X, Y.nv * 32 + i, &txt0[(size_t)i * D]); }
    else { // test: x in token order from XT (T x 2560, then text rows) instead of the latents
      auto xin = rdf(argv[2], (size_t)(T + TXT) * D); par(T, [&](int t) { put_row(X, Y.row_of_tok[t], &xin[(size_t)t * D]); }); for (int i = 0; i < TXT; i++) put_row(X, Y.nv * 32 + i, &xin[(size_t)(T + i) * D]); } }
  // ---- layers
  auto load = [&](int i, uint8_t* dst) { char p[512]; snprintf(p, sizeof p, "%s/blk%02d.bin", pack.c_str(), i); FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    size_t n = 0, g; while ((g = fread(dst + n, 1, 8 << 20, f)) > 0) n += g; fclose(f); sync(dst, true); return n; };
  load(L0, wb[0]);
  double tdsp = 0, tperm = 0; const char* dump = getenv("SVDUMP");
  for (int i = L0; i < L1; i++) {
    const int pr = i % 2; const Layout& Y = LY[pr]; uint8_t* wcur = wb[(i - L0) % 2];
    std::thread pre; if (i + 1 < L1) pre = std::thread([&] { load(i + 1, wb[(i + 1 - L0) % 2]); });
    int m[M_NMETA]; memset(m, 0, sizeof m);
    m[M_X] = oX; m[M_COS] = oC[pr]; m[M_SIN] = oS[pr]; m[M_H] = oH; m[M_Q] = oQ; m[M_K] = oK; m[M_V] = oV; m[M_O] = oO; m[M_T] = oT; m[M_MID] = oM; m[M_LQ] = oLQ; m[M_LO] = oLO; m[M_LU] = oLU; m[M_LD] = oLD[pr];
    m[M_NTOK] = Y.ns * 32; m[M_NIMG] = Y.nv; m[M_NCB] = NCBH; m[M_MODE] = 100; m[M_A0] = (int)Y.wins.size(); m[M_NW] = 3;
    m[M_FIRST] = i == L0 ? 1 : 2; m[M_REL] = i == L1 - 1 ? 1 : 2; if (getenv("SVSTOP")) m[M_STOP] = atoi(getenv("SVSTOP"));
    int times[32] = {0}, rc = 0; double a = now(); sync(blob, true);
    r = mega_block(hd, m, M_NMETA, rpcmem_to_fd(wcur), rpcmem_to_fd(blob), (const uint8_t*)"\0\0\0\0\0\0\0\0", 8, times, 32, &rc);
    sync(blob, false); double b = now(); tdsp += b - a;
    if (r || rc) { fprintf(stderr, "layer %d failed r=0x%x rc=%d\n", i, r, rc); return 1; }
    if (i < 2 || i == L1 - 1) { printf("layer %d %.0f ms | stages(us):", i, (b - a) * 1e3); for (int k = 0; k < 9; k++) printf(" %d", times[k]); printf("\n"); }
    if (pre.joinable()) pre.join();
    if (dump) { std::vector<float> o((size_t)(T + TXT) * D); par(T, [&](int t) { get_row(X, Y.row_of_tok[t], &o[(size_t)t * D]); }); for (int k = 0; k < TXT; k++) get_row(X, Y.nv * 32 + k, &o[(size_t)(T + k) * D]);
      char p[512]; snprintf(p, sizeof p, "%s/x_%02d.f32", dump, i); FILE* f = fopen(p, "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f); }
    if (i + 1 < L1) {   // to the next partition
      double c = now(); const Layout& N = LY[(i + 1) % 2]; std::vector<float> nx((size_t)N.ns * NCBH * 1024, 0.f);
      par(T, [&](int t) { std::vector<float> v(D); get_row(X, Y.row_of_tok[t], v.data()); put_row(nx.data(), N.row_of_tok[t], v.data()); });
      for (int k = 0; k < TXT; k++) { std::vector<float> v(D); get_row(X, Y.nv * 32 + k, v.data()); put_row(nx.data(), N.nv * 32 + k, v.data()); }
      memcpy(X, nx.data(), nx.size() * 4); tperm += now() - c; }
  }
  printf("dsp %.2fs, permutations %.2fs\n", tdsp, tperm);
  const Layout& Y = LY[(L1 - 1) % 2];
  if (partial) { std::vector<float> o((size_t)(T + TXT) * D); par(T, [&](int t) { get_row(X, Y.row_of_tok[t], &o[(size_t)t * D]); }); for (int k = 0; k < TXT; k++) get_row(X, Y.nv * 32 + k, &o[(size_t)(T + k) * D]);
    FILE* f = fopen(argv[4], "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f); mega_close(hd); return 0; }
  // ---- final: rms * out_norm * out_scale + out_shift, Linear 2560 -> 64, unpatchify
  auto onorm = rdf(pack + "/glue/out_norm", D), oscale = rdf(pack + "/glue/out_scale", D), oshift = rdf(pack + "/glue/out_shift", D), ow = rdf(pack + "/glue/out_w", (size_t)64 * D), ob = rdf(pack + "/glue/out_b", 64);
  std::vector<float> outv((size_t)16 * h8 * w8);
  par(h, [&](int ty) { std::vector<float> v(D), y(D); for (int tx = 0; tx < w; tx++) {
    get_row(X, Y.row_of_tok[(size_t)ty * w + tx], v.data()); double ms = 0; for (int k = 0; k < D; k++) ms += (double)v[k] * v[k]; float rs = 1.f / sqrtf((float)(ms / D) + 1e-5f);
    for (int k = 0; k < D; k++) y[k] = v[k] * rs * onorm[k] * oscale[k] + oshift[k];
    for (int n = 0; n < 64; n++) { const float* wr = &ow[(size_t)n * D]; float a = ob[n]; for (int k = 0; k < D; k++) a += wr[k] * y[k];
      int hh = n / 32, ww = (n / 16) % 2, c = n % 16; outv[(size_t)c * h8 * w8 + (size_t)(2 * ty + hh) * w8 + 2 * tx + ww] = a; } } });
  FILE* f = fopen(argv[4], "wb"); fwrite(outv.data(), 4, outv.size(), f); fclose(f);
  printf("total %.2fs\n", now() - t0); mega_close(hd); return 0;
}
