// MiniMax-H3 video VAE decoder (ViT, 36 blocks, MODE 82) on the phone.
// usage: vaerun DIR T LH LW LAT.f32 OUT.f32   DIR: glue/ vbNN.bin (anima/h3/vae/export_vae.py); LAT (24,T,LH,LW) = the DiT latents (normalised), OUT (3, 4T, 16LH, 16LW) fp32 = decoder output (ImageNet-normalised pixels)
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <future>
#include <string>
#include <sys/ioctl.h>
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
static const int D = 2048, PD = 3072, NBLK = 36; static const size_t WT = 67355136u;
static std::string MD;
template <class T> static std::vector<T> load(const std::string &name, size_t n = 0) {
  FILE *f = fopen((MD + "/" + name).c_str(), "rb"); if (!f) { fprintf(stderr, "missing %s\n", name.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET); std::vector<T> v(sz / sizeof(T)); if (fread(v.data(), 1, sz, f) != sz) { fprintf(stderr, "short %s\n", name.c_str()); exit(1); } fclose(f);
  if (n && v.size() != n) { fprintf(stderr, "%s: %zu elements, want %zu\n", name.c_str(), v.size(), n); exit(1); }
  return v;
}
static void parfor(int n, const std::function<void(int)> &fn, int nth = 8) { std::atomic<int> nxt(0); std::vector<std::thread> th; for (int t = 0; t < nth; t++) th.emplace_back([&] { for (int i; (i = nxt++) < n;) fn(i); }); for (auto &t : th) t.join(); }
struct dma_sync_ { unsigned long long flags; };
static void sync_fd(int fd, unsigned long long f) { dma_sync_ s = {f}; ioctl(fd, _IOW('b', 0, dma_sync_), &s); }
static void read_file(const std::string &path, uint8_t *dst, size_t size) {
  int fd = open(path.c_str(), O_RDONLY); if (fd < 0) { fprintf(stderr, "open %s\n", path.c_str()); exit(1); }
  std::vector<std::thread> th; const size_t chunk = 8u << 20; std::atomic<size_t> nxt(0);
  for (int t = 0; t < 4; t++) th.emplace_back([&] { for (;;) { size_t o = nxt.fetch_add(chunk); if (o >= size) break; size_t len = std::min(chunk, size - o), got = 0; while (got < len) { ssize_t r = pread(fd, dst + o + got, len - got, o + got); if (r <= 0) { fprintf(stderr, "read %s\n", path.c_str()); exit(1); } got += r; } } });
  for (auto &t : th) t.join(); close(fd);
}
static size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }

// Full decode as the official AutoencoderKLLegacy.decode_base with the bundled settings (clip_length 17, token_drop 3, decoder_tiling 256 px / overlap 64, isolated frames off):
//  time: latents are padded (last frame repeated) to a multiple of 5 after +3, chunk i = latents [5i, 5i+7) -> 28 frames; frames [3,20) of it (17, the first 5 blended with the kept tail of the previous chunk) are output, frames [23,28) are kept as the tail; total = 17 nc + 5 minus the padding frames
//  space: tiles of 256 px (16 latents) with overlaps >= 64 px, blended along y then x as in tiled_decode; each tile is one ViT pass over 7 x 16 x 16 tokens (+ 4 register + 1 cls)
// usage: vaerun DIR T LH LW LAT OUT [FRAMES]   OUT = fp32 [F][3][H][W] ImageNet-normalised pixels (frames appended chunk by chunk)
static void split_tiles(int len, int tile, int ov_min, std::vector<int> &idx, std::vector<int> &ln, std::vector<int> &ov) {
  idx = {0}; ln = {len}; ov.clear(); if (tile >= len) return;
  int Nn = (len + tile - 1) / tile, remaining; std::vector<int> o;
  for (;;) { o.assign(Nn - 1, ov_min); remaining = tile * Nn - ov_min * (Nn - 1) - len; if (remaining < 0) Nn++; else break; }
  for (int i = 0; i < remaining / 16; i++) o[i % (Nn - 1)] += 16;
  idx = {0}; for (int i = 0; i < Nn - 1; i++) idx.push_back(idx.back() + tile - o[i]); ln.assign(Nn, tile); ov = o;
}
typedef std::vector<f32> Vf;
// blend (official): a = previous, b = current; along the dimension of size n with stride st inside [outer][n][inner]; result = b with its first ext entries replaced by a_tail * (1 - p/ext) + b_head * (p/ext)
int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage: vaerun DIR T LH LW LAT OUT [FRAMES]\n"); return 1; }
  MD = argv[1]; const int T = atoi(argv[2]), LH = atoi(argv[3]), LW = atoi(argv[4]); const int want_frames = argc > 7 ? atoi(argv[7]) : 0; setvbuf(stdout, 0, _IONBF, 0); double T0 = now();
  const int TOK = 5, DROP = 3, OV = (-DROP % TOK + TOK) % TOK, CLIPT = TOK + OV, CHF = TOK * 4, PRE = 3, FOV = std::max(OV * 4 - PRE, 0);
  int pseudo = T + DROP, pad = 0; if (pseudo % TOK) { pad = TOK - pseudo % TOK; pseudo += pad; }
  const int nc = pseudo / TOK - 1; if (nc < 1) { fprintf(stderr, "T=%d: the official chunking needs T >= 7 (T = 5n + 2, n >= 1)\n", T); return 1; }
  int pad_frames = 0; for (int k = 0; k < pad; k++) pad_frames += ((T + k) % TOK == 0) ? 1 : 4;   // 17 % 4 == 1 for the frame that ends a chunk
  const int total_frames = 17 * nc + 5 - pad_frames, out_frames = want_frames > 0 ? std::min(want_frames, total_frames) : total_frames, skip = total_frames - out_frames;   // trim_output keeps the LAST frames (causal encoder)
  auto xe_w = load<f32>("glue/xe_w.f32", (size_t)D * 24), xe_b = load<f32>("glue/xe_b.f32", D), reg = load<f32>("glue/reg.f32", 4 * D), no_w = load<f32>("glue/no_w.f32", D), no_b = load<f32>("glue/no_b.f32", D);
  auto po_w = load<__fp16>("glue/po_w.f16", (size_t)PD * D); auto po_b = load<f32>("glue/po_b.f32", PD), lm = load<f32>("glue/lat_mean.f32", 24), ls = load<f32>("glue/lat_std.f32", 24);
  Vf lat((size_t)24 * T * LH * LW); { FILE *f = fopen(argv[5], "rb"); if (!f || fread(lat.data(), 4, lat.size(), f) != lat.size()) { fprintf(stderr, "lat %s\n", argv[5]); return 1; } fclose(f); }
  const int Tp = T + pad, H = 16 * LH, W = 16 * LW;
  auto zat = [&](int c, int t, int y, int x) { return lat[(((size_t)c * T + std::min(t, T - 1)) * LH + y) * LW + x]; };   // padding repeats the last latent frame
  std::vector<int> yi, yl, yo, xi, xl, xo; split_tiles(H, 256, 64, yi, yl, yo); split_tiles(W, 256, 64, xi, xl, xo);
  const int ni = yi.size(), nj = xi.size(), nt = ni * nj, tlh = yl[0] / 16, tlw = xl[0] / 16, nl = CLIPT * tlh * tlw, N = nl + 5, ntok = (N + 31) / 32 * 32, ns = ntok / 32;
  printf("T %d -> %d chunks (pad %d tokens / %d frames), %d frames out; %dx%d px: %dx%d tiles of %dx%d latents, %d tokens per pass\n", T, nc, pad, pad_frames, out_frames, W, H, ni, nj, tlh, tlw, N);
  int meta[M_NMETA]; memset(meta, 0, sizeof meta);
  struct { int id; size_t bytes; } ent[] = {{M_X, (size_t)ntok * D * 4}, {M_COS, (size_t)ns * 4096}, {M_SIN, (size_t)ns * 4096}, {M_HN, 131072}, {M_H, (size_t)ntok * D * 2}, {M_HR, (size_t)ntok * 4 * D * 2},
    {M_LQ, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}, {M_DLT, 4096}, {M_Q, (size_t)ntok * D * 2}, {M_K, (size_t)ntok * D * 2}, {M_V, (size_t)ntok * D * 2}, {M_O, (size_t)ntok * D * 2}, {M_T, (size_t)ntok * D * 2}, {M_MID, (size_t)ntok * 4 * D * 2}};
  size_t off = 0; for (auto &e : ent) { meta[e.id] = (int)off; off += up(e.bytes); }
  rpcmem_init();
  uint8_t *wb[2]; for (int i = 0; i < 2; i++) { wb[i] = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WT); if (!wb[i]) { fprintf(stderr, "rpcmem weights\n"); return 1; } }
  uint8_t *blob = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, off); if (!blob) { fprintf(stderr, "rpcmem blob (%zu MB)\n", off >> 20); return 1; } memset(blob, 0, off);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; if (mega_open(mega_URI CDSP_DOMAIN, &h)) { fprintf(stderr, "mega_open failed\n"); return 1; }
  for (int i = 0; i < 2; i++) if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(wb[i]), wb[i], 0, WT, FASTRPC_MAP_FD)) { fprintf(stderr, "mmap w\n"); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(blob), blob, 0, off, FASTRPC_MAP_FD)) { fprintf(stderr, "mmap blob\n"); return 1; }
  { auto hn = load<uint8_t>("glue/hn.bin", 131072); memcpy(blob + meta[M_HN], hn.data(), 131072); }
  { f32 *cs = (f32 *)(blob + meta[M_COS]), *sn = (f32 *)(blob + meta[M_SIN]); const int dims[3] = {CLIPT, tlh, tlw};   // pair i < 24 = axis (i / 8) coordinate x 2 pi x 100^(-(i % 8) / 8); pairs 24..31 and the suffix tokens: identity
    for (int r = 0; r < ntok; r++) for (int c = 0; c < 32; c++) { float a = 0; if (r < nl && c < 24) { const int ax = c / 8, k = c % 8; int idx[3] = {r / (tlh * tlw), (r / tlw) % tlh, r % tlw};
        const double co = ((idx[ax] + 0.5) / dims[ax]) * 2.0 - 1.0; a = (float)(2.0 * M_PI * co * pow(100.0, -k / 8.0)); }
      const size_t o = (size_t)(r / 32) * 1024 + (r % 32) * 32 + c; cs[o] = cosf(a); sn[o] = sinf(a); } }
  const int fdb = rpcmem_to_fd(blob);
  meta[M_MODE] = 82; meta[M_NCB] = 64; meta[M_NW] = 3; meta[M_NTOK] = ntok; meta[M_NVALID] = N; meta[M_NIMG] = 0; meta[M_STOP] = 0; meta[M_FMT] = 0; meta[M_LORA] = 0; meta[M_FIRST] = 1; meta[M_REL] = 1;
  std::vector<f32> mod(64, 0.f); auto path = [&](int l) { char nm[32]; snprintf(nm, sizeof nm, "/vb%02d.bin", l); return MD + nm; };
  const size_t XB = (size_t)ntok * D; std::vector<Vf> X(nt, Vf(XB));
  FILE *fo = fopen(argv[6], "wb"); if (!fo) { fprintf(stderr, "open %s\n", argv[6]); return 1; }
  const bool pm1 = getenv("VAE_PM1") != nullptr; Vf tail; int written = 0, seen = 0;   // frames produced so far (before the skip), the kept 5-frame tail [3][5][H][W]
  auto emit = [&](const Vf &fr, int nfr) {   // fr [3][nfr][H][W] -> frames appended as [F][3][H][W]
    for (int t = 0; t < nfr; t++, seen++) { if (seen < skip || written >= out_frames) continue; Vf o((size_t)3 * H * W); for (int c = 0; c < 3; c++) memcpy(&o[(size_t)c * H * W], &fr[((size_t)c * nfr + t) * H * W], (size_t)H * W * 4);
      if (pm1) { static const float mn[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f}; for (int c = 0; c < 3; c++) for (size_t k = 0; k < (size_t)H * W; k++) { float v = o[(size_t)c * H * W + k] * sd[c] + mn[c]; v = v < 0.f ? 0.f : v > 1.f ? 1.f : v; o[(size_t)c * H * W + k] = v * 2.f - 1.f; } }   // VAE_PM1=1: the app's raw video format, [-1, 1]
      fwrite(o.data(), 4, o.size(), fo); written++; } };
  std::future<void> pre;
  for (int ch = 0; ch < nc; ch++) {
    double tc0 = now();
    // embeddings of every tile of this chunk
    for (int ti = 0; ti < ni; ti++) for (int tj = 0; tj < nj; tj++) { Vf &x = X[ti * nj + tj]; std::fill(x.begin(), x.end(), 0.f); const int y0 = yi[ti] / 16, x0 = xi[tj] / 16;
      parfor(nl, [&](int t) { f32 z[24]; const int tt = t / (tlh * tlw), hh = (t / tlw) % tlh, ww = t % tlw; for (int c = 0; c < 24; c++) z[c] = zat(c, TOK * ch + tt, y0 + hh, x0 + ww) * ls[c] + lm[c];
        for (int i = 0; i < D; i++) { float a = xe_b[i]; const f32 *w = &xe_w[(size_t)i * 24]; for (int c = 0; c < 24; c++) a += w[c] * z[c]; x[((size_t)(t / 32) * 64 + i / 32) * 1024 + (t % 32) * 32 + i % 32] = a; } });
      for (int r = 0; r < 4; r++) for (int i = 0; i < D; i++) x[((size_t)((nl + r) / 32) * 64 + i / 32) * 1024 + ((nl + r) % 32) * 32 + i % 32] = reg[r * D + i]; }
    pre = std::async(std::launch::async, [&] { read_file(path(0), wb[0], WT); });
    for (int l = 0; l < NBLK; l++) {
      pre.get(); if (l + 1 < NBLK) { uint8_t *dst = wb[(l + 1) & 1]; std::string p = path(l + 1); pre = std::async(std::launch::async, [p, dst] { read_file(p, dst, WT); }); }
      sync_fd(rpcmem_to_fd(wb[l & 1]), 7);
      for (int u = 0; u < nt; u++) {
        memcpy(blob + meta[M_X], X[u].data(), XB * 4); sync_fd(fdb, 7);
        int times[32]; int rc = 0, r = 0;
        for (int attempt = 0; attempt < 8; attempt++) { rc = 0; r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wb[l & 1]), fdb, (const uint8_t *)mod.data(), (int)(mod.size() * 4), times, 32, &rc);
          if (!r && rc == -77) { fprintf(stderr, "HMX self-test failed, retry %d\n", attempt + 1); usleep(400000); continue; } break; }
        if (r || rc) { fprintf(stderr, "mega_block chunk %d block %d tile %d r=0x%x rc=%d\n", ch, l, u, r, rc); return 1; }
        sync_fd(fdb, 3); memcpy(X[u].data(), blob + meta[M_X], XB * 4);
      }
    }
    printf("chunk %d: blocks %.1f s\n", ch, now() - tc0);
    // tail on the CPU: LayerNorm + proj_out + unpack -> tile pixels [3][28][tile h][tile w]
    const int CF = CLIPT * 4; std::vector<Vf> tile(nt); const int th = yl[0], tw = xl[0];
    for (int u = 0; u < nt; u++) { tile[u].assign((size_t)3 * CF * th * tw, 0.f); const Vf &x = X[u]; Vf &o = tile[u];
      parfor(nl, [&](int t) { std::vector<f32> xv(D), y(D); for (int c = 0; c < D; c++) xv[c] = x[((size_t)(t / 32) * 64 + c / 32) * 1024 + (t % 32) * 32 + c % 32];
        double m = 0, v = 0; for (int c = 0; c < D; c++) m += xv[c]; m /= D; for (int c = 0; c < D; c++) v += (xv[c] - m) * (xv[c] - m); v /= D; const float rs = 1.f / sqrtf((float)v + 1e-5f);
        for (int c = 0; c < D; c++) y[c] = ((xv[c] - (float)m) * rs) * no_w[c] + no_b[c];
        const int tt = t / (tlh * tlw), hh = (t / tlw) % tlh, ww = t % tlw;
        for (int oo = 0; oo < PD; oo++) { const __fp16 *w = &po_w[(size_t)oo * D]; float a = po_b[oo]; for (int c = 0; c < D; c++) a += (float)w[c] * y[c];
          const int cc = oo / 1024, pt = (oo / 256) % 4, i = (oo / 16) % 16, j = oo % 16; o[(((size_t)cc * CF + tt * 4 + pt) * th + hh * 16 + i) * tw + ww * 16 + j] = a; } }, 8); }
    // spatial assembly (tiled_decode): blend along y with the raw tile above, along x with the raw tile on the left, then cut the overlaps at the far edges
    Vf vid((size_t)3 * CF * H * W);
    auto raw = [&](int ti, int tj) -> const Vf & { return tile[ti * nj + tj]; };
    for (int ti = 0; ti < ni; ti++) for (int tj = 0; tj < nj; tj++) {
      Vf t = raw(ti, tj); const int hh = th, ww = tw;
      if (ti > 0) { const Vf &a = raw(ti - 1, tj); const int e = std::min(yo[ti - 1], hh); for (int c = 0; c < 3 * CF; c++) for (int p = 0; p < e; p++) { const float wa = 1.f - (float)p / e, wb = (float)p / e;
          for (int xx = 0; xx < ww; xx++) { const size_t ib = ((size_t)c * hh + p) * ww + xx, ia = ((size_t)c * hh + (hh - e + p)) * ww + xx; t[ib] = a[ia] * wa + t[ib] * wb; } } }
      if (tj > 0) { const Vf &a = raw(ti, tj - 1); const int e = std::min(xo[tj - 1], ww); for (int c = 0; c < 3 * CF; c++) for (int yy = 0; yy < hh; yy++) for (int p = 0; p < e; p++) { const float wa = 1.f - (float)p / e, wb = (float)p / e;
          const size_t ib = ((size_t)c * hh + yy) * ww + p, ia = ((size_t)c * hh + yy) * ww + (ww - e + p); t[ib] = a[ia] * wa + t[ib] * wb; } }
      const int keh = ti < ni - 1 ? hh - yo[ti] : hh, kew = tj < nj - 1 ? ww - xo[tj] : ww;
      int oy = 0, ox = 0; for (int k = 0; k < ti; k++) oy += yl[k] - yo[k]; for (int k = 0; k < tj; k++) ox += xl[k] - xo[k];
      for (int c = 0; c < 3 * CF; c++) for (int yy = 0; yy < keh; yy++) memcpy(&vid[((size_t)c * H + oy + yy) * W + ox], &t[((size_t)c * hh + yy) * ww], (size_t)kew * 4); }
    // temporal assembly: frames [3, 20) (17) of the chunk, the first FOV blended with the kept tail; frames [23, 28) are the new tail
    auto frames = [&](int f0, int n) { Vf o((size_t)3 * n * H * W); for (int c = 0; c < 3; c++) memcpy(&o[(size_t)c * n * H * W], &vid[((size_t)c * CF + f0) * H * W], (size_t)n * H * W * 4); return o; };
    Vf cur = frames(PRE, CHF - PRE);   // [3][17][H][W]
    const int n17 = CHF - PRE;
    if (!tail.empty()) for (int c = 0; c < 3; c++) for (int p = 0; p < FOV; p++) { const float wa = 1.f - (float)p / FOV, wb = (float)p / FOV; float *b = &cur[((size_t)c * n17 + p) * H * W]; const float *a = &tail[((size_t)c * 5 + (5 - FOV + p)) * H * W];
        for (size_t k = 0; k < (size_t)H * W; k++) b[k] = a[k] * wa + b[k] * wb; }
    emit(cur, n17); tail = frames(CHF + PRE, CF - CHF - PRE);
    printf("chunk %d done %.1f s, %d frames written\n", ch, now() - tc0, written);
  }
  emit(tail, 5); fclose(fo);
  printf("done %.1f s total, %d frames (%dx%d)\n", now() - T0, written, W, H); return 0;
}
