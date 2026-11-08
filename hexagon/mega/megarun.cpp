#include <sys/stat.h>
#include <sys/statvfs.h>
#include <dirent.h>
// HMX block runner (replaces chainrun): holds the 28 fp16 crouton block weights in rpcmem and runs them on the DSP with the mega kernels.
// stdin: "run X_IN EMB LORA CTX X_OUT [GH GW]" -> "done <ms>"; "quit".   X_IN/X_OUT fp32 [D][T], EMB fp32[2048], LORA fp32[6144], CTX fp32 [1024][512]
// usage: megarun HMXDIR [FIRST COUNT]
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <iostream>
#include <sstream>
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
#include "blk_wlayout.h"
#include "lora_fold.h"

struct dma_buf_sync_ { unsigned long long flags; };
#define DMA_BUF_IOCTL_SYNC_ _IOW('b', 0, dma_buf_sync_)
static const int D = 2048, NCTX = 512, TMAX = 6144;
typedef float f32;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<char> readfile(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); std::vector<char> b; if (!f) return b;
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET); b.resize(n);
  if (fread(b.data(), 1, n, f) != n) b.clear();
  fclose(f); return b;
}
// MEGA_STREAM=1 (low-RAM mode): only two block buffers stay in memory; the blocks are read from flash for every run while the DSP works on the previous one
// (3-4 GB of pinned weights do not fit next to Android on an 8 GB phone). ~1.4 GB/s from UFS -> about 2.5 s per step for the 2.9B models.
static size_t read_into(const std::string& path, uint8_t* dst, size_t maxb) {   // 4 parallel preads (one stream reads ~0.9 GB/s, four ~1.5 GB/s)
  int fd = open(path.c_str(), O_RDONLY); if (fd < 0) return 0;
  struct stat st; fstat(fd, &st); size_t n = std::min((size_t)st.st_size, maxb);
  const size_t chunk = 8u << 20; std::atomic<size_t> next(0), got(0);
  auto work = [&] { for (;;) { size_t o = next.fetch_add(chunk); if (o >= n) break; size_t len = std::min(chunk, n - o), d = 0; while (d < len) { ssize_t r = pread(fd, dst + o + d, len - d, o + d); if (r <= 0) return; d += r; } got += d; } };
  std::vector<std::thread> th; for (int i = 0; i < 3; i++) th.emplace_back(work); work(); for (auto& t : th) t.join();
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); return got.load();
}
static size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }
// fp16 block (W_TOTAL) -> q8 block (WQ_TOTAL), the format export_all.py --q8 writes for the 2.9B models: every 32x32 fp16 tile (2048 B) becomes a 1152 B
// record = 64 fp16 column scales (halfword l -> column l/2, max|w|/127) + 1024 offset-binary int8 in the same order; gains and modulation weights are copied.
// LoRA-folded blocks are kept this way: half the RAM and half the LoRA cache on flash, the same kernels the 2.9B models use.
static void to_q8(const uint8_t* src, uint8_t* dst) {
  const size_t nt = W_GAIN / 2048;
  auto job = [&](size_t t0, size_t t1) {
    for (size_t t = t0; t < t1; t++) {
      const __fp16* a = (const __fp16*)(src + t * 2048); uint8_t* rec = dst + t * 1152; __fp16* sv = (__fp16*)rec; uint8_t* q = rec + 128;
      for (int c = 0; c < 32; c++) {
        float mx = 0; for (int r16 = 0; r16 < 16; r16++) for (int p = 0; p < 2; p++) mx = std::max(mx, std::fabs((float)a[(r16 * 32 + c) * 2 + p]));
        float v = mx / 127.0f; __fp16 sc = (__fp16)(v == 0 ? 1.0f : v); float sf = (float)sc;
        sv[2 * c] = sv[2 * c + 1] = sc;
        for (int r16 = 0; r16 < 16; r16++) for (int p = 0; p < 2; p++) {
          int i = (r16 * 32 + c) * 2 + p; float qv = std::nearbyint((float)a[i] / sf); qv = std::min(127.f, std::max(-127.f, qv)); q[i] = (uint8_t)((int)qv + 128);
        }
      }
    }
  };
  std::vector<std::thread> th; for (int i = 0; i < 4; i++) th.emplace_back(job, nt * i / 4, nt * (i + 1) / 4); for (auto& x : th) x.join();
  memcpy(dst + WQ_GAIN, src + W_GAIN, W_TOTAL - W_GAIN);
}

struct Layout { int meta[M_NMETA]; size_t total; };
static bool g_q8 = false;
static Layout layout(int T) {
  Layout L; memset(&L, 0, sizeof L);
  size_t sz[M_NMETA] = {0};
  sz[M_X] = (size_t)T * D * 4; sz[M_CTX] = (size_t)NCTX * 1024 * 2; sz[M_MOD] = 9 * D * 4;
  sz[M_COS] = sz[M_SIN] = (size_t)(T / 32) * 2 * 1024 * 4;
  sz[M_H] = sz[M_Q] = sz[M_K] = sz[M_V] = sz[M_O] = sz[M_T] = (size_t)T * D * 2;
  sz[M_MID] = (size_t)T * 8192 * 2; sz[M_CK] = sz[M_CV] = (size_t)NCTX * D * 2;
  static const int order[] = {M_X, M_CTX, M_MOD, M_COS, M_SIN, M_H, M_Q, M_K, M_V, M_O, M_T, M_MID, M_CK, M_CV};
  size_t off = 0;
  for (int id : order) { L.meta[id] = (int)off; off += up(sz[id]); }
#define WO(x) (g_q8 ? WQ(x) : (x))
  L.meta[M_WSQ] = WO(W_SQ); L.meta[M_WSK] = WO(W_SK); L.meta[M_WSV] = WO(W_SV); L.meta[M_WSO] = WO(W_SO); L.meta[M_WCQ] = WO(W_CQ); L.meta[M_WCK] = WO(W_CK); L.meta[M_WCV] = WO(W_CV);
  L.meta[M_WCO] = WO(W_CO); L.meta[M_WM1] = WO(W_M1); L.meta[M_WM2] = WO(W_M2); L.meta[M_GAIN] = WO(W_GAIN); L.meta[M_Q8] = g_q8;
  L.meta[M_NTOK] = T; L.meta[M_NCTX] = NCTX; L.meta[M_NW] = 3; L.meta[M_REL] = 1; L.meta[M_FIRST] = 1;
  L.total = off; return L;
}

// [D][T] fp32 -> x32 tiles [slab][cb] of 32x32 row-major (row = token, col = channel)
static void to_x32(const f32* src, f32* dst, int T) {
  for (int cb = 0; cb < D / 32; cb++) for (int c = 0; c < 32; c++) {
    const f32* row = src + (size_t)(cb * 32 + c) * T;
    for (int s = 0; s < T / 32; s++) { f32* tile = dst + ((size_t)s * (D / 32) + cb) * 1024; for (int r = 0; r < 32; r++) tile[r * 32 + c] = row[s * 32 + r]; }
  }
}
static void from_x32(const f32* src, f32* dst, int T) {
  for (int cb = 0; cb < D / 32; cb++) for (int c = 0; c < 32; c++) {
    f32* row = dst + (size_t)(cb * 32 + c) * T;
    for (int s = 0; s < T / 32; s++) { const f32* tile = src + ((size_t)s * (D / 32) + cb) * 1024; for (int r = 0; r < 32; r++) row[s * 32 + r] = tile[r * 32 + c]; }
  }
}
// ctx fp32 [1024 ch][512 tok] -> fp16 act tiles of matrix [512 rows][1024 cols]
static void to_ctx16(const f32* ct, __fp16* dst) {
  for (int c = 0; c < 1024; c++) for (int i = 0; i < NCTX; i++)
    dst[(((size_t)(i >> 5) * 32 + (c >> 5)) * 1024) + ((i & 31) >> 1) * 64 + (c & 31) * 2 + (i & 1)] = (__fp16)ct[(size_t)c * NCTX + i];
}
static void make_rope(int gh, int gw, f32* cs, f32* sn) {   // tiles [slab][j][32 rows][32 cols] (angle index d = 32 j + col)
  const int dim_h = 128 / 6 * 2, dim_t = 128 - 2 * dim_h, nt = dim_t / 2, nh = dim_h / 2;
  std::vector<double> fh(nh);
  for (int i = 0; i < nh; i++) fh[i] = 1.0 / pow(10000.0, (2.0 * i) / dim_h);
  int T = gh * gw;
  for (int tk = 0; tk < T; tk++) {
    int hh = tk / gw, ww = tk % gw;
    for (int d = 0; d < 64; d++) {
      double ang = d < nt ? 0.0 : d < nt + nh ? hh * fh[d - nt] : ww * fh[d - nt - nh];
      size_t idx = (((size_t)(tk / 32) * 2 + (d / 32)) * 1024) + (tk % 32) * 32 + (d % 32);
      cs[idx] = (f32)cos(ang); sn[idx] = (f32)sin(ang);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: megarun HMXDIR [FIRST COUNT]\n"); return 1; }
  std::string dir = argv[1]; int first = argc > 2 ? atoi(argv[2]) : 0, count = argc > 3 ? atoi(argv[3]) : 28;
  rpcmem_init();
  const bool stream = getenv("MEGA_STREAM") != nullptr; const int nbuf = stream ? std::min(2, count) : count;
  std::vector<uint8_t*> wts(nbuf);
  { char p[512]; snprintf(p, sizeof p, "%s/blk%02d.bin", dir.c_str(), first); struct stat st; if (!stat(p, &st) && (size_t)st.st_size == WQ_TOTAL) g_q8 = true; }   // q8 weight files are recognised by size
  size_t wtot = g_q8 ? WQ_TOTAL : W_TOTAL, wmodw = g_q8 ? WQ_MODW : W_MODW;
  // LoRA cache (MEGA_LORA_CACHE=<dir>): blocks already folded with this exact LoRA set (files, sizes, dates, strengths) are read from <dir>/<key>/
  // instead of folding again; after a fold the blocks are written there in the background (only the last set is kept). Cached blocks are always q8.
  std::string cdir; bool cached = false;
  const char* lv0 = getenv("MEGA_LORA");
  if (lv0 && *lv0 && getenv("MEGA_LORA_CACHE")) {
    std::string k = dir + "|" + lv0 + "|" + (getenv("MEGA_LORA_S") ? getenv("MEGA_LORA_S") : "");
    std::string all = lv0;
    for (size_t p = 0; p < all.size();) {
      size_t q = all.find('|', p); if (q == std::string::npos) q = all.size();
      std::string item = all.substr(p, q - p); p = q + 1; size_t s = item.rfind('*'); if (s != std::string::npos) item.resize(s);
      struct stat st; if (!stat(item.c_str(), &st)) k += "|" + std::to_string((long long)st.st_size) + ":" + std::to_string((long long)st.st_mtime);
    }
    uint64_t h = 1469598103934665603ull; for (unsigned char c : k) { h ^= c; h *= 1099511628211ull; }
    char hx[24]; snprintf(hx, sizeof hx, "%016llx", (unsigned long long)h);
    cdir = std::string(getenv("MEGA_LORA_CACHE")) + "/" + hx;
    cached = true;
    for (int i = 0; i < count && cached; i++) { char p[600]; snprintf(p, sizeof p, "%s/blk%02d.bin", cdir.c_str(), first + i); struct stat st; cached = !stat(p, &st) && (size_t)st.st_size == WQ_TOTAL; }
    if (cached) { g_q8 = true; wtot = WQ_TOTAL; wmodw = WQ_MODW; }
  }
  double t0 = now();
  for (int i = 0; i < nbuf; i++) {
    char p[600]; snprintf(p, sizeof p, "%s/blk%02d.bin", cached ? cdir.c_str() : dir.c_str(), first + i);
    wts[i] = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, wtot);
    if (stream) { if (!wts[i]) { fprintf(stderr, "rpcmem_alloc failed\n"); return 1; } continue; }   // streaming: the buffers are filled for every run
    int fd = open(p, O_RDONLY); if (fd < 0 || !wts[i]) { fprintf(stderr, "cannot load %s\n", p); return 1; }
    size_t got = 0; while (got < wtot) { ssize_t r = read(fd, wts[i] + got, wtot - got); if (r <= 0) break; got += r; }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd);
    if (got != wtot) { fprintf(stderr, "short read %s\n", p); return 1; }
  }
  fprintf(stderr, "megarun: weights loaded %.1fs%s\n", now() - t0, cached ? " (LoRA cache)" : "");
  if (cached) fprintf(stderr, "megarun: LoRA from cache %s, blocks %d..%d\n", cdir.c_str(), first, first + count - 1);
  bool folded = false;
  // user LoRAs (Anima 2B), folded into the fp16 weights in RAM (lora_fold.h): MEGA_LORA = "file*strength|file*strength|..." (a bare file takes MEGA_LORA_S, default 1)
  if (stream && !cached && lv0 && *lv0) fprintf(stderr, "megarun: LoRA skipped in low-memory mode (no folded copy in the LoRA cache yet: load once with the mode off)\n");
  const std::string sdir = cached ? cdir : dir;   // where the streaming loader reads the blocks
  if (const char* lv = cached || stream ? nullptr : getenv("MEGA_LORA")) {
    const float s0 = getenv("MEGA_LORA_S") ? (float)atof(getenv("MEGA_LORA_S")) : 1.f;
    std::string all = lv;
    for (size_t p = 0; p < all.size();) {
      size_t q = all.find('|', p); if (q == std::string::npos) q = all.size();
      std::string item = all.substr(p, q - p); p = q + 1; if (item.empty()) continue;
      float s = s0; size_t st = item.rfind('*'); if (st != std::string::npos) { s = (float)atof(item.c_str() + st + 1); item.resize(st); }
      LoraFile L; double tl = now();
      if (!L.open(item.c_str())) { fprintf(stderr, "megarun: LoRA skipped: cannot read %s\n", item.c_str()); continue; }
      struct stat st40; char p40[512]; snprintf(p40, sizeof p40, "%s/blk39.bin", dir.c_str());
      const int model_blocks = stat(p40, &st40) == 0 ? 40 : 28, lora_n = lora_blocks(L);   // 2B LoRAs on 2.9B models (and back) through FROM2B
      int nm = 0; bool fit = true;
      for (int i = 0; i < count && fit; i++) { int src = lora_source(first + i, model_blocks, lora_n); if (src < 0) continue; int k = lora_fold_block(L, s, src, wts[i], g_q8); if (k < 0) fit = false; else nm += k; }
      if (nm > 0) folded = true;
      if (!fit) fprintf(stderr, "megarun: LoRA %s does not fit this model (shapes), partly applied\n", item.c_str());
      fprintf(stderr, "megarun: LoRA %s: %d matrices of blocks %d..%d, strength %.2f, %.1fs\n", item.c_str(), nm, first, first + count - 1, s, now() - tl);
    }
  }
  if (folded && !g_q8) {   // fp16 model with LoRAs: keep the folded blocks as q8 (see to_q8)
    double tq = now();
    for (int i = 0; i < count; i++) {
      uint8_t* q = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WQ_TOTAL);
      if (!q) { fprintf(stderr, "rpcmem_alloc failed (q8)\n"); return 1; }
      to_q8(wts[i], q); rpcmem_free(wts[i]); wts[i] = q;
    }
    g_q8 = true; wtot = WQ_TOTAL; wmodw = WQ_MODW;
    fprintf(stderr, "megarun: LoRA blocks kept as q8, %.1fs\n", now() - tq);
  }
  struct remote_rpc_control_unsigned_module ud = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof ud);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h);
  if (r) { fprintf(stderr, "mega_open 0x%x\n", r); return 1; }
  for (int i = 0; i < nbuf; i++) {   // map once (a DSP process takes at most ~13 of these buffers: run several megarun processes for 28 blocks)
    int fd = rpcmem_to_fd(wts[i]);
    dma_buf_sync_ sy = {(1ull << 2) | 3};   // DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW: clean CPU caches so the DSP sees the weights
    ioctl(fd, DMA_BUF_IOCTL_SYNC_, &sy);
    int mr = fastrpc_mmap(CDSP_DOMAIN_ID, fd, wts[i], 0, wmodw, FASTRPC_MAP_FD);
    if (mr) { fprintf(stderr, "fastrpc_mmap block %d rc=%d\n", first + i, mr); return 1; }
  }
  uint8_t* blob = nullptr; int blobT = 0, blobfd = -1;   // activations: one rpcmem buffer mapped into the DSP; the CPU/DSP cache sync is done by hand (dma-buf sync here, dccleaninva on the DSP)

  // make the CPU-written weights visible to the DSP (clean CPU caches), then map every buffer into the DSP once (extended address space: 4 GB do not fit in the normal one)
  for (int i = 0; i < nbuf; i++) {
    int fd = rpcmem_to_fd(wts[i]);
    dma_buf_sync_ sy = {(1ull << 2) | 3};   // DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW
    ioctl(fd, DMA_BUF_IOCTL_SYNC_, &sy);
  }
  if (folded && !cdir.empty()) {   // save the folded blocks for the next load (the DSP only reads them, so a CPU thread can copy them out meanwhile)
    std::thread([cdir, wts, wtot, first, count]() {
      const std::string root = cdir.substr(0, cdir.rfind('/')), mine = cdir.substr(cdir.rfind('/') + 1);
      mkdir(root.c_str(), 0700);
      if (DIR* d = opendir(root.c_str())) {   // keep only this LoRA set: other keys are removed
        for (struct dirent* e; (e = readdir(d));) {
          std::string n = e->d_name; if (n == "." || n == ".." || n == mine) continue;
          std::string sub = root + "/" + n;
          if (DIR* s = opendir(sub.c_str())) { for (struct dirent* f; (f = readdir(s));) if (f->d_name[0] != '.') unlink((sub + "/" + f->d_name).c_str()); closedir(s); }
          rmdir(sub.c_str());
        }
        closedir(d);
      }
      mkdir(cdir.c_str(), 0700);
      struct statvfs vs; if (statvfs(cdir.c_str(), &vs) || (double)vs.f_bavail * vs.f_frsize < (double)wtot * count + 2e9) { fprintf(stderr, "megarun: LoRA cache skipped (storage)\n"); return; }
      double tw = now();
      for (int i = 0; i < count; i++) {
        char p[600], t[620]; snprintf(p, sizeof p, "%s/blk%02d.bin", cdir.c_str(), first + i); snprintf(t, sizeof t, "%s.tmp", p);
        int fd = open(t, O_WRONLY | O_CREAT | O_TRUNC, 0600); if (fd < 0) return;
        size_t put = 0; while (put < wtot) { ssize_t r = write(fd, wts[i] + put, wtot - put); if (r <= 0) break; put += r; }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd);
        if (put != wtot) { unlink(t); return; }
        rename(t, p);
      }
      fprintf(stderr, "megarun: LoRA cache written, blocks %d..%d, %.1fs\n", first, first + count - 1, now() - tw);
    }).detach();
  }
  printf("ready\n"); fflush(stdout);
  int cur_gh = -1, cur_gw = -1;
  std::vector<f32> xin((size_t)D * TMAX), emb(D), lora(6144), ctxf((size_t)1024 * NCTX), mod((size_t)count * 9 * D);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd; ss >> cmd;
    if (cmd == "quit") break;
    if (cmd != "run") continue;
    std::string xi, em, lo, cx, xo; int gh = 48, gw = 32; unsigned skipmask = 0; ss >> xi >> em >> lo >> cx >> xo >> gh >> gw >> skipmask;
    int T = gh * gw;
    auto skipped = [&](int k) { return first + k < 32 && ((skipmask >> (first + k)) & 1); };   // the mask has 32 bits: blocks 32+ (2.9B models) are never skipped
    auto xb = readfile(xi), eb = readfile(em), lb = readfile(lo), cb = readfile(cx);
    if (T % 32 || T > TMAX || xb.size() != (size_t)D * T * 4 || eb.size() != D * 4 || lb.size() != 6144 * 4 || cb.size() != (size_t)1024 * NCTX * 4) { fprintf(stderr, "bad input T=%d\n", T); printf("error\n"); fflush(stdout); continue; }
    double ts = now(), t_in = ts;
    if (T > blobT) {
      if (blob) { fastrpc_munmap(CDSP_DOMAIN_ID, blobfd, blob, 0); rpcmem_free(blob); }
      Layout Lb = layout(T); blobT = T;
      blob = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, Lb.total);
      if (!blob) { fprintf(stderr, "blob alloc failed\n"); printf("error\n"); fflush(stdout); continue; }
      blobfd = rpcmem_to_fd(blob); memset(blob, 0, Lb.total); cur_gh = -1;
      if (fastrpc_mmap(CDSP_DOMAIN_ID, blobfd, blob, 0, Lb.total, FASTRPC_MAP_FD)) { fprintf(stderr, "blob mmap failed\n"); printf("error\n"); fflush(stdout); continue; }
    }
    Layout L = layout(T);
    if (gh != cur_gh || gw != cur_gw) { make_rope(gh, gw, (f32*)(blob + L.meta[M_COS]), (f32*)(blob + L.meta[M_SIN])); cur_gh = gh; cur_gw = gw; }
    to_ctx16((const f32*)cb.data(), (__fp16*)(blob + L.meta[M_CTX]));
    bool xin32 = xi.size() > 4 && xi.compare(xi.size() - 4, 4, ".x32") == 0;
    if (xin32) memcpy(blob + L.meta[M_X], xb.data(), (size_t)D * T * 4); else to_x32((const f32*)xb.data(), (f32*)(blob + L.meta[M_X]), T);
    { dma_buf_sync_ sy = {(1ull << 2) | 3}; ioctl(blobfd, DMA_BUF_IOCTL_SYNC_, &sy); }   // END|RW: flush CPU writes
    // modulation vectors of all blocks on a helper thread; block k is waited for before its DSP call
    const f32* e = (const f32*)eb.data(); const f32* lr = (const f32*)lb.data();
    std::atomic<int> mdone(0);
    std::vector<f32> sv(D);
    for (int i = 0; i < D; i++) sv[i] = e[i] / (1.f + expf(-e[i]));
    auto compute_mod = [&](int k, const uint8_t* wk) {
      std::vector<f32> h1(256);
      const __fp16* mw = (const __fp16*)(wk + wmodw);
      for (int site = 0; site < 3; site++) {
        const __fp16* A = mw + (size_t)site * (256 * 2048 + 6144 * 256); const __fp16* B = A + 256 * 2048;
        for (int j = 0; j < 256; j++) { float a = 0; const __fp16* row = A + (size_t)j * 2048; for (int i = 0; i < D; i++) a += (float)row[i] * sv[i]; h1[j] = a; }
        f32* out = &mod[((size_t)k * 9 + site * 3) * D];
        for (int o = 0; o < 6144; o++) { float a = lr[o]; const __fp16* row = B + (size_t)o * 256; for (int j = 0; j < 256; j++) a += (float)row[j] * h1[j]; out[o] = a; }
      }
    };
    std::vector<int> order;   // blocks to run (skipped ones pass x through)
    for (int k = 0; k < count; k++) if (!skipped(k)) order.push_back(k);
    std::mutex smu; std::condition_variable scv; int sready = -1, sfreed = -2; std::atomic<bool> sfail(false);
    std::thread mt;
    if (!stream) mt = std::thread([&] { for (int k = 0; k < count; k++) { compute_mod(k, wts[k]); mdone.store(k + 1, std::memory_order_release); } });
    else mt = std::thread([&] {   // loader: block i+1 is read (and its modulation computed) while the DSP runs block i
      for (size_t oi = 0; oi < order.size(); oi++) {
        { std::unique_lock<std::mutex> lk(smu); scv.wait(lk, [&] { return sfreed >= (int)oi - 2; }); }
        int k = order[oi]; uint8_t* b = wts[oi % nbuf]; char p[600]; snprintf(p, sizeof p, "%s/blk%02d.bin", sdir.c_str(), first + k);
        if (read_into(p, b, wtot) != wtot) { fprintf(stderr, "cannot read %s\n", p); sfail = true; }
        else { dma_buf_sync_ sy = {(1ull << 2) | 3}; ioctl(rpcmem_to_fd(b), DMA_BUF_IOCTL_SYNC_, &sy); compute_mod(k, b); }
        { std::lock_guard<std::mutex> lk(smu); sready = (int)oi; } scv.notify_all();
        if (sfail) break;
      }
    });
    double t_conv = now();
    bool ok = true; int times[32];
    int firstRun = -1, lastRun = -1;   // blocks in skipmask are skipped (x passes through unchanged)
    for (int k = 0; k < count; k++) if (!skipped(k)) { if (firstRun < 0) firstRun = k; lastRun = k; }
    for (int k = 0; k < count && ok; k++) {
      if (skipped(k)) continue;
      int oi = (int)(std::find(order.begin(), order.end(), k) - order.begin());
      if (!stream) while (mdone.load(std::memory_order_acquire) <= k) {}
      else { std::unique_lock<std::mutex> lk(smu); scv.wait(lk, [&] { return sready >= oi || sfail.load(); }); if (sfail) { ok = false; break; } }
      uint8_t* wk = stream ? wts[oi % nbuf] : wts[k];
      L.meta[M_FIRST] = k == firstRun; L.meta[M_REL] = k == lastRun;   // VTCM, pool and power votes are held across the blocks of one run, released after the last
      double tm1 = now();
      int rc = 0; int rr = mega_block(h, L.meta, M_NMETA, rpcmem_to_fd(wk), blobfd, (const uint8_t*)&mod[(size_t)k * 9 * D], 9 * D * 4, times, 32, &rc);
      if (getenv("MEGA_DBG") && k == 5) { char lp[256]; snprintf(lp, sizeof lp, "megarun_%d.log", first); FILE* lf = fopen(lp, "a"); if (lf) { fprintf(lf, "blk5 stages(us):"); for (int q = 0; q < 25; q++) fprintf(lf, " %d", times[q]); fprintf(lf, " | call %.1f ms\n", (now() - tm1) * 1e3); fclose(lf); } }
      if (rr || rc) { fprintf(stderr, "block %d failed r=0x%x rc=%d\n", first + k, rr, rc); ok = false; }
      if (stream) { std::lock_guard<std::mutex> lk(smu); sfreed = oi; scv.notify_all(); }
    }
    double t_dsp = now();
    if (stream) { std::lock_guard<std::mutex> lk(smu); sfreed = 1 << 30; scv.notify_all(); }   // let the loader finish when a block failed
    mt.join();
    if (!ok) { printf("error\n"); fflush(stdout); continue; }
    { dma_buf_sync_ sy = {(0ull << 2) | 3}; ioctl(blobfd, DMA_BUF_IOCTL_SYNC_, &sy); }   // START|RW: invalidate CPU caches before reading the result
    bool xout32 = xo.size() > 4 && xo.compare(xo.size() - 4, 4, ".x32") == 0;
    if (!xout32) from_x32((const f32*)(blob + L.meta[M_X]), xin.data(), T);
    FILE* f = fopen(xo.c_str(), "wb"); fwrite(xout32 ? (const void*)(blob + L.meta[M_X]) : (const void*)xin.data(), 4, (size_t)D * T, f); fclose(f);
    double t_end = now();
    if (getenv("MEGA_DBG")) { char lp[256]; snprintf(lp, sizeof lp, "megarun_%d.log", first); FILE* lf = fopen(lp, "a"); if (lf) { fprintf(lf, "run: in+conv %.1f ms, blocks %.1f ms, out %.1f ms, total %.1f ms\n", (t_conv - ts) * 1e3, (t_dsp - t_conv) * 1e3, (t_end - t_dsp) * 1e3, (t_end - ts) * 1e3); fclose(lf); } }
    printf("done %.0f\n", (t_end - ts) * 1000); fflush(stdout);
  }
  mega_close(h);
  return 0;
}
