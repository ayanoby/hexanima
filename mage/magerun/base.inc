// magerun: Mage-Flow-Edit-Turbo on the Hexagon NPU (derived from qwenrun). See ../NOTES.md
// MODELDIR: glue.bin embd.q4k te/te00..te32.bin ref/{proj,ref0,ref1}.bin dit/blk00..blk41.bin
#include <arm_neon.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
extern "C" {
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
}
#include "blk_meta.h"

typedef float f32;
struct dma_buf_sync_ { unsigned long long flags; };
#define DMA_BUF_IOCTL_SYNC_ _IOW('b', 0, dma_buf_sync_)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static size_t up4k(size_t x) { return (x + 4095) & ~(size_t)4095; }

template <class F> static void par_for(int n, F f) {
  int nt = n < 4 ? n : 4; if (nt < 2) { for (int i = 0; i < n; i++) f(i); return; }
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
  for (auto& x : th) x.join();
}

// ------------------------------------------------------------------ rpcmem buffers
// A DSP process can hold only ~1.5 GB of fastrpc_mmap'd buffers (measured), but mapping costs ~0.6 ms per 128 MB: weights live in rpcmem and are mapped on demand (LRU window).
struct Buf { uint8_t* p = nullptr; int fd = -1; size_t sz = 0; bool mapped = false, pinned = false; unsigned long long use = 0; };
static std::vector<Buf*> g_all; static size_t g_mapped = 0; static unsigned long long g_tick = 0; static const size_t MAP_BUDGET = 1000u << 20;
static bool map_buf(Buf& b) {
  b.use = ++g_tick; if (b.mapped) return true;
  while (g_mapped + b.sz > MAP_BUDGET) {
    Buf* v = nullptr; for (Buf* x : g_all) if (x->mapped && !x->pinned && (!v || x->use < v->use)) v = x;
    if (!v) break; fastrpc_munmap(CDSP_DOMAIN_ID, v->fd, v->p, 0); v->mapped = false; g_mapped -= v->sz;
  }
  { int rr = fastrpc_mmap(CDSP_DOMAIN_ID, b.fd, b.p, 0, b.sz, FASTRPC_MAP_FD); if (rr) { fprintf(stderr, "fastrpc_mmap %zu failed rc=%d errno=%d fd=%d (mapped %zu MB)\n", b.sz, rr, errno, b.fd, g_mapped >> 20); return false; } }
  b.mapped = true; g_mapped += b.sz; return true;
}
static void unmap_buf(Buf& b) { if (b.mapped) { fastrpc_munmap(CDSP_DOMAIN_ID, b.fd, b.p, 0); b.mapped = false; g_mapped -= b.sz; } }
static Buf make_buf(size_t sz, bool pin = false) {   // pinned buffers are mapped now and never evicted
  Buf b; b.sz = sz; b.p = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, sz);
  if (!b.p) { fprintf(stderr, "rpcmem_alloc %zu failed\n", sz); return b; }
  b.fd = rpcmem_to_fd(b.p); b.pinned = pin;
  if (pin && !map_buf(b)) { b.p = nullptr; }
  return b;
}
static void sync_clean(const Buf& b) { dma_buf_sync_ s = {(1ull << 2) | 3}; ioctl(b.fd, DMA_BUF_IOCTL_SYNC_, &s); }   // END|RW: CPU writes become visible to the DSP
static void sync_inval(const Buf& b) { dma_buf_sync_ s = {(0ull << 2) | 3}; ioctl(b.fd, DMA_BUF_IOCTL_SYNC_, &s); }  // START|RW before the CPU reads DSP results
static size_t read_into(const std::string& path, uint8_t* dst, size_t maxb, bool keep = false) {   // 4 parallel preads: one stream reads ~0.9 GB/s from the phone's UFS, four reach ~1.5 GB/s
  int fd = open(path.c_str(), O_RDONLY); if (fd < 0) return 0;
  struct stat st; fstat(fd, &st); size_t n = (size_t)st.st_size < maxb ? (size_t)st.st_size : maxb;
  const size_t chunk = 8u << 20; std::atomic<size_t> next(0), got(0);
  auto work = [&] { for (;;) { size_t o = next.fetch_add(chunk); if (o >= n) break; size_t len = std::min(chunk, n - o), d = 0; while (d < len) { ssize_t r = pread(fd, dst + o + d, len - d, o + d); if (r <= 0) return; d += r; } got += d; } };
  std::vector<std::thread> th; const int nrd = getenv("MAGE_RD") ? atoi(getenv("MAGE_RD")) : 3; for (int i = 0; i < nrd; i++) th.emplace_back(work); work(); for (auto& t : th) t.join();
  if (!keep) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // keep = leave the pages in the kernel's page cache (it drops them by itself under memory pressure)
  close(fd); return got.load();
}
// stream a list of pack files through the two g_te buffers: a loader thread reads file i+1 while use(i) runs on the DSP
#include <functional>
static int g_keep_blocks = 0; static double g_waitio = 0;   // seconds the consumer waited for the next pack to be read
static std::string g_ovl;   // MOSS_LORA=dir: user-LoRA patches (blkNN.ovl) laid over the DiT packs' rank-128 LoRA slots after each read
static void apply_ovl(const std::string& pack, uint8_t* dst, size_t maxb) {
  if (g_ovl.empty() || pack.find("/dit/") == std::string::npos) return;
  std::string nm = pack.substr(pack.rfind('/') + 1); nm = g_ovl + "/" + nm.substr(0, nm.size() - 4) + ".ovl";
  FILE* f = fopen(nm.c_str(), "rb"); if (!f) { fprintf(stderr, "no overlay %s\n", nm.c_str()); return; }
  uint32_t n = 0; if (fread(&n, 4, 1, f) != 1) { fclose(f); return; }
  for (uint32_t i = 0; i < n; i++) { uint64_t o; uint32_t l; if (fread(&o, 8, 1, f) != 1 || fread(&l, 4, 1, f) != 1 || o + l > maxb || fread(dst + o, 1, l, f) != l) break; }
  fclose(f);
}   // the first g_keep_blocks DiT packs stay in the page cache between steps (their reads become memcpy)
static bool stream_files(const std::vector<std::string>& files, Buf* bufs, const std::function<bool(int, Buf&)>& use, int keep_first = 0) {
  std::mutex mu; std::condition_variable cv; int ready = -1, freed = -2; std::atomic<bool> fail(false); const int n = (int)files.size();
  std::thread loader([&] {
    for (int i = 0; i < n; i++) {
      { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return freed >= i - 2; }); }
      Buf& b = bufs[i & 1]; size_t got = read_into(files[i], b.p, b.sz, i < keep_first); if (!got) { fprintf(stderr, "cannot read %s\n", files[i].c_str()); fail = true; }
      else { apply_ovl(files[i], b.p, b.sz); sync_clean(b); }
      { std::lock_guard<std::mutex> lk(mu); ready = i; } cv.notify_all();
      if (fail) break;
    }
  });
  bool ok = true;
  for (int i = 0; i < n && ok; i++) {
    { double tw = now(); std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return ready >= i || fail.load(); }); g_waitio += now() - tw; }
    if (fail) { ok = false; break; }
    if (!use(i, bufs[i & 1])) ok = false;
    { std::lock_guard<std::mutex> lk(mu); freed = i; } cv.notify_all();
  }
  { std::lock_guard<std::mutex> lk(mu); freed = 1 << 30; } cv.notify_all(); loader.join();
  return ok;
}

// ------------------------------------------------------------------ glue.bin
struct Glue {
  uint8_t* base = nullptr; size_t len = 0; std::map<std::string, std::pair<const void*, int>> t;   // name -> (ptr, dtype)
  bool load(const std::string& p) {
    int fd = open(p.c_str(), O_RDONLY); if (fd < 0) return false; struct stat st; fstat(fd, &st); len = st.st_size;
    base = (uint8_t*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); if (base == MAP_FAILED) return false;
    if (memcmp(base, "MGL1", 4)) return false; int n; memcpy(&n, base + 4, 4); const uint8_t* e = base + 8;
    for (int i = 0; i < n; i++, e += 76) { char name[41] = {0}; memcpy(name, e, 40); int dt; memcpy(&dt, e + 40, 4); long long off; memcpy(&off, e + 60, 8); t[name] = {base + off, dt}; }
    return true;
  }
  const f32* f(const char* n) const { return (const f32*)t.at(n).first; }
  const __fp16* h(const char* n) const { return (const __fp16*)t.at(n).first; }
};

// ------------------------------------------------------------------ Q4_K embedding rows
static void q4k_row(const uint8_t* row, f32* out) {   // 4096 values = 16 super-blocks of 144 bytes
  for (int b = 0; b < 16; b++) {
    const uint8_t* p = row + b * 144; __fp16 dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2); const float d = (float)dh, dm = (float)mh; const uint8_t* s = p + 4; const uint8_t* qs = p + 16;
    float sc[8], mn[8];
    for (int j = 0; j < 4; j++) { sc[j] = s[j] & 63; mn[j] = s[j + 4] & 63; }
    for (int j = 4; j < 8; j++) { sc[j] = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); mn[j] = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
    for (int c = 0; c < 4; c++) for (int i = 0; i < 32; i++) {
      const uint8_t q = qs[c * 32 + i];
      out[b * 256 + c * 64 + i] = d * sc[2 * c] * (q & 15) - dm * mn[2 * c];
      out[b * 256 + c * 64 + 32 + i] = d * sc[2 * c + 1] * (q >> 4) - dm * mn[2 * c + 1];
    }
  }
}

// ------------------------------------------------------------------ tile helpers
// rows [nrows][cols] fp32 -> x32 tiles [slab][cb][32][32]
static void rows_to_x32(const f32* rows, int nrows, int cols, f32* x32, int ns) {
  const int ncb = cols / 32;
  par_for(ns, [&](int s) {
    for (int cb = 0; cb < ncb; cb++) { f32* tile = x32 + ((size_t)s * ncb + cb) * 1024;
      for (int r = 0; r < 32; r++) { const int row = s * 32 + r; if (row < nrows) memcpy(tile + r * 32, rows + (size_t)row * cols + cb * 32, 128); else memset(tile + r * 32, 0, 128); } }
  });
}
static void x32_to_rows(const f32* x32, int nrows, int cols, f32* rows) {
  const int ncb = cols / 32;
  par_for((nrows + 31) / 32, [&](int s) {
    for (int cb = 0; cb < ncb; cb++) { const f32* tile = x32 + ((size_t)s * ncb + cb) * 1024;
      for (int r = 0; r < 32; r++) { const int row = s * 32 + r; if (row < nrows) memcpy(rows + (size_t)row * cols + cb * 32, tile + r * 32, 128); } }
  });
}

struct Layout { int meta[M_NMETA]; size_t total; };
static remote_handle64 g_h = 0;
static Glue g_glue; static uint8_t* g_embd = nullptr; static size_t g_embd_len = 0; static std::string g_dir;
static Buf g_te[2];                       // streaming pack buffers (TE layers and DiT blocks)
static Buf g_blobTE; static Layout g_LT;  // TE blob (MAXT tokens)
static Buf g_blobP, g_blobD; static Layout g_LP, g_LD; static int g_P_ntok = -1, g_D_ntok = -1, g_D_extra = -1;

static int g_profn = 0;
static int call(const Layout& L, int* meta, Buf& w, size_t woff, const Buf& blob, const f32* mod, int nmod) {
  int times[32]; int rc = 0; meta[M_WOFF] = (int)woff;
  if (!map_buf(w)) return -1;
  int r = 0; for (int attempt = 0; attempt < 8; attempt++) { rc = 0; r = mega_block(g_h, meta, M_NMETA, w.fd, blob.fd, (const uint8_t*)mod, nmod * 4, times, 32, &rc); if (!r && rc == -77) { fprintf(stderr, "HMX self-test failed, retry %d\n", attempt + 1); usleep(400000); continue; } break; }   // rc -77: nothing was computed, try again
  if (getenv("MAGE_PROF") && (g_profn++ % 12) == 5) { fprintf(stderr, "prof mode %d us:", meta[M_MODE]); for (int i = 0; i < 6; i++) fprintf(stderr, " %d", times[i]); fprintf(stderr, " | attn(setup qk pv scale wait flush):"); for (int i = 17; i < 23; i++) fprintf(stderr, " %d", times[i]); fprintf(stderr, " | mode50(ln1 q k lnv v o ln2 fc1 fc2):"); for (int i = 8; i < 17; i++) fprintf(stderr, " %d", times[i]); fprintf(stderr, "\n"); }
  if (r || rc) { fprintf(stderr, "mega_block r=0x%x rc=%d mode=%d\n", r, rc, meta[M_MODE]); return -1; }
  return 0;
}

// Y[n][out] = X[n][in] * W[out][in]^T (W fp16, X/Y fp32, in % 4 == 0). 4 weight rows are widened to fp32 once, then a 4-token x 4-output register tile runs 16 FMAs per 8 loads; blocks of outputs are handed out dynamically to all cores (big and little).
static void gemm_f16(const __fp16* W, const f32* X, f32* Y, int n, int out, int in) {
  const int OB = 4, nblk = (out + OB - 1) / OB; std::atomic<int> next(0);
  auto worker = [&] {
    std::vector<f32> w((size_t)OB * in + 16);
    for (;;) {
      const int blk = next.fetch_add(1); if (blk >= nblk) break;
      const int o0 = blk * OB, ob = std::min(OB, out - o0);
      for (int r = 0; r < OB; r++) { f32* wr = &w[(size_t)r * in]; if (r < ob) { const __fp16* src = W + (size_t)(o0 + r) * in; for (int k = 0; k < in; k += 4) vst1q_f32(wr + k, vcvt_f32_f16(vld1_f16(src + k))); } else memset(wr, 0, (size_t)in * 4); }
      const f32 *w0 = &w[0], *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in; int t = 0;
      for (; t + 4 <= n; t += 4) {
        const f32 *x0 = X + (size_t)t * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
        float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00, a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
        for (int k = 0; k < in; k += 4) {
          const float32x4_t v0 = vld1q_f32(w0 + k), v1 = vld1q_f32(w1 + k), v2 = vld1q_f32(w2 + k), v3 = vld1q_f32(w3 + k);
          float32x4_t xv = vld1q_f32(x0 + k); a00 = vfmaq_f32(a00, xv, v0); a01 = vfmaq_f32(a01, xv, v1); a02 = vfmaq_f32(a02, xv, v2); a03 = vfmaq_f32(a03, xv, v3);
          xv = vld1q_f32(x1 + k); a10 = vfmaq_f32(a10, xv, v0); a11 = vfmaq_f32(a11, xv, v1); a12 = vfmaq_f32(a12, xv, v2); a13 = vfmaq_f32(a13, xv, v3);
          xv = vld1q_f32(x2 + k); a20 = vfmaq_f32(a20, xv, v0); a21 = vfmaq_f32(a21, xv, v1); a22 = vfmaq_f32(a22, xv, v2); a23 = vfmaq_f32(a23, xv, v3);
          xv = vld1q_f32(x3 + k); a30 = vfmaq_f32(a30, xv, v0); a31 = vfmaq_f32(a31, xv, v1); a32 = vfmaq_f32(a32, xv, v2); a33 = vfmaq_f32(a33, xv, v3);
        }
        const float32x4_t acc[4][4] = {{a00, a01, a02, a03}, {a10, a11, a12, a13}, {a20, a21, a22, a23}, {a30, a31, a32, a33}};
        for (int i = 0; i < 4; i++) for (int r = 0; r < ob; r++) Y[(size_t)(t + i) * out + o0 + r] = vaddvq_f32(acc[i][r]);
      }
      for (; t < n; t++) {
        const f32* x0 = X + (size_t)t * in; float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int k = 0; k < in; k += 4) { const float32x4_t xv = vld1q_f32(x0 + k); a0 = vfmaq_f32(a0, xv, vld1q_f32(w0 + k)); a1 = vfmaq_f32(a1, xv, vld1q_f32(w1 + k)); a2 = vfmaq_f32(a2, xv, vld1q_f32(w2 + k)); a3 = vfmaq_f32(a3, xv, vld1q_f32(w3 + k)); }
        const float32x4_t acc[4] = {a0, a1, a2, a3}; for (int r = 0; r < ob; r++) Y[(size_t)t * out + o0 + r] = vaddvq_f32(acc[r]);
      }
    }
  };
  const int nt = std::max(1, std::min<int>(8, (int)std::thread::hardware_concurrency())); std::vector<std::thread> th;
  for (int i = 1; i < nt; i++) th.emplace_back(worker); worker(); for (auto& x : th) x.join();
}
