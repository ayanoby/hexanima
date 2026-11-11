// qwenrun (derived from mossrun): Qwen-Image-2.1-Turbo on the Hexagon NPU in one process.
//   tokens -> Qwen3.5-9B text encoder (32 layers, HMX/HVX) -> text refiner -> DiT (42 blocks, 4-step rapid-moss sampler) -> VAE latent z
// The VAE (FLUX.2 decoder, QNN context) is run by vaerun on the z file written here.
// stdin:  "load MODELDIR"                         read glue + DiT packs (resident), prints "loaded <s>"
//         "gen IDS.i32 H W SEED STEPS OUT.z"      prints "step i <ms>" lines and "done <ms>"; OUT = fp32 [32][H/8][W/8] (BN-denormalised, unpatchified)
//         "quit"
// MODELDIR: glue.bin embd.q4k te/te00..te32.bin ref/{proj,ref0,ref1}.bin dit/blk00..blk41.bin
#include <arm_neon.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
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
static const int NLAYER_DIT = 32, NLAYER_TE = 36, TE_BUF = 262144000, MAXT = 2048;

template <class F> static void par_for(int n, F f) {
  int nt = n < 4 ? n : 4; if (nt < 2) { for (int i = 0; i < n; i++) f(i); return; }
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
  for (auto& x : th) x.join();
}

// ------------------------------------------------------------------ rpcmem buffers
// A DSP process can hold only ~1.5 GB of fastrpc_mmap'd buffers (measured), but mapping costs ~0.6 ms per 128 MB: weights live in rpcmem and are mapped on demand (LRU window).
struct Buf { uint8_t* p = nullptr; int fd = -1; size_t sz = 0; bool mapped = false, pinned = false; unsigned long long use = 0; };
static std::vector<Buf*> g_all; static size_t g_mapped = 0; static unsigned long long g_tick = 0; static const size_t MAP_BUDGET = 1400u << 20;
static bool map_buf(Buf& b) {
  b.use = ++g_tick; if (b.mapped) return true;
  while (g_mapped + b.sz > MAP_BUDGET) {
    Buf* v = nullptr; for (Buf* x : g_all) if (x->mapped && !x->pinned && (!v || x->use < v->use)) v = x;
    if (!v) break; fastrpc_munmap(CDSP_DOMAIN_ID, v->fd, v->p, 0); v->mapped = false; g_mapped -= v->sz;
  }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, b.fd, b.p, 0, b.sz, FASTRPC_MAP_FD)) { fprintf(stderr, "fastrpc_mmap %zu failed (mapped %zu MB)\n", b.sz, g_mapped >> 20); return false; }
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
  std::vector<std::thread> th; for (int i = 0; i < 3; i++) th.emplace_back(work); work(); for (auto& t : th) t.join();
  if (!keep) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // keep = leave the pages in the kernel's page cache (it drops them by itself under memory pressure)
  close(fd); return got.load();
}
// stream a list of pack files through the two g_te buffers: a loader thread reads file i+1 while use(i) runs on the DSP
#include <functional>
static int g_keep_blocks = 0;
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
    { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return ready >= i || fail.load(); }); }
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

