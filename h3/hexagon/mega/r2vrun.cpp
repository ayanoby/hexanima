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
static std::string MD;
static std::string LD; static const size_t LORASZ = 26607616u;   // H3_LORA dir: loraNN.bin / lorar0,1.bin (export_r2v_lora.py), read after every DiT block pack
static const size_t DITSZ = 385660928u, ADASZ = 260886528u;   // block pack, adaln pack (int8 [96768][2688] + f32 scale + f32 bias)
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


// MiniMax-H3 Ref2VA DiT on the phone (text-to-video rows only, one forward pass for tests): 50 x MODE 50 blocks on the NPU, glue and the adaln layers (int8, ConvRot group 64) on the CPU.
// usage: r2vrun DIR LH LW T Z.f32 TXT.f32 LT TVAL OUT_V.f32 [NBLK]   DIR: blkNN.bin adalnNN.bin r0.bin r1.bin + glue files (anima/hexanima/h3/export_h3_ref2va.py, export_r2v_glue.py)
//   Z.f32 (24,T,LH,LW) latent noise, TXT.f32 (LT,5120) text-encoder hidden state; output v (T*LH/2*LW/2, 96) = final_layer.video_out of the video rows at timestep TVAL (t = 1 - sigma)
int main(int argc, char **argv) {
  if (argc < 10) { fprintf(stderr, "usage: r2vrun DIR LH LW T Z TXT LT TVAL OUT_V [NBLK]\n"); return 1; }
  MD = argv[1]; if (getenv("H3_LORA")) LD = getenv("H3_LORA"); const int lh = atoi(argv[2]), lw = atoi(argv[3]), T = atoi(argv[4]); int nt = atoi(argv[7]); const double tval = atof(argv[8]); const char *outp = argv[9]; const int nblk = argc > 10 ? atoi(argv[10]) : NBLK;
  const std::string a6 = argv[6]; const bool use_ids = nt == 0 || (a6.size() > 4 && a6.substr(a6.size() - 4) == ".ids");   // TXT: a .ids file (int32 token ids, the text encoder runs on the NPU) or fp32 hidden states
  std::vector<int32_t> ids; if (use_ids) { FILE *f = fopen(argv[6], "rb"); if (!f) { fprintf(stderr, "ids\n"); return 1; } fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(sz / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) return 1; fclose(f); nt = (int)ids.size(); }
  const std::string TED = getenv("H3_TE") ? getenv("H3_TE") : MD + "/te";
  const int steps = getenv("R2V_STEPS") ? atoi(getenv("R2V_STEPS")) : 0; const double shift = getenv("R2V_SHIFT") ? atof(getenv("R2V_SHIFT")) : 12.0;   // steps > 0: sampling (OUT = latents), 0: one forward at TVAL (OUT = v)
  const int th = lh / 2, tw = lw / 2, ni = T * th * tw; if (ni % 32) { fprintf(stderr, "video rows %d must be a multiple of 32\n", ni); return 1; }
  // reference images (R2V_REF=lat.f32:rh:rw[,...] = normalised VAE latents (24,1,rh,rw)): rows [video | references | text]; reference rows are clean condition rows (timestep R2V_REFT, default 0.999) with their own modulation set
  struct Ref { std::string f; int rh, rw; }; std::vector<Ref> refs; if (getenv("R2V_REF")) { std::string a = getenv("R2V_REF"); size_t p0 = 0; while (p0 < a.size()) { size_t p1 = a.find(',', p0); if (p1 == std::string::npos) p1 = a.size(); std::string it = a.substr(p0, p1 - p0); size_t c2 = it.rfind(':'), c1 = it.rfind(':', c2 - 1); refs.push_back({it.substr(0, c1), atoi(it.substr(c1 + 1, c2 - c1 - 1).c_str()), atoi(it.substr(c2 + 1).c_str())}); p0 = p1 + 1; } }
  int nr = 0; for (auto &rf : refs) { const int rr = (rf.rh / 2) * (rf.rw / 2); if (rr % 32) { fprintf(stderr, "reference rows %d must be a multiple of 32\n", rr); return 1; } nr += rr; }
  const double tcond = getenv("R2V_REFT") ? atof(getenv("R2V_REFT")) : 0.999;
  const int ntok = (ni + nr + nt + 31) / 32 * 32, ns = ntok / 32, nimgslabs = ni / 32, nrefslabs = nr / 32, nvalid = ni + nr + nt, ntk = (nt + 31) / 32 * 32;
  // windowed DiT (like the LTX port): the activations X / rope tables of all rows stay in phone memory, the DSP sees a window of R2V_WIN slabs (32 rows each) and the keys/values of ALL rows (global K/V buffers); automatic above 4000 rows, R2V_WIN=0 forces the plain path
  const int WSL = getenv("R2V_WIN") ? atoi(getenv("R2V_WIN")) : 32; const bool win = getenv("R2V_WIN") ? WSL > 0 : ntok > 4000; const int wrows = win ? std::max(WSL * 32, ntk) : ntok, wns = wrows / 32;
  setvbuf(stdout, 0, _IONBF, 0); double T0 = now();
  auto vpp_w = load<f32>("vpp_w.f32", (size_t)HID * PD), vpp_b = load<f32>("vpp_b.f32", HID);
  auto cond_w = load<f32>("cond_w.f32", (size_t)HID * TXTD), cond_b = load<f32>("cond_b.f32", HID);
  auto te_in_w = load<f32>("te_in_w.f32", (size_t)HID * 256), te_in_b = load<f32>("te_in_b.f32", HID), te_out_w = load<f32>("te_out_w.f32", (size_t)2688 * HID), te_out_b = load<f32>("te_out_b.f32", 2688);
  auto fin_w = load<f32>("fin_ada_w.f32", (size_t)10752 * 2688), fin_b = load<f32>("fin_ada_b.f32", 10752), fin_n = load<f32>("fin_norm.f32", HID);
  auto vout_w = load<f32>("vout_w.f32", (size_t)PD * HID), vout_b = load<f32>("vout_b.f32", PD), rfin = load<f32>("rfin.f32", HID), rinv = load<f32>("rope_inv.f32", 16), H64 = load<f32>("H64.f32", 4096);
  auto H = load<f32>("H256.f32", 65536); for (auto &v : H) v /= 16.0f;
  std::vector<f32> z((size_t)24 * T * lh * lw); if (!strncmp(argv[5], "seed:", 5)) { std::mt19937_64 rng(strtoull(argv[5] + 5, 0, 10)); std::normal_distribution<float> nd(0.f, 1.f); for (auto &v : z) v = nd(rng); } else { FILE *f = fopen(argv[5], "rb"); if (!f || fread(z.data(), 4, z.size(), f) != z.size()) { fprintf(stderr, "z\n"); return 1; } fclose(f); }
  std::vector<f32> te((size_t)nt * TXTD); if (!use_ids) { FILE *f = fopen(argv[6], "rb"); if (!f || fread(te.data(), 4, te.size(), f) != te.size()) { fprintf(stderr, "txt\n"); return 1; } fclose(f); }
  printf("glue loaded %.1f s\n", now() - T0);
  int meta[M_NMETA]; memset(meta, 0, sizeof meta);
  const bool hasL = !LD.empty(); const size_t bigFF = (size_t)wrows * FF * 2;
  struct { int id; size_t bytes; } ent[] = {{M_X, (size_t)wrows * HID * 4}, {M_COS, (size_t)wns * 8192}, {M_SIN, (size_t)wns * 8192}, {M_HN, 131072}, {M_H, (size_t)wrows * HID * 2}, {M_HR, bigFF},
    {M_LQ, hasL ? (size_t)wns * 12 * 2048 : 4096}, {M_LO, hasL ? (size_t)wns * 4 * 2048 : 4096}, {M_LU, hasL ? (size_t)wns * 4 * 2048 : 4096}, {M_LD, hasL ? (size_t)wns * 4 * 2048 : 4096}, {M_DLT, hasL ? (size_t)wrows * 28672 * 2 : 4096}, {M_Q, (size_t)wrows * QW * 2}, {M_K, (size_t)ntok * QW * 2}, {M_V, (size_t)ntok * QW * 2}, {M_O, (size_t)wrows * QW * 2}, {M_T, (size_t)wrows * HID * 2}, {M_MID, bigFF}};
  size_t off = 0; for (auto &e : ent) { meta[e.id] = (int)off; off += up(e.bytes); }
  rpcmem_init();
  uint8_t *wb[2]; for (int i = 0; i < 2; i++) { wb[i] = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WTA); if (!wb[i]) { fprintf(stderr, "rpcmem weights\n"); return 1; } }
  uint8_t *blob = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, off); if (!blob) { fprintf(stderr, "rpcmem blob\n"); return 1; } memset(blob, 0, off);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; if (mega_open(mega_URI CDSP_DOMAIN, &h)) { fprintf(stderr, "mega_open failed\n"); return 1; }
  for (int i = 0; i < 2; i++) if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(wb[i]), wb[i], 0, WTA, FASTRPC_MAP_FD)) { fprintf(stderr, "mmap w\n"); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(blob), blob, 0, off, FASTRPC_MAP_FD)) { fprintf(stderr, "mmap blob\n"); return 1; }
  { auto hn = load<uint8_t>("hn.bin", 131072); memcpy(blob + meta[M_HN], hn.data(), 131072); }
  meta[M_MODE] = 50; meta[M_NCB] = 168; meta[M_NW] = 3;
  const int fdb = rpcmem_to_fd(blob);
  auto put_rows = [&](const f32 *rows, int n, int hid = HID) { f32 *X = (f32 *)(blob + meta[M_X]); memset(X, 0, ent[0].bytes); const int ncb = hid / 32;
    for (int r = 0; r < n; r++) for (int c = 0; c < hid; c++) X[((size_t)(r / 32) * ncb + c / 32) * 1024 + (r % 32) * 32 + c % 32] = rows[(size_t)r * hid + c]; };
  auto get_rows = [&](f32 *rows, int n, int hid = HID) { sync_fd(fdb, 3); const f32 *X = (const f32 *)(blob + meta[M_X]); const int ncb = hid / 32;
    for (int r = 0; r < n; r++) for (int c = 0; c < hid; c++) rows[(size_t)r * hid + c] = X[((size_t)(r / 32) * ncb + c / 32) * 1024 + (r % 32) * 32 + c % 32]; };
  auto put_tab = [&](int id, const std::vector<f32> &t) { f32 *Tb = (f32 *)(blob + meta[id]); memset(Tb, 0, ent[id == M_COS ? 1 : 2].bytes);
    for (int r = 0; r < wns * 32; r++) for (int c = 0; c < 64; c++) Tb[((size_t)(r / 32) * 2 + c / 32) * 1024 + (r % 32) * 32 + c % 32] = r < (int)(t.size() / 64) ? t[(size_t)r * 64 + c] : (id == M_COS ? 1.0f : 0.0f); };
  auto run_block = [&](int wi, const std::vector<f32> &mod, bool first, bool last, int nvalid_, int nimg_slabs, int ntok_) {
    meta[M_NTOK] = ntok_; meta[M_LORA] = (!LD.empty() && meta[M_MODE] == 50) ? 1 : 0; meta[M_NVALID] = nvalid_; meta[M_NIMG] = nimg_slabs; meta[M_FIRST] = first; meta[M_REL] = last;
    if (first) sync_fd(fdb, 7);
    sync_fd(rpcmem_to_fd(wb[wi]), 7);
    int times[32]; int rc = 0, r = 0;
    for (int attempt = 0; attempt < 8; attempt++) {   // rc -77 = the HMX self-test failed at the start of the call (nothing was computed): wait and try again
      rc = 0; r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wb[wi]), fdb, (const uint8_t *)mod.data(), (int)(mod.size() * 4), times, 32, &rc);
      if (!r && rc == -77) { fprintf(stderr, "HMX self-test failed, retry %d\n", attempt + 1); usleep(400000); continue; }
      break; }
    if (r || rc) { fprintf(stderr, "mega_block r=0x%x rc=%d\n", r, rc); exit(1); } };
  // ---- windowed DiT blocks: host copies of all rows (fp32 tiles [slab][168][1024]) and rope tables ([slab][2][1024] cos / sin)
  const size_t XS = (size_t)168 * 1024, CS = (size_t)2 * 1024; std::vector<f32> Xh, Ch, Sh; if (win) { Xh.assign((size_t)ns * XS, 0.f); Ch.assign((size_t)ns * CS, 1.f); Sh.assign((size_t)ns * CS, 0.f); }
  auto put_rows_h = [&](const f32 *rows, int n) { std::fill(Xh.begin(), Xh.end(), 0.f); parfor(n, [&](int r) { for (int c = 0; c < HID; c++) Xh[((size_t)(r / 32) * 168 + c / 32) * 1024 + (r % 32) * 32 + c % 32] = rows[(size_t)r * HID + c]; }); };
  auto get_rows_h = [&](f32 *rows, int n) { parfor(n, [&](int r) { for (int c = 0; c < HID; c++) rows[(size_t)r * HID + c] = Xh[((size_t)(r / 32) * 168 + c / 32) * 1024 + (r % 32) * 32 + c % 32]; }); };
  auto put_tab_h = [&](std::vector<f32> &Tb, const std::vector<f32> &t, float pad) { for (int r = 0; r < ns * 32; r++) for (int c = 0; c < 64; c++) Tb[((size_t)(r / 32) * 2 + c / 32) * 1024 + (r % 32) * 32 + c % 32] = r < (int)(t.size() / 64) ? t[(size_t)r * 64 + c] : pad; };
  auto run_block_w = [&](int wi, const std::vector<f32> &mod, bool first, bool last) {   // per block: pass 0 = keys/values of every window into the global K/V buffers, pass 1 = full block per window (attention over all keys)
    const int WS = wns, nwin = (ns + WS - 1) / WS;
    for (int pass = 0; pass < 2; pass++) for (int wn = 0; wn < nwin; wn++) {
      const int w0 = wn * WS, nws = std::min(WS, ns - w0), wtok = nws * 32, ns0w = std::max(0, std::min(nimgslabs - w0, nws));
      memcpy(blob + meta[M_X], &Xh[(size_t)w0 * XS], (size_t)nws * XS * 4); memcpy(blob + meta[M_COS], &Ch[(size_t)w0 * CS], (size_t)nws * CS * 4); memcpy(blob + meta[M_SIN], &Sh[(size_t)w0 * CS], (size_t)nws * CS * 4);
      sync_fd(fdb, 7); sync_fd(rpcmem_to_fd(wb[wi]), 7);
      meta[M_NTOK] = wtok; meta[M_LORA] = !LD.empty() ? 1 : 0; meta[M_NVALID] = nvalid; meta[M_NIMG] = ns0w; meta[M_FIRST] = (first && pass == 0 && wn == 0) ? 1 : 2; meta[M_REL] = (last && pass == 1 && wn == nwin - 1) ? 1 : 2; meta[M_A2] = ntok; meta[M_A4] = pass == 0 ? 1 : 2; meta[M_A9] = w0;
      { const int ra = nimgslabs, rb = nimgslabs + nrefslabs; int rl = std::max(ra - w0, 0), rh2 = std::min(rb - w0, nws); if (rh2 <= rl) rl = rh2 = 0; meta[M_A5] = rl; meta[M_A6] = rh2; }
      int times[32]; int rc = 0, r = 0;
      for (int attempt = 0; attempt < 8; attempt++) { rc = 0; r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wb[wi]), fdb, (const uint8_t *)mod.data(), (int)(mod.size() * 4), times, 32, &rc);
        if (!r && rc == -77) { fprintf(stderr, "HMX self-test failed, retry %d\n", attempt + 1); usleep(400000); continue; } break; }
      if (r || rc) { fprintf(stderr, "mega_block window pass %d w0 %d r=0x%x rc=%d\n", pass, w0, r, rc); exit(1); }
      if (pass == 1) { sync_fd(fdb, 3); memcpy(&Xh[(size_t)w0 * XS], blob + meta[M_X], (size_t)nws * XS * 4); }
    }
    meta[M_A4] = 0; meta[M_A2] = 0; meta[M_A9] = 0; meta[M_A5] = 0; meta[M_A6] = 0; };
  // ---------------- text encoder on the NPU (Qwen3-VL layers 0..49, MODE 51): embeddings -> 50 layers -> hidden state after layer 50
  if (use_ids) {
    double t0 = now(); std::vector<f32> x((size_t)nt * TXTD); { FILE *f = fopen((TED + "/embed.bf16").c_str(), "rb"); if (!f) { fprintf(stderr, "embed.bf16\\n"); return 1; } std::vector<uint16_t> row(TXTD);
      for (int r = 0; r < nt; r++) { fseek(f, (long)ids[r] * TXTD * 2, SEEK_SET); if (fread(row.data(), 2, TXTD, f) != (size_t)TXTD) return 1; for (int i = 0; i < TXTD; i++) { uint32_t u = (uint32_t)row[i] << 16; memcpy(&x[(size_t)r * TXTD + i], &u, 4); } } fclose(f); }
    std::vector<f32> cs((size_t)nt * 64), sn((size_t)nt * 64);
    for (int r = 0; r < nt; r++) for (int c = 0; c < 64; c++) { double a = (float)r * (double)(float)pow(5000000.0, -(double)c / 64.0); cs[(size_t)r * 64 + c] = cosf((float)a); sn[(size_t)r * 64 + c] = sinf((float)a); }
    put_tab(M_COS, cs); put_tab(M_SIN, sn); put_rows(x.data(), nt, TXTD); meta[M_MODE] = 51; meta[M_NCB] = 160;
    const int nl = getenv("H3_TE_LAYERS") ? atoi(getenv("H3_TE_LAYERS")) : 50; std::vector<f32> dm(16, 0.f); std::future<void> pre;
    { char nm[32]; snprintf(nm, sizeof nm, "/lay%02d.bin", 0); std::string p = TED + nm; pre = std::async(std::launch::async, [p, &wb] { read_file(p, wb[0], 487940096u); }); }
    for (int l = 0; l < nl; l++) {
      pre.get();
      if (l + 1 < nl) { char nm[32]; snprintf(nm, sizeof nm, "/lay%02d.bin", l + 1); std::string p = TED + nm; uint8_t *dst = wb[(l + 1) & 1]; pre = std::async(std::launch::async, [p, dst] { read_file(p, dst, 487940096u); }); }
      run_block(l & 1, dm, l == 0, l == nl - 1, nt, 0, ntk);
    }
    get_rows(te.data(), nt, TXTD); meta[M_MODE] = 50; meta[M_NCB] = 168;
    printf("text encoder %.2f s (%d layers)\\n", now() - t0, nl);
    if (getenv("H3_TE_OUT")) { FILE *f = fopen(getenv("H3_TE_OUT"), "wb"); fwrite(te.data(), 4, te.size(), f); fclose(f); } }
  // ---------------- text: condition_proj (CPU, bf16 weights as f32) + 2 refiner blocks (NPU, no modulation / rope) + final norm
  std::vector<f32> txt((size_t)nt * HID);
  { double t0 = now();
    parfor(HID, [&](int o) { const f32 *w = &cond_w[(size_t)o * TXTD]; for (int r = 0; r < nt; r++) { float s = cond_b[o]; const f32 *x = &te[(size_t)r * TXTD]; for (int i = 0; i < TXTD; i++) s += w[i] * x[i]; txt[(size_t)r * HID + o] = s; } });
    std::vector<f32> mod(MODB / 4, 0.f); for (int i = 0; i < HID; i++) mod[6 * HID + 2 * HID + i] = mod[6 * HID + 5 * HID + i] = 1.0f;   // kind 1: shift 0 scale 0 gate 1
    std::vector<f32> one(0); put_tab(M_COS, one); put_tab(M_SIN, one);
    put_rows(txt.data(), nt);
    for (int i = 0; i < 2; i++) { read_file(MD + "/r" + std::to_string(i) + ".bin", wb[i & 1], DITSZ); if (!LD.empty()) read_file(LD + "/lorar" + std::to_string(i) + ".bin", wb[i & 1] + DITSZ, LORASZ); run_block(i & 1, mod, i == 0, i == 1, nt, 0, ntk); }
    get_rows(txt.data(), nt);
    for (int r = 0; r < nt; r++) rms_row(&txt[(size_t)r * HID], rfin.data(), &txt[(size_t)r * HID], HID, 1e-5f);
    printf("text path %.2f s\n", now() - t0); }
  // ---------------- positions -> rope tables (video rows first, then text rows), engine channel order (i < 48 real, 48..63 identity)
  { std::vector<f32> cs((size_t)(ni + nr + nt) * 64), sn((size_t)(ni + nr + nt) * 64);
    const double sq = sqrt((double)lh * lw); auto grid = [&](int dim, int k) { double ratio = dim / sq, left = (1.0 - ratio) / 2.0; return (left + k * ratio / (dim / 2)) * 32.0; };
    std::vector<double> tg(T); { double acc = nt + (double)refs.size(); static const int FP[5] = {1, 4, 4, 4, 4}; for (int k = 0; k < T; k++) { tg[k] = acc; acc += 5.0 / 3.0 * FP[k % 5]; } }
    std::vector<int> rstart; { int o = ni; for (auto &rf : refs) { rstart.push_back(o); o += (rf.rh / 2) * (rf.rw / 2); } }
    auto rgrid = [&](int dim, int sqr, int k) { double ratio = dim / sqrt((double)sqr), left = (1.0 - ratio) / 2.0; return (left + k * ratio / (dim / 2)) * 32.0; };
    for (int r = 0; r < ni + nr + nt; r++) {
      double p[3]; if (r < ni) { const int tt = r / (th * tw), rem = r % (th * tw); p[0] = tg[tt]; p[1] = grid(lh, rem / tw); p[2] = grid(lw, rem % tw); }
      else if (r < ni + nr) { int k = 0; while (k + 1 < (int)refs.size() && r >= rstart[k + 1]) k++; const int rem = r - rstart[k], rtw = refs[k].rw / 2; p[0] = nt + k; p[1] = rgrid(refs[k].rh, refs[k].rh * refs[k].rw, rem / rtw); p[2] = rgrid(refs[k].rw, refs[k].rh * refs[k].rw, rem % rtw); }
      else { p[0] = r - ni - nr; p[1] = p[2] = 0; }
      for (int c = 0; c < 64; c++) { if (c >= 48) { cs[(size_t)r * 64 + c] = 1; sn[(size_t)r * 64 + c] = 0; continue; }
        double a = (float)p[c / 16] * (double)rinv[c % 16]; cs[(size_t)r * 64 + c] = cosf((float)a); sn[(size_t)r * 64 + c] = sinf((float)a); } }
    if (win) { put_tab_h(Ch, cs, 1.0f); put_tab_h(Sh, sn, 0.0f); } else { put_tab(M_COS, cs); put_tab(M_SIN, sn); } }
  // ---------------- video rows: patchify (t,h,w | c, p, q) and video_patch_proj
  std::vector<f32> x_img((size_t)ni * PD);
  for (int tt = 0; tt < T; tt++) for (int ph = 0; ph < th; ph++) for (int pw = 0; pw < tw; pw++) for (int c = 0; c < 24; c++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
    x_img[(size_t)((tt * th + ph) * tw + pw) * PD + c * 4 + a * 2 + b] = z[(((size_t)c * T + tt) * lh + ph * 2 + a) * lw + pw * 2 + b];
  std::vector<f32> rows((size_t)(ni + nr + nt) * HID), v((size_t)ni * PD); std::vector<uint8_t> ada(ADASZ); std::vector<f32> mod(18 * HID);
  std::vector<double> SGM; if (steps > 0) { for (int i = 0; i < steps; i++) { const double bb = 1.0 - (double)i / steps, sg = shift * bb / (1.0 + (shift - 1.0) * bb); SGM.push_back(std::min(sg, 1.0)); } SGM.push_back(0.0); }
  const int nsteps = steps > 0 ? steps : 1; double tc = 0; int step0 = 0; const std::string ckpt = std::string(outp) + ".ckpt";
  if (steps > 0 && getenv("R2V_RESUME")) { FILE *f = fopen(ckpt.c_str(), "rb"); int sv = 0; if (f && fread(&sv, 4, 1, f) == 1 && fread(x_img.data(), 4, x_img.size(), f) == x_img.size()) { step0 = sv; printf("resumed at step %d from %s\n", step0, ckpt.c_str()); } if (f) fclose(f); }
  // reference rows (once): noise-augmented clean latents -> patchify -> video_patch_proj; the reference timestep embedding / ConvRot-rotated silu(t_emb) for its modulation set
  std::vector<f32> refrows((size_t)nr * HID); f32 xr2[2688];
  { size_t o = 0; std::mt19937_64 rng(12345); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &rf : refs) { const int rh = rf.rh, rw = rf.rw, rth = rh / 2, rtw = rw / 2; std::vector<f32> zz((size_t)24 * rh * rw); { FILE *f = fopen(rf.f.c_str(), "rb"); if (!f || fread(zz.data(), 4, zz.size(), f) != zz.size()) { fprintf(stderr, "ref latent %s\n", rf.f.c_str()); return 1; } fclose(f); }
      for (auto &x : zz) x = (float)tcond * x + (1.f - (float)tcond) * nd(rng);
      std::vector<f32> px((size_t)rth * rtw * PD); for (int ph = 0; ph < rth; ph++) for (int pw = 0; pw < rtw; pw++) for (int c = 0; c < 24; c++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) px[(size_t)(ph * rtw + pw) * PD + c * 4 + a * 2 + b] = zz[((size_t)c * rh + ph * 2 + a) * rw + pw * 2 + b];
      parfor(rth * rtw, [&](int r) { for (int oo = 0; oo < HID; oo++) { float sm = vpp_b[oo]; const f32 *w = &vpp_w[(size_t)oo * PD], *x = &px[(size_t)r * PD]; for (int i = 0; i < PD; i++) sm += w[i] * x[i]; refrows[(o + r) * HID + oo] = sm; } }); o += (size_t)rth * rtw; }
    if (nr > 0) { f32 tf[256], hh[HID], tee[2688], se2[2688]; for (int i = 0; i < 128; i++) { double a = tcond * exp(-log(10000.0) * i / 128.0); tf[i] = (float)cos(a); tf[128 + i] = (float)sin(a); }
      for (int oo = 0; oo < HID; oo++) { float sm = te_in_b[oo]; for (int i = 0; i < 256; i++) sm += te_in_w[(size_t)oo * 256 + i] * tf[i]; hh[oo] = sm / (1.f + expf(-sm)); }
      for (int oo = 0; oo < 2688; oo++) { float sm = te_out_b[oo]; for (int i = 0; i < HID; i++) sm += te_out_w[(size_t)oo * HID + i] * hh[i]; tee[oo] = sm; }
      for (int oo = 0; oo < 2688; oo++) se2[oo] = tee[oo] / (1.f + expf(-tee[oo]));
      for (int j = 0; j < 42; j++) for (int n = 0; n < 64; n++) { float sm = 0; for (int m = 0; m < 64; m++) sm += se2[j * 64 + m] * H64[m * 64 + n]; xr2[j * 64 + n] = sm; } printf("reference: %d image(s), %d rows, timestep %.3f\n", (int)refs.size(), nr, tcond); } }
  for (int step = step0; step < nsteps; step++) {
    const double tcur = steps > 0 ? 1.0 - SGM[step] : tval, ts = now();
  parfor(ni, [&](int r) { for (int o = 0; o < HID; o++) { float s = vpp_b[o]; const f32 *w = &vpp_w[(size_t)o * PD], *x = &x_img[(size_t)r * PD]; for (int i = 0; i < PD; i++) s += w[i] * x[i]; rows[(size_t)r * HID + o] = s; } });
  memcpy(&rows[(size_t)ni * HID], refrows.data(), (size_t)nr * HID * 4); memcpy(&rows[(size_t)(ni + nr) * HID], txt.data(), (size_t)nt * HID * 4);
  if (win) put_rows_h(rows.data(), ni + nr + nt); else put_rows(rows.data(), ni + nr + nt);
  // ---------------- time embedding -> silu -> adaln of every block (CPU, int8 ConvRot group 64) -> blocks
  f32 se[2688], tf[256], hh[HID]; { for (int i = 0; i < 128; i++) { double a = tcur * exp(-log(10000.0) * i / 128.0); tf[i] = (float)cos(a); tf[128 + i] = (float)sin(a); }
    for (int o = 0; o < HID; o++) { float s = te_in_b[o]; for (int i = 0; i < 256; i++) s += te_in_w[(size_t)o * 256 + i] * tf[i]; hh[o] = s / (1.f + expf(-s)); }
    f32 tee[2688]; for (int o = 0; o < 2688; o++) { float s = te_out_b[o]; for (int i = 0; i < HID; i++) s += te_out_w[(size_t)o * HID + i] * hh[i]; tee[o] = s; }
    for (int o = 0; o < 2688; o++) se[o] = tee[o] / (1.f + expf(-tee[o])); }
  f32 xr[2688]; for (int j = 0; j < 42; j++) for (int n = 0; n < 64; n++) { float s = 0; for (int m = 0; m < 64; m++) s += se[j * 64 + m] * H64[m * 64 + n]; xr[j * 64 + n] = s; }
  std::future<void> pre = std::async(std::launch::async, [&] { read_file(MD + "/blk00.bin", wb[0], DITSZ); if (!LD.empty()) read_file(LD + "/lora00.bin", wb[0] + DITSZ, LORASZ); });
  for (int b = 0; b < nblk; b++) {
    pre.get();
    if (b + 1 < nblk) { char nm[32]; snprintf(nm, sizeof nm, "/blk%02d.bin", b + 1); std::string p = MD + nm; uint8_t *dst = wb[(b + 1) & 1]; int bn = b + 1; pre = std::async(std::launch::async, [p, dst, bn] { read_file(p, dst, DITSZ); if (!LD.empty()) { char n2[32]; snprintf(n2, sizeof n2, "/lora%02d.bin", bn); read_file(LD + n2, dst + DITSZ, LORASZ); } }); }
    { char nm[32]; snprintf(nm, sizeof nm, "/adaln%02d.bin", b); FILE *f = fopen((MD + nm).c_str(), "rb"); if (!f || fread(ada.data(), 1, ADASZ, f) != ADASZ) { fprintf(stderr, "adaln %d\n", b); return 1; } fclose(f); }
    const int8_t *q = (const int8_t *)ada.data(); const f32 *sc = (const f32 *)(ada.data() + (size_t)96768 * 2688), *bi = sc + 96768;
    parfor(64512, [&](int row) { const int8_t *w = q + (size_t)row * 2688; float s = 0; for (int i = 0; i < 2688; i++) s += (float)w[i] * xr[i]; mod[row] = s * sc[row] + bi[row]; });   // rows [tag][chunk][5376]: tag 0 = video, 1 = text
    if (nr > 0) parfor(6 * HID, [&](int row) { const int8_t *w = q + (size_t)row * 2688; float s = 0; for (int i = 0; i < 2688; i++) s += (float)w[i] * xr2[i]; mod[12 * HID + row] = s * sc[row] + bi[row]; });   // third set: video tag at the reference timestep
    double c0 = now(); if (win) run_block_w(b & 1, mod, b == 0, b == nblk - 1); else { meta[M_A5] = nimgslabs; meta[M_A6] = nimgslabs + nrefslabs; run_block(b & 1, mod, b == 0, b == nblk - 1, nvalid, nimgslabs, ntok); meta[M_A5] = meta[M_A6] = 0; } tc += now() - c0;
  }
  if (win) get_rows_h(rows.data(), ni + nr + nt); else get_rows(rows.data(), ni + nr + nt);
  if (getenv("R2V_XOUT")) { FILE *f = fopen(getenv("R2V_XOUT"), "wb"); fwrite(rows.data(), 4, rows.size(), f); fclose(f); }
  // ---------------- final layer: adaln (bf16 as f32, silu(t_emb)) -> norm -> (1+scale), shift -> video_out
  f32 sh[HID], scl[HID];
  for (int i = 0; i < 2 * HID; i++) { float s = fin_b[i]; const f32 *w = &fin_w[(size_t)i * 2688]; for (int k = 0; k < 2688; k++) s += w[k] * se[k]; (i < HID ? sh[i] : scl[i - HID]) = s; }
  parfor(ni, [&](int r) { std::vector<f32> hn(HID); rms_row(&rows[(size_t)r * HID], fin_n.data(), hn.data(), HID, 1e-5f); for (int i = 0; i < HID; i++) hn[i] = hn[i] * (1 + scl[i]) + sh[i];
    for (int o = 0; o < PD; o++) { float s = vout_b[o]; const f32 *w = &vout_w[(size_t)o * HID]; for (int i = 0; i < HID; i++) s += w[i] * hn[i]; v[(size_t)r * PD + o] = s; } });
    if (steps > 0) { const float dsg = (float)(SGM[step] - SGM[step + 1]); double sd = 0; for (size_t i = 0; i < x_img.size(); i++) { x_img[i] += dsg * v[i]; sd += (double)x_img[i] * x_img[i]; } printf("step %d/%d: sigma %.4f t %.4f  %.1f s  |x| rms %.3f\n", step + 1, steps, SGM[step], tcur, now() - ts, sqrt(sd / x_img.size())); { int nx = step + 1; FILE *f = fopen((ckpt + ".tmp").c_str(), "wb"); fwrite(&nx, 4, 1, f); fwrite(x_img.data(), 4, x_img.size(), f); fclose(f); rename((ckpt + ".tmp").c_str(), ckpt.c_str()); } }
  }
  if (steps > 0) {   // unpatchify -> latents (24, T, lh, lw), still normalised
    std::vector<f32> lat((size_t)24 * T * lh * lw);
    for (int tt = 0; tt < T; tt++) for (int ph = 0; ph < th; ph++) for (int pw = 0; pw < tw; pw++) for (int c = 0; c < 24; c++) for (int a = 0; a < 2; a++) for (int b2 = 0; b2 < 2; b2++)
      lat[(((size_t)c * T + tt) * lh + ph * 2 + a) * lw + pw * 2 + b2] = x_img[(size_t)((tt * th + ph) * tw + pw) * PD + c * 4 + a * 2 + b2];
    FILE *f = fopen(outp, "wb"); fwrite(lat.data(), 4, lat.size(), f); fclose(f);
  } else { FILE *f = fopen(outp, "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f); }
  printf("forward done: %d blocks %.2f s, total %.1f s\n", nblk, tc, now() - T0); mega_close(h); return 0;
}
