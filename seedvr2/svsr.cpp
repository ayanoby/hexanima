// svsr: SeedVR2-3B one-step image super-resolution on the phone. Orchestrates: VAE encoder tiles (QNN, vaerun) -> DiT (svrun, DSP MODE 100) -> VAE decoder tiles (QNN, vaerun).
// usage: svsr WORKDIR PACK VAEDIR IN.rgb W0 H0 OUT.rgb W H [SEED]
//   IN.rgb: uint8 RGB [H0][W0][3] (any size); it is resized bicubically to W x H (multiples of 16, W0 = W and H0 = H: no resize); OUT.rgb: uint8 RGB [H][W][3]. WORKDIR: scratch directory.
// env: SV_BIN = directory with the executables (vaerun, svrun; default: the directory of svsr), names SV_VAERUN / SV_SVRUN (default vaerun / svrun, or libvaerun.so / libsvrun.so when SV_LIB=1)
//      SV_QNN = directory with libQnnHtp.so / libQnnSystem.so (default SV_BIN); the VAE contexts are VAEDIR/svr_enc_t64.bin and svr_dec_t64.bin (512 px tiles = 64 latents)
// progress on stdout: "stage <name> <done> <total>".
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>

static const int TP = 512, TL = 64;        // tile in pixels / latents
static const float SCALE = 0.9152f;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void die(const char* m, const std::string& a = "") { fprintf(stderr, "svsr: %s %s\n", m, a.c_str()); exit(1); }
static void wr(const std::string& p, const void* d, size_t n) { FILE* f = fopen(p.c_str(), "wb"); if (!f) die("cannot write", p); fwrite(d, 1, n, f); fclose(f); }
static std::vector<float> rdf(const std::string& p, size_t n) { std::vector<float> v(n); FILE* f = fopen(p.c_str(), "rb"); if (!f) die("cannot read", p); size_t g = fread(v.data(), 4, n, f); fclose(f); if (g != n) die("short read", p); return v; }

struct Child { FILE* in; FILE* out; pid_t pid; };
static Child spawn(const std::string& exe, const std::vector<std::string>& args) {
  int pin[2], pout[2]; if (pipe(pin) || pipe(pout)) die("pipe");
  pid_t pid = fork();
  if (pid == 0) {
    dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2); close(pin[1]); close(pout[0]);
    std::vector<char*> av; av.push_back((char*)exe.c_str()); for (auto& a : args) av.push_back((char*)a.c_str()); av.push_back(nullptr);
    execv(exe.c_str(), av.data()); _exit(127);
  }
  close(pin[0]); close(pout[1]); return {fdopen(pin[1], "w"), fdopen(pout[0], "r"), pid};
}
static bool waitline(Child& c, const char* a) { char l[4096]; while (fgets(l, sizeof l, c.out)) { if (!strncmp(l, a, strlen(a))) return true; if (!strncmp(l, "error", 5)) return false; if (strstr(l, "ERROR") || strstr(l, "failed")) fprintf(stderr, "child: %s", l); } return false; }
static void quit(Child& c) { fprintf(c.in, "quit\n"); fflush(c.in); fclose(c.in); int st; waitpid(c.pid, &st, 0); fclose(c.out); }

// tile starts covering [0, n) with tiles of size t (n >= t): 0, step, ... and the last one flush with the end
static std::vector<int> starts(int n, int t, int step) { std::vector<int> s; if (n <= t) { s.push_back(0); return s; } for (int p = 0; p + t < n; p += step) s.push_back(p); s.push_back(n - t); s.erase(std::unique(s.begin(), s.end()), s.end()); return s; }
static inline int mirror(int i, int n) { if (n == 1) return 0; int p = 2 * (n - 1); i %= p; if (i < 0) i += p; return i < n ? i : p - i; }
// blend weight of position i inside a tile of size t: ramps up over `ov` samples at the sides that touch another tile
static inline float ramp(int i, int t, int ov, bool left, bool right) { float w = 1.f; if (left && i < ov) w = std::min(w, (i + 0.5f) / ov); if (right && t - 1 - i < ov) w = std::min(w, (t - 1 - i + 0.5f) / ov); return w; }

int main(int argc, char** argv) {
  if (argc < 10) { fprintf(stderr, "usage: svsr WORKDIR PACK VAEDIR IN.rgb W0 H0 OUT.rgb W H [SEED]\n"); return 1; }
  const std::string wd = argv[1], pack = argv[2], vd = argv[3]; const int W0 = atoi(argv[5]), H0 = atoi(argv[6]), W = atoi(argv[8]), H = atoi(argv[9]); const unsigned seed = argc > 10 ? (unsigned)atoi(argv[10]) : 666;
  if (W % 16 || H % 16 || W < 16 || H < 16) die("size must be a multiple of 16");
  setvbuf(stdout, 0, _IONBF, 0); double t0 = now();
  std::string self = argv[0]; std::string bin = getenv("SV_BIN") ? getenv("SV_BIN") : self.substr(0, self.rfind('/') == std::string::npos ? 0 : self.rfind('/'));
  const bool lib = getenv("SV_LIB") != nullptr; std::string vaerun = bin + (lib ? "/libvaerun.so" : "/vaerun"), svrun = bin + (lib ? "/libsvrun.so" : "/svrun"), qnn = getenv("SV_QNN") ? getenv("SV_QNN") : bin;
  const int h8 = H / 8, w8 = W / 8;
  std::vector<uint8_t> img((size_t)W * H * 3);
  { std::vector<uint8_t> src((size_t)W0 * H0 * 3); FILE* f = fopen(argv[4], "rb"); if (!f || fread(src.data(), 1, src.size(), f) != src.size()) die("cannot read image"); fclose(f);
    if (W0 == W && H0 == H) img = src;
    else {   // bicubic (Keys, a = -0.5 like PIL), separable, edge clamped
      auto cub = [](float x) { x = fabsf(x); return x < 1 ? (1.5f * x - 2.5f) * x * x + 1 : x < 2 ? ((-0.5f * x + 2.5f) * x - 4) * x + 2 : 0.f; };
      const bool up = W >= W0; const float sx = (float)W0 / W, sy = (float)H0 / H;   // only upscaling is expected; a downscale is a plain (aliasing) bicubic
      (void)up; std::vector<float> tmp((size_t)H0 * W * 3);
      for (int x = 0; x < W; x++) { float c = (x + 0.5f) * sx - 0.5f; int i0 = (int)floorf(c) - 1; float wt[4], sum = 0; for (int k = 0; k < 4; k++) { wt[k] = cub(c - (i0 + k)); sum += wt[k]; }
        for (int y = 0; y < H0; y++) for (int ch = 0; ch < 3; ch++) { float a = 0; for (int k = 0; k < 4; k++) a += wt[k] * src[((size_t)y * W0 + std::min(std::max(i0 + k, 0), W0 - 1)) * 3 + ch]; tmp[((size_t)y * W + x) * 3 + ch] = a / sum; } }
      for (int y = 0; y < H; y++) { float c = (y + 0.5f) * sy - 0.5f; int i0 = (int)floorf(c) - 1; float wt[4], sum = 0; for (int k = 0; k < 4; k++) { wt[k] = cub(c - (i0 + k)); sum += wt[k]; }
        for (int x = 0; x < W; x++) for (int ch = 0; ch < 3; ch++) { float a = 0; for (int k = 0; k < 4; k++) a += wt[k] * tmp[((size_t)std::min(std::max(i0 + k, 0), H0 - 1) * W + x) * 3 + ch]; img[((size_t)y * W + x) * 3 + ch] = (uint8_t)std::min(255.f, std::max(0.f, a / sum + 0.5f)); } }
    } }
  // ---------------- encode (tiles of 512 px, overlap >= 128 px; images smaller than a tile are mirror padded)
  std::vector<float> lat((size_t)16 * h8 * w8, 0.f), lw((size_t)h8 * w8, 0.f);
  {
    Child c = spawn(vaerun, {qnn + "/libQnnHtp.so", qnn + "/libQnnSystem.so", vd + "/svr_enc_t64.bin"}); if (!waitline(c, "ready")) die("encoder runner failed");
    const auto xs = starts(W, TP, TP - 128), ys = starts(H, TP, TP - 128); int done = 0, tot = (int)(xs.size() * ys.size());
    std::vector<float> in((size_t)3 * TP * TP);
    for (int ty : ys) for (int tx : xs) {
      for (int c3 = 0; c3 < 3; c3++) for (int y = 0; y < TP; y++) { int sy = ty + y; if (H < TP) sy = mirror(y, H); for (int x = 0; x < TP; x++) { int sx = tx + x; if (W < TP) sx = mirror(x, W); in[((size_t)c3 * TP + y) * TP + x] = img[((size_t)sy * W + sx) * 3 + c3] / 127.5f - 1.f; } }
      wr(wd + "/enc_in.raw", in.data(), in.size() * 4); fprintf(c.in, "run %s/enc_in.raw %s/enc_out.raw\n", wd.c_str(), wd.c_str()); fflush(c.in); if (!waitline(c, "done")) die("encode failed");
      auto z = rdf(wd + "/enc_out.raw", (size_t)16 * TL * TL);
      const int lx0 = tx / 8, ly0 = ty / 8, tw = std::min(TL, w8 - lx0), th = std::min(TL, h8 - ly0); const bool l = tx > 0, r = tx + TP < W, u = ty > 0, d = ty + TP < H;
      for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) { float wgt = ramp(x, TL, 16, l, r) * ramp(y, TL, 16, u, d); size_t o = (size_t)(ly0 + y) * w8 + lx0 + x; lw[o] += wgt; for (int c3 = 0; c3 < 16; c3++) lat[(size_t)c3 * h8 * w8 + o] += wgt * z[((size_t)c3 * TL + y) * TL + x]; }
      printf("stage encode %d %d\n", ++done, tot);
    }
    quit(c);
    for (size_t o = 0; o < lw.size(); o++) for (int c3 = 0; c3 < 16; c3++) lat[(size_t)c3 * h8 * w8 + o] = lat[(size_t)c3 * h8 * w8 + o] / lw[o] * SCALE;
  }
  printf("encoded %.1fs\n", now() - t0);
  // ---------------- DiT (one step)
  std::vector<float> noise((size_t)16 * h8 * w8); { std::mt19937 g(seed); std::normal_distribution<float> nd(0.f, 1.f); for (auto& v : noise) v = nd(g); }
  wr(wd + "/xt.raw", noise.data(), noise.size() * 4); wr(wd + "/cond.raw", lat.data(), lat.size() * 4);
  { printf("stage dit 0 1\n"); double td = now(); Child c = spawn(svrun, {pack, wd + "/xt.raw", wd + "/cond.raw", wd + "/v.raw", std::to_string(h8), std::to_string(w8)});
    char l[4096]; while (fgets(l, sizeof l, c.out)) fputs(l, stdout); int st; fclose(c.in); waitpid(c.pid, &st, 0); fclose(c.out); if (!WIFEXITED(st) || WEXITSTATUS(st)) die("svrun failed"); printf("stage dit 1 1\ndit %.1fs\n", now() - td); }
  auto v = rdf(wd + "/v.raw", (size_t)16 * h8 * w8);
  for (size_t i = 0; i < v.size(); i++) lat[i] = (noise[i] - v[i]) / SCALE;
  // ---------------- decode (latent tiles of 64, overlap >= 16)
  std::vector<float> out((size_t)3 * H * W, 0.f), ow((size_t)H * W, 0.f);
  {
    Child c = spawn(vaerun, {qnn + "/libQnnHtp.so", qnn + "/libQnnSystem.so", vd + "/svr_dec_t64.bin"}); if (!waitline(c, "ready")) die("decoder runner failed");
    const auto xs = starts(w8, TL, TL - 16), ys = starts(h8, TL, TL - 16); int done = 0, tot = (int)(xs.size() * ys.size()); std::vector<float> in((size_t)16 * TL * TL);
    for (int ty : ys) for (int tx : xs) {
      for (int c3 = 0; c3 < 16; c3++) for (int y = 0; y < TL; y++) { int sy = h8 < TL ? mirror(y, h8) : ty + y; for (int x = 0; x < TL; x++) { int sx = w8 < TL ? mirror(x, w8) : tx + x; in[((size_t)c3 * TL + y) * TL + x] = lat[((size_t)c3 * h8 + sy) * w8 + sx]; } }
      wr(wd + "/dec_in.raw", in.data(), in.size() * 4); fprintf(c.in, "run %s/dec_in.raw %s/dec_out.raw\n", wd.c_str(), wd.c_str()); fflush(c.in); if (!waitline(c, "done")) die("decode failed");
      auto o = rdf(wd + "/dec_out.raw", (size_t)3 * TP * TP);
      const int px0 = tx * 8, py0 = ty * 8, tw = std::min(TP, W - px0), th = std::min(TP, H - py0); const bool l = tx > 0, r = tx + TL < w8, u = ty > 0, d = ty + TL < h8;
      for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) { float wgt = ramp(x, TP, 128, l, r) * ramp(y, TP, 128, u, d); size_t p = (size_t)(py0 + y) * W + px0 + x; ow[p] += wgt; for (int c3 = 0; c3 < 3; c3++) out[(size_t)c3 * H * W + p] += wgt * o[((size_t)c3 * TP + y) * TP + x]; }
      printf("stage decode %d %d\n", ++done, tot);
    }
    quit(c);
  }
  std::vector<uint8_t> rgb((size_t)W * H * 3);
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int c3 = 0; c3 < 3; c3++) { float p = out[(size_t)c3 * H * W + (size_t)y * W + x] / ow[(size_t)y * W + x]; p = (p + 1.f) * 127.5f; rgb[((size_t)y * W + x) * 3 + c3] = (uint8_t)std::min(255.f, std::max(0.f, p + 0.5f)); }
  wr(argv[7], rgb.data(), rgb.size()); printf("done %.1fs\n", now() - t0); return 0;
}
