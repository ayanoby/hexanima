// henc: H3 video-VAE image encoder tile (anima/h3/vae/enc2d_h3.py, one 256x256 QNN context) - derived from lvae (LTX)
// kvae (FLUX.3 Klein; derived from vaetile of the Qwen port): usage kvae BACKEND SYSTEM CONTEXT.bin [vae_bn.f32]; commands: run Z W H OUT (Z = fp32 [32][H/8][W/8]) | runtok TOK W H OUT (TOK = fp32 [H/16*W/16][128], the DiT output)
// ORIGINAL: vaetile (Qwen-Image-2.1): tiled decode with ONE fixed-size decoder context (z tile [64,T,T] -> 16T px, RGB of the 4 output channels). Derived from vaerun.
// commands: "run Z W H OUT" (Z = fp32 [64][H/16][W/16]; OUT = fp32 [3][H][W] in [-1,1]), "quit".
// (vaerun original header:) persistent runner for ONE single-input/single-output QNN HTP context binary (the TAEHV VAE decoder). Tensors may be fp32 or fp16 in the
// binary; files are always fp32. Commands on stdin: "run IN OUT" -> prints "done <ms>"; "quit".
// usage: vaerun BACKEND.so SYSTEM.so CONTEXT.bin
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

static QNN_INTERFACE_VER_TYPE qi;
static QNN_SYSTEM_INTERFACE_VER_TYPE qsi;

static bool check(Qnn_ErrorHandle_t e, const char* what) {
  if (e != QNN_SUCCESS) { fprintf(stderr, "ERROR %s: 0x%lx\n", what, (unsigned long)e); return false; }
  return true;
}

static std::vector<uint8_t> readFile(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  std::vector<uint8_t> v;
  if (!f) return v;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  v.resize(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) v.clear();
  fclose(f);
  return v;
}

static void setBuf(Qnn_Tensor_t& t, void* data, uint32_t bytes) {
  if (t.version == QNN_TENSOR_VERSION_1) {
    t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf.data = data; t.v1.clientBuf.dataSize = bytes;
  } else {
    t.v2.memType = QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf.data = data; t.v2.clientBuf.dataSize = bytes;
  }
}
static float g_bn[256];   // BatchNorm of the latent tokens: mean[128], sqrt(var + eps)[128]
static const char* tName(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_1 ? t.v1.name : t.v2.name; }
static uint32_t tElems(const Qnn_Tensor_t& t) {
  uint32_t rank = t.version == QNN_TENSOR_VERSION_1 ? t.v1.rank : t.v2.rank;
  const uint32_t* d = t.version == QNN_TENSOR_VERSION_1 ? t.v1.dimensions : t.v2.dimensions;
  uint32_t n = 1;
  for (uint32_t i = 0; i < rank; i++) n *= d[i];
  return n;
}


static uint32_t tType(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_1 ? t.v1.dataType : t.v2.dataType; }


int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: vaerun BACKEND SYSTEM CONTEXT.bin\n"); return 2; }
  void* hb = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* hs = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!hb || !hs) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
  typedef Qnn_ErrorHandle_t (*GetP)(const QnnInterface_t***, uint32_t*);
  typedef Qnn_ErrorHandle_t (*GetSP)(const QnnSystemInterface_t***, uint32_t*);
  const QnnInterface_t** provs = nullptr; uint32_t np = 0;
  if (!check(((GetP)dlsym(hb, "QnnInterface_getProviders"))(&provs, &np), "getProviders")) return 1;
  qi = provs[0]->QNN_INTERFACE_VER_NAME;
  const QnnSystemInterface_t** sprovs = nullptr; uint32_t nsp = 0;
  if (!check(((GetSP)dlsym(hs, "QnnSystemInterface_getProviders"))(&sprovs, &nsp), "getSystemProviders")) return 1;
  qsi = sprovs[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  Qnn_BackendHandle_t backend = nullptr;
  if (!check(qi.backendCreate(nullptr, nullptr, &backend), "backendCreate")) return 1;
  Qnn_DeviceHandle_t device = nullptr;
  if (!check(qi.deviceCreate(nullptr, nullptr, &device), "deviceCreate")) return 1;
    // burst power profile (what qnn-net-run --perf_profile burst does)
  QnnDevice_Infrastructure_t infra = nullptr;
  if (qi.deviceGetInfrastructure && qi.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
    auto* hi = (QnnHtpDevice_Infrastructure_t*)infra;
    uint32_t pid = 0;
    if (hi->perfInfra.createPowerConfigId(0, 0, &pid) == QNN_SUCCESS) {
      QnnHtpPerfInfrastructure_PowerConfig_t dc = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
      dc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
      auto& d = dc.dcvsV3Config;
      d.contextId = pid; d.setDcvsEnable = 1; d.dcvsEnable = 0;
      d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
      d.setSleepLatency = 1; d.sleepLatency = 40; d.setSleepDisable = 1; d.sleepDisable = 1;
      d.setBusParams = 1; d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      d.setCoreParams = 1; d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      QnnHtpPerfInfrastructure_PowerConfig_t rc = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
      rc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
      rc.rpcControlLatencyConfig = 100;
      const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dc, &rc, nullptr};
      hi->perfInfra.setPowerConfig(pid, cfgs);
    }
  }


  std::vector<uint8_t> bin = readFile(argv[3]);
  if (bin.empty()) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
  QnnSystemContext_Handle_t sh = nullptr;
  if (!check(qsi.systemContextCreate(&sh), "systemContextCreate")) return 1;
  const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
  if (!check(qsi.systemContextGetBinaryInfo(sh, bin.data(), bin.size(), &bi, &bis), "getBinaryInfo")) return 1;
  const QnnSystemContext_GraphInfo_t* g = nullptr;
  if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) g = bi->contextBinaryInfoV1.graphs;
  else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) g = bi->contextBinaryInfoV2.graphs;
  else g = bi->contextBinaryInfoV3.graphs;
  const char* gname; uint32_t nin, nout; Qnn_Tensor_t *gin, *gout;
  if (g->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { auto& v = g->graphInfoV1; gname = v.graphName; nin = v.numGraphInputs; gin = v.graphInputs; nout = v.numGraphOutputs; gout = v.graphOutputs; }
  else if (g->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { auto& v = g->graphInfoV2; gname = v.graphName; nin = v.numGraphInputs; gin = v.graphInputs; nout = v.numGraphOutputs; gout = v.graphOutputs; }
  else { auto& v = g->graphInfoV3; gname = v.graphName; nin = v.numGraphInputs; gin = v.graphInputs; nout = v.numGraphOutputs; gout = v.graphOutputs; }
  Qnn_ContextHandle_t ctx = nullptr; Qnn_GraphHandle_t graph = nullptr;
  if (!check(qi.contextCreateFromBinary(backend, device, nullptr, bin.data(), bin.size(), &ctx, nullptr), "contextCreateFromBinary")) return 1;
  if (!check(qi.graphRetrieve(ctx, gname, &graph), "graphRetrieve")) return 1;
  if (nin != 1 || nout != 1) { fprintf(stderr, "expected 1 input and 1 output\n"); return 1; }
  Qnn_Tensor_t tin = gin[0], tout = gout[0];
  uint32_t ni = tElems(tin), no = tElems(tout);
  bool in16 = tType(tin) == QNN_DATATYPE_FLOAT_16, out16 = tType(tout) == QNN_DATATYPE_FLOAT_16;
  std::vector<uint8_t> ibuf(ni * (in16 ? 2 : 4)), obuf(no * (out16 ? 2 : 4));
  setBuf(tin, ibuf.data(), ibuf.size()); setBuf(tout, obuf.data(), obuf.size());
  // tile: latent [TT][128][TH][TW] -> pixels [PF][3][TH*32][TW*32]

  const bool full_enc = ni == 3 * 256 * 256 && no == 48 * 16 * 16;   // other shapes (cut graphs for debugging) only support "raw"
  fprintf(stderr, "henc: tile 256x256 -> moments 48x16x16, in %s out %s\n", in16 ? "fp16" : "fp32", out16 ? "fp16" : "fp32");
  printf("ready\n"); fflush(stdout);
  std::string line;
  auto runtile = [&](const float* z, float* o) -> bool {   // z [TT][128][TH][TW] fp32
    if (in16) { __fp16* d = (__fp16*)ibuf.data(); for (uint32_t i = 0; i < ni; i++) d[i] = (__fp16)z[i]; } else memcpy(ibuf.data(), z, (size_t)ni * 4);
    Qnn_ErrorHandle_t er = qi.graphExecute(graph, &tin, 1, &tout, 1, nullptr, nullptr); if (er != QNN_SUCCESS) { fprintf(stderr, "execute error 0x%lx\n", (unsigned long)er); return false; }
    if (out16) { const __fp16* sp = (const __fp16*)obuf.data(); for (uint32_t i = 0; i < no; i++) o[i] = (float)sp[i]; } else memcpy(o, obuf.data(), (size_t)no * 4); return true; };
  // official split_tiles (klvae.py): tile 256, minimal overlap 64, overlaps grown in steps of 16 px
  auto split_tiles = [&](int len, std::vector<int>& idx, std::vector<int>& ov) {
    idx = {0}; ov.clear(); const int tile = 256, omin = 64; if (tile >= len) return;
    int N = (len + tile - 1) / tile, remaining; std::vector<int> o;
    for (;;) { o.assign(N - 1, omin); remaining = tile * N - omin * (N - 1) - len; if (remaining < 0) N++; else break; }
    for (int i = 0; i < remaining / 16; i++) o[i % (N - 1)] += 16;
    for (int i = 0; i < N - 1; i++) idx.push_back(idx.back() + tile - o[i]); ov = o; };
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd, fi, fo, fm; int H = 0, W = 0; ss >> cmd >> fi >> H >> W >> fo >> fm;
    if (cmd == "quit") break;
    if (cmd == "raw") {   // raw IN.f32 0 0 OUT.f32 : run the context once on fp32 input of ni floats, write no floats
      auto b = readFile(fi); if (b.size() != (size_t)ni * 4) { fprintf(stderr, "raw: need %u floats\n", ni); printf("error\n"); fflush(stdout); continue; }
      std::vector<float> o(no); if (!runtile((const float*)b.data(), o.data())) { printf("error\n"); fflush(stdout); continue; } FILE* f = fopen(fo.c_str(), "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f); printf("done %u\n", no); fflush(stdout); continue; }
    if (cmd != "enc" || !full_enc) continue;   // enc IMG.f32 H W OUT.f32 GLUEDIR : image fp32 [3][H][W] ImageNet-normalised (H, W multiples of 16, >= 256) -> normalised latent fp32 [24][H/16][W/16] = (sample(moments) - mean) / std
    auto b = readFile(fi); if (H < 256 || W < 256 || H % 16 || W % 16 || b.size() != (size_t)3 * H * W * 4) { fprintf(stderr, "bad image %dx%d\n", H, W); printf("error\n"); fflush(stdout); continue; }
    const float* img = (const float*)b.data(); std::vector<int> yi, yo, xi, xo; split_tiles(H, yi, yo); split_tiles(W, xi, xo);
    const int ni_ = yi.size(), nj_ = xi.size(); std::vector<std::vector<float>> mom((size_t)ni_ * nj_, std::vector<float>(no)); std::vector<float> tl((size_t)ni), o2(no); bool ok = true; auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ni_ && ok; i++) for (int j = 0; j < nj_ && ok; j++) {
      for (int c = 0; c < 3; c++) for (int y = 0; y < 256; y++) memcpy(&tl[((size_t)c * 256 + y) * 256], &img[((size_t)c * H + yi[i] + y) * W + xi[j]], 256 * 4);
      ok = runtile(tl.data(), mom[(size_t)i * nj_ + j].data()); }
    if (!ok) { printf("error\n"); fflush(stdout); continue; }
    // blend (official tiled_encode, in latent units): y with the raw tile above, then x with the raw tile on the left, then cut the overlaps at the far edges
    const int LH = H / 16, LW = W / 16; std::vector<float> z((size_t)48 * LH * LW, 0.f);
    for (int i = 0; i < ni_; i++) for (int j = 0; j < nj_; j++) {
      std::vector<float> t = mom[(size_t)i * nj_ + j]; const int th = 16, tw = 16;
      if (i > 0) { const auto& a = mom[(size_t)(i - 1) * nj_ + j]; const int e = std::min(yo[i - 1] / 16, th); for (int c = 0; c < 48; c++) for (int p = 0; p < e; p++) { const float wa = 1.f - (float)p / e, wb = (float)p / e; for (int x = 0; x < tw; x++) t[((size_t)c * th + p) * tw + x] = a[((size_t)c * th + th - e + p) * tw + x] * wa + t[((size_t)c * th + p) * tw + x] * wb; } }
      if (j > 0) { const auto& a = mom[(size_t)i * nj_ + j - 1]; const int e = std::min(xo[j - 1] / 16, tw); for (int c = 0; c < 48; c++) for (int y = 0; y < th; y++) for (int p = 0; p < e; p++) { const float wa = 1.f - (float)p / e, wb = (float)p / e; t[((size_t)c * th + y) * tw + p] = a[((size_t)c * th + y) * tw + tw - e + p] * wa + t[((size_t)c * th + y) * tw + p] * wb; } }
      const int keh = i < ni_ - 1 ? th - yo[i] / 16 : th, kew = j < nj_ - 1 ? tw - xo[j] / 16 : tw; int oy = 0, ox = 0; for (int k = 0; k < i; k++) oy += 16 - yo[k] / 16; for (int k = 0; k < j; k++) ox += 16 - xo[k] / 16;
      for (int c = 0; c < 48; c++) for (int y = 0; y < keh; y++) memcpy(&z[((size_t)c * LH + oy + y) * LW + ox], &t[((size_t)c * th + y) * tw], (size_t)kew * 4); }
    // sample (DiagonalGaussian: logvar clamped to [-30, 20], std = exp(0.5 logvar)), then normalise with the VAE latent statistics
    std::vector<float> lm(24), ls(24); { auto a = readFile(fm + "/lat_mean.f32"), c = readFile(fm + "/lat_std.f32"); if (a.size() != 96 || c.size() != 96) { fprintf(stderr, "lat_mean/std missing in %s\n", fm.c_str()); printf("error\n"); fflush(stdout); continue; } memcpy(lm.data(), a.data(), 96); memcpy(ls.data(), c.data(), 96); }
    std::mt19937_64 rng(1234); std::normal_distribution<float> nd(0.f, 1.f); std::vector<float> out((size_t)24 * LH * LW);
    for (int c = 0; c < 24; c++) for (size_t k = 0; k < (size_t)LH * LW; k++) { const float mean = z[(size_t)c * LH * LW + k], lv = std::min(20.f, std::max(-30.f, z[(size_t)(24 + c) * LH * LW + k])); out[(size_t)c * LH * LW + k] = ((mean + (getenv("HENC_MEAN") ? 0.f : expf(0.5f * lv) * nd(rng))) - lm[c]) / ls[c]; }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count(); FILE* f = fopen(fo.c_str(), "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f); printf("done %.0f\n", ms); fflush(stdout);
  }
  return 0;
}
