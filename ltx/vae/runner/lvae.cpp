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
  const int TT = 3, TH = ni == 3 * 256 * 256 ? 8 : (int)lround(sqrt(ni / (128.0 * TT))), TW = TH, PF = 8 * (TT - 1) + 1, TP = TH * 32, OV = 2;
  fprintf(stderr, "lvae: tile %d latent frames x %dx%d (%d frames, %d px), in %u (%s) out %u (%s)\n", TT, TH, TW, PF, TP, ni, in16 ? "fp16" : "fp32", no, out16 ? "fp16" : "fp32");
  printf("ready\n"); fflush(stdout);
  std::string line;
  auto runtile = [&](const float* z, float* o) -> bool {   // z [TT][128][TH][TW] fp32
    if (in16) { __fp16* d = (__fp16*)ibuf.data(); for (uint32_t i = 0; i < ni; i++) d[i] = (__fp16)z[i]; } else memcpy(ibuf.data(), z, (size_t)ni * 4);
    Qnn_ErrorHandle_t er = qi.graphExecute(graph, &tin, 1, &tout, 1, nullptr, nullptr); if (er != QNN_SUCCESS) { fprintf(stderr, "execute error 0x%lx\n", (unsigned long)er); return false; }
    if (out16) { const __fp16* sp = (const __fp16*)obuf.data(); for (uint32_t i = 0; i < no; i++) o[i] = (float)sp[i]; } else memcpy(o, obuf.data(), (size_t)no * 4); return true; };
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd, fi, fo; int FL = 0, HL = 0, WL = 0; ss >> cmd >> fi >> FL >> HL >> WL >> fo;
    if (cmd == "quit") break;
    if (cmd == "tile") {   // tile Z.f32 3 TH TW OUT.f32 : one tile, Z fp32 [3][128][TH][TW]
      auto b = readFile(fi); if (b.size() != (size_t)ni * 4) { fprintf(stderr, "bad input size\n"); printf("error\n"); fflush(stdout); continue; }
      std::vector<float> o(no); auto s0 = std::chrono::steady_clock::now(); bool ok = runtile((const float*)b.data(), o.data()); double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
      if (!ok) { printf("error\n"); fflush(stdout); continue; } FILE* f = fopen(fo.c_str(), "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f); printf("done %.0f\n", ms); fflush(stdout); continue; }
    if (cmd == "enc") {   // enc IMG.f32 HL WL OUT.f32 : image fp32 [3][32HL][32WL] in [-1,1] -> normalised latent tokens fp32 [HL*WL][128] (needs the encoder context: input 3x256x256); tiles of 8x8 latents, overlap 2, linear blend
      if (ni != 3 * 256 * 256) { fprintf(stderr, "this context is not the image encoder\n"); printf("error\n"); fflush(stdout); continue; }
      HL = FL; WL = (int)lround(0); std::istringstream s2(line); std::string c2, f2, f3; int a1 = 0, a2 = 0; s2 >> c2 >> f2 >> a1 >> a2 >> f3; HL = a1; WL = a2; fo = f3;
      const int EH = HL * 32, EW = WL * 32, ET = 8, OVL = 2;
      auto b = readFile(fi); if (b.size() != (size_t)3 * EH * EW * 4) { fprintf(stderr, "bad image size\n"); printf("error\n"); fflush(stdout); continue; }
      const float* img = (const float*)b.data();
      auto st2 = [&](int n) { std::vector<int> v; if (n <= ET) { v.push_back(0); return v; } int k = std::max(2, (n - OVL + (ET - OVL) - 1) / (ET - OVL)); for (int i = 0; i < k; i++) v.push_back((int)lround((double)i * (n - ET) / (k - 1))); v.erase(std::unique(v.begin(), v.end()), v.end()); return v; };
      std::vector<int> ys2 = st2(HL), xs2 = st2(WL); std::vector<float> acc2((size_t)HL * WL * 128, 0.f), ws2((size_t)HL * WL, 0.f), tl((size_t)ni), o2(no); bool ok2 = true; auto s0 = std::chrono::steady_clock::now();
      for (int y : ys2) for (int x : xs2) {
        for (int c = 0; c < 3; c++) for (int i = 0; i < 256; i++) for (int j = 0; j < 256; j++) { const int yy = std::min(y * 32 + i, EH - 1), xx = std::min(x * 32 + j, EW - 1); tl[((size_t)c * 256 + i) * 256 + j] = img[((size_t)c * EH + yy) * EW + xx]; }
        if (!runtile(tl.data(), o2.data())) { ok2 = false; break; }
        for (int i = 0; i < ET && y + i < HL; i++) {
          float wy = 1.f; if (y > 0 && i < OVL) wy = (i + 1.f) / (OVL + 1.f); if (y + ET < HL && i >= ET - OVL) wy = std::min(wy, (ET - i) / (OVL + 1.f));
          for (int j = 0; j < ET && x + j < WL; j++) {
            float wx = 1.f; if (x > 0 && j < OVL) wx = (j + 1.f) / (OVL + 1.f); if (x + ET < WL && j >= ET - OVL) wx = std::min(wx, (ET - j) / (OVL + 1.f));
            const float w = wy * wx; const size_t at = (size_t)(y + i) * WL + x + j; ws2[at] += w;
            for (int c = 0; c < 128; c++) acc2[at * 128 + c] += o2[((size_t)c * ET + i) * ET + j] * w; } }
      }
      if (!ok2) { printf("error\n"); fflush(stdout); continue; }
      for (size_t i = 0; i < (size_t)HL * WL; i++) { const float inv = 1.f / std::max(ws2[i], 1e-6f); for (int c = 0; c < 128; c++) acc2[i * 128 + c] *= inv; }
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
      FILE* f = fopen(fo.c_str(), "wb"); fwrite(acc2.data(), 4, acc2.size(), f); fclose(f); printf("done %.0f\n", ms); fflush(stdout); continue; }
    if (cmd != "dec") continue;   // dec TOK.f32 FL HL WL OUT.f32 : DiT tokens [FL*HL*WL][128] (normalised latent) -> raw fp32 [8*(FL-1)+1][3][32HL][32WL] in [-1,1]; FL >= 3: windows of 3 latent frames at stride 1, window 0 gives all 17 frames, window w>0 its last 8 (frames 9..16) = the pixels of latent w+2
    if (FL < TT) { fprintf(stderr, "need at least %d latent frames\n", TT); printf("error\n"); fflush(stdout); continue; }
    auto b = readFile(fi); if (b.size() != (size_t)FL * HL * WL * 128 * 4) { fprintf(stderr, "bad token file\n"); printf("error\n"); fflush(stdout); continue; }
    const float* tok = (const float*)b.data(); const int H = HL * 32, W = WL * 32;
    auto starts = [&](int n) { std::vector<int> v; if (n <= TH) { v.push_back(0); return v; } int k = std::max(2, (n - OV + (TH - OV) - 1) / (TH - OV)); for (int i = 0; i < k; i++) v.push_back((int)lround((double)i * (n - TH) / (k - 1))); v.erase(std::unique(v.begin(), v.end()), v.end()); return v; };
    std::vector<int> ys = starts(HL), xs = starts(WL);
    std::vector<float> acc((size_t)PF * 3 * H * W), wsum((size_t)H * W), tile((size_t)ni), o(no); bool ok = true; double ms = 0;
    FILE* fout = fopen(fo.c_str(), "wb"); if (!fout) { printf("error\n"); fflush(stdout); continue; }
    for (int w0 = 0; w0 + TT <= FL && ok; w0++) {
      std::fill(acc.begin(), acc.end(), 0.f); std::fill(wsum.begin(), wsum.end(), 0.f); auto s0 = std::chrono::steady_clock::now();
      for (int y : ys) for (int x : xs) {
        for (int t = 0; t < TT; t++) for (int c = 0; c < 128; c++) for (int i = 0; i < TH; i++) for (int j = 0; j < TW; j++) {
          const int yy = std::min(y + i, HL - 1), xx = std::min(x + j, WL - 1); tile[(((size_t)t * 128 + c) * TH + i) * TW + j] = tok[(((size_t)(w0 + t) * HL + yy) * WL + xx) * 128 + c]; }
        if (!runtile(tile.data(), o.data())) { ok = false; break; }
        const int px = x * 32, py = y * 32, r = OV * 32, M = std::min(16, r / 4);
        const bool l = x > 0, rr = x + TW < WL, u = y > 0, d = y + TH < HL;
        for (int i = 0; i < TP && py + i < H; i++) {
          float wy = 1.f; if (u && i < r) wy = i < M ? 0.f : (i - M + 1.f) / (r - M + 1.f); if (d && i >= TP - r) wy = std::min(wy, TP - i <= M ? 0.f : (TP - i - M) / (r - M + 1.f));
          for (int j = 0; j < TP && px + j < W; j++) {
            float wx = 1.f; if (l && j < r) wx = j < M ? 0.f : (j - M + 1.f) / (r - M + 1.f); if (rr && j >= TP - r) wx = std::min(wx, TP - j <= M ? 0.f : (TP - j - M) / (r - M + 1.f));
            const float w = wy * wx; const size_t at = (size_t)(py + i) * W + px + j; wsum[at] += w;
            for (int f = 0; f < PF; f++) for (int c = 0; c < 3; c++) acc[(((size_t)f * 3 + c) * H) * W + at] += std::min(1.f, std::max(-1.f, o[(((size_t)f * 3 + c) * TP + i) * TP + j])) * w; } }
      }
      if (!ok) break;
      ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
      for (size_t i = 0; i < (size_t)H * W; i++) { const float inv = 1.f / std::max(wsum[i], 1e-6f); for (int f = 0; f < PF; f++) for (int c = 0; c < 3; c++) acc[(((size_t)f * 3 + c) * H) * W + i] *= inv; }
      const int f0 = w0 == 0 ? 0 : PF - 8;   // window 0: frames 0..16, later windows: the last 8 frames
      fwrite(acc.data() + (size_t)f0 * 3 * H * W, 4, (size_t)(PF - f0) * 3 * H * W, fout);
      fprintf(stderr, "lvae: window %d/%d\n", w0 + 1, FL - TT + 1);
    }
    fclose(fout);
    if (!ok) { printf("error\n"); fflush(stdout); continue; }
    printf("done %.0f\n", ms); fflush(stdout);
  }
  return 0;
}
