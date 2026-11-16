// h3vae (MiniMax-H3-Image): tiled decode with ONE fixed-size ViT decoder context (z tile [24,16,16] -> 256x256 px), tiling and blending exactly like diffusers' AutoencoderKLMiniMaxH3Image._decode
// (tile 256 px, minimum overlap 64 px, linear blend). Derived from vaetile.
// commands: "run Z W H OUT" (Z = fp32 [24][H/16][W/16] already denormalised (latents*std+mean); OUT = fp32 [3][H][W] in [-1,1] = 2*clamp(img*pixel_std+pixel_mean,0,1)-1), "quit". H, W >= 256.
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
  fprintf(stderr, "vaerun: in %u elems (%s), out %u elems (%s)\n", ni, in16 ? "fp16" : "fp32", no, out16 ? "fp16" : "fp32");
  const int T = 16, TP = 256;
  if (ni != (uint32_t)24 * T * T || no != (uint32_t)3 * TP * TP) { fprintf(stderr, "unexpected context shapes\n"); return 1; }
  printf("ready\n"); fflush(stdout);
  static const float PM[3] = {0.485f, 0.456f, 0.406f}, PS[3] = {0.229f, 0.224f, 0.225f};
  auto split = [&](int length, std::vector<int>& st, std::vector<int>& ov) {   // diffusers _split_tiles (tile 256, min overlap 64, ratio 16)
    st.clear(); ov.clear(); if (TP >= length) { st.push_back(0); return; }
    int nt = (length + TP - 1) / TP; while (TP * nt - 64 * (nt - 1) - length < 0) nt++;
    ov.assign(nt - 1, 64); int rem = TP * nt - 64 * (nt - 1) - length; for (int i = 0; i < rem / 16; i++) ov[i % (nt - 1)] += 16;
    st.push_back(0); for (int i = 0; i < nt - 1; i++) st.push_back(st.back() + TP - ov[i]); };
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd, fi, fo; int W = 0, H = 0; ss >> cmd >> fi >> W >> H >> fo;
    if (cmd == "quit") break;
    if (cmd != "run" || W % 16 || H % 16 || W < TP || H < TP) { printf("error\n"); fflush(stdout); continue; }
    const int zw = W / 16, zh = H / 16; auto b = readFile(fi);
    if (b.size() != (size_t)24 * zh * zw * 4) { fprintf(stderr, "bad input size %s\n", fi.c_str()); printf("error\n"); fflush(stdout); continue; }
    const float* z = (const float*)b.data();
    std::vector<int> ys, yo, xs, xo; split(H, ys, yo); split(W, xs, xo);
    const int ny = ys.size(), nx = xs.size(); std::vector<std::vector<float>> tiles((size_t)ny * nx); std::vector<float> tile((size_t)24 * T * T);
    auto s0 = std::chrono::steady_clock::now(); bool ok = true;
    for (int a = 0; a < ny && ok; a++) for (int c2 = 0; c2 < nx && ok; c2++) {
      const int y = ys[a] / 16, x = xs[c2] / 16;
      for (int c = 0; c < 24; c++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) tile[((size_t)c * T + i) * T + j] = z[((size_t)c * zh + y + i) * zw + x + j];
      if (in16) { __fp16* d = (__fp16*)ibuf.data(); for (uint32_t i = 0; i < ni; i++) d[i] = (__fp16)tile[i]; } else memcpy(ibuf.data(), tile.data(), (size_t)ni * 4);
      if (qi.graphExecute(graph, &tin, 1, &tout, 1, nullptr, nullptr) != QNN_SUCCESS) { fprintf(stderr, "execute error at tile %d,%d\n", a, c2); ok = false; break; }
      std::vector<float>& o = tiles[(size_t)a * nx + c2]; o.resize((size_t)no);
      if (out16) { const __fp16* sp = (const __fp16*)obuf.data(); for (uint32_t i = 0; i < no; i++) o[i] = (float)sp[i]; } else memcpy(o.data(), obuf.data(), (size_t)no * 4);
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
    if (!ok) { printf("error\n"); fflush(stdout); continue; }
    // stitch: blend each tile with the ORIGINAL up / left neighbour over the overlap, then drop the trailing overlap (diffusers _stitch_tiles)
    std::vector<float> out((size_t)3 * H * W);
    for (int a = 0; a < ny; a++) for (int c2 = 0; c2 < nx; c2++) {
      std::vector<float> t = tiles[(size_t)a * nx + c2];   // [3][256][256]
      if (a > 0) { const std::vector<float>& up = tiles[(size_t)(a - 1) * nx + c2]; const int e = yo[a - 1];
        for (int c = 0; c < 3; c++) for (int i = 0; i < e; i++) { const float wb = (float)i / e, wa = 1.f - wb; for (int j = 0; j < TP; j++) t[((size_t)c * TP + i) * TP + j] = up[((size_t)c * TP + TP - e + i) * TP + j] * wa + t[((size_t)c * TP + i) * TP + j] * wb; } }
      if (c2 > 0) { const std::vector<float>& lf = tiles[(size_t)a * nx + c2 - 1]; const int e = xo[c2 - 1];
        for (int c = 0; c < 3; c++) for (int i = 0; i < TP; i++) for (int j = 0; j < e; j++) { const float wb = (float)j / e, wa = 1.f - wb; t[((size_t)c * TP + i) * TP + j] = lf[((size_t)c * TP + i) * TP + TP - e + j] * wa + t[((size_t)c * TP + i) * TP + j] * wb; } }
      const int rows = a < ny - 1 ? TP - yo[a] : TP, cols = c2 < nx - 1 ? TP - xo[c2] : TP;
      for (int c = 0; c < 3; c++) for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) {
        float v = t[((size_t)c * TP + i) * TP + j] * PS[c] + PM[c]; v = std::min(1.f, std::max(0.f, v));
        out[((size_t)c * H + ys[a] + i) * W + xs[c2] + j] = 2.f * v - 1.f; } }
    FILE* f = fopen(fo.c_str(), "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
    printf("done %.0f\n", ms); fflush(stdout);
  }
  return 0;
}
