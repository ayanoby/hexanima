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

