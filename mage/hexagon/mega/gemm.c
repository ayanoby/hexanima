/* HMX fp16 GEMM on v69, all operands in "crouton" layout: 32x32 fp16 tiles (2KB), rows paired/interleaved:
   halfword index of (r,c) inside a tile = (r/2)*64 + (c%32)*2 + (r&1).
   Activation/out: tiles ordered [row slab][col block]. Weights: [n block][k block] (K tiles contiguous per n block). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "mega.h"
#include "qhblas_hmx.h"

typedef uint16_t u16;
void hmx_acc(uint8_t*,uint32_t,uint8_t*,uint32_t);
void hmx_store(uint8_t*,uint32_t);
#define KCH 32  /* k blocks per HMX call: ctrl length field caps at 64KB */
static inline size_t tin(int r, int c) { return ((r & 31) >> 1) * 64 + (c & 31) * 2 + (r & 1); }
static float gen(unsigned a, unsigned b, unsigned s) { unsigned h = a * 2654435761u + b * 40503u + s; h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15; return ((int)(h & 255) - 128) / 256.f; }

int mega_bench(remote_handle64 h, int M, int K, int N, int64_t *us, float *err, int *rcout) {
  *us = 0; *err = -1; *rcout = 0;
  const int rs = M / 32, kb = K / 32, nb = N / 32;
  const size_t actB = (size_t)M * K * 2, wB = (size_t)K * N * 2, outB = (size_t)M * N * 2;
  if (actB + wB + outB > (8u << 20) - 8192) { *rcout = -3; return 0; }
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000);
  if (!ctx) { *rcout = -1; return 0; }
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  __fp16 *A = (__fp16 *)vt, *W = (__fp16 *)(vt + actB), *O = (__fp16 *)(vt + actB + wB);
  for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) A[((size_t)(m >> 5) * kb + (k >> 5)) * 1024 + tin(m, k)] = (__fp16)gen(m, k, 7);
  for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) W[((size_t)(n >> 5) * kb + (k >> 5)) * 1024 + tin(k, n)] = (__fp16)gen(k, n, 11);
  HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)h, &rq);
  if (HAP_compute_res_hmx_lock(ctx)) { *rcout = -2; HAP_compute_res_release(ctx); return 0; }
  uint64_t t0 = HAP_perf_get_time_us();
  for (int n = 0; n < nb; n++)
    for (int r = 0; r < rs; r++) {
      for (int k0 = 0; k0 < kb; k0 += KCH) {
        int c = kb - k0 < KCH ? kb - k0 : KCH;
        hmx_acc((uint8_t *)A + ((size_t)r * kb + k0) * 2048, (c << 11) | 0x7C, (uint8_t *)W + ((size_t)n * kb + k0) * 2048, (c << 11) - 1);
      }
      hmx_store((uint8_t *)O + ((size_t)r * nb + n) * 2048, 0);
    }
  *us = HAP_perf_get_time_us() - t0;
  float me = 0;
  for (int s = 0; s < 200; s++) {
    int m = (s * 37) % M, n = (s * 91 + 5) % N; double a = 0;
    for (int k = 0; k < K; k++) a += (double)(float)(__fp16)gen(m, k, 7) * (float)(__fp16)gen(k, n, 11);
    float g = (float)O[((size_t)(m >> 5) * nb + (n >> 5)) * 1024 + tin(m, n)];
    float e = fabsf(g - (float)a); if (e > me) me = e;
  }
  *err = me;
  HAP_compute_res_hmx_unlock(ctx);
  HAP_compute_res_release(ctx);
  return 0;
}
