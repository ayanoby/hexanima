/* weight streaming test: panels (2 n-blocks, K rows) copied DDR->VTCM by a helper thread while the main thread runs HMX on the previous panel */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "mega.h"

void hmx_acc(uint8_t *, uint32_t, uint8_t *, uint32_t);
void hmx_store(uint8_t *, uint32_t);
#define KCH 32

static void vcopy(void *dst, const void *src, size_t bytes) {
  HVX_Vector *d = (HVX_Vector *)dst; const HVX_Vector *s = (const HVX_Vector *)src; size_t n = bytes / 128;
  for (size_t i = 0; i < n; i++) {
    if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(s + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32));
    d[i] = s[i];
  }
}
typedef struct { void *dst; const void *src; size_t bytes; volatile int go, done, quit; } job_t;
static void *worker(void *p) {
  job_t *j = (job_t *)p;
  for (;;) {
    while (!j->go && !j->quit) {}
    if (j->quit) return 0;
    j->go = 0; vcopy(j->dst, j->src, j->bytes); j->done = 1;
  }
}

int mega_stream(remote_handle64 h, int M, int K, int N, int mode, int64_t *us, int64_t *cus, int *rcout) {
  *us = *cus = 0; *rcout = 0;
  const int rs = M / 32, kb = K / 32, nb = N / 32; const size_t panel = (size_t)kb * 2 * 2048;
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) { *rcout = -1; return 0; }
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  uint8_t *A = vt, *W0 = vt + (size_t)M * K * 2, *W1 = W0 + panel, *O = W1 + panel;
  size_t total = panel * (nb / 2);
  uint8_t *ddr = (uint8_t *)memalign(128, total); if (!ddr) { *rcout = -4; HAP_compute_res_release(ctx); return 0; }
  memset(ddr, 0x3c, total); memset(A, 0x3c, (size_t)M * K * 2);
  HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)h, &rq);
  if (HAP_compute_res_hmx_lock(ctx)) { *rcout = -2; HAP_compute_res_release(ctx); return 0; }
  job_t job = {0}; pthread_t th; pthread_create(&th, 0, worker, &job);
  int np = nb / 2; uint8_t *Wb[2] = {W0, W1};
  uint64_t t0 = HAP_perf_get_time_us();
  if (mode == 1) {  /* copy only, main thread */
    for (int p = 0; p < np; p++) vcopy(Wb[p & 1], ddr + (size_t)p * panel, panel);
    *cus = HAP_perf_get_time_us() - t0;
  } else {
    if (mode == 2) vcopy(Wb[0], ddr, panel);
    else { job.dst = Wb[0]; job.src = ddr; job.bytes = panel; job.done = 0; job.go = 1; while (!job.done) {} }
    for (int p = 0; p < np; p++) {
      if (mode == 3 && p + 1 < np) { job.dst = Wb[(p + 1) & 1]; job.src = ddr + (size_t)(p + 1) * panel; job.bytes = panel; job.done = 0; job.go = 1; }
      uint8_t *w = Wb[p & 1];
      for (int r = 0; r < rs; r++) {
        for (int j = 0; j < 2; j++) {
          for (int k0 = 0; k0 < kb; k0 += KCH) { int c = kb - k0 < KCH ? kb - k0 : KCH;
            hmx_acc(A + ((size_t)r * kb + k0) * 2048, (c << 11) | 0x7C, w + ((size_t)j * kb + k0) * 2048, (c << 11) - 1); }
          hmx_store(O + ((size_t)r * 2 + j) * 2048, 0);
        }
      }
      if (mode == 2 && p + 1 < np) vcopy(Wb[(p + 1) & 1], ddr + (size_t)(p + 1) * panel, panel);
      if (mode == 3 && p + 1 < np) while (!job.done) {}
    }
  }
  *us = HAP_perf_get_time_us() - t0;
  job.quit = 1; pthread_join(th, 0);
  free(ddr); HAP_compute_res_hmx_unlock(ctx); HAP_compute_res_release(ctx);
  return 0;
}
