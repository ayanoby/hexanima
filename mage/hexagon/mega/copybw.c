#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "mega.h"
typedef struct { uint8_t *dst; const uint8_t *src; size_t bytes; } cj;
static void *cw(void *p) {
  cj *j = (cj *)p; HVX_Vector *d = (HVX_Vector *)j->dst; const HVX_Vector *s = (const HVX_Vector *)j->src; size_t n = j->bytes / 128;
  for (size_t i = 0; i < n; i++) {
    if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(s + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32));
    d[i] = s[i];
  }
  return 0;
}
int mega_copybw(remote_handle64 h, int T, int mb, int64_t *us) {
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) return -1;
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  size_t total = (size_t)mb << 20, per = 1u << 20; uint8_t *ddr = (uint8_t *)memalign(128, total); memset(ddr, 1, total);
  uint64_t t0 = HAP_perf_get_time_us();
  /* each pass: T threads each copy a 1MB chunk (distinct VTCM region) from successive DDR ranges */
  size_t chunks = total / per; pthread_t th[8]; cj jobs[8];
  for (size_t c = 0; c < chunks; c += T) {
    int n = (int)(chunks - c < (size_t)T ? chunks - c : T);
    for (int t = 0; t < n; t++) { jobs[t].dst = vt + t * per; jobs[t].src = ddr + (c + t) * per; jobs[t].bytes = per; pthread_create(&th[t], 0, cw, &jobs[t]); }
    for (int t = 0; t < n; t++) pthread_join(th[t], 0);
  }
  *us = HAP_perf_get_time_us() - t0;
  free(ddr); HAP_compute_res_release(ctx); return 0;
}
