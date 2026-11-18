#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "HAP_farf.h"
#include "qurt_hmx.h"
#include "AEEStdErr.h"
#include "remote.h"
void hmx_bias_setup(uint8_t *);
#include "mega.h"
#include "qhblas_hmx.h"

void hmx8_tile(uint8_t*,uint32_t,uint8_t*,uint32_t,uint8_t*,uint32_t);
int mega_open(const char *uri, remote_handle64 *h) { *h = (remote_handle64)malloc(1); return *h ? 0 : -1; }
void mega_blk_release(void);
int mega_close(remote_handle64 h) { mega_blk_release(); free((void *)h); return 0; }

#define M 32
#define N 64
static inline size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }

int mega_probe(remote_handle64 h, int K, int reps, int stage, int64_t *cyc, float *maxerr, int *lockrc, int *vtcmrc) {
  *cyc = 0; *maxerr = -1; *lockrc = -999; *vtcmrc = -999;
  compute_res_attr_t attr;
  HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 1 << 20, 1);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000);
  *vtcmrc = ctx ? 0 : -1;
  if (!ctx) return 0;
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr); hmx_bias_setup(vt + (8u << 20) - 4096);
  if (!vt) { HAP_compute_res_release(ctx); *vtcmrc = -2; return 0; }
  __fp16 *A = (__fp16 *)memalign(128, M * K * 2), *B = (__fp16 *)memalign(128, K * N * 2), *C = (__fp16 *)memalign(128, M * N * 2);
  float *R = (float *)malloc(M * N * 4);
  unsigned s = 1;
  for (int i = 0; i < M * K; i++) { s = s * 1103515245 + 12345; A[i] = (__fp16)(((s >> 16) & 255) / 128.f - 1.f); }
  for (int i = 0; i < K * N; i++) { s = s * 1103515245 + 12345; B[i] = (__fp16)(((s >> 16) & 255) / 128.f - 1.f); }
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { float a = 0; for (int k = 0; k < K; k++) a += (float)A[m * K + k] * (float)B[k * N + n]; R[m * N + n] = a; }
  uint8_t *vA = vt, *vB = vt + up(M * K * 2), *vO = vB + up(K * N * 2 + M * N * 2 + 4096);
  qhblas_hmx_transfer_row_ahf(A, vA, K);
  qhblas_hmx_transfer_column_ahf(B, vB, K, N);
  if (stage == 0) { *vtcmrc = 11; goto done; }
  HAP_power_request_t rq; memset(&rq, 0, sizeof rq);
  rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE;
  *vtcmrc = 100 + HAP_power_set((void *)h, &rq);
  int rc = HAP_compute_res_hmx_lock(ctx);
  *lockrc = rc;
  if (rc == 0 && stage == 1) { HAP_compute_res_hmx_unlock(ctx); goto done; }
  if (rc == 0 && stage == 3) {
    memset(vt, 0, 65536);
    uint64_t t0 = HAP_perf_get_pcycles();
    hmx8_tile(vt, (1 << 11) | 0x7C, vt + 8192, (1 << 11) - 1, vt + 16384, 0);
    *cyc = HAP_perf_get_pcycles() - t0; *maxerr = 0;
    HAP_compute_res_hmx_unlock(ctx); goto done;
  }
  if (rc == 0) {
    qhblas_hmx_matrix_matrix_mpy_ahf(vA, vB, vO, C, M, K, N);
    float me = 0; for (int i = 0; i < M * N; i++) { float e = fabsf((float)C[i] - R[i]); if (e > me) me = e; }
    *maxerr = me;
    uint32_t nb = K >> 5, ac = (nb << 11) | 0x7C, wc = (nb << 11) - 1;
    uint64_t t0 = HAP_perf_get_pcycles();
    for (int r = 0; r < reps; r++) qhblas_hmx_matrix_matrix_mpy_hf_sm(vA, ac, vB, wc, vO, 0, vB);
    *cyc = (HAP_perf_get_pcycles() - t0) / reps;
    HAP_compute_res_hmx_unlock(ctx);
  }
done:
  HAP_compute_res_release(ctx);
  free(A); free(B); free(C); free(R);
  return 0;
}

/* crtbegin references EH frame registration; no libc++abi here, so stub them out (else the DSP loader rejects the .so) */
void __register_frame_info_bases(void) {}
void __deregister_frame_info_bases(void) {}
