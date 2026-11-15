#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <pthread.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "HAP_farf.h"
#include "qurt_hmx.h"
#include "AEEStdErr.h"
#include "remote.h"
#include "mega.h"
#include "qhblas_hmx.h"

void hmx8_tile(uint8_t*,uint32_t,uint8_t*,uint32_t,uint8_t*,uint32_t);
void hmx8_acc(uint8_t*,uint32_t,uint8_t*,uint32_t);
void hmx8_store(uint8_t*,uint32_t);
void hmx_acc(uint8_t*,uint32_t,uint8_t*,uint32_t);
void hmx_store(uint8_t*,uint32_t);
void hmx8_store_before(uint8_t*,uint32_t);
void power_max(void *);
void hmx8_bias(uint8_t*);
void hmx_clr(void);
#ifdef V79
void hmx79_st(uint8_t*, uint32_t, uint32_t);
void hmx79_acc_p(uint8_t*, uint32_t, uint8_t*, uint32_t);
void hmx79_acc_s(uint8_t*, uint32_t, uint8_t*, uint32_t);
#endif
void hmx8_clr(void);
void hmx8_bias2(uint8_t*);
void hmx8_acc_cm(uint8_t*,uint32_t,uint8_t*,uint32_t);
void hmx8_st_cmsat(uint8_t*,uint32_t);
void hmx8_st_sat(uint8_t*,uint32_t);
void hmx8_st_uh21(uint8_t*,uint32_t);
void hmx8_st_uh22(uint8_t*,uint32_t);
void hmx8_st_ret(uint8_t*,uint32_t);
int mega_open(const char *uri, remote_handle64 *h) { *h = (remote_handle64)malloc(1); return *h ? 0 : -1; }
void mega_blk_release(void);
int mega_close(remote_handle64 h) { mega_blk_release(); free((void *)h); return 0; }

#define M 32
#define N 64
static inline size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }


/* DDR -> VTCM copy variants (stage 11..17): K = MB to copy; *cyc = microseconds, *maxerr = GB/s */
static void copy_variant(int v, uint8_t *dst, const uint8_t *src, size_t bytes) {
  HVX_Vector *d = (HVX_Vector *)dst; const HVX_Vector *s = (const HVX_Vector *)src; size_t n = bytes / 128;
  if (v == 17) { memcpy(dst, src, bytes); return; }
  for (size_t i = 0; i < n; i++) {
    if (v == 11 && (i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(s + i + 64), (uint64_t)(128 * 32) | (1ull << 16) | (128ull << 32));
    if (v == 13 && (i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(s + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32));
    if (v == 14 && i + 16 < n) Q6_dcfetch_A((void *)(s + i + 16));
    if (v == 15 && i + 8 < n) Q6_l2fetch_AP((void *)(s + i + 8), (uint64_t)128 | (1ull << 16) | (128ull << 32));
    d[i] = s[i];
  }
}
int mega_probe(remote_handle64 h, int K, int reps, int stage, int64_t *cyc, float *maxerr, int *lockrc, int *vtcmrc) {
  *cyc = 0; *maxerr = -1; *lockrc = -999; *vtcmrc = -999;
  if (stage >= 11 && stage <= 17) {
    compute_res_attr_t ca; HAP_compute_res_attr_init(&ca); HAP_compute_res_attr_set_vtcm_param(&ca, 8u << 20, 1);
    uint32_t cx = HAP_compute_res_acquire(&ca, 100000); if (!cx) { *vtcmrc = -1; return 0; }
    uint8_t *vt2 = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&ca);
    size_t per = 1u << 20, total = (size_t)K << 20; uint8_t *ddr = (uint8_t *)memalign(128, total); memset(ddr, 1, total);
    power_max((void *)h);
    copy_variant(stage, vt2, ddr, per);   /* warm */
    uint64_t t0 = HAP_perf_get_time_us();
    for (size_t o = 0; o < total; o += per) copy_variant(stage, vt2, ddr + o, per);
    uint64_t us = HAP_perf_get_time_us() - t0;
    *cyc = (int64_t)us; *maxerr = (float)((double)total / 1e9 / (us / 1e6)); *vtcmrc = 0;
    free(ddr); HAP_compute_res_release(cx); return 0;
  }
  if (stage == 20) {   /* VTCM size: *cyc = total bytes, *maxerr = available bytes */
    unsigned int tot = 0, av = 0; compute_res_vtcm_page_t tl, al; memset(&tl, 0, sizeof tl); memset(&al, 0, sizeof al);
    int rc = HAP_compute_res_query_VTCM(0, &tot, &tl, &av, &al);
    *cyc = (int64_t)tot; *maxerr = (float)av; *vtcmrc = rc; return 0;
  }
  if (stage == 9 || stage == 10) {   /* DSP clock: pcycles per microsecond of a busy loop (9: as the system left it, 10: after power_max) */
    if (stage == 10) power_max((void *)h);
    volatile uint32_t sink = 0; uint64_t c0 = HAP_perf_get_pcycles(), t0 = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < 3000000; i++) sink += i;
    uint64_t c1 = HAP_perf_get_pcycles(), t1 = HAP_perf_get_time_us();
    *cyc = (int64_t)(c1 - c0); *maxerr = (float)(t1 - t0); *vtcmrc = 0; return 0;
  }
  compute_res_attr_t attr;
  HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 1 << 20, 1);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000);
  *vtcmrc = ctx ? 0 : -1;
  if (!ctx) return 0;
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
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
  if (rc == 0 && stage >= 5 && stage <= 8) {   /* timing only: K = bytes/2048 tiles per call (<=32), data is garbage, HMX time does not depend on it */
    uint32_t c = K >> 5; if (c < 1) c = 1; if (c > 32) c = 32;
    uint32_t sh = (stage == 8) ? 10 : 11;       /* 8: ub with 1 KB tiles (c<<10) */
    uint32_t ac = (c << sh) | 0x7C, wc = (c << sh) - 1;
    memset(vt, 0x11, 262144);
    uint8_t *va = vt, *vw = vt + 65536, *vo = vt + 131072;
    uint64_t t0 = HAP_perf_get_pcycles();
    for (int r = 0; r < reps; r++) {
      if (stage == 5) hmx_acc(va, ac, vw, wc); else hmx8_acc(va, ac, vw, wc);
      if ((r & 7) == 7) { if (stage == 5) hmx_store(vo, 0); else hmx8_store(vo, 0); }
    }
    *cyc = (HAP_perf_get_pcycles() - t0) / reps; *maxerr = 0;
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



#ifdef V79
/* scan HMX ctrl words: records (form, ac, wc, first nonzero halfword) for every combination that produces a non-zero tile (ones x ones) */
static int hmx_scan(uint8_t *vt, int32_t *rec, int maxrec) {
  uint8_t *va = vt, *vw = vt + 65536, *vo = vt + 131072, *vb = vt + 196608;
  __fp16 *a = (__fp16 *)va, *w = (__fp16 *)vw; for (int i = 0; i < 1024; i++) { a[i] = (__fp16)1.0f; w[i] = (__fp16)1.0f; }
  float *bf = (float *)vb; for (int i = 0; i < 32; i++) { bf[i] = 0.0f; bf[32 + i] = 1.0f; }
  int nrec = 0; static const uint32_t WC[4] = {0x7ff, 0x3ff, 0xfff, 0x7fe}; static const uint32_t NS[7] = {0, 1, 2, 3, 4, 8, 16};
  for (int form = 0; form < 3; form++) for (int ni = 0; ni < 7; ni++) for (uint32_t low = 0; low < 0x800; low += 4) for (int wi = 0; wi < 4; wi++) {
    uint32_t ac = (NS[ni] << 11) | low, wc = WC[wi];
    memset(vo, 0, 4096); hmx_clr(); hmx8_bias2(vb);
    if (form == 0) hmx_acc(va, ac, vw, wc); else if (form == 1) hmx79_acc_p(va, ac, vw, wc); else hmx79_acc_s(va, ac, vw, wc);
    hmx79_st(vo, 0, 0);
    const uint16_t *o = (const uint16_t *)vo; uint16_t nz = 0; for (int i = 0; i < 1024; i++) if (o[i]) { nz = o[i]; break; }
    if (nz && nrec < maxrec) { rec[nrec * 4] = form; rec[nrec * 4 + 1] = (int32_t)ac; rec[nrec * 4 + 2] = (int32_t)wc; rec[nrec * 4 + 3] = nz; nrec++; }
  }
  return nrec;
}
#endif
/* dump: one int8 (or fp16 if mode&0x80) HMX call on caller-provided tiles; output region is pre-filled with 0xEE to see what the store writes.
   mode&15: 0 after.ub, 1 after.hf, 2 before.ub; mode&0x80: use the fp16 .hf pair instead of ub/b; mode>>8: number of 2 KB blocks c */

/* copy into VTCM with HVX stores (scalar stores to VTCM may sit in the L1 data cache where the HMX does not see them) */
static void vtcm_put(uint8_t *dst, const uint8_t *src, size_t bytes, int hvx) {
  if (!hvx) { memcpy(dst, src, bytes); return; }
  size_t n = (bytes + 127) & ~(size_t)127; uint8_t *tmp = (uint8_t *)memalign(128, n); memset(tmp, 0, n); memcpy(tmp, src, bytes);
  HVX_Vector *d = (HVX_Vector *)dst; const HVX_Vector *s = (const HVX_Vector *)tmp; for (size_t i = 0; i < n / 128; i++) d[i] = s[i];
  asm volatile("barrier" ::: "memory"); free(tmp);
}
int mega_dump(remote_handle64 h, int mode, const uint8_t *act, int actLen, const uint8_t *wgt, int wgtLen, uint8_t *out, int outLen, int *rcout) {
  *rcout = 0;
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 1 << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) { *rcout = -1; return 0; }
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; int pr1 = HAP_power_set((void *)h, &rq); power_max((void *)h);
  int pr2 = 99;
  if (mode & 0x40000000) { memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX_v2; rq.hmx_v2.set_power = TRUE; rq.hmx_v2.power_up = TRUE; rq.hmx_v2.set_clock = TRUE; rq.hmx_v2.pick_default = TRUE; pr2 = HAP_power_set((void *)h, &rq); }
  unsigned int *hbusy = 0; int lrc = 0, l4rc = -7, l3rc = -7; unsigned int caps = 0; compute_res_hmx_mutex_t mtx; memset(&mtx, 0, sizeof mtx);
#ifdef V79
  int qrc = HAP_compute_res_query_capability(HAP_COMPUTE_RES_PREEMPTION_CAPABILITY, &caps);
  if (qrc) caps = 0;
  l4rc = HAP_compute_res_hmx_lock4(ctx, &hbusy, 100000); lrc = l4rc;
  if (l4rc || !hbusy) { hbusy = 0; l3rc = HAP_compute_res_hmx_lock3(ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &mtx, 100000); lrc = l3rc; if (l3rc) { mtx.lock = 0; lrc = HAP_compute_res_hmx_lock(ctx); } }
  caps = (unsigned)qrc;
#else
  lrc = HAP_compute_res_hmx_lock(ctx);
#endif
  if (lrc) { *rcout = -2; HAP_compute_res_release(ctx); return 0; }
  if (hbusy) *hbusy = 1; else if (mtx.lock) mtx.lock(mtx.mutex);
  if (mode & 0x800) { uint64_t t0 = HAP_perf_get_time_us(); while (HAP_perf_get_time_us() - t0 < 200000) {} }   /* let the HMX power domain settle */
#ifdef V79
  if (mode & 0x80000000) { int32_t *rec = (int32_t *)out; int n = hmx_scan(vt, rec + 4, (outLen / 4 - 4) / 4); rec[0] = n; HAP_compute_res_hmx_unlock(ctx); HAP_compute_res_release(ctx); return 0; }
#endif
  uint32_t c = ((mode >> 8) & 3) ? ((mode >> 8) & 3) : 1;
  uint8_t *va = vt, *vw = vt + 65536, *vo = vt + 131072, *vd = vt + 262144;
  memset(vt, 0, 524288);
  { int32_t *ri = (int32_t *)(vt + 131072 + 16376); (void)ri; }
  { int hv = (mode & 0x400) != 0; vtcm_put(va, act, actLen, hv); vtcm_put(vw, wgt, wgtLen > 65536 ? 65536 : wgtLen, hv); memset(vo, 0xEE, 16384); }
  uint8_t *vb = vt + 196608; if (wgtLen > (int)(c << 11)) memcpy(vb, wgt + (c << 11), wgtLen - (c << 11));   /* tail of wgt = bias block */
  if (mode & 0x40) hmx8_bias(vb);
  if (mode & 0x200) {   /* QNN-style int8: tail of wgt = [256 B bias block][u32 act ctrl][u32 weight ctrl][u32 n tile pairs]; tiles are 1 KB */
    const uint8_t *tl = wgt + 2048 * ((mode >> 8) & 0xFF ? 0 : 0);
    uint32_t ac2, wc2, np; memcpy(&ac2, wgt + wgtLen - 12, 4); memcpy(&wc2, wgt + wgtLen - 8, 4); memcpy(&np, wgt + wgtLen - 4, 4);
    int woff = wgtLen - 12 - 256;      /* bias block position, weights are the first woff bytes */
    memset(vw, 0, 65536); memcpy(vw, wgt, woff); memcpy(vb, wgt + woff, 256);
    hmx8_clr(); hmx8_bias2(vb);
    for (uint32_t i = 0; i < np; i++) { if (mode & 0x10) hmx8_acc_cm(va + 1024 * i, ac2, vw + 1024 * i, wc2); else hmx8_acc(va + 1024 * i, ac2, vw + 1024 * i, wc2); }
    hmx8_st_cmsat(vo, 0);
    { int32_t *ri = (int32_t *)(vo + 16352); ri[0] = pr1; ri[1] = pr2; ri[2] = lrc; ri[3] = (int32_t)caps; ri[4] = l4rc; ri[5] = l3rc; } memcpy(out, vo, outLen > 16384 ? 16384 : outLen);
    HAP_compute_res_hmx_unlock(ctx); HAP_compute_res_release(ctx);
    return 0;
  }
  hmx_store(vd, 0);                         /* flush a possibly non-zero accumulator */
  if (mode & 0x2000) hmx_clr();             /* mxclracc.hf like QNN */
  if (mode & 0x4000) { hmx8_bias(vb); }     /* bias = mxmem(zero block) like QNN */
  uint32_t low = (mode & 0x1000000) ? 0x70 : (mode & 0x2000000) ? 0x7C : 0x7C; if (mode & 0x4000000) low = 0;
  uint32_t ac = (c << 11) | low, wc = (c << 11) - 1;
#ifdef V79
  if (mode & 0x38000000) {
    static const uint32_t T[8] = {0x7ff, 0x70, 0x7f0, 0x7c, 0x1c, 0x3ff, 0x77f, 0x7f}; uint32_t a1 = T[(mode >> 24) & 7], w1 = 0x7ff;
    if (mode & 0x08000000) { memcpy(&a1, wgt + wgtLen - 8, 4); memcpy(&w1, wgt + wgtLen - 4, 4); }
    if (mode & 0x20000000) hmx79_acc_s(va, a1, vw, w1); else if (mode & 0x10000000) hmx79_acc_p(va, a1, vw, w1); else hmx_acc(va, a1, vw, w1);
  }
  else
#endif
  if (mode & 0x80) hmx_acc(va, ac, vw, wc); else hmx8_acc(va, ac, vw, wc);
#ifdef V79
  if (mode & 0x8000) { hmx8_bias2(vb); hmx79_st(vo, 0, (mode >> 16) & 0xFF); { int32_t *ri = (int32_t *)(vo + 16352); ri[0] = pr1; ri[1] = pr2; ri[2] = lrc; ri[3] = (int32_t)caps; ri[4] = l4rc; ri[5] = l3rc; } memcpy(out, vo, outLen > 16384 ? 16384 : outLen); if (hbusy) *hbusy = 0; else if (mtx.unlock) mtx.unlock(mtx.mutex); HAP_compute_res_hmx_unlock(ctx); HAP_compute_res_release(ctx); return 0; }
#endif
  int st = mode & 15;
  if (st == 1) hmx_store(vo, 0); else if (st == 2) hmx8_store_before(vo, 0); else if (st == 3) hmx8_st_sat(vo, 0); else if (st == 4) hmx8_st_uh21(vo, 0); else if (st == 5) hmx8_st_uh22(vo, 0); else if (st == 6) hmx8_st_ret(vo, 0); else hmx8_store(vo, 0);
  { int32_t *ri = (int32_t *)(vo + 16352); ri[0] = pr1; ri[1] = pr2; ri[2] = lrc; ri[3] = (int32_t)caps; ri[4] = l4rc; ri[5] = l3rc; } memcpy(out, vo, outLen > 16384 ? 16384 : outLen);
  HAP_compute_res_hmx_unlock(ctx); HAP_compute_res_release(ctx);
  return 0;
}

/* crtbegin references EH frame registration; no libc++abi here, so stub them out (else the DSP loader rejects the .so) */
void __register_frame_info_bases(void) {}
void __deregister_frame_info_bases(void) {}
