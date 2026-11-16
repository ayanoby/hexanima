#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_perf.h"
#include "mega.h"
#include "kern.h"
#include "pool.h"
#include <pthread.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_compute_res.h"
struct LA { float *x; uint16_t *o; HVX_Vector *M, *N; };
static void lnf(void *p, int i, int n) { struct LA *q = (struct LA *)p; if (i == 0) k_lnmod_s(q->x, q->o, 1, q->M, q->N, 1e-6f); }
struct SA { const uint16_t *S; uint16_t *P; const uint16_t *et; HVX_Vector *scr; int reps; };
static void smf(void *p, int i, int n) { struct SA *q = (struct SA *)p; int per = 16 / n; for (int r = 0; r < q->reps; r++) k_softmax2p(q->S, q->P + (size_t)i * 0, NCB, q->et, q->scr + 8 * i, i * per, (i + 1) * per, 0, 0); }
static float rnd(unsigned *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) & 0xFFFF) / 32768.f - 1.f; }
int mega_ktest(remote_handle64 h, int which, float *err, int64_t *us, int64_t *cyc) {
  const int NS = 2, R = NS * 32;
  float *x = (float *)memalign(128, (size_t)NS * NCB * 4096), *x0 = (float *)malloc((size_t)NS * NCB * 4096);
  uint16_t *o = (uint16_t *)memalign(128, (size_t)NS * NCB * 2048);
  float *sh = (float *)memalign(128, DIM * 4), *sc = (float *)memalign(128, DIM * 4);
  unsigned s = 5;
  for (int sl = 0; sl < NS; sl++) for (int cb = 0; cb < NCB; cb++) for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++)
    x[((size_t)sl * NCB + cb) * 1024 + r * 32 + c] = rnd(&s) * 300.f + 50.f * (c + cb);
  memcpy(x0, x, (size_t)NS * NCB * 4096);
  for (int i = 0; i < DIM; i++) { sh[i] = rnd(&s); sc[i] = rnd(&s) * .5f; }
  *err = 0; *us = 0; *cyc = 0;
  if (which == 0) {
    uint64_t t0 = HAP_perf_get_time_us(), c0 = HAP_perf_get_pcycles();
    k_lnmod(x, o, NS, sh, sc, 1e-6f);
    *us = HAP_perf_get_time_us() - t0; *cyc = HAP_perf_get_pcycles() - c0;
    for (int row = 0; row < R; row += 5) {
      int sl = row / 32, r = row % 32; double m = 0, q = 0;
      for (int col = 0; col < DIM; col++) { double v = x0[((size_t)sl * NCB + col / 32) * 1024 + r * 32 + col % 32]; m += v; q += v * v; }
      m /= DIM; double var = q / DIM - m * m, rs = 1.0 / sqrt(var + 1e-6);
      for (int col = 0; col < DIM; col += 7) {
        double v = x0[((size_t)sl * NCB + col / 32) * 1024 + r * 32 + col % 32];
        double ref = (v - m) * rs * (1 + sc[col]) + sh[col];
        __fp16 g; memcpy(&g, &o[((size_t)sl * NCB + col / 32) * 1024 + (r / 2) * 64 + (col % 32) * 2 + (r & 1)], 2);
        float e = fabsf((float)g - (float)ref); if (e > *err) *err = e;
      }
    }
  } else if (which == 1) {
    for (size_t i = 0; i < (size_t)NS * NCB * 1024; i++) { __fp16 g = (__fp16)(rnd(&s) * 4); memcpy(&o[i], &g, 2); }
    uint64_t t0 = HAP_perf_get_time_us(), c0 = HAP_perf_get_pcycles();
    k_resadd(x, o, NS, sc);
    *us = HAP_perf_get_time_us() - t0; *cyc = HAP_perf_get_pcycles() - c0;
    for (int row = 0; row < R; row += 3) {
      int sl = row / 32, r = row % 32;
      for (int col = 0; col < DIM; col += 5) {
        size_t cr = (size_t)sl * NCB + col / 32; __fp16 g; memcpy(&g, &o[cr * 1024 + (r / 2) * 64 + (col % 32) * 2 + (r & 1)], 2);
        float ref = x0[cr * 1024 + r * 32 + col % 32] + sc[col] * (float)g, got = x[cr * 1024 + r * 32 + col % 32];
        float e = fabsf(got - ref) / (fabsf(ref) + 1.f); if (e > *err) *err = e;
      }
    }
  }
  else if (which == 2) {  /* gelu */
    size_t nv = (size_t)NS * NCB * 16;
    for (size_t i = 0; i < nv * 64; i++) { __fp16 g = (__fp16)(rnd(&s) * 6); memcpy(&o[i], &g, 2); }
    uint16_t *o0 = (uint16_t *)malloc(nv * 128); memcpy(o0, o, nv * 128);
    uint64_t t0 = HAP_perf_get_time_us(); k_gelu(o, nv); *us = HAP_perf_get_time_us() - t0;
    for (size_t i = 0; i < nv * 64; i += 3) { __fp16 a, b; memcpy(&a, &o0[i], 2); memcpy(&b, &o[i], 2); double xv = (float)a, ref = 0.5 * xv * (1 + erf(xv / sqrt(2.0)));
      float e = fabsf((float)b - (float)ref); if (e > *err) *err = e; }
    free(o0);
  } else if (which == 3) {  /* softmax over 2*? keys: use NCB tiles per slab as keys */
    for (size_t i = 0; i < (size_t)NS * NCB * 1024; i++) { __fp16 g = (__fp16)(rnd(&s) * 8); memcpy(&o[i], &g, 2); }
    uint16_t *P = (uint16_t *)memalign(128, (size_t)NCB * 2048); float *R = (float *)memalign(128, 4096);
    uint64_t t0 = HAP_perf_get_time_us(); k_softmax(o, P, NCB, R); *us = HAP_perf_get_time_us() - t0;
    for (int r = 0; r < 32; r += 3) {
      double mx = -1e30, sm = 0; for (int c = 0; c < DIM; c++) { __fp16 g; memcpy(&g, &o[(size_t)(c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + (r & 1)], 2); if ((float)g > mx) mx = (float)g; }
      for (int c = 0; c < DIM; c++) { __fp16 g; memcpy(&g, &o[(size_t)(c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + (r & 1)], 2); sm += exp((float)g - mx); }
      for (int c = 0; c < DIM; c += 11) { __fp16 g, pg; size_t ix = (size_t)(c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + (r & 1); memcpy(&g, &o[ix], 2); memcpy(&pg, &P[ix], 2);
        float e = fabsf((float)pg - (float)exp((float)g - mx)); if (e > *err) *err = e; }
      float rr = ((float *)R)[r * 32]; float e2 = fabsf(rr * (float)sm - 1.f); if (e2 > *err) *err = e2;
    }
    free(P); free(R);
  } else if (which == 5 || which == 6) {
    uint16_t *z = (uint16_t *)memalign(128, (size_t)NCB * 2048); memset(z, 0, (size_t)NCB * 2048);
    float *g = (float *)memalign(128, 512), *cs = (float *)memalign(128, 8192);
    memset(g, 0, 512); memset(cs, 0, 8192);
    if (which == 5) k_rmsrope(z, g, cs, cs, 0); else k_rmsrope(z, g, cs, cs, 1);
  } else if (which == 4) {  /* rmsrope (1 slab, 64 cb) */
    for (size_t i = 0; i < (size_t)NCB * 1024; i++) { __fp16 g = (__fp16)(rnd(&s) * 20 + 3); memcpy(&o[i], &g, 2); }
    uint16_t *o0 = (uint16_t *)malloc((size_t)NCB * 2048); memcpy(o0, o, (size_t)NCB * 2048);
    float *g = (float *)memalign(128, 128 * 4), *cs = (float *)memalign(128, 2 * 4096), *sn = (float *)memalign(128, 2 * 4096);
    for (int i = 0; i < 128; i++) g[i] = 0.5f + rnd(&s) * .3f;
    for (int j = 0; j < 2; j++) for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++) { double ang = (r + 1) * (j * 32 + c + 1) * 0.01; cs[j * 1024 + r * 32 + c] = cos(ang); sn[j * 1024 + r * 32 + c] = sin(ang); }
    uint64_t t0 = HAP_perf_get_time_us(); k_rmsrope(o, g, cs, sn, 1); *us = HAP_perf_get_time_us() - t0;
    #define EL(buf, r_, c_) ({ int rr_ = (r_), cc_ = (c_); __fp16 t_; memcpy(&t_, &buf[(size_t)(cc_ / 32) * 1024 + (rr_ / 2) * 64 + (cc_ % 32) * 2 + (rr_ & 1)], 2); (double)(float)t_; })
    for (int r = 0; r < 32; r += 5) for (int hd = 0; hd < 16; hd += 3) {
      double ss = 0; for (int d = 0; d < 128; d++) { double v = EL(o0, r, hd * 128 + d); ss += v * v; }
      double rs = 1.0 / sqrt(ss / 128 + 1e-6);
      for (int d = 0; d < 64; d += 5) {
        double a = EL(o0, r, hd * 128 + d) * rs * g[d], b = EL(o0, r, hd * 128 + d + 64) * rs * g[d + 64];
        double c = cs[(d / 32) * 1024 + r * 32 + d % 32], sv = sn[(d / 32) * 1024 + r * 32 + d % 32];
        double e1 = fabs(EL(o, r, hd * 128 + d) - (a * c - b * sv)), e2 = fabs(EL(o, r, hd * 128 + d + 64) - (a * sv + b * c));
        if (e1 > *err) *err = e1; if (e2 > *err) *err = e2;
      }
    }
    free(o0); free(g); free(cs); free(sn);
  }
  else if (which == 4 + 5) {  /* vgather speed/correctness: y = tab[(x bits)*2] for 1M halfwords */
    compute_res_attr_t attr; HAP_compute_res_attr_init(&attr); HAP_compute_res_attr_set_vtcm_param(&attr, 1u << 20, 1);
    uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) return -1;
    uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
    uint16_t *tab = (uint16_t *)vt;                 /* 32768 entries = 64 KB */
    for (int i = 0; i < 32768; i++) tab[i] = (uint16_t)(i * 3 + 1);
    HVX_Vector *dstv = (HVX_Vector *)(vt + 65536);   /* gather destination in VTCM */
    const int NV = 16384;                            /* vectors of 64 halfwords */
    uint16_t *idx = (uint16_t *)memalign(128, NV * 128), *out = (uint16_t *)memalign(128, NV * 128);
    for (int i = 0; i < NV * 64; i++) idx[i] = ((i * 7919) & 32767) * 2;   /* byte offsets */
    uint64_t t0 = HAP_perf_get_time_us(), c0 = HAP_perf_get_pcycles();
    for (int v = 0; v < NV; v++) {
      Q6_vgather_ARMVh(dstv, (uint32_t)vt, 65535, *(HVX_Vector *)(idx + v * 64));
      ((HVX_Vector *)out)[v] = *dstv;
    }
    *us = HAP_perf_get_time_us() - t0; *cyc = (HAP_perf_get_pcycles() - c0) / NV;
    for (int i = 0; i < NV * 64; i += 13) { float e = fabsf((float)out[i] - (float)(uint16_t)(((i * 7919) & 32767) * 3 + 1)); if (e > *err) *err = e; }
    free(idx); free(out); HAP_compute_res_release(ctx);
  }
  else if (which == 10 || which == 11) {
    compute_res_attr_t attr; HAP_compute_res_attr_init(&attr); HAP_compute_res_attr_set_vtcm_param(&attr, 1u << 20, 1);
    uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) return -1;
    uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
    uint16_t *et = (uint16_t *)vt, *t2 = (uint16_t *)(vt + 65536); HVX_Vector *scr = (HVX_Vector *)(vt + 131072);
    tab_init(et, t2);
    if (which == 10) {
      size_t nv = (size_t)NS * NCB * 16;
      for (size_t i = 0; i < nv * 64; i++) { __fp16 g = (__fp16)(rnd(&s) * 6); memcpy(&o[i], &g, 2); }
      uint16_t *o0 = (uint16_t *)malloc(nv * 128); memcpy(o0, o, nv * 128);
      uint64_t t0 = HAP_perf_get_time_us(); k_gelu2(o, nv, t2, scr); *us = HAP_perf_get_time_us() - t0;
      for (size_t i = 0; i < nv * 64; i += 3) { __fp16 a, b; memcpy(&a, &o0[i], 2); memcpy(&b, &o[i], 2); double xv = (float)a, ref = 0.5 * xv * (1 + erf(xv / sqrt(2.0)));
        float e = fabsf((float)b - (float)ref); if (e > *err) *err = e; }
      free(o0);
    } else {
      for (size_t i = 0; i < (size_t)NS * NCB * 1024; i++) { __fp16 g = (__fp16)(rnd(&s) * 8); memcpy(&o[i], &g, 2); }
      uint16_t *P = (uint16_t *)memalign(128, (size_t)NCB * 2048);
      uint64_t t0 = HAP_perf_get_time_us(); k_softmax2(o, P, NCB, et, scr); *us = HAP_perf_get_time_us() - t0;
      for (int r = 0; r < 32; r += 3) {
        double mx = -1e30; for (int c = 0; c < DIM; c++) { __fp16 g; memcpy(&g, &o[(size_t)(c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + (r & 1)], 2); if ((float)g > mx) mx = (float)g; }
        for (int c = 0; c < DIM; c += 11) { __fp16 g, pg; size_t ix = (size_t)(c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + (r & 1); memcpy(&g, &o[ix], 2); memcpy(&pg, &P[ix], 2);
          float e = fabsf((float)pg - (float)exp((float)g - mx)); if (e > *err) *err = e; }
      }
      free(P);
    }
    HAP_compute_res_release(ctx);
  }
  else if (which == 12) {   /* LN inside a pool worker */
    pthread_attr_t at; size_t ss = 0; pthread_attr_init(&at); pthread_attr_getstacksize(&at, &ss); *cyc = (int64_t)ss;
    static HVX_Vector M[MAXNCB] __attribute__((aligned(128))), N[MAXNCB] __attribute__((aligned(128)));
    k_ln_prep(M, N, sh, sc);
    struct LA a = {x, o, M, N};
    pool_init(1); uint64_t t0 = HAP_perf_get_time_us(); pool_run(lnf, &a); *us = HAP_perf_get_time_us() - t0; pool_quit();
  }
  else if (which >= 13 && which <= 16) {   /* softmax via pool, nw = which-12 workers (1,2,4 -> uses 16/n split) */
    int nw = which == 13 ? 1 : which == 14 ? 2 : which == 15 ? 4 : 8;
    compute_res_attr_t attr; HAP_compute_res_attr_init(&attr); HAP_compute_res_attr_set_vtcm_param(&attr, 2u << 20, 1);
    uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) return -1;
    uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
    uint16_t *et = (uint16_t *)vt, *t2 = (uint16_t *)(vt + 65536); HVX_Vector *scr = (HVX_Vector *)(vt + 131072);
    uint16_t *S = (uint16_t *)(vt + 262144), *P = (uint16_t *)(vt + 262144 + 98304);
    tab_init(et, t2);
    for (size_t i = 0; i < 48 * 1024; i++) { __fp16 g = (__fp16)(rnd(&s) * 8); memcpy(&S[i], &g, 2); }
    struct SA a = {S, P, et, scr, 20};
    pool_init(nw); uint64_t t0 = HAP_perf_get_time_us(); pool_run(smf, &a); *us = HAP_perf_get_time_us() - t0; pool_quit();
    *us = *us / 20; *cyc = nw;
    HAP_compute_res_release(ctx);
  }
  else if (which == 20) {   /* search a 16x16 transpose network of 64-bit units with vshuff/vdeal */
    static const int perm[24][4] = {{0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},{1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
                                    {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},{3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}};
    HVX_Vector in[16], v[16]; uint16_t *ip = (uint16_t *)in;
    for (int p = 0; p < 16; p++) for (int u = 0; u < 16; u++) for (int h = 0; h < 4; h++) ip[p * 64 + u * 4 + h] = p * 16 + u;
    int found = -1;
    for (int opm = 0; opm < 16 && found < 0; opm++) for (int swm = 0; swm < 16 && found < 0; swm++) for (int pd = 0; pd < 24 && found < 0; pd++) for (int pr = 0; pr < 24 && found < 0; pr++) {
      memcpy(v, in, sizeof in);
      for (int st = 0; st < 4; st++) {
        int d = 1 << perm[pd][st], R = 8 << perm[pr][st];
        for (int i = 0; i < 16; i++) if (!(i & d)) {
          HVX_VectorPair w = (opm >> st & 1) == 0 ? Q6_W_vshuff_VVR(v[i + d], v[i], -R) : Q6_W_vdeal_VVR(v[i + d], v[i], -R);
          int sw = swm >> st & 1; v[i] = sw ? Q6_V_hi_W(w) : Q6_V_lo_W(w); v[i + d] = sw ? Q6_V_lo_W(w) : Q6_V_hi_W(w);
        }
      }
      uint16_t *vp = (uint16_t *)v; int ok = 1; int64_t map = 0; int used = 0;
      for (int jv = 0; jv < 16 && ok; jv++) {
        int q = vp[jv * 64] & 15; int pq = vp[jv * 64] >> 4;
        if (pq != 0) { ok = 0; break; }
        for (int u = 0; u < 16 && ok; u++) for (int h = 0; h < 4; h++) if (vp[jv * 64 + u * 4 + h] != u * 16 + q) { ok = 0; break; }
        if (used >> q & 1) ok = 0; used |= 1 << q; map |= (int64_t)q << (4 * jv);
      }
      if (ok) *us = map;
      if (ok) found = ((opm * 16 + swm) * 24 + pd) * 24 + pr;
    }
    *cyc = found;
  }
  else if (which == 21) {   /* tile transpose vs scalar */
    uint16_t *a = (uint16_t *)memalign(128, 2048), *b = (uint16_t *)memalign(128, 2048);
    for (int i = 0; i < 1024; i++) a[i] = (uint16_t)(i * 37 + 5);
    k_transpose_tile(a, b); int bad = 0;
    for (int key = 0; key < 32; key++) for (int d = 0; d < 32; d++) if (b[(d >> 1) * 64 + key * 2 + (d & 1)] != a[(key >> 1) * 64 + d * 2 + (key & 1)]) bad++;
    *err = bad;
    uint64_t t0 = HAP_perf_get_pcycles(); for (int r = 0; r < 1000; r++) k_transpose_tile(a, b); *cyc = (HAP_perf_get_pcycles() - t0) / 1000;
    free(a); free(b);
  }
  free(x); free(x0); free(o); free(sh); free(sc);
  return 0;
}
