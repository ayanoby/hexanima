/* One Anima DiT block (self-attn, cross-attn, MLP) on HMX, activations in crouton tiles, shared blob in DDR. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_mem.h"
#include "HAP_power.h"
#include "mega.h"
void power_max(void *);
#include "gemm2.h"
#include "pool.h"
#include "kern.h"
#include "blk_meta.h"

typedef uint16_t u16;
#define T16 1024          /* halfwords per h16 tile */
#define VT_TAB 0x7B0000u   /* tables (about 260 KB) at the very top of the 8 MB VTCM: the attention may use everything below (up to 231 key slabs = 7392 tokens) */

static u16 *g_et, *g_t2; static HVX_Vector *g_scr;   /* tables + per-worker gather scratch */

static void vcp(void *d, const void *s, size_t bytes) {
  HVX_Vector *dv = (HVX_Vector *)d; const HVX_Vector *sv = (const HVX_Vector *)s; size_t n = bytes / 128;
  for (size_t i = 0; i < n; i++) { if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(sv + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32)); dv[i] = sv[i]; }
}

/* ---------------- simple parallel kernels ---------------- */
typedef struct { const float *x; u16 *h; const HVX_Vector *M, *N; int ns; float *xm; const u16 *t; const float *gate; const float *g; const float *c, *s; u16 *q; int rope; size_t nvec; } par_t;
static void p_ln(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_lnmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, p->M, p->N, 1e-6f); }
static void p_add(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_resadd(p->xm + (size_t)s * NCB * 1024, p->t + (size_t)s * NCB * T16, 1, p->gate); }
static void p_rope(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_rmsrope(p->q + (size_t)s * NCB * T16, p->g, p->rope ? p->c + (size_t)s * 2048 : 0, p->rope ? p->s + (size_t)s * 2048 : 0, p->rope); }
static void p_gelu(void *a, int i, int n) { par_t *p = (par_t *)a; size_t per = (p->nvec + n - 1) / n, o = per * i, e = o + per > p->nvec ? p->nvec : o + per; if (o < e) k_gelu2(p->q + o * 64, e - o, g_t2, g_scr + 8 * i); }

/* ---------------- NAG: normalized attention guidance on the cross-attention output ----------------
   z = positive attention output (o16, in place), n = the one with the negative context. Per token: zt = s*z - (s-1)*n; if |zt|_1 / |z|_1 > tau: zt *= tau * |z|_1 / |zt|_1; out = alpha*zt + (1-alpha)*z.
   Scalar code (a test path): fp16 <-> fp32 by hand, tile element (row r, col c) of a 32x32 tile is at (r>>1)*64 + c*2 + (r&1). */
static inline float h2f(u16 h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, r;
  if (e == 0) { if (m == 0) r = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; r = s | ((e + 112) << 23) | (m << 13); } }
  else if (e == 31) r = s | 0x7f800000u | (m << 13); else r = s | ((e + 112) << 23) | (m << 13);
  union { uint32_t u; float f; } c; c.u = r; return c.f;
}
static inline u16 f2h(float f) {
  union { uint32_t u; float f; } c; c.f = f; uint32_t x = c.u, s = (x >> 16) & 0x8000, e = (x >> 23) & 255, m = x & 0x7fffff, h;
  if (e >= 143) return (u16)(s | 0x7bff);
  if (e <= 102) return (u16)s;
  if (e < 113) { m |= 0x800000; uint32_t sh = 126 - e; h = m >> sh; uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1); if (rem > half || (rem == half && (h & 1))) h++; return (u16)(s | h); }
  h = ((e - 112) << 10) | (m >> 13); uint32_t rem = m & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
  return (u16)(s | h);
}
typedef struct { u16 *z; const u16 *n; int ns; float s, tau, alpha; } nag_t;
static float g_nagbuf[8][2048] __attribute__((aligned(128)));
static HVX_Vector g_nagv[8][2 * NCB] __attribute__((aligned(128)));
static void p_nag_hvx(void *a, int idx, int nw) { nag_t *p = (nag_t *)a; for (int sl = idx; sl < p->ns; sl += nw) k_nag_slab(p->z + (size_t)sl * NCB * T16, p->n + (size_t)sl * NCB * T16, p->s, p->tau, p->alpha, g_nagv[idx & 7]); }
static void p_nag(void *a, int idx, int nw) {
  nag_t *p = (nag_t *)a; float *b = g_nagbuf[idx & 7];
  for (int sl = idx; sl < p->ns; sl += nw)
    for (int r = 0; r < 32; r++) {
      const size_t ro = (size_t)(r >> 1) * 64 + (r & 1); float lp = 0, lt = 0;
      for (int cb = 0; cb < NCB; cb++) {
        const u16 *zp = p->z + ((size_t)sl * NCB + cb) * T16 + ro, *np = p->n + ((size_t)sl * NCB + cb) * T16 + ro;
        for (int c = 0; c < 32; c++) { float zv = h2f(zp[c * 2]), nv = h2f(np[c * 2]), t = p->s * zv - (p->s - 1.f) * nv; b[cb * 32 + c] = t; lp += zv < 0 ? -zv : zv; lt += t < 0 ? -t : t; }
      }
      float sc = 1.f; if (lp > 0 && lt > p->tau * lp) sc = p->tau * lp / lt;
      for (int cb = 0; cb < NCB; cb++) {
        u16 *zp = p->z + ((size_t)sl * NCB + cb) * T16 + ro;
        for (int c = 0; c < 32; c++) zp[c * 2] = f2h(p->alpha * sc * b[cb * 32 + c] + (1.f - p->alpha) * h2f(zp[c * 2]));
      }
    }
}

/* ---------------- attention ---------------- */
typedef struct { const u16 *k, *v, *q; u16 *o; int nkb, h, nq; u16 *KT, *Vw, *Qh; } ah_t;
static void p_attn_setup(void *a, int idx, int n) {   /* build K^T and V(+ones) weight tiles for head h, copy Q tiles */
  ah_t *p = (ah_t *)a;
  for (int kb = idx; kb < p->nkb; kb += n)
    for (int j = 0; j < 4; j++) {
      const u16 *ks = p->k + ((size_t)kb * 64 + p->h * 4 + j) * T16; u16 *kd = p->KT + ((size_t)kb * 4 + j) * T16;
      k_transpose_tile(ks, kd);
      vcp(p->Vw + ((size_t)j * p->nkb + kb) * T16, p->v + ((size_t)kb * 64 + p->h * 4 + j) * T16, 2048);
    }
  for (int s = idx; s < p->nq; s += n) vcp(p->Qh + (size_t)s * 4 * T16, p->q + ((size_t)s * 64 + p->h * 4) * T16, 8192);
}
static void p_attn_flush(void *a, int idx, int n) { ah_t *p = (ah_t *)a; for (int s = idx; s < p->nq; s += n) vcp(p->o + ((size_t)s * 64 + p->h * 4) * T16, p->Qh + (size_t)s * 4 * T16, 8192); }
typedef struct { const u16 *S; u16 *P; int nkb; } sm_t;
static void p_softmax(void *a, int i, int n) { sm_t *p = (sm_t *)a; k_softmax2p(p->S, p->P, p->nkb, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n); }


/* ---------------- chunked attention (no limit on the number of tokens) ----------------
   The plain attention keeps K^T, V and Q of ALL tokens of one head and a full score row in the 8 MB VTCM (231 key slabs = 7392 tokens). This version works on QB query slabs
   at a time and streams the keys through VTCM in chunks of C key tiles (K and V stay in DDR; each chunk is transposed on its way in): pass 0 finds the row max over all chunks,
   pass 1 computes P = exp(S - max) per chunk and adds P*V (plus a ones column = row sums) into fp32 accumulators; the end divides by the sums. Same result as the plain one up to
   the summation order. Port of the flux3 version for heads of 4 tiles (128), 16 heads, no key mask (the token count is a multiple of 32). */
typedef struct { const u16 *k, *v, *q; u16 *o; u16 *KTc, *Vc, *Qb; int kb0, nkc, C, h, withV, s0, nqb; } cl_t;
static void p_cl_load(void *a, int idx, int n) {
  cl_t *p = (cl_t *)a;
  for (int i = idx; i < p->nkc; i += n) {
    const int kb = p->kb0 + i;
    for (int j = 0; j < 4; j++) k_transpose_tile(p->k + ((size_t)kb * NCB + p->h * 4 + j) * T16, p->KTc + ((size_t)i * 4 + j) * T16);
    if (p->withV) for (int j = 0; j < 4; j++) vcp(p->Vc + ((size_t)j * p->C + i) * T16, p->v + ((size_t)kb * NCB + p->h * 4 + j) * T16, 2048);
  }
}
static void p_cq_load(void *a, int idx, int n) { cl_t *p = (cl_t *)a; for (int s = idx; s < p->nqb; s += n) vcp(p->Qb + (size_t)s * 4 * T16, p->q + ((size_t)(p->s0 + s) * NCB + p->h * 4) * T16, 8192); }
static void p_cq_flush(void *a, int idx, int n) { cl_t *p = (cl_t *)a; for (int s = idx; s < p->nqb; s += n) vcp(p->o + ((size_t)(p->s0 + s) * NCB + p->h * 4) * T16, p->Qb + (size_t)s * 4 * T16, 8192); }
typedef struct { const u16 *S; u16 *P; int nkc; HVX_Vector *mx; } rm_t;
static void p_rowmax(void *a, int i, int n) { rm_t *p = (rm_t *)a; k_rowmax_chunk(p->S, p->nkc, i * 16 / n, (i + 1) * 16 / n, 0, p->mx); }
static void p_smfix(void *a, int i, int n) { rm_t *p = (rm_t *)a; k_softmax_fixed(p->S, p->P, p->nkc, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, 0, p->mx); }
static size_t attn_plain_bytes(int nq, int nkb) { return ((size_t)13 * nkb + (size_t)4 * nq + 5) * 2048; }
static int g_force_c = 0, g_force_q = 0;   /* test overrides of C / QB (0 = automatic) */
static void attn_chunked(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, uint8_t *vt) {
  const int nkb = nkv / 32, hdt = 4;
  const size_t BUDGET = 6815744;   /* 6.5 MB: the tables sit at 7.7 MB */
  int C = nkb < 128 ? nkb : 128; if (g_force_c > 0 && g_force_c < C) C = g_force_c;
  while (C > 8 && (size_t)C * (4 + hdt + hdt + 1) * 2048 > BUDGET / 2) C = (C + 1) / 2;
  const size_t chunkB = (size_t)C * (4 + hdt + hdt + 1) * 2048 + (size_t)(hdt + 1) * 2048, perSlab = (size_t)hdt * 2048 + (size_t)(hdt + 1) * 4096 + 2048;
  int QB = (int)((BUDGET - chunkB) / perSlab); if (QB > 64) QB = 64; if (QB > nq) QB = nq; if (g_force_q > 0 && g_force_q < QB) QB = g_force_q; if (QB < 1) QB = 1;
  u16 *KTc = (u16 *)vt, *Vc = KTc + (size_t)C * hdt * T16, *Qb = Vc + (size_t)C * (hdt + 1) * T16, *Sb[2], *Pb[2];
  Sb[0] = Qb + (size_t)QB * hdt * T16; Sb[1] = Sb[0] + (size_t)C * T16; Pb[0] = Sb[1] + (size_t)C * T16; Pb[1] = Pb[0] + (size_t)C * T16;
  u16 *Ob = Pb[1] + (size_t)C * T16; HVX_Vector *Oacc = (HVX_Vector *)(((uintptr_t)(Ob + (size_t)(hdt + 1) * T16) + 127) & ~(uintptr_t)127); HVX_Vector *MX = Oacc + (size_t)QB * (hdt + 1) * 32;
  for (size_t i = 0; i < (size_t)C * T16; i++) Vc[(size_t)hdt * C * T16 + i] = 0x3C00;   /* ones column (row sums through P*V) */
  const HVX_Vector neginf = Q6_Vh_vsplat_R(0xFBFF);
  for (int h = 0; h < 16; h++) {
    for (int s0 = 0; s0 < nq; s0 += QB) {
      const int nqb = nq - s0 < QB ? nq - s0 : QB;
      cl_t cl = {k, v, q, o, KTc, Vc, Qb, 0, 0, C, h, 0, s0, nqb};
      pool_run(p_cq_load, &cl);
      for (int i = 0; i < nqb * 16; i++) MX[i] = neginf;
      for (int pass = 0; pass < 2; pass++) {
        for (int c0 = 0; c0 < nkb; c0 += C) {
          const int nkc = nkb - c0 < C ? nkb - c0 : C;
          cl.kb0 = c0; cl.nkc = nkc; cl.withV = pass; pool_run(p_cl_load, &cl);
          for (int s = 0; s <= nqb; s++) {
            const int b = s & 1; rm_t rm = {Sb[b], Pb[b], nkc, MX + (size_t)s * 16};
            rm_t pr = {Sb[(s - 1) & 1], 0, nkc, MX + (size_t)(s > 0 ? s - 1 : 0) * 16};
            if (pass == 0) {
              if (s > 0) pool_start(p_rowmax, &pr);
              if (s < nqb) for (int nb = 0; nb < nkc; nb++) { hmx_acc((uint8_t *)(Qb + (size_t)s * hdt * T16), (hdt << 11) | 0x7C, (uint8_t *)(KTc + (size_t)nb * hdt * T16), (hdt << 11) - 1); hmx_store((uint8_t *)(Sb[b] + (size_t)nb * T16), 0); }
              if (s > 0) pool_wait();
            } else {
              if (s < nqb) for (int nb = 0; nb < nkc; nb++) { hmx_acc((uint8_t *)(Qb + (size_t)s * hdt * T16), (hdt << 11) | 0x7C, (uint8_t *)(KTc + (size_t)nb * hdt * T16), (hdt << 11) - 1); hmx_store((uint8_t *)(Sb[b] + (size_t)nb * T16), 0); }
              if (s < nqb) pool_start(p_smfix, &rm);
              if (s > 0) {   /* P*V for slab s-1 while the workers run the softmax of slab s */
                const u16 *P = Pb[(s - 1) & 1];
                for (int j = 0; j < hdt + 1; j++) {
                  for (int k0 = 0; k0 < nkc; k0 += 32) { int cc = nkc - k0 < 32 ? nkc - k0 : 32; hmx_acc((uint8_t *)(P + (size_t)k0 * T16), (cc << 11) | 0x7C, (uint8_t *)(Vc + ((size_t)j * C + k0) * T16), (cc << 11) - 1); }
                  hmx_store((uint8_t *)(Ob + (size_t)j * T16), 0);
                  k_acc_tile(Oacc + ((size_t)(s - 1) * (hdt + 1) + j) * 32, Ob + (size_t)j * T16, c0 == 0);
                }
              }
              if (s < nqb) pool_wait();
            }
          }
        }
      }
      for (int s = 0; s < nqb; s++) k_final_acc(Qb + (size_t)s * hdt * T16, Oacc + (size_t)s * (hdt + 1) * 32, hdt);
      pool_run(p_cq_flush, &cl);
    }
  }
}

static uint64_t g_acc[8];
#define PC() HAP_perf_get_time_us()
static void attn_plain(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, uint8_t *vt) {
  const int nkb = nkv / 32;
  u16 *KT = (u16 *)vt, *Vw = KT + (size_t)4 * nkb * T16, *Qh = Vw + (size_t)5 * nkb * T16;           /* Qh also receives O */
  u16 *Sb[2], *Pb[2]; Sb[0] = Qh + (size_t)nq * 4 * T16; Sb[1] = Sb[0] + (size_t)nkb * T16; Pb[0] = Sb[1] + (size_t)nkb * T16; Pb[1] = Pb[0] + (size_t)nkb * T16;
  u16 *Ob = Pb[1] + (size_t)nkb * T16;                                                                  /* 5 tiles: 4 + sum */
  for (size_t i = 0; i < (size_t)nkb * T16; i++) Vw[(size_t)4 * nkb * T16 + i] = 0x3C00;               /* ones n-block */
  for (int h = 0; h < 16; h++) {
    ah_t a = {k, v, q, o, nkb, h, nq, KT, Vw, Qh};
    uint64_t c0 = PC(); pool_run(p_attn_setup, &a); g_acc[0] += PC() - c0;
    for (int s = 0; s <= nq; s++) {
      const int b = s & 1;
      uint64_t c1 = PC();
      if (s < nq) {
        for (int nb = 0; nb < nkb; nb++) {
          hmx_acc((uint8_t *)(Qh + (size_t)s * 4 * T16), (4 << 11) | 0x7C, (uint8_t *)(KT + (size_t)nb * 4 * T16), (4 << 11) - 1);
          hmx_store((uint8_t *)(Sb[b] + (size_t)nb * T16), 0);
        }
      }
      g_acc[1] += PC() - c1;
      sm_t sm = {Sb[b], Pb[b], nkb};
      uint64_t c2 = PC();
      if (s < nq) pool_start(p_softmax, &sm);
      if (s > 0) {   /* PV for slab s-1 while the workers run softmax(s) */
        const u16 *P = Pb[(s - 1) & 1];
        for (int j = 0; j < 5; j++) {
          for (int k0 = 0; k0 < nkb; k0 += 32) { int c = nkb - k0 < 32 ? nkb - k0 : 32; hmx_acc((uint8_t *)(P + (size_t)k0 * T16), (c << 11) | 0x7C, (uint8_t *)(Vw + ((size_t)j * nkb + k0) * T16), (c << 11) - 1); }
          hmx_store((uint8_t *)(Ob + (size_t)j * T16), 0);
        }
        uint64_t c3 = PC(); g_acc[2] += c3 - c2;
        k_rowscale_sum(Ob, 4, Ob + (size_t)4 * T16);
        g_acc[3] += PC() - c3;
        vcp(Qh + (size_t)(s - 1) * 4 * T16, Ob, 4 * 2048);   /* O replaces the (already consumed) Q slab */
      }
      uint64_t c4 = PC(); if (s < nq) pool_wait(); g_acc[4] += PC() - c4;
    }
    uint64_t c5 = PC(); pool_run(p_attn_flush, &a); g_acc[5] += PC() - c5;
  }
}

static void attn_run(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, uint8_t *vt) {
  if (g_force_c > 0 || attn_plain_bytes(nq, nkv / 32) > VT_TAB - 65536) attn_chunked(q, k, v, o, nq, nkv, vt);   /* the keys do not fit in VTCM (or a test forces it) */
  else attn_plain(q, k, v, o, nq, nkv, vt);
}

/* ---------------- block ---------------- */
static uint32_t us_now(void) { return (uint32_t)HAP_perf_get_time_us(); }

static uint32_t g_ctx; static uint8_t *g_vt; static int g_init; static u16 *g_et_h, *g_t2_h;
void mega_blk_release(void) { if (!g_init) return; pool_quit(); if (g_vt) HAP_compute_res_release(g_ctx); g_vt = 0; g_init = 0; }
static int vtcm_get(void) {
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  g_ctx = HAP_compute_res_acquire(&attr, 5000000); if (!g_ctx) return -1;
  g_vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  g_et = (u16 *)(g_vt + VT_TAB); g_t2 = g_et + 32768; g_scr = (HVX_Vector *)(g_vt + VT_TAB + 262144);
  memcpy(g_et, g_et_h, 65536); memcpy(g_t2, g_t2_h, 65536);
  return 0;
}

static uint8_t *g_blob; static size_t g_blobsz;
static int mega_block_impl(remote_handle64 hd, const int *meta, int nmeta, int wfd, const uint8_t *modp, int modlen, int *times, int ntimes, int *rc) {
  uint8_t *blob = g_blob; if (!blob) { *rc = -6; return 0; }
  uint32_t te0 = us_now();
  *rc = 0; for (int i = 0; i < ntimes; i++) times[i] = 0;
  void *wp = 0; uint64 wpa = 0;
  if (HAP_mmap_get(wfd, &wp, &wpa) || !wp) { *rc = -5; return 0; }
  if (meta[M_STOP] == 99) return 0;
  const uint8_t *wts = (const uint8_t *)wp;
  if (!g_init) {   /* one-time: worker pool + lookup tables (heap copies; VTCM copies are made at every acquire) */
    g_et_h = (u16 *)memalign(128, 65536); g_t2_h = (u16 *)memalign(128, 65536);
    tab_init(g_et_h, g_t2_h);
    g_init = 1;
  }
  g_force_c = meta[M_FORCE] & 0xFFFF; g_force_q = (meta[M_FORCE] >> 16) & 0xFFFF;
  gemm_set_q8(meta[M_Q8]);
  if (!pool_n()) pool_init(meta[M_NW] ? meta[M_NW] : 3);   /* workers spin: with several processes they only exist during a call */
  if (!g_vt && vtcm_get()) { *rc = -1; return 0; }
  uint32_t ctx = g_ctx; uint8_t *vt = g_vt;
  if (meta[M_FIRST]) { HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)hd, &rq); power_max((void *)hd); }   /* once per run of blocks */
  if (HAP_compute_res_hmx_lock(ctx)) { *rc = -2; return 0; }
  const int ntok = meta[M_NTOK], nctx = meta[M_NCTX], nc = nctx / 32, win = meta[M_WN] > 0, w0 = win ? meta[M_W0] : 0, ns = win ? meta[M_WN] : ntok / 32, nw = ns * 32, kvonly = win && meta[M_PASS] == 1;   /* nw: rows of this call (one window or all tokens) */
#define BUF(m) (blob + meta[m])
#define WB(m) ((const uint8_t *)wts + meta[m])
  float *x = (float *)BUF(M_X) + (size_t)w0 * NCB * 1024; const float *mod = (const float *)modp, *gain = (const float *)WB(M_GAIN), *cosT = (const float *)BUF(M_COS) + (size_t)w0 * 2048, *sinT = (const float *)BUF(M_SIN) + (size_t)w0 * 2048;
  u16 *h16 = (u16 *)BUF(M_H), *q16 = (u16 *)BUF(M_Q), *k16 = (u16 *)BUF(M_K) + (size_t)w0 * NCB * T16, *v16 = (u16 *)BUF(M_V) + (size_t)w0 * NCB * T16, *kall = (u16 *)BUF(M_K), *vall = (u16 *)BUF(M_V), *o16 = (u16 *)BUF(M_O), *t16 = (u16 *)BUF(M_T), *mid = (u16 *)BUF(M_MID), *ck16 = (u16 *)BUF(M_CK), *cv16 = (u16 *)BUF(M_CV);
  static HVX_Vector LM[NCB] __attribute__((aligned(128))), LN[NCB] __attribute__((aligned(128)));
  if (ntimes > 24) times[24] = (int)(us_now() - te0);   /* entry -> start of work: mmap_get, acquire, pool, HMX lock */
  int ti = 0; uint32_t t0 = us_now();
#define TICK() do { uint32_t t1 = us_now(); if (ti < ntimes) times[ti++] = (int)(t1 - t0); t0 = t1; if (meta[M_STOP] && ti >= meta[M_STOP]) goto done; } while (0)
  if (meta[M_STOP] == 92) {   /* gemm weights/acts from blob vs from DSP heap */
    size_t wb = (size_t)DIM * DIM * 2;
    u16 *Wh = (u16 *)memalign(128, wb), *Ah = (u16 *)memalign(128, (size_t)ntok * DIM * 2), *Oh = (u16 *)memalign(128, (size_t)ntok * DIM * 2);
    memcpy(Wh, WB(M_WSQ), wb); memcpy(Ah, h16, (size_t)ntok * DIM * 2);
    uint32_t a0 = us_now(); gemm_run(h16, ntok, DIM, (u16 *)WB(M_WSQ), DIM, q16, vt); times[0] = us_now() - a0;           /* all blob */
    a0 = us_now(); gemm_run(Ah, ntok, DIM, Wh, DIM, Oh, vt); times[1] = us_now() - a0;                                   /* all heap */
    a0 = us_now(); gemm_run(h16, ntok, DIM, Wh, DIM, Oh, vt); times[2] = us_now() - a0;                                   /* blob act, heap w */
    a0 = us_now(); gemm_run(Ah, ntok, DIM, (u16 *)WB(M_WSQ), DIM, Oh, vt); times[3] = us_now() - a0;                    /* heap act, blob w */
    a0 = us_now(); gemm_run(Ah, ntok, DIM, Wh, DIM, q16, vt); times[4] = us_now() - a0;                                   /* blob out */
    free(Wh); free(Ah); free(Oh); goto done;
  }
  par_t P; memset(&P, 0, sizeof P); P.ns = ns;
  for (int site = 0; site < 3; site++) {
    const float *sh = mod + (size_t)(site * 3 + 0) * DIM, *sc = mod + (size_t)(site * 3 + 1) * DIM, *gt = mod + (size_t)(site * 3 + 2) * DIM;
    k_ln_prep(LM, LN, sh, sc);
    if (meta[M_STOP] == 95) goto done;
    P.x = x; P.h = h16; P.M = LM; P.N = LN;
    if (meta[M_STOP] == 94) { p_ln(&P, 0, 1); goto done; }
    if (meta[M_STOP] == 93) { P.ns = 2; p_ln(&P, 0, 1); goto done; }
    pool_run(p_ln, &P); TICK();
    if (site == 0) {
      if (!kvonly) gemm_run(h16, nw, DIM, (u16 *)WB(M_WSQ), DIM, q16, vt);
      if (!win || kvonly) { gemm_run(h16, nw, DIM, (u16 *)WB(M_WSK), DIM, k16, vt); gemm_run(h16, nw, DIM, (u16 *)WB(M_WSV), DIM, v16, vt); }   /* windows: K and V of every window are made first (pass 1), the full block follows */
      TICK();
      P.g = gain; P.c = cosT; P.s = sinT; P.rope = 1; P.q = q16; if (!kvonly) pool_run(p_rope, &P);
      if (!win || kvonly) { P.g = gain + 128; P.q = k16; pool_run(p_rope, &P); }
      TICK();
      if (kvonly) goto done;
      attn_run(q16, kall, vall, o16, ns, ntok, vt); TICK();
      { gemm_epi_t e = {1, x, gt, 0, 0}; gemm_run_ex(o16, nw, DIM, (u16 *)WB(M_WSO), DIM, t16, vt, &e); } TICK();
    } else if (site == 1) {
      gemm_run(h16, nw, DIM, (u16 *)WB(M_WCQ), DIM, q16, vt);
      gemm_run((u16 *)BUF(M_CTX), nctx, 1024, (u16 *)WB(M_WCK), DIM, ck16, vt); gemm_run((u16 *)BUF(M_CTX), nctx, 1024, (u16 *)WB(M_WCV), DIM, cv16, vt);
      u16 *ck2 = (u16 *)BUF(M_CK2), *cv2 = (u16 *)BUF(M_CV2); const int nag = meta[M_NAG];
      if (nag) { gemm_run((u16 *)BUF(M_CTX2), nctx, 1024, (u16 *)WB(M_WCK), DIM, ck2, vt); gemm_run((u16 *)BUF(M_CTX2), nctx, 1024, (u16 *)WB(M_WCV), DIM, cv2, vt); }
      TICK();
      P.rope = 0; P.g = gain + 256; P.q = q16; pool_run(p_rope, &P);
      P.ns = nc; P.g = gain + 384; P.q = ck16; pool_run(p_rope, &P); if (nag) { P.q = ck2; pool_run(p_rope, &P); } P.ns = ns; TICK();
      attn_run(q16, ck16, cv16, o16, ns, nctx, vt);
      if (nag) {   /* the negative context: its output goes to t16 (free until the output projection), then both are combined into o16 */
        attn_run(q16, ck2, cv2, t16, ns, nctx, vt);
        nag_t Nt; Nt.z = o16; Nt.n = t16; Nt.ns = ns; memcpy(&Nt.s, &meta[M_NAGS], 4); memcpy(&Nt.tau, &meta[M_NAGT], 4); memcpy(&Nt.alpha, &meta[M_NAGA], 4);
        pool_run(nag == 2 ? p_nag : p_nag_hvx, &Nt);   /* 2 = scalar reference (MEGA_NAG_REF), 1 = HVX */
      }
      TICK();
      { gemm_epi_t e = {1, x, gt, 0, 0}; gemm_run_ex(o16, nw, DIM, (u16 *)WB(M_WCO), DIM, t16, vt, &e); } TICK();
    } else {
      { gemm_epi_t e = {2, 0, 0, g_t2, g_scr}; gemm_run_ex(h16, nw, DIM, (u16 *)WB(M_WM1), 8192, mid, vt, &e); } TICK();
      { gemm_geo_t g0 = {256, 256, 0}, g1 = {256, 256, 128};   /* two K halves: weights are re-read 2x less; partial 1 -> t16, partial 2 + residual */
        gemm_run_geo(mid, nw, 4096, (u16 *)WB(M_WM2), DIM, t16, vt, 0, &g0);
        gemm_epi_t e = {4, x, gt}; e.prev = t16; gemm_run_geo(mid, nw, 4096, (u16 *)WB(M_WM2), DIM, t16, vt, &e, &g1); } TICK();
    }
  }
done:
  for (int i = 0; i < 6 && 17 + i < ntimes; i++) times[17 + i] = (int)g_acc[i];
  memset(g_acc, 0, sizeof g_acc);
  HAP_compute_res_hmx_unlock(ctx);
  if (meta[M_REL]) { pool_quit(); HAP_compute_res_release(g_ctx); g_vt = 0; }
  return 0;
}

static void cache_sync(void *p, size_t bytes) { for (char *c = (char *)p; c < (char *)p + bytes; c += 32) Q6_dccleaninva_A(c); }
int mega_block(remote_handle64 hd, const int *meta, int nmeta, int wfd, int bfd, const uint8_t *mod, int modlen, int *times, int ntimes, int *rc) {
  void *bp = 0; uint64 bpa = 0;
  if (HAP_mmap_get(bfd, &bp, &bpa) || !bp) { *rc = -7; return 0; }
  g_blob = (uint8_t *)bp;
  const size_t xb = (size_t)meta[M_NTOK] * DIM * 4;
  if (meta[M_FIRST]) { cache_sync(g_blob + meta[M_X], xb); cache_sync(g_blob + meta[M_CTX], 1048576); cache_sync(g_blob + meta[M_COS], (size_t)meta[M_NTOK] / 32 * 8192); cache_sync(g_blob + meta[M_SIN], (size_t)meta[M_NTOK] / 32 * 8192); }
  int r = mega_block_impl(hd, meta, nmeta, wfd, mod, modlen, times, ntimes, rc);
  if (meta[M_REL]) cache_sync(g_blob + meta[M_X], xb);   /* make the result visible to the CPU */
  HAP_mmap_put(bfd);
  if (*rc != -5) HAP_mmap_put(wfd);
  return r;
}
