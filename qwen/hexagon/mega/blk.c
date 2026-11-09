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
void hmx_clr(void);
#include "pool.h"
#include "kern.h"
#include "blk_meta.h"

typedef uint16_t u16;
#define T16 1024          /* halfwords per h16 tile */
#define VT_TAB (7u << 20) /* tables at the top 1 MB of VTCM */

static u16 *g_et, *g_tabs, *g_tabs2, *g_t2; static HVX_Vector *g_scr;   /* tables (exp, 3 SiTU) + per-worker gather scratch */

static void vcp(void *d, const void *s, size_t bytes) {
  HVX_Vector *dv = (HVX_Vector *)d; const HVX_Vector *sv = (const HVX_Vector *)s; size_t n = bytes / 128;
  for (size_t i = 0; i < n; i++) { if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(sv + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32)); dv[i] = sv[i]; }
}


/* ---------------- simple parallel kernels ---------------- */
typedef struct { const float *x; u16 *h; const HVX_Vector *M, *N; int ns; } par_t;
static void p_ln(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_rmsmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, p->M, p->N, 1e-6f); }

static void p_lnn(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_lnmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, p->M, p->N, 1e-6f); }

/* ---------------- attention (joint, NHEAD heads, last key tile masked) ---------------- */
static int g_qcb, g_kcb, g_grp;   /* tiles per slab of q/o, of k/v, and query heads per kv head (DiT: NCB, NCB, 1; padded ViT: 64, 64, 1; Qwen3-VL text encoder: 128, 32, 4) */
typedef struct { const u16 *k, *v, *q; u16 *o; int nkb, h, nq; u16 *KT, *Vw, *Qh; } ah_t;
static void p_attn_setup(void *a, int idx, int n) {   /* build K^T and V(+ones) weight tiles for head h, copy Q tiles */
  ah_t *p = (ah_t *)a;
  for (int kb = idx; kb < p->nkb; kb += n)
    for (int j = 0; j < 4; j++) {
      const u16 *ks = p->k + ((size_t)kb * g_kcb + (p->h / g_grp) * 4 + j) * T16; u16 *kd = p->KT + ((size_t)kb * 4 + j) * T16;
      k_transpose_tile(ks, kd);
      vcp(p->Vw + ((size_t)j * p->nkb + kb) * T16, p->v + ((size_t)kb * g_kcb + (p->h / g_grp) * 4 + j) * T16, 2048);
    }
  for (int s = idx; s < p->nq; s += n) vcp(p->Qh + (size_t)s * 4 * T16, p->q + ((size_t)s * g_qcb + p->h * 4) * T16, 8192);
}
static void p_attn_flush(void *a, int idx, int n) { ah_t *p = (ah_t *)a; for (int s = idx; s < p->nq; s += n) vcp(p->o + ((size_t)s * g_qcb + p->h * 4) * T16, p->Qh + (size_t)s * 4 * T16, 8192); }
typedef struct { const u16 *S; u16 *P; int nkb; const HVX_Vector *km; int kst; int np; const int *pk; const int *pmi; const HVX_Vector *pm; } sm_t;
static void p_softmax(void *a, int i, int n) { sm_t *p = (sm_t *)a;
  if (p->pm) k_softmax2pm(p->S, p->P, p->nkb, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, p->pk, p->pmi, p->pm, p->np);
  else k_softmax2p(p->S, p->P, p->nkb, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, p->km, p->kst); }

static uint64_t g_acc[8];
#define PC() HAP_perf_get_time_us()
static u16 g_km[64] __attribute__((aligned(128)));
static u16 g_cm[16 * 64] __attribute__((aligned(128)));   /* causal diagonal-tile masks, one vector per row pair: lane l = key c*2 + (row&1); masked when c > row */
#define MAXPART 16
#define MAXPT 512
static u16 g_pm[MAXPART * 16 * 64] __attribute__((aligned(128))); static int g_pk[MAXPT], g_pmi[MAXPT];
/* causal == 2: block-causal; each bidirectional block (condition image) [ba,bb) of bl[0..5] = {ba,bb,...} (three at most; ba == bb: unused).
   Row i sees the keys j < lim(i), lim(i) = i + 1, or the end of its block when it lies inside one. */
static int row_lim(int i, const int *bl) { int l = i + 1; for (int b = 0; b < 3; b++) if (i >= bl[2 * b] && i < bl[2 * b + 1] && bl[2 * b + 1] > l) l = bl[2 * b + 1]; return l; }
static int nks_of(int s, int nkb, int causal, const int *bl) {
  if (!causal) return nkb;
  int n = s + 1;
  if (causal == 2) for (int b = 0; b < 3; b++) if (32 * s + 31 >= bl[2 * b] && 32 * s < bl[2 * b + 1]) { const int e = (bl[2 * b + 1] + 31) / 32; if (e > n) n = e; }
  return n < nkb ? n : nkb;
}
static int build_partial(int s, int nks, const int *bl) {   /* partially visible key tiles of slab s -> g_pk / g_pmi / g_pm; returns the tile count (-1: out of table space) */
  const int r0 = 32 * s; int lim[32], mn = 1 << 30, np = 0, nm = 0; unsigned char cl[MAXPART][32];
  for (int r = 0; r < 32; r++) { lim[r] = row_lim(r0 + r, bl); if (lim[r] < mn) mn = lim[r]; }
  for (int kt = 0; kt < nks; kt++) {
    const int k0 = 32 * kt; if (k0 + 32 <= mn) continue;   /* fully visible */
    unsigned char c[32]; for (int r = 0; r < 32; r++) { int v = lim[r] - k0; c[r] = v < 0 ? 0 : v > 32 ? 32 : v; }
    int m = 0; for (; m < nm; m++) if (!memcmp(cl[m], c, 32)) break;
    if (m == nm) { if (nm == MAXPART) return -1; memcpy(cl[nm], c, 32);
      for (int p = 0; p < 16; p++) for (int l = 0; l < 64; l++) g_pm[(nm * 16 + p) * 64 + l] = (l >> 1) < c[2 * p + (l & 1)] ? 0 : 0xFFFF;
      nm++; }
    if (np == MAXPT) return -1;
    g_pk[np] = kt; g_pmi[np] = m; np++;
  }
  return np;
}
static void attn_run(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, int nvalid, int causal, const int *bl, int qcb, int kcb, int grp, uint8_t *vt) {
  const int nkb = nkv / 32; g_qcb = qcb; g_kcb = kcb; g_grp = grp;
  const int nv = nvalid - (nkb - 1) * 32;   /* valid keys in the last tile */
  const HVX_Vector *km = 0;
  if (causal == 1) for (int p = 0; p < 16; p++) for (int l = 0; l < 64; l++) g_cm[p * 64 + l] = (l >> 1) > 2 * p + (l & 1) ? 0xFFFF : 0;
  if (!causal && nv < 32) { for (int l = 0; l < 64; l++) g_km[l] = (l >> 1) >= nv ? 0xFFFF : 0; km = (const HVX_Vector *)g_km; }
  u16 *KT = (u16 *)vt, *Vw = KT + (size_t)4 * nkb * T16, *Qh = Vw + (size_t)5 * nkb * T16;           /* Qh also receives O */
  u16 *Sb[2], *Pb[2]; Sb[0] = Qh + (size_t)nq * 4 * T16; Sb[1] = Sb[0] + (size_t)nkb * T16; Pb[0] = Sb[1] + (size_t)nkb * T16; Pb[1] = Pb[0] + (size_t)nkb * T16;
  u16 *Ob = Pb[1] + (size_t)nkb * T16;                                                                  /* 5 tiles: 4 + sum */
  for (size_t i = 0; i < (size_t)nkb * T16; i++) Vw[(size_t)4 * nkb * T16 + i] = 0x3C00;               /* ones n-block */
  for (int h = 0; h < qcb / 4; h++) {
    ah_t a = {k, v, q, o, nkb, h, nq, KT, Vw, Qh};
    uint64_t c0 = PC(); pool_run(p_attn_setup, &a); g_acc[0] += PC() - c0;
    for (int s = 0; s <= nq; s++) {
      const int b = s & 1;
      uint64_t c1 = PC();
      const int nks = nks_of(s, nkb, causal, bl);   /* key tiles visible to slab s */
      if (s < nq) {
        for (int nb = 0; nb < nks; nb++) {
          hmx_acc((uint8_t *)(Qh + (size_t)s * 4 * T16), (4 << 11) | 0x7C, (uint8_t *)(KT + (size_t)nb * 4 * T16), (4 << 11) - 1);
          hmx_store((uint8_t *)(Sb[b] + (size_t)nb * T16), 0);
        }
      }
      g_acc[1] += PC() - c1;
      int npart = 0; if (causal == 2 && s < nq) npart = build_partial(s, nks, bl);
      sm_t sm = {Sb[b], Pb[b], nks, causal ? (const HVX_Vector *)g_cm : km, causal == 1, npart, g_pk, g_pmi, causal == 2 ? (const HVX_Vector *)g_pm : 0};
      uint64_t c2 = PC();
      if (s < nq) pool_start(p_softmax, &sm);
      if (s > 0) {   /* PV for slab s-1 while the workers run softmax(s) */
        const u16 *P = Pb[(s - 1) & 1];
        const int nkp = nks_of(s - 1, nkb, causal, bl);   /* tiles of P for slab s-1 */
        for (int j = 0; j < 5; j++) {
          for (int k0 = 0; k0 < nkp; k0 += 32) { int c = nkp - k0 < 32 ? nkp - k0 : 32; hmx_acc((uint8_t *)(P + (size_t)k0 * T16), (c << 11) | 0x7C, (uint8_t *)(Vw + ((size_t)j * nkb + k0) * T16), (c << 11) - 1); }
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

/* ---------------- block ---------------- */
static uint32_t us_now(void) { return (uint32_t)HAP_perf_get_time_us(); }

static uint32_t g_ctx; static uint8_t *g_vt; static int g_init; static u16 *g_et_h, *g_tabs_h, *g_tabs2_h, *g_t2_h;
void mega_blk_release(void) { if (!g_init) return; pool_quit(); if (g_vt) HAP_compute_res_release(g_ctx); g_vt = 0; g_init = 0; }
static int vtcm_get(void) {
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  g_ctx = HAP_compute_res_acquire(&attr, 5000000); if (!g_ctx) return -1;
  g_vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  g_et = (u16 *)(g_vt + VT_TAB); g_tabs = g_et + 32768; g_tabs2 = g_tabs + 3 * 32768; g_scr = (HVX_Vector *)(g_vt + VT_TAB + 393216); g_t2 = (u16 *)(g_vt + VT_TAB + 458752);
  memcpy(g_et, g_et_h, 65536); memcpy(g_tabs, g_tabs_h, 3 * 65536); memcpy(g_tabs2, g_tabs2_h, 2 * 65536); memcpy(g_t2, g_t2_h, 65536);
  return 0;
}

/* HMX self-test: one 32x32x32 fp16 tile of ones x ones must give 32.0 */
static int hmx_selftest(uint8_t *vt) {
  uint8_t *a = vt + (8u << 20) - 16384, *w = a + 2048, *o = w + 2048;
  HVX_Vector one = Q6_Vh_vsplat_R(0x3C003C00); HVX_Vector *pa = (HVX_Vector *)a, *pw = (HVX_Vector *)w, *po = (HVX_Vector *)o;
  for (int i = 0; i < 16; i++) { pa[i] = one; pw[i] = one; po[i] = Q6_V_vzero(); }
  asm volatile("barrier" ::: "memory");
  hmx_clr(); hmx_acc(a, (1 << 11) | 0x7C, w, (1 << 11) - 1); hmx_store(o, 0);
  volatile uint16_t *r = (volatile uint16_t *)o; return r[0] == 0x5000 ? 0 : (int)r[0] + 1;
}
typedef struct { const u16 *t; u16 *t2; float *r; const float *a, *b, *c, *d, *e, *f; float *o; int ns, nct, T; } tej_t;
static void j_t2r(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_t2r(p->t, p->r, p->ns, p->nct, i, n); }
static void j_r2t(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_r2t(p->r, p->t2, p->ns, p->nct, i, n); }
static void j_conv(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_conv_silu(p->a, p->o, p->b, p->T, 8192, i, n); }
static void j_delta(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_delta(p->a, p->b, p->c, p->o, p->d, p->e, p->f, p->T, i, n); }
static void j_aprep(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_attn_prep(p->r, p->o, p->a, p->b, p->c, p->T, i, n); }
static void j_attn(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_attn(p->a, p->b, p->o, p->T, i, n); }
static void j_aprep3(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_attn_prep3(p->r, p->o, p->a, p->b, p->c, p->T, i, n); }
static void j_attn3(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_attn3(p->a, p->b, p->o, p->r, p->T, i, n); }
static uint8_t *g_blob;
static int mega_block_impl(remote_handle64 hd, const int *meta, int nmeta, int wfd, const uint8_t *modp, int modlen, int *times, int ntimes, int *rc) {
  uint8_t *blob = g_blob; if (!blob) { *rc = -6; return 0; }
  uint32_t te0 = us_now();
  *rc = 0; for (int i = 0; i < ntimes; i++) times[i] = 0;
  void *wp = 0; uint64 wpa = 0;
  if (HAP_mmap_get(wfd, &wp, &wpa) || !wp) { *rc = -5; return 0; }
  if (meta[M_STOP] == 99) return 0;
  const uint8_t *wts = (const uint8_t *)wp + meta[M_WOFF];
  if (!g_init) {
    g_et_h = (u16 *)memalign(128, 65536); g_tabs_h = (u16 *)memalign(128, 3 * 65536);
    { u16 *t2 = (u16 *)memalign(128, 65536); tab_init(g_et_h, t2); free(t2); }
    tab_init_situ(g_tabs_h);
    g_tabs2_h = (u16 *)memalign(128, 2 * 65536); tab_init_swiglu(g_tabs2_h);
    g_t2_h = (u16 *)memalign(128, 65536); tab_init_gelut(g_t2_h);
    g_init = 1;
  }
  gemm_set_q8(1);
  if (!pool_n()) pool_init(meta[M_NW] ? meta[M_NW] : 3);
  if (!g_vt && vtcm_get()) { *rc = -1; return 0; }
  uint32_t ctx = g_ctx; uint8_t *vt = g_vt;
  if (meta[M_FIRST]) { HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)hd, &rq); power_max((void *)hd); }
  if (HAP_compute_res_hmx_lock(ctx)) { *rc = -2; return 0; }
  { int tries = 0, tr;
    while ((tr = hmx_selftest(vt)) != 0 && tries < 3) {
      tries++; HAP_compute_res_hmx_unlock(ctx);
      { uint64_t t0 = HAP_perf_get_time_us(); while (HAP_perf_get_time_us() - t0 < 30000) {} }
      if (HAP_compute_res_hmx_lock(ctx)) { *rc = -2; return 0; }
    }
    if (tr) { *rc = -77; HAP_compute_res_hmx_unlock(ctx); return 0; } }
  hmx_clr();
  g_ncb = meta[M_NCB] ? meta[M_NCB] : 96;
  const int ntok = meta[M_NTOK], ns = ntok / 32;
#define BUF(m) (blob + meta[m])
#define WB(o) ((const u16 *)(wts + (o)))
  float *x = (float *)BUF(M_X); const float *mod = (const float *)modp, *gain = (const float *)(wts + W_GAIN), *cosT = (const float *)BUF(M_COS), *sinT = (const float *)BUF(M_SIN);
  u16 *h16 = (u16 *)BUF(M_H), *q16 = (u16 *)BUF(M_Q), *k16 = (u16 *)BUF(M_K), *v16 = (u16 *)BUF(M_V), *o16 = (u16 *)BUF(M_O), *t16 = (u16 *)BUF(M_T), *mid = (u16 *)BUF(M_MID);
  u16 *Lq = (u16 *)BUF(M_LQ), *Lo = (u16 *)BUF(M_LO), *Lu = (u16 *)BUF(M_LU), *Ld = (u16 *)BUF(M_LD);
  static HVX_Vector LM[MAXNCB] __attribute__((aligned(128))), LN[MAXNCB] __attribute__((aligned(128)));
  static float g_zero[4096] __attribute__((aligned(128)));
  if (ntimes > 24) times[24] = (int)(us_now() - te0);
  int ti = 0; uint32_t t0 = us_now();
#define TICK() do { uint32_t t1 = us_now(); if (ti < ntimes) times[ti++] = (int)(t1 - t0); t0 = t1; if (meta[M_STOP] && ti >= meta[M_STOP]) goto done; } while (0)
  par_t P; memset(&P, 0, sizeof P); P.ns = ns; P.x = x; P.h = h16; P.M = LM; P.N = LN;
  const float *sh_a = mod, *sc_a = mod + DIM, *gt_a = mod + 2 * DIM, *sh_f = mod + 3 * DIM, *sc_f = mod + 4 * DIM, *gt_f = mod + 5 * DIM;
  const gemm_geo_t gq = {NCB, NCB + 4, 0, Lq, 12, 4}, gk = {NCB, NCB + 4, 0, Lq + 4 * T16, 12, 4}, gv = {NCB, NCB + 4, 0, Lq + 8 * T16, 12, 4};

  if (meta[M_MODE] == 20 || meta[M_MODE] == 21 || meta[M_MODE] == 22) {   /* Qwen3.5 layer: x += attn(rms(x)); x += ffn(rms(x)); everything on 4096-wide tiles */
    const te_hdr_t *H = (const te_hdr_t *)wts; const float *vec = (const float *)(wts + H->voff);
    const int T = meta[M_NVALID]; float *R = (float *)BUF(M_ROWS); const int full = meta[M_MODE] >= 21, v3 = meta[M_MODE] == 22;   /* 22: Qwen3-VL layer (q 4096 out, no gate, hd128, cs rows [T][128]) */
    const float *ones = mod + 2 * 4096;
    const int igu = full ? 3 : 4, idn = full ? 4 : 5;
    k_rms_prep(LM, LN, vec, g_zero, g_zero); pool_run(p_ln, &P); TICK();
    tej_t J; memset(&J, 0, sizeof J); J.ns = ns; J.T = T;
    gemm_epi_t eo; memset(&eo, 0, sizeof eo); eo.mode = 1; eo.x32 = x; eo.gate = ones;
    if (!full) {
      float *R0 = R, *R1 = R0 + (size_t)ntok * 8192, *R2 = R1 + (size_t)ntok * 8192, *R3 = R2 + (size_t)ntok * 4096, *R4 = R3 + (size_t)ntok * 64;
      gemm_set_q8(H->fmt[0]); gemm_run_geo(h16, ntok, 4096, WB(H->off[0]), 8192, mid, vt, 0, 0);
      gemm_set_q8(H->fmt[1]); gemm_run_geo(h16, ntok, 4096, WB(H->off[1]), 4096, q16, vt, 0, 0);
      gemm_set_q8(H->fmt[2]); gemm_run_geo(h16, ntok, 4096, WB(H->off[2]), 64, t16, vt, 0, 0); TICK();
      J.t = mid; J.r = R0; J.nct = 256; pool_run(j_t2r, &J); J.t = q16; J.r = R2; J.nct = 128; pool_run(j_t2r, &J); J.t = t16; J.r = R3; J.nct = 2; pool_run(j_t2r, &J); TICK();
      memset(R4, 0, (size_t)ntok * 4096 * 4);
      J.a = R0; J.o = R1; J.b = vec + 8192; pool_run(j_conv, &J);
      J.a = R1; J.b = R3; J.c = R2; J.o = R4; J.d = vec + 8192 + 32768; J.e = J.d + 32; J.f = J.d + 64; pool_run(j_delta, &J); TICK();
      J.r = R4; J.t2 = o16; J.nct = 128; pool_run(j_r2t, &J); TICK();
      gemm_set_q8(H->fmt[3]); gemm_run_geo(o16, ntok, 4096, WB(H->off[3]), 4096, t16, vt, &eo, 0); TICK();
    } else {
      if (v3 && meta[M_FMT] == 1) {   /* Qwen3-VL: tile-domain like the DiT: q/k per-head RMSNorm + rope in the GEMM epilogue (cos/sin tiles in M_COS/M_SIN), HMX causal attention with 8 kv heads for 32 q heads, no row-domain round trip */
        static float g_qg[128] __attribute__((aligned(128)));
        for (int i2 = 0; i2 < 128; i2++) g_qg[i2] = vec[8192 + i2] * 0.08838834764831845f;   /* attention scale 1/sqrt(128) folded into q like the DiT pack */
        gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.cosT = cosT; e.sinT = sinT; e.gain = g_qg;
        gemm_set_q8(H->fmt[0]); gemm_run_geo(h16, ntok, 4096, WB(H->off[0]), 4096, q16, vt, &e, 0);
        const uint8_t *kvp = (const uint8_t *)WB(H->off[1]);   /* kv pack = [64 n-blocks][128 kb] records: k = first 32 n-blocks, v = last 32 */
        e.gain = vec + 8192 + 128; gemm_set_q8(H->fmt[1]); gemm_run_geo(h16, ntok, 4096, (const u16 *)kvp, 1024, k16, vt, &e, 0);
        gemm_run_geo(h16, ntok, 4096, (const u16 *)(kvp + (size_t)32 * 128 * (H->fmt[1] == 2 ? 768 : 1152)), 1024, v16, vt, 0, 0); TICK();
        attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 1, 0, 128, 32, 4, vt); TICK();
      } else {
      const int qw = v3 ? 4096 : 8192;
      float *Rq = R, *Rkv = Rq + (size_t)ntok * qw, *Ro = Rkv + (size_t)ntok * 2048, *Kts = Ro + (size_t)ntok * 4096;
      gemm_set_q8(H->fmt[0]); gemm_run_geo(h16, ntok, 4096, WB(H->off[0]), qw, mid, vt, 0, 0);
      gemm_set_q8(H->fmt[1]); gemm_run_geo(h16, ntok, 4096, WB(H->off[1]), 2048, q16, vt, 0, 0); TICK();
      J.t = mid; J.r = Rq; J.nct = qw / 32; pool_run(j_t2r, &J); J.t = q16; J.r = Rkv; J.nct = 64; pool_run(j_t2r, &J); TICK();
      if (v3) { J.r = Rq; J.o = Rkv; J.a = vec + 8192; J.b = vec + 8192 + 128; J.c = (const float *)BUF(M_CS); pool_run(j_aprep3, &J);
                J.a = Rq; J.b = Rkv; J.o = Ro; J.r = Kts; pool_run(j_attn3, &J); }
      else { memset(Ro, 0, (size_t)ntok * 4096 * 4); J.r = Rq; J.o = Rkv; J.a = vec + 8192; J.b = vec + 8192 + 256; J.c = (const float *)BUF(M_CS); pool_run(j_aprep, &J);
             J.a = Rq; J.b = Rkv; J.o = Ro; pool_run(j_attn, &J); } TICK();
      J.r = Ro; J.t2 = o16; J.nct = 128; pool_run(j_r2t, &J); TICK();
      }
      gemm_set_q8(H->fmt[2]); gemm_run_geo(o16, ntok, 4096, WB(H->off[2]), 4096, t16, vt, &eo, 0); TICK();
    }
    /* ffn */
    k_rms_prep(LM, LN, vec + 4096, g_zero, g_zero); pool_run(p_ln, &P);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr; gemm_set_q8(H->fmt[igu]);
      gemm_run_geo(h16, ntok, 4096, WB(H->off[igu]), 24576, mid, vt, &e, 0); }
    { gemm_epi_t e1; memset(&e1, 0, sizeof e1); e1.mode = 4; e1.x32 = x; e1.gate = ones; e1.prev = t16; gemm_set_q8(H->fmt[idn]);
      const gemm_geo_t d0 = {384, 384, 0, 0, 0, 0}, d1 = {384, 384, 192, 0, 0, 0};
      gemm_run_geo(mid, ntok, 6144, WB(H->off[idn]), 4096, t16, vt, 0, &d0);
      gemm_run_geo(mid, ntok, 6144, WB(H->off[idn]), 4096, t16, vt, &e1, &d1); } TICK();
    gemm_set_q8(1); goto done;
  }
  if (meta[M_MODE] == 40) {   /* Qwen-Image-2.1 DiT block. rows = own tokens (ns slabs); keys = own rows + meta[M_NPRE] cached prefix slabs placed after them in K/V; causal = prefix prefill.
                                 mod = [scale1, tanh(gate1), scale2, tanh(gate2)] x 4096 f32 (host side) */
    const float *s1 = mod, *g1 = mod + DIM, *s2 = mod + 2 * DIM, *g2 = mod + 3 * DIM; const float *gn;
    /* pack v2 (4096 B header: magic, offsets of q,k,v,o,gate/proj,down,A_qkv,A_o,A_gp,A_down,gain, tile formats of the six big matrices: 1 = q8, 2 = q4a (native Q4_K)); v1 = fixed offsets, all q8 */
    uint32_t qo[11] = {QW_Q, QW_K, QW_V, QW_O, QW_GP, QW_DN, QW_AQKV, QW_AO, QW_AGP, QW_ADN, QW_GAIN}; int qf[6] = {1, 1, 1, 1, 1, 1};
    { const int32_t *h4 = (const int32_t *)wts; if (h4[0] == 0x00325751) { for (int i = 0; i < 11; i++) qo[i] = (uint32_t)h4[1 + i]; for (int i = 0; i < 6; i++) qf[i] = h4[12 + i]; } }
    gn = (const float *)(wts + qo[10]);
    const int npre = meta[M_NPRE], causal = meta[M_CAUS];
    k_ln_prep(LM, LN, g_zero, s1); pool_run(p_lnn, &P); TICK();
    gemm_set_q8(1); gemm_run(h16, ntok, DIM, WB(qo[6]), 384, Lq, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = gn; e.cosT = cosT; e.sinT = sinT;
      gemm_set_q8(qf[0]); gemm_run_geo(h16, ntok, DIM + 128, WB(qo[0]), DIM, q16, vt, &e, &gq);
      e.gain = gn + 128; gemm_set_q8(qf[1]); gemm_run_geo(h16, ntok, DIM + 128, WB(qo[1]), DIM, k16, vt, &e, &gk);
      gemm_set_q8(qf[2]); gemm_run_geo(h16, ntok, DIM + 128, WB(qo[2]), DIM, v16, vt, 0, &gv); } TICK();
    const int bl[6] = {meta[M_BA], meta[M_BB], meta[M_B2A], meta[M_B2B], meta[M_B3A], meta[M_B3B]};
    attn_run(q16, k16, v16, o16, ns, (ns + npre) * 32, meta[M_NVALID], causal, bl, NCB, NCB, 1, vt); TICK();
    { gemm_set_q8(1); gemm_run(o16, ntok, DIM, WB(qo[7]), 128, Lo, vt);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = g1;
      const gemm_geo_t go = {NCB, NCB + 4, 0, Lo, 4, 4};
      gemm_set_q8(qf[3]); gemm_run_geo(o16, ntok, DIM + 128, WB(qo[3]), DIM, t16, vt, &e, &go); } TICK();
    k_ln_prep(LM, LN, g_zero, s2); pool_run(p_lnn, &P); TICK();
    { gemm_set_q8(1); gemm_run(h16, ntok, DIM, WB(qo[8]), 128, Lu, vt);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr;   /* silu(gate_j) * proj_j, n-blocks interleaved (gate, proj) */
      const gemm_geo_t gu = {NCB, NCB + 4, 0, Lu, 4, 4};
      gemm_set_q8(qf[4]); gemm_run_geo(h16, ntok, DIM + 128, WB(qo[4]), 24576, mid, vt, &e, &gu); } TICK();
    { gemm_set_q8(1); gemm_run(mid, ntok, 12288, WB(qo[9]), 128, Ld, vt); gemm_set_q8(qf[5]);
      const gemm_geo_t g0 = {384, 388, 0, 0, 0, 0}, g1_ = {384, 388, 192, Ld, 4, 4};
      gemm_run_geo(mid, ntok, 6144, WB(qo[5]), DIM, t16, vt, 0, &g0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = g2; e.prev = t16;
      gemm_run_geo(mid, ntok, 6144 + 128, WB(qo[5]), DIM, t16, vt, &e, &g1_); } TICK();
    gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 23) {   /* Qwen3-VL vision layer (qwen/export_vit.py): LN(+bias) -> q,k,v (+bias, rope only, no norm) -> attention (hd128-padded heads, no mask) -> o (+bias) -> LN -> fc1 (+bias, tanh GELU) -> fc2 (+bias); mod[0..1151] = 1 (residual gate) */
    const float *gw = (const float *)(wts + QV_GAIN), *ones = mod;
    for (int cb = 0; cb < NCB; cb++) { LM[cb] = *(const HVX_Vector *)(gw + cb * 32); LN[cb] = *(const HVX_Vector *)(gw + 1152 + cb * 32); }
    pool_run(p_lnn, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = 0; e.cosT = cosT; e.sinT = sinT;
      gemm_run_geo(h16, ntok, DIM + 128, WB(QV_Q), 2048, q16, vt, &e, &gq);
      gemm_run_geo(h16, ntok, DIM + 128, WB(QV_K), 2048, k16, vt, &e, &gk);
      gemm_run_geo(h16, ntok, DIM + 128, WB(QV_V), 2048, v16, vt, 0, &gv); } TICK();
    attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 0, 0, 64, 64, 1, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = ones; const gemm_geo_t go = {64, 68, 0, Lo, 4, 4};
      gemm_run_geo(o16, ntok, 2048 + 128, WB(QV_O), DIM, t16, vt, &e, &go); } TICK();
    for (int cb = 0; cb < NCB; cb++) { LM[cb] = *(const HVX_Vector *)(gw + 2 * 1152 + cb * 32); LN[cb] = *(const HVX_Vector *)(gw + 3 * 1152 + cb * 32); }
    pool_run(p_lnn, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 2; e.t2 = g_t2; e.scr = g_scr; const gemm_geo_t gu = {NCB, NCB + 4, 0, Lu, 4, 4};
      gemm_run_geo(h16, ntok, DIM + 128, WB(QV_FC1), 4352, mid, vt, &e, &gu); } TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = ones; const gemm_geo_t gd = {136, 140, 0, Ld, 4, 4};
      gemm_run_geo(mid, ntok, 4352 + 128, WB(QV_FC2), DIM, t16, vt, &e, &gd); } TICK();
    goto done;
  }
  if (meta[M_MODE] == 30) {   /* final norm: h16 = rms(x) * w (pack = f32 weight vector) */
    k_rms_prep(LM, LN, (const float *)wts, g_zero, g_zero); pool_run(p_ln, &P); TICK(); goto done;
  }
  if (meta[M_MODE] == 11) { *rc = 111; gemm_dq_var(meta[M_STOP]); gemm_dq_tiles((uint8_t *)mid, wts, 16, meta[M_FMT]); goto done; }   /* dequant test: first 16 tiles -> mid */
  if (meta[M_MODE] == 10) {   /* Qwen3.5 FFN: x += down(silu(gate(rms(x)))*up(rms(x))); pack: [gate/up interleaved q4a nb 768 kb 128][down q8 nb 128 kb 384][post norm w f32 4096] */
    const size_t O_DN = 75497472u, O_G = O_DN + 56623104u;
    k_rms_prep(LM, LN, (const float *)(wts + O_G), g_zero, g_zero);
    pool_run(p_ln, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = meta[M_FMT] == 1 ? 0 : 6; e.tabs = g_tabs2; e.scr = g_scr; gemm_set_q8(2);
      gemm_run_geo(h16, ntok, 4096, WB(0), 24576, mid, vt, &e, 0); gemm_set_q8(1); } TICK();
    { gemm_epi_t e1; memset(&e1, 0, sizeof e1); e1.mode = 4; e1.x32 = x; e1.gate = (const float *)(mod + 2 * 4096); e1.prev = t16;
      const gemm_geo_t d0 = {384, 384, 0, 0, 0, 0}, d1 = {384, 384, 192, 0, 0, 0};
      gemm_run_geo(mid, ntok, 6144, WB(O_DN), 4096, t16, vt, 0, &d0);
      gemm_run_geo(mid, ntok, 6144, WB(O_DN), 4096, t16, vt, &e1, &d1); } TICK();
    goto done;
  }
  if (meta[M_MODE] == 2) {   /* text projection: h16 = text tiles [slab][128 cb]; weights = pack at offset 0 [96 nb][132 kb] */
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = gt_a;
    const gemm_geo_t gp = {4096 / 32, 4096 / 32 + 4, 0, Lq, 4, 4};
    gemm_run_geo(h16, ntok, 4096 + 128, WB(0), DIM, t16, vt, &e, &gp); TICK(); goto done;
  }
  /* ---- attention half ---- */
  k_rms_prep(LM, LN, gain + 256, sh_a, sc_a);
  pool_run(p_ln, &P); TICK();
  const int nolora = meta[M_MODE] == 1;
  if (!nolora) gemm_run(h16, ntok, DIM, WB(W_AQKV), 384, Lq, vt); TICK();
  { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = gain; e.cosT = cosT; e.sinT = sinT;
    gemm_run_geo(h16, ntok, DIM + 128, WB(W_Q), DIM, q16, vt, &e, &gq);
    e.gain = gain + 128; gemm_run_geo(h16, ntok, DIM + 128, WB(W_K), DIM, k16, vt, &e, &gk);
    gemm_run_geo(h16, ntok, DIM + 128, WB(W_V), DIM, v16, vt, 0, &gv); } TICK();
  attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 0, 0, NCB, NCB, 1, vt); TICK();
  { if (!nolora) gemm_run(o16, ntok, DIM, WB(W_AO), 128, Lo, vt);
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = gt_a;
    const gemm_geo_t go = {NCB, NCB + 4, 0, Lo, 4, 4};
    gemm_run_geo(o16, ntok, DIM + 128, WB(W_O), DIM, t16, vt, &e, &go); } TICK();
  /* ---- FFN half ---- */
  k_rms_prep(LM, LN, gain + 256 + DIM, sh_f, sc_f);
  pool_run(p_ln, &P); TICK();
  { if (!nolora) gemm_run(h16, ntok, DIM, WB(W_AUG), 128, Lu, vt);
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 5; e.tabs = g_tabs; e.scr = g_scr;
    const gemm_geo_t gu = {NCB, NCB + 4, 0, Lu, 4, 4};
    gemm_run_geo(h16, ntok, DIM + 128, WB(W_UG), 16384, mid, vt, &e, &gu); } TICK();
  { if (!nolora) gemm_run(mid, ntok, 8192, WB(W_ADN), 128, Ld, vt);
    const gemm_geo_t g0 = {256, 260, 0, 0, 0, 0}, g1 = {256, 260, 128, Ld, 4, 4};
    gemm_run_geo(mid, ntok, 4096, WB(W_DN), DIM, t16, vt, 0, &g0);
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = gt_f; e.prev = t16;
    gemm_run_geo(mid, ntok, 4096 + 128, WB(W_DN), DIM, t16, vt, &e, &g1); } TICK();
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
  g_ncb = meta[M_NCB] ? meta[M_NCB] : 96;
  if (HAP_mmap_get(bfd, &bp, &bpa) || !bp) { *rc = -7; return 0; }
  g_blob = (uint8_t *)bp;
  const size_t xb = (size_t)meta[M_NTOK] * DIM * 4;
  if (meta[M_FIRST]) { cache_sync(g_blob + meta[M_X], xb); cache_sync(g_blob + meta[M_COS], (size_t)meta[M_NTOK] / 32 * 8192); cache_sync(g_blob + meta[M_SIN], (size_t)meta[M_NTOK] / 32 * 8192); }
  if (meta[M_NPRE]) { const size_t own = (size_t)meta[M_NTOK] * DIM * 2, pre = (size_t)meta[M_NPRE] * 32 * DIM * 2; cache_sync(g_blob + meta[M_K] + own, pre); cache_sync(g_blob + meta[M_V] + own, pre); }
  int r = mega_block_impl(hd, meta, nmeta, wfd, mod, modlen, times, ntimes, rc);
  if (meta[M_REL]) cache_sync(g_blob + meta[M_X], xb);
  HAP_mmap_put(bfd);
  if (*rc != -5) HAP_mmap_put(wfd);
  return r;
}
