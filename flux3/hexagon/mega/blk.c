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

#ifdef SIM_NOFD   /* hexagon-sim harness (sim.sh): no FastRPC fds, the "fd" is an index into g_sim_ptr (1 = weights, 2 = blob) */
void *g_sim_ptr[4];
static int sim_mmap_get(int fd, void **p, unsigned long long *pa) { *p = g_sim_ptr[fd]; *pa = 0; return 0; }
#define HAP_mmap_get(fd, p, pa) sim_mmap_get((fd), (void **)(p), (unsigned long long *)(pa))
#define HAP_mmap_put(fd) 0
#endif


typedef uint16_t u16;
#define T16 1024          /* halfwords per h16 tile */
#define VT_TAB (7u << 20) /* tables at the top 1 MB of VTCM */

static u16 *g_et, *g_tabs, *g_tabs2, *g_t2, *g_t2s; static HVX_Vector *g_scr;   /* tables (exp, 3 SiTU) + per-worker gather scratch */

static void vcp(void *d, const void *s, size_t bytes) {
  HVX_Vector *dv = (HVX_Vector *)d; const HVX_Vector *sv = (const HVX_Vector *)s; size_t n = bytes / 128;
  for (size_t i = 0; i < n; i++) { if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(sv + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32)); dv[i] = sv[i]; }
}


/* ---------------- simple parallel kernels ---------------- */
typedef struct { const float *x; u16 *h; const HVX_Vector *M, *N; int ns; const HVX_Vector *M1, *N1; int ns0; float eps; const HVX_Vector *M2, *N2; int ns2, ns3; } par_t;
static void p_ln(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_rmsmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, p->M, p->N, 1e-6f); }

typedef struct { float *x; const u16 *t; const float *w; float sc; int ns; } pn_t;
static void p_pn(void *a, int i, int n) { pn_t *p = (pn_t *)a; for (int s = i; s < p->ns; s += n) k_postnorm_slab(p->x + (size_t)s * NCB * 1024, p->t + (size_t)s * NCB * T16, p->w, p->sc); }
typedef struct { u16 *q, *k; const float *qg, *kg, *cs, *sn; int ns, nct, hdt; size_t tbl; } qn_t;
static void p_qn(void *a, int i, int n) { qn_t *p = (qn_t *)a; for (int s = i; s < p->ns; s += n) {
  k_qknorm_rope_slab(p->q + (size_t)s * p->nct * T16, p->nct, p->hdt, p->qg, p->cs + s * p->tbl, p->sn + s * p->tbl, 1);
  k_qknorm_rope_slab(p->k + (size_t)s * p->nct * T16, p->nct, p->hdt, p->kg, p->cs + s * p->tbl, p->sn + s * p->tbl, 1); } }
typedef struct { u16 *x; const float *w, *cs, *sn; int ns, nct, hdt, rope; size_t tbl; } qn1_t;   /* q or k alone (LTX DiT attention ops) */
static void p_qn1(void *a, int i, int n) { qn1_t *p = (qn1_t *)a; for (int s = i; s < p->ns; s += n) k_qknorm_rope_slab(p->x + (size_t)s * p->nct * T16, p->nct, p->hdt, p->w, p->rope ? p->cs + s * p->tbl : 0, p->rope ? p->sn + s * p->tbl : 0, p->rope); }
typedef struct { u16 *o; const u16 *gl; int ns, nct, hdt, nh; } gt_t;
static void p_gt(void *a, int i, int n) { gt_t *p = (gt_t *)a; for (int s = i; s < p->ns; s += n) k_gate_slab(p->o + (size_t)s * p->nct * T16, p->gl + (size_t)s * 2 * T16, p->nh, p->hdt); }
/* H3: slabs < ns0 (image rows) use M/N, the others (text rows) M1/N1; RMSNorm eps from P.eps */
static void p_ln2(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) { const int r = s >= p->ns2 && s < p->ns3, k = s >= p->ns0; const HVX_Vector *Mv = r ? p->M2 : k ? p->M1 : p->M, *Nv = r ? p->N2 : k ? p->N1 : p->N; k_rmsmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, Mv, Nv, p->eps); } }
/* ConvRot: O = A * blockdiag(Hn) per 256-column block (8 tiles) on HMX: Hn tiles (64 fp16 tiles, [out tile u][in tile t]) resident at the start of VTCM. A, O: [slab][nct tiles] (nct % 8 == 0) */
typedef struct { const u16 *src; u16 *dst; size_t rowB; int n; } cp_t;
static void p_cprows(void *a, int i, int n) { cp_t *c = (cp_t *)a; for (int r = i; r < c->n; r += n) vcp((uint8_t *)c->dst + (size_t)r * c->rowB, (const uint8_t *)c->src + (size_t)r * c->rowB, c->rowB); }
static void rot_run(const u16 *A, u16 *O, int ns, int nct, const u16 *hn_dram, uint8_t *vt) {
  hmx_clr();
  u16 *hn = (u16 *)vt; vcp(hn, hn_dram, 131072);
  const size_t rowB = (size_t)nct * 2048; int G = (int)((5u << 20) / (2 * rowB)); if (G < 1) G = 1; if (G > ns) G = ns;
  uint8_t *ain = vt + 131072, *aout = ain + (size_t)G * rowB;
  for (int g0 = 0; g0 < ns; g0 += G) {
    const int g = ns - g0 < G ? ns - g0 : G;
    cp_t ci = {A + (size_t)g0 * nct * T16, (u16 *)ain, rowB, g}; pool_run(p_cprows, &ci);
    for (int r = 0; r < g; r++)
      for (int b = 0; b < nct / 8; b++)
        for (int u = 0; u < 8; u++) {
          hmx_acc(ain + (size_t)r * rowB + (size_t)b * 8 * 2048, (8 << 11) | 0x7C, (uint8_t *)hn + (size_t)u * 8 * 2048, (8 << 11) - 1);
          hmx_store(aout + (size_t)r * rowB + ((size_t)b * 8 + u) * 2048, 0);
        }
    cp_t co = {(const u16 *)aout, O + (size_t)g0 * nct * T16, rowB, g}; pool_run(p_cprows, &co);
  }
}
/* LayerNorm (with mean) with two row kinds: slabs < ns0 use M/N, the others M1/N1 */
static void p_lnn2(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) { const int k = s >= p->ns0; k_lnmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, k ? p->M1 : p->M, k ? p->N1 : p->N, p->eps); } }
static void p_lnn(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_lnmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, p->M, p->N, 1e-6f); }

/* ---------------- attention (joint, NHEAD heads, last key tile masked) ---------------- */
static int g_qcb, g_kcb, g_grp, g_hdt = 4;   /* g_hdt: tiles per head (4 = head_dim 128; Gemma 4: 8 and 16) */   /* tiles per slab of q/o, of k/v, and query heads per kv head (DiT: NCB, NCB, 1; padded ViT: 64, 64, 1; Qwen3-VL text encoder: 128, 32, 4) */
typedef struct { const u16 *k, *v, *q; u16 *o; int nkb, h, nq; u16 *KT, *Vw, *Qh; } ah_t;
static void p_attn_setup(void *a, int idx, int n) {   /* build K^T and V(+ones) weight tiles for head h, copy Q tiles */
  ah_t *p = (ah_t *)a; const int hdt = g_hdt;
  for (int kb = idx; kb < p->nkb; kb += n)
    for (int j = 0; j < hdt; j++) {
      const u16 *ks = p->k + ((size_t)kb * g_kcb + (p->h / g_grp) * hdt + j) * T16; u16 *kd = p->KT + ((size_t)kb * hdt + j) * T16;
      k_transpose_tile(ks, kd);
      vcp(p->Vw + ((size_t)j * p->nkb + kb) * T16, p->v + ((size_t)kb * g_kcb + (p->h / g_grp) * hdt + j) * T16, 2048);
    }
  for (int s = idx; s < p->nq; s += n) vcp(p->Qh + (size_t)s * hdt * T16, p->q + ((size_t)s * g_qcb + p->h * hdt) * T16, hdt * 2048);
}
static void p_attn_flush(void *a, int idx, int n) { ah_t *p = (ah_t *)a; const int hdt = g_hdt; for (int s = idx; s < p->nq; s += n) vcp(p->o + ((size_t)s * g_qcb + p->h * hdt) * T16, p->Qh + (size_t)s * hdt * T16, hdt * 2048); }
typedef struct { const u16 *S; u16 *P; int nkb; const HVX_Vector *km; int kst; int np; const int *pk; const int *pmi; const HVX_Vector *pm; } sm_t;
static void p_softmax(void *a, int i, int n) { sm_t *p = (sm_t *)a;
  if (p->pm) k_softmax2pm(p->S, p->P, p->nkb, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, p->pk, p->pmi, p->pm, p->np);
  else k_softmax2p(p->S, p->P, p->nkb, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, p->km, p->kst); }

static uint64_t g_acc[24];
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
/* ---------------- chunked attention (no limit on the number of tokens) ----------------
   (K stays in DDR as it is; each chunk is transposed on its way into VTCM.) The plain attn_run keeps K^T, V and Q of ALL tokens of one head (about 830 B per token for 128-wide heads) and a full score row in the 8 MB VTCM: ~6400 tokens at most.
   This version works on a block of QB query slabs at a time and streams the keys through VTCM in chunks of C key blocks (from DDR, where K was transposed in place once):
   pass 1: row max over all key chunks; pass 2: P = exp(S - max) per chunk, P*V (+ ones column = row sums) per chunk added into fp32 accumulators; the end divides by the sums.
   Same result as attn_run up to the summation order. Non-causal only (causal == 0). g_force_c / g_force_q: test overrides of C / QB (0 = automatic). */
static int g_force_c = 0, g_force_q = 0;
typedef struct { const u16 *k, *v, *q; u16 *o; u16 *KTc, *Vc, *Qb; int kb0, nkc, C, h, withV, s0, nqb; } cl_t;
static void p_cl_load(void *a, int idx, int n) {
  cl_t *p = (cl_t *)a; const int hdt = g_hdt, hk = p->h / g_grp;
  for (int i = idx; i < p->nkc; i += n) {
    const int kb = p->kb0 + i;
    for (int j = 0; j < hdt; j++) k_transpose_tile(p->k + ((size_t)kb * g_kcb + hk * hdt + j) * T16, p->KTc + ((size_t)i * hdt + j) * T16);   /* K^T into VTCM; K itself stays untouched (the next window / head reads it again) */
    if (p->withV) for (int j = 0; j < hdt; j++) vcp(p->Vc + ((size_t)j * p->C + i) * T16, p->v + ((size_t)kb * g_kcb + hk * hdt + j) * T16, 2048);
  }
}
static void p_cq_load(void *a, int idx, int n) { cl_t *p = (cl_t *)a; const int hdt = g_hdt; for (int s = idx; s < p->nqb; s += n) vcp(p->Qb + (size_t)s * hdt * T16, p->q + ((size_t)(p->s0 + s) * g_qcb + p->h * hdt) * T16, hdt * 2048); }
static void p_cq_flush(void *a, int idx, int n) { cl_t *p = (cl_t *)a; const int hdt = g_hdt; for (int s = idx; s < p->nqb; s += n) vcp(p->o + ((size_t)(p->s0 + s) * g_qcb + p->h * hdt) * T16, p->Qb + (size_t)s * hdt * T16, hdt * 2048); }
typedef struct { u16 *k; int kcb; size_t ntile; } kt_t;
static void p_kt_all(void *a, int idx, int n) { kt_t *p = (kt_t *)a; for (size_t t = idx; t < p->ntile; t += n) k_transpose_tile(p->k + t * T16, p->k + t * T16); }
typedef struct { const u16 *S; u16 *P; int nkc; const HVX_Vector *km; HVX_Vector *mx; } rm_t;
static void p_rowmax(void *a, int i, int n) { rm_t *p = (rm_t *)a; k_rowmax_chunk(p->S, p->nkc, i * 16 / n, (i + 1) * 16 / n, p->km, p->mx); }
static void p_smfix(void *a, int i, int n) { rm_t *p = (rm_t *)a; k_softmax_fixed(p->S, p->P, p->nkc, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, p->km, p->mx); }
static size_t attn_plain_bytes(int nq, int nkb, int hdt) { return ((size_t)(2 * hdt + 1) * nkb + (size_t)nq * hdt + 4 * (size_t)nkb + 5) * 2048; }
static void attn_chunked(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, int nvalid, int qcb, int kcb, int grp, uint8_t *vt) {
  const int nkb = nkv / 32, hdt = g_hdt; g_qcb = qcb; g_kcb = kcb; g_grp = grp;
  const int nv = nvalid - (nkb - 1) * 32;   /* valid keys in the last tile */
  const HVX_Vector *kmLast = 0; if (nv < 32) { for (int l = 0; l < 64; l++) g_km[l] = (l >> 1) >= nv ? 0xFFFF : 0; kmLast = (const HVX_Vector *)g_km; }
  const size_t BUDGET = 6815744;   /* 6.5 MB: the tables sit at 7 MB */
  int C = nkb < 128 ? nkb : 128; if (g_force_c > 0 && g_force_c < C) C = g_force_c;
  while (C > 8 && (size_t)C * (4 + hdt + hdt + 1) * 2048 > BUDGET / 2) C = (C + 1) / 2;
  const size_t chunkB = (size_t)C * (4 + hdt + hdt + 1) * 2048 + (size_t)(hdt + 1) * 2048, perSlab = (size_t)hdt * 2048 + (size_t)(hdt + 1) * 4096 + 2048;
  int QB = (int)((BUDGET - chunkB) / perSlab); if (QB > 64) QB = 64; if (QB > nq) QB = nq; if (g_force_q > 0 && g_force_q < QB) QB = g_force_q; if (QB < 1) QB = 1;
  u16 *KTc = (u16 *)vt, *Vc = KTc + (size_t)C * hdt * T16, *Qb = Vc + (size_t)C * (hdt + 1) * T16, *Sb[2], *Pb[2];
  Sb[0] = Qb + (size_t)QB * hdt * T16; Sb[1] = Sb[0] + (size_t)C * T16; Pb[0] = Sb[1] + (size_t)C * T16; Pb[1] = Pb[0] + (size_t)C * T16;
  u16 *Ob = Pb[1] + (size_t)C * T16; HVX_Vector *Oacc = (HVX_Vector *)(((uintptr_t)(Ob + (size_t)(hdt + 1) * T16) + 127) & ~(uintptr_t)127); HVX_Vector *MX = Oacc + (size_t)QB * (hdt + 1) * 32;
  for (size_t i = 0; i < (size_t)C * T16; i++) Vc[(size_t)hdt * C * T16 + i] = 0x3C00;   /* ones column (row sums through P*V) */
  const HVX_Vector neginf = Q6_Vh_vsplat_R(0xFBFF);
  for (int h = 0; h < qcb / hdt; h++) {
    for (int s0 = 0; s0 < nq; s0 += QB) {
      const int nqb = nq - s0 < QB ? nq - s0 : QB;
      cl_t cl = {k, v, q, o, KTc, Vc, Qb, 0, 0, C, h, 0, s0, nqb};
      pool_run(p_cq_load, &cl);
      for (int i = 0; i < nqb * 16; i++) MX[i] = neginf;
      for (int pass = 0; pass < 2; pass++) {
        for (int c0 = 0; c0 < nkb; c0 += C) {
          const int nkc = nkb - c0 < C ? nkb - c0 : C; const int last = c0 + nkc == nkb;
          cl.kb0 = c0; cl.nkc = nkc; cl.withV = pass; pool_run(p_cl_load, &cl);
          for (int s = 0; s <= nqb; s++) {
            const int b = s & 1; rm_t rm = {Sb[b], Pb[b], nkc, last ? kmLast : 0, MX + (size_t)s * 16};
            rm_t pr = {Sb[(s - 1) & 1], 0, nkc, last ? kmLast : 0, MX + (size_t)(s > 0 ? s - 1 : 0) * 16};
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
static void attn_run(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, int nvalid, int causal, const int *bl, int qcb, int kcb, int grp, uint8_t *vt) {
  if (causal == 0 && (g_force_c > 0 || attn_plain_bytes(nq, nkv / 32, g_hdt) > (7u << 20) - 65536)) { attn_chunked(q, k, v, o, nq, nkv, nvalid, qcb, kcb, grp, vt); return; }   /* the keys do not fit in VTCM (or a test forces it) */
  const int nkb = nkv / 32; g_qcb = qcb; g_kcb = kcb; g_grp = grp; const int hdt = g_hdt;
  const int nv = nvalid - (nkb - 1) * 32;   /* valid keys in the last tile */
  const HVX_Vector *km = 0;
  if (causal == 1) for (int p = 0; p < 16; p++) for (int l = 0; l < 64; l++) g_cm[p * 64 + l] = (l >> 1) > 2 * p + (l & 1) ? 0xFFFF : 0;
  if (!causal && nv < 32) { for (int l = 0; l < 64; l++) g_km[l] = (l >> 1) >= nv ? 0xFFFF : 0; km = (const HVX_Vector *)g_km; }
  u16 *KT = (u16 *)vt, *Vw = KT + (size_t)hdt * nkb * T16, *Qh = Vw + (size_t)(hdt + 1) * nkb * T16;           /* Qh also receives O */
  u16 *Sb[2], *Pb[2]; Sb[0] = Qh + (size_t)nq * hdt * T16; Sb[1] = Sb[0] + (size_t)nkb * T16; Pb[0] = Sb[1] + (size_t)nkb * T16; Pb[1] = Pb[0] + (size_t)nkb * T16;
  u16 *Ob = Pb[1] + (size_t)nkb * T16;                                                                  /* 5 tiles: 4 + sum */
  for (size_t i = 0; i < (size_t)nkb * T16; i++) Vw[(size_t)hdt * nkb * T16 + i] = 0x3C00;               /* ones n-block */
  for (int h = 0; h < qcb / hdt; h++) {
    ah_t a = {k, v, q, o, nkb, h, nq, KT, Vw, Qh};
    uint64_t c0 = PC(); pool_run(p_attn_setup, &a); g_acc[0] += PC() - c0;
    for (int s = 0; s <= nq; s++) {
      const int b = s & 1;
      uint64_t c1 = PC();
      const int nks = nks_of(s, nkb, causal, bl);   /* key tiles visible to slab s */
      if (s < nq) {
        for (int nb = 0; nb < nks; nb++) {
          hmx_acc((uint8_t *)(Qh + (size_t)s * hdt * T16), (hdt << 11) | 0x7C, (uint8_t *)(KT + (size_t)nb * hdt * T16), (hdt << 11) - 1);
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
        for (int j = 0; j < hdt + 1; j++) {
          for (int k0 = 0; k0 < nkp; k0 += 32) { int c = nkp - k0 < 32 ? nkp - k0 : 32; hmx_acc((uint8_t *)(P + (size_t)k0 * T16), (c << 11) | 0x7C, (uint8_t *)(Vw + ((size_t)j * nkb + k0) * T16), (c << 11) - 1); }
          hmx_store((uint8_t *)(Ob + (size_t)j * T16), 0);
        }
        uint64_t c3 = PC(); g_acc[2] += c3 - c2;
        k_rowscale_sum(Ob, hdt, Ob + (size_t)hdt * T16);
        g_acc[3] += PC() - c3;
        vcp(Qh + (size_t)(s - 1) * hdt * T16, Ob, hdt * 2048);   /* O replaces the (already consumed) Q slab */
      }
      uint64_t c4 = PC(); if (s < nq) pool_wait(); g_acc[4] += PC() - c4;
    }
    uint64_t c5 = PC(); pool_run(p_attn_flush, &a); g_acc[5] += PC() - c5;
  }
}

/* ---------------- block ---------------- */
static uint32_t us_now(void) { return (uint32_t)HAP_perf_get_time_us(); }

static uint32_t g_ctx; static uint8_t *g_vt; static int g_init; static u16 *g_et_h, *g_tabs_h, *g_tabs2_h, *g_t2_h, *g_tabs3_h, *g_t2s_h;
void mega_blk_release(void) { if (!g_init) return; pool_quit(); if (g_vt) HAP_compute_res_release(g_ctx); g_vt = 0; g_init = 0; }
static int vtcm_get(void) {
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  g_ctx = HAP_compute_res_acquire(&attr, 5000000); if (!g_ctx) return -1;
  g_vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr); hmx_bias_setup(g_vt + (8u << 20) - 4096);
  g_et = (u16 *)(g_vt + VT_TAB); g_tabs = g_et + 32768; g_tabs2 = g_tabs + 3 * 32768; g_scr = (HVX_Vector *)(g_vt + VT_TAB + 393216); g_t2 = (u16 *)(g_vt + VT_TAB + 458752); g_t2s = (u16 *)(g_vt + VT_TAB + 524288);
  memcpy(g_et, g_et_h, 65536); memcpy(g_tabs, g_tabs_h, 3 * 65536); memcpy(g_tabs2, g_tabs2_h, 2 * 65536); memcpy(g_t2, g_t2_h, 65536); memcpy(g_t2s, g_t2s_h, 65536);
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
  g_force_c = meta[M_A10]; g_force_q = meta[M_A11];   /* tests: force the chunked attention with C key blocks / QB query slabs */
  const uint8_t *wts = (const uint8_t *)wp + meta[M_WOFF];
  if (!g_init) {
    g_et_h = (u16 *)memalign(128, 65536); g_tabs_h = (u16 *)memalign(128, 3 * 65536);
    { u16 *t2 = (u16 *)memalign(128, 65536); tab_init(g_et_h, t2); free(t2); }
    tab_init_situ(g_tabs_h);
    g_tabs2_h = (u16 *)memalign(128, 2 * 65536); tab_init_swiglu(g_tabs2_h);
    g_t2_h = (u16 *)memalign(128, 65536); tab_init_gelut(g_t2_h);
    g_t2s_h = (u16 *)memalign(128, 65536); tab_init_gelut_s(g_t2s_h, 16.0f);   /* Mage: GELU(16 x) / 16 */
    g_tabs3_h = (u16 *)memalign(128, 2 * 65536); tab_init_gelumul(g_tabs3_h);
    g_init = 1;
  }
  gemm_set_q8(1);
  if (!pool_n()) pool_init(meta[M_NW] ? meta[M_NW] : 3);
  if (!g_vt && vtcm_get()) { *rc = -1; return 0; }
  uint32_t ctx = g_ctx; uint8_t *vt = g_vt;
  if (meta[M_FIRST] == 1) {   /* M_FIRST 2 (windowed runs): only the cache sync of x / rope tables in mega_block, no power request */ HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)hd, &rq); power_max((void *)hd); }
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
  static float g_zero[8192] __attribute__((aligned(128)));
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
        for (int i2 = 0; i2 < 128; i2++) g_qg[i2] = vec[2 * DIM + i2] * 0.08838834764831845f;   /* attention scale 1/sqrt(128) folded into q like the DiT pack */
        gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.cosT = cosT; e.sinT = sinT; e.gain = g_qg;
        gemm_set_q8(H->fmt[0]); gemm_run_geo(h16, ntok, DIM, WB(H->off[0]), 4096, q16, vt, &e, 0);
        const uint8_t *kvp = (const uint8_t *)WB(H->off[1]);   /* kv pack = [64 n-blocks][128 kb] records: k = first 32 n-blocks, v = last 32 */
        e.gain = vec + 2 * DIM + 128; gemm_set_q8(H->fmt[1]); gemm_run_geo(h16, ntok, DIM, (const u16 *)kvp, 1024, k16, vt, &e, 0);
        gemm_run_geo(h16, ntok, DIM, (const u16 *)(kvp + (size_t)32 * H->kb[1] * (H->fmt[1] == 2 ? 768 : 1152)), 1024, v16, vt, 0, 0); TICK();
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
      gemm_set_q8(H->fmt[2]); gemm_run_geo(o16, ntok, 4096, WB(H->off[2]), DIM, t16, vt, &eo, 0); TICK();
    }
    /* ffn */
    k_rms_prep(LM, LN, vec + DIM, g_zero, g_zero); pool_run(p_ln, &P);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr; gemm_set_q8(H->fmt[igu]);
      gemm_run_geo(h16, ntok, DIM, WB(H->off[igu]), H->nb[igu] * 32, mid, vt, &e, 0); }
    { gemm_epi_t e1; memset(&e1, 0, sizeof e1); e1.mode = 4; e1.x32 = x; e1.gate = ones; e1.prev = t16; gemm_set_q8(H->fmt[idn]);
      const int fk = H->kb[idn], hk = fk / 2; const gemm_geo_t d0 = {fk, fk, 0, 0, 0, 0}, d1 = {fk, fk, hk, 0, 0, 0};
      gemm_run_geo(mid, ntok, hk * 32, WB(H->off[idn]), DIM, t16, vt, 0, &d0);
      gemm_run_geo(mid, ntok, (fk - hk) * 32, WB(H->off[idn]), DIM, t16, vt, &e1, &d1); } TICK();
    gemm_set_q8(1); goto done;
  }
  if (meta[M_MODE] == 51) {   /* MiniMax-H3 text-encoder layer (Qwen3-VL text, hidden 5120 = 160 tiles, 64 q heads / 8 kv heads x 128, ffn 25600), causal, ConvRot int8; x += attn(rms(x)); x += ffn(rms(x)) */
    const u16 *hnd = (const u16 *)BUF(M_HN); u16 *hr = (u16 *)BUF(M_HR); const float *gn = (const float *)(wts + TE_GAIN);
    gemm_set_q8(3);
    static float g_ones[8192] __attribute__((aligned(128))); static int ones_init; if (!ones_init) { for (int i = 0; i < 8192; i++) g_ones[i] = 1.0f; ones_init = 1; }
    k_rms_prep(LM, LN, gn + 256, g_zero, g_zero); pool_run(p_ln, &P); TICK();
    rot_run(h16, hr, ns, 160, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = gn; e.cosT = cosT; e.sinT = sinT;
      gemm_run_geo(hr, ntok, 5120, WB(TE_Q), 8192, q16, vt, &e, 0);
      e.gain = gn + 128; gemm_run_geo(hr, ntok, 5120, WB(TE_K), 1024, k16, vt, &e, 0);
      gemm_run_geo(hr, ntok, 5120, WB(TE_V), 1024, v16, vt, 0, 0); } TICK();
    attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 1, 0, 256, 32, 8, vt); TICK();
    rot_run(o16, hr, ns, 256, hnd, vt);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = g_ones;
      gemm_run_geo(hr, ntok, 8192, WB(TE_O), 5120, t16, vt, &e, 0); } TICK();
    k_rms_prep(LM, LN, gn + 256 + 5120, g_zero, g_zero); pool_run(p_ln, &P);
    rot_run(h16, hr, ns, 160, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr;
      gemm_run_geo(hr, ntok, 5120, WB(TE_GU), 51200, mid, vt, &e, 0); } TICK();
    rot_run(mid, hr, ns, 800, hnd, vt);
    { const gemm_geo_t d0 = {800, 800, 0, 0, 0, 0}, d1 = {800, 800, 400, 0, 0, 0};
      gemm_run_geo(hr, ntok, 12800, WB(TE_DN), 5120, t16, vt, 0, &d0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = g_ones; e.prev = t16;
      gemm_run_geo(hr, ntok, 12800, WB(TE_DN), 5120, t16, vt, &e, &d1); } TICK();
    gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 80) {   /* LTX-2.5 Gemma 4 text-encoder layer (anima/ltx, hidden 3840 = 120 tiles, ffn 15360, 16 q heads), causal, ConvRot int8. M_FMT: 0 sliding layer (head_dim 256, 8 kv heads), 1 full layer (head_dim 512, 1 kv head, v = raw k_proj output).
       x += post_attn_norm(attn(input_norm(x))); x += post_ffn_norm(mlp(pre_ffn_norm(x))); x *= layer_scalar (sandwich norms; q/k RMSNorm with weight, v RMSNorm without weight, no 1/sqrt(d); mlp = down(gelu_tanh(gate) * up)) */
    const int full = meta[M_FMT] == 1, hdt = full ? 16 : 8, QN = 512 * hdt, KN = full ? 512 : 2048, HD = 32 * hdt;
#define NBS(K) (128u + (size_t)((K) / 32) * 1024u)
    const size_t oQ = 0, oK = oQ + (size_t)(QN / 32) * NBS(3840), oV = oK + (size_t)(KN / 32) * NBS(3840), oO = oV + (full ? 0 : (size_t)(KN / 32) * NBS(3840)),
                 oGU = oO + (size_t)120 * NBS(QN), oDN = oGU + (size_t)960 * NBS(3840), oG = oDN + (size_t)120 * NBS(15360);
    const float *gn = (const float *)(wts + oG), *gq = gn + 4 * 3840, *gk = gq + HD, *gs = gk + HD;   /* f32: input_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln [3840 each], q_norm [HD], k_norm [HD], layer_scalar x 32 */
    static float g_one512[512] __attribute__((aligned(128))); static int one_init; if (!one_init) { for (int i = 0; i < 512; i++) g_one512[i] = 1.0f; one_init = 1; }
    const u16 *hnd = (const u16 *)BUF(M_HN); u16 *hr = (u16 *)BUF(M_HR);
    g_hdt = hdt; gemm_set_q8(3); memcpy(g_tabs2, g_tabs3_h, 2 * 65536);
    k_rms_prep(LM, LN, gn, g_zero, g_zero); pool_run(p_ln, &P); TICK();
    rot_run(h16, hr, ns, 120, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 7; e.hdt = hdt; e.rope = 1; e.cosT = cosT; e.sinT = sinT;
      e.gain = gq; gemm_run_geo(hr, ntok, 3840, WB(oQ), QN, q16, vt, &e, 0);
      e.gain = gk; gemm_run_geo(hr, ntok, 3840, WB(oK), KN, k16, vt, &e, 0);
      e.gain = g_one512; e.rope = 0; gemm_run_geo(hr, ntok, 3840, WB(full ? oK : oV), KN, v16, vt, &e, 0); } TICK();
    attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 1, 0, QN / 32, KN / 32, full ? 16 : 2, vt); TICK();
    rot_run(o16, hr, ns, QN / 32, hnd, vt);
    gemm_run_geo(hr, ntok, QN, WB(oO), 3840, t16, vt, 0, 0); TICK();
    { pn_t pn = {x, t16, gn + 3840, 1.0f, ns}; pool_run(p_pn, &pn); } TICK();
    k_rms_prep(LM, LN, gn + 2 * 3840, g_zero, g_zero); pool_run(p_ln, &P);
    rot_run(h16, hr, ns, 120, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr;
      gemm_run_geo(hr, ntok, 3840, WB(oGU), 30720, mid, vt, &e, 0); } TICK();
    rot_run(mid, hr, ns, 480, hnd, vt);
    gemm_run_geo(hr, ntok, 15360, WB(oDN), 3840, t16, vt, 0, 0); TICK();
    { pn_t pn = {x, t16, gn + 3 * 3840, gs[0], ns}; pool_run(p_pn, &pn); } TICK();
    memcpy(g_tabs2, g_tabs2_h, 2 * 65536); g_hdt = 4; gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 91) {   /* LTX DiT op: modulated RMSNorm h = rms(x) * M + N (M = 1 + scale, N = shift; mod = [M (D) | N (D)] f32) written to the fp16 tiles at meta[M_A0] */
    const int nct = meta[M_NCB];
    for (int cb = 0; cb < nct; cb++) { LM[cb] = *(const HVX_Vector *)(mod + cb * 32); LN[cb] = *(const HVX_Vector *)(mod + nct * 32 + cb * 32); }
    P.h = (u16 *)BUF(M_A0);
    if (meta[M_A8] > 0) {   /* image-to-video: the first M_A8 slabs (latent frame 0, timestep 0) use the first [M|N] set, the others a second set that follows it */
      static HVX_Vector LM1[MAXNCB] __attribute__((aligned(128))), LN1[MAXNCB] __attribute__((aligned(128)));
      for (int cb = 0; cb < nct; cb++) { LM1[cb] = *(const HVX_Vector *)(mod + 2 * nct * 32 + cb * 32); LN1[cb] = *(const HVX_Vector *)(mod + 3 * nct * 32 + cb * 32); }
      P.M1 = LM1; P.N1 = LN1; P.ns0 = meta[M_A8]; P.eps = 1e-6f; pool_run(p_ln2, &P);
    } else pool_run(p_ln, &P);
    TICK();
    goto done;
  }
  if (meta[M_MODE] == 90) {   /* LTX DiT attention op (any query/key widths; 32 heads): x += gate * to_out(softmax(norm(q) norm(k)^T) v * 2 sigmoid(gate logits)), no mask. Weights pack: Q [I x Dq] K [I x Dk] V [I x Dk] O [Dq x I] GATE [64 x Dq] (int8 without ConvRot),
       f32 q gain (1/sqrt(hd) folded) [I], k gain [I], fp16 bias vectors bq bk bv bo bg. meta: M_NCB Dq/32, M_NTOK query tokens, M_NVALID key tokens, M_A0 query input (fp16 tiles, un-rotated), M_A1 key input (== M_A0: self attention), M_A2 Dk/32, M_A3 I/32,
       M_A8 (modes 90-92): slabs [0, M_A8) use the first modulation set/gate, the others the second one that follows it in mod (0 = a single set), M_A4 flags (1 rope q (M_COS/M_SIN tables), 2 rope k (M_A5/M_A6 offsets), 4 gate logits present, 8 keys/values already in the K/V buffers (windowed queries: nk = all keys), 16 only project keys/values into the K/V buffers at slab offset M_A9), M_A7 offset of the rotated key input buffer; mod = residual gate vector [Dq] f32 */
    const int Dqt = meta[M_NCB], Dkt = meta[M_A2], It = meta[M_A3], fl = meta[M_A4], nk = meta[M_NVALID], hdt = It / 32, NH = 32, nqs = ns, nks = nk / 32, self = meta[M_A1] == meta[M_A0];
#define NBS(K) (128u + (size_t)((K) / 32) * 1024u)
#define BVS(n) ((size_t)((n) / 32) * 128u)
    const size_t oQ = 0, oK = oQ + (size_t)It * NBS(Dqt * 32), oV = oK + (size_t)It * NBS(Dkt * 32), oO = oV + (size_t)It * NBS(Dkt * 32), oG = oO + (size_t)Dqt * NBS(It * 32), oVec = oG + (size_t)2 * NBS(Dqt * 32);
    const float *qg = (const float *)(wts + oVec), *kg = qg + It * 32; const size_t oB = oVec + 8u * It * 32;
    const u16 *bq = (const u16 *)(wts + oB), *bk = (const u16 *)(wts + oB + BVS(It * 32)), *bv = (const u16 *)(wts + oB + 2 * BVS(It * 32)), *bo = (const u16 *)(wts + oB + 3 * BVS(It * 32)), *bg = (const u16 *)(wts + oB + 4 * BVS(It * 32));
    const u16 *qin = (const u16 *)BUF(M_A0), *kin = (const u16 *)BUF(M_A1); u16 *hr = (u16 *)BUF(M_HR), *hr2 = self ? hr : (u16 *)BUF(M_A7), *gl = (u16 *)BUF(M_LQ);
    const float *ck = (const float *)BUF(M_A5), *sk = (const float *)BUF(M_A6); const size_t tbl = (size_t)NH * (hdt / 2) * 1024;
    const u16 *hnd = (const u16 *)BUF(M_HN); g_hdt = hdt; gemm_set_q8(3);
    const int onlyKV = (fl & 16) != 0, haveKV = (fl & 8) != 0; u16 *k16w = k16 + (size_t)meta[M_A9] * It * T16, *v16w = v16 + (size_t)meta[M_A9] * It * T16;
    if (onlyKV) {   /* windowed runs: project the keys/values of nk tokens (input at M_A1, rotated into the M_A7 buffer) into the global K/V buffers at slab offset M_A9 */
      u16 *hrk = (u16 *)BUF(M_A7);
      rot_run(kin, hrk, nks, Dkt, hnd, vt); TICK();
      { gemm_epi_t e; memset(&e, 0, sizeof e); e.bias16 = bk; gemm_run_geo(hrk, nk, Dkt * 32, WB(oK), It * 32, k16w, vt, &e, 0); e.bias16 = bv; gemm_run_geo(hrk, nk, Dkt * 32, WB(oV), It * 32, v16w, vt, &e, 0); } TICK();
      { qn1_t b = {k16w, kg, ck, sk, nks, It, hdt, (fl >> 1) & 1, tbl}; pool_run(p_qn1, &b); } TICK();
      g_hdt = 4; gemm_set_q8(1);
      goto done;
    }
    rot_run(qin, hr, nqs, Dqt, hnd, vt); if (!self && !haveKV) rot_run(kin, hr2, nks, Dkt, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e);
      e.bias16 = bq; gemm_run_geo(hr, ntok, Dqt * 32, WB(oQ), It * 32, q16, vt, &e, 0);
      if (!haveKV) { e.bias16 = bk; gemm_run_geo(hr2, nk, Dkt * 32, WB(oK), It * 32, k16, vt, &e, 0);
      e.bias16 = bv; gemm_run_geo(hr2, nk, Dkt * 32, WB(oV), It * 32, v16, vt, &e, 0); }
      if (fl & 4) { e.bias16 = bg; gemm_run_geo(qin, ntok, Dqt * 32, WB(oG), 64, gl, vt, &e, 0); } } TICK();
    { qn1_t a = {q16, qg, cosT, sinT, nqs, It, hdt, fl & 1, tbl}; pool_run(p_qn1, &a); if (!haveKV) { qn1_t b = {k16, kg, ck, sk, nks, It, hdt, (fl >> 1) & 1, tbl}; pool_run(p_qn1, &b); } } TICK();
    attn_run(q16, k16, v16, o16, nqs, nk, nk, 0, 0, It, It, 1, vt); TICK();
    if (fl & 4) { gt_t gt = {o16, gl, nqs, It, hdt, NH}; pool_run(p_gt, &gt); } TICK();
    rot_run(o16, hr, nqs, It, hnd, vt);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = mod; if (meta[M_A8] > 0) { e.gate2 = mod + Dqt * 32; e.ns0 = meta[M_A8]; } e.bias16 = bo; gemm_run_geo(hr, ntok, It * 32, WB(oO), Dqt * 32, t16, vt, &e, 0); } TICK();
    g_hdt = 4; gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 92) {   /* LTX DiT op: feed-forward x += gate * W2 gelu_tanh(W1 h), no biases; h = fp16 tiles at meta[M_A0] (output of the modulated norm op), D = M_NCB*32, ffn = M_A1*32; pack: W1 [ffn x D] W2 [D x ffn]; mod = gate [D] f32 */
    const int Dt = meta[M_NCB], Ft = meta[M_A1], D = Dt * 32, FFN = Ft * 32;
#define NBS(K) (128u + (size_t)((K) / 32) * 1024u)
    const size_t oF1 = 0, oF2 = (size_t)Ft * NBS(D);
    const u16 *hnd = (const u16 *)BUF(M_HN), *hin = (const u16 *)BUF(M_A0); u16 *hr = (u16 *)BUF(M_HR); gemm_set_q8(3);
    rot_run(hin, hr, ns, Dt, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 2; e.t2 = g_t2; e.scr = g_scr; gemm_run_geo(hr, ntok, D, WB(oF1), FFN, mid, vt, &e, 0); } TICK();
    rot_run(mid, hr, ns, Ft, hnd, vt);
    { const gemm_geo_t d0 = {Ft, Ft, 0, 0, 0, 0}, d1 = {Ft, Ft, Ft / 2, 0, 0, 0};
      gemm_run_geo(hr, ntok, FFN / 2, WB(oF2), D, t16, vt, 0, &d0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = mod; if (meta[M_A8] > 0) { e.gate2 = mod + D; e.ns0 = meta[M_A8]; } e.prev = t16;
      gemm_run_geo(hr, ntok, FFN / 2, WB(oF2), D, t16, vt, &e, &d1); } TICK();
    gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 81) {   /* LTX-2.5 text-embeddings connector block (anima/ltx; video: D 4096, 32 heads x 128; audio (M_FMT 1): D 2048, 32 heads x 64; ffn 4D), ConvRot int8, NO mask (all rows attend to all rows):
       x += attn(rms(x)); x += ff(rms(x)); attn: q,k,v (+bias), q/k RMSNorm over the whole row (weight, 1/sqrt(head_dim) folded into q), split-half rope with per-head tables, softmax attention, * 2 sigmoid(gate logits of rms(x)) per head, out (+bias); ff: gelu_tanh(x W1 + b1) W2 + b2 */
    const int aud = meta[M_FMT] == 1, D = aud ? 2048 : 4096, FFN = 4 * D, hdt = aud ? 2 : 4, NH = 32, nct = D / 32;
#define NBS(K) (128u + (size_t)((K) / 32) * 1024u)
#define BVS(n) ((size_t)((n) / 32) * 128u)
    const size_t Wsz = (size_t)(D / 32) * NBS(D), oQ = 0, oK = oQ + Wsz, oV = oK + Wsz, oO = oV + Wsz, oF1 = oO + Wsz, oF2 = oF1 + (size_t)(FFN / 32) * NBS(D), oG = oF2 + (size_t)(D / 32) * NBS(FFN), oVec = oG + (size_t)2 * NBS(D);
    const float *qg = (const float *)(wts + oVec), *kg = qg + D; const size_t oB = oVec + 8u * D;   /* f32 q gain, k gain, then fp16 bias vectors */
    const u16 *bq = (const u16 *)(wts + oB), *bk = (const u16 *)(wts + oB + BVS(D)), *bv = (const u16 *)(wts + oB + 2 * BVS(D)), *bo = (const u16 *)(wts + oB + 3 * BVS(D)), *bg = (const u16 *)(wts + oB + 4 * BVS(D)),
              *bf1 = (const u16 *)(wts + oB + 4 * BVS(D) + 256), *bf2 = (const u16 *)(wts + oB + 4 * BVS(D) + 256 + BVS(FFN));
    static float g_ones4k[4096] __attribute__((aligned(128))); static int ones_init; if (!ones_init) { for (int i = 0; i < 4096; i++) g_ones4k[i] = 1.0f; ones_init = 1; }
    const u16 *hnd = (const u16 *)BUF(M_HN); u16 *hr = (u16 *)BUF(M_HR), *gl = (u16 *)BUF(M_LQ); const size_t tbl = (size_t)NH * (hdt / 2) * 1024;
    g_hdt = hdt; gemm_set_q8(3);
    k_rms_prep(LM, LN, g_ones4k, g_zero, g_zero); pool_run(p_ln, &P); TICK();
    rot_run(h16, hr, ns, nct, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e);
      e.bias16 = bq; gemm_run_geo(hr, ntok, D, WB(oQ), D, q16, vt, &e, 0);
      e.bias16 = bk; gemm_run_geo(hr, ntok, D, WB(oK), D, k16, vt, &e, 0);
      e.bias16 = bv; gemm_run_geo(hr, ntok, D, WB(oV), D, v16, vt, &e, 0);
      e.bias16 = bg; gemm_run_geo(h16, ntok, D, WB(oG), 64, gl, vt, &e, 0); } TICK();
    { qn_t qn = {q16, k16, qg, kg, cosT, sinT, ns, nct, hdt, tbl}; pool_run(p_qn, &qn); } TICK();
    attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 0, 0, nct, nct, 1, vt); TICK();
    { gt_t gt = {o16, gl, ns, nct, hdt, NH}; pool_run(p_gt, &gt); } TICK();
    rot_run(o16, hr, ns, nct, hnd, vt);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = g_ones4k; e.bias16 = bo; gemm_run_geo(hr, ntok, D, WB(oO), D, t16, vt, &e, 0); } TICK();
    k_rms_prep(LM, LN, g_ones4k, g_zero, g_zero); pool_run(p_ln, &P);
    rot_run(h16, hr, ns, nct, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 2; e.t2 = g_t2; e.scr = g_scr; e.bias16 = bf1; gemm_run_geo(hr, ntok, D, WB(oF1), FFN, mid, vt, &e, 0); } TICK();
    rot_run(mid, hr, ns, FFN / 32, hnd, vt);
    { const gemm_geo_t d0 = {FFN / 32, FFN / 32, 0, 0, 0, 0}, d1 = {FFN / 32, FFN / 32, FFN / 64, 0, 0, 0};
      gemm_run_geo(hr, ntok, FFN / 2, WB(oF2), D, t16, vt, 0, &d0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = g_ones4k; e.prev = t16; e.bias16 = bf2;
      gemm_run_geo(hr, ntok, FFN / 2, WB(oF2), D, t16, vt, &e, &d1); } TICK();
    g_hdt = 4; gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 82) {   /* MiniMax-H3 video VAE decoder ViT block (anima/h3/vae): D 2048 (64 tiles), 32 heads x 64, ffn 8192 gated silu, ConvRot int8, NO mask, no modulation, 1/sqrt(64) folded into the q gain;
       x += s1 * (O(attn(rms(x) * g1)) + bo); x += s2 * (W2(silu(gate) * up) + b2), gate/up = rms(x) * g2 W1 (+ b1). Per-head q/k RMSNorm (no weight) + split-half rope in the GEMM epilogue (mode 7, tables [slab][32 rows][32] shared by all heads).
       Pack: Q K V O [D x D], W1 [2*FFN x D] (n-blocks interleaved gate_j, up_j), W2 [D x FFN] (each matrix: per n-block 128 B scales + K/32 tiles), then f32 g1 g2 s1 s2 [D each], gq gk [64 each], then fp16 bias vectors (128 B per 32 columns) bq bk bv bo [D/32], bf1 [2*FFN/32], bf2 [D/32] */
    const int D = 2048, FFN = 8192, hdt = 2, nct = 64;
#define NBS(K) (128u + (size_t)((K) / 32) * 1024u)
#define BVS(n) ((size_t)((n) / 32) * 128u)
    const size_t Wsz = (size_t)(D / 32) * NBS(D), oQ = 0, oK = oQ + Wsz, oV = oK + Wsz, oO = oV + Wsz, oF1 = oO + Wsz, oF2 = oF1 + (size_t)(2 * FFN / 32) * NBS(D), oVec = oF2 + (size_t)(D / 32) * NBS(FFN);
    const float *g1 = (const float *)(wts + oVec), *g2 = g1 + D, *s1 = g2 + D, *s2 = s1 + D, *gq = s2 + D, *gk = gq + 64; const size_t oB = oVec + (size_t)(4 * D + 128) * 4;
    const u16 *bq = (const u16 *)(wts + oB), *bk = (const u16 *)(wts + oB + BVS(D)), *bv = (const u16 *)(wts + oB + 2 * BVS(D)), *bo = (const u16 *)(wts + oB + 3 * BVS(D)),
              *bf1 = (const u16 *)(wts + oB + 4 * BVS(D)), *bf2 = (const u16 *)(wts + oB + 4 * BVS(D) + BVS(2 * FFN));
    const u16 *hnd = (const u16 *)BUF(M_HN); u16 *hr = (u16 *)BUF(M_HR);
    g_hdt = hdt; gemm_set_q8(3);
    k_rms_prep(LM, LN, g1, g_zero, g_zero); pool_run(p_ln, &P); TICK();
    rot_run(h16, hr, ns, nct, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 7; e.hdt = hdt; e.rope = 1; e.cosT = cosT; e.sinT = sinT; e.bias16 = bq; e.gain = gq; gemm_run_geo(hr, ntok, D, WB(oQ), D, q16, vt, &e, 0);
      e.bias16 = bk; e.gain = gk; gemm_run_geo(hr, ntok, D, WB(oK), D, k16, vt, &e, 0);
      gemm_epi_t ev; memset(&ev, 0, sizeof ev); ev.bias16 = bv; gemm_run_geo(hr, ntok, D, WB(oV), D, v16, vt, &ev, 0); } TICK();
    attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 0, 0, nct, nct, 1, vt); TICK();
    rot_run(o16, hr, ns, nct, hnd, vt);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = s1; e.bias16 = bo; gemm_run_geo(hr, ntok, D, WB(oO), D, t16, vt, &e, 0); } TICK();
    k_rms_prep(LM, LN, g2, g_zero, g_zero); pool_run(p_ln, &P);
    rot_run(h16, hr, ns, nct, hnd, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr; e.bias16 = bf1; gemm_run_geo(hr, ntok, D, WB(oF1), 2 * FFN, mid, vt, &e, 0); } TICK();
    rot_run(mid, hr, ns, FFN / 32, hnd, vt);
    { const gemm_geo_t d0 = {FFN / 32, FFN / 32, 0, 0, 0, 0}, d1 = {FFN / 32, FFN / 32, FFN / 64, 0, 0, 0};
      gemm_run_geo(hr, ntok, FFN / 2, WB(oF2), D, t16, vt, 0, &d0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = s2; e.prev = t16; e.bias16 = bf2;
      gemm_run_geo(hr, ntok, FFN / 2, WB(oF2), D, t16, vt, &e, &d1); } TICK();
    g_hdt = 4; gemm_set_q8(1);
    goto done;
  }
  const int h3lora = meta[M_MODE] == 50 && meta[M_LORA];
  if (meta[M_MODE] == 50) {   /* MiniMax-H3-Image DiT block (anima/h3): slabs [0, M_NIMG) image rows, the rest text rows (+ padding at the very end, masked through M_NVALID);
                                 mod = 2 kinds (image, text) x [shift1, scale1, gate1, shift2, scale2, gate2] x 5376 f32; ConvRot: activations rotated by Hn before every int8 GEMM */
    const int nimg = meta[M_NIMG]; const float *mk[2] = {mod, mod + 6 * 5376}; const float *gn = (const float *)(wts + H3_GAIN);
    const u16 *hnd = (const u16 *)BUF(M_HN); u16 *hr = (u16 *)BUF(M_HR);
    static HVX_Vector LM1[MAXNCB] __attribute__((aligned(128))), LN1[MAXNCB] __attribute__((aligned(128))), LM2[MAXNCB] __attribute__((aligned(128))), LN2[MAXNCB] __attribute__((aligned(128)));
    /* Ref2VA reference-image rows (clean condition rows, their own timestep): slabs [M_A5, M_A6) use the third modulation set mod[12 * 5376 ..] (video tag at the condition timestep) */
    const int rs2 = meta[M_A5], rs3 = meta[M_A6]; const float *mkR = mod + 12 * 5376;
    g_rope_eps = 1e-5f; gemm_set_q8(3);
    /* windowed runs (anima/h3 r2vrun, like the LTX ops): M_A4 bit 0 = only project this window's keys/values into the GLOBAL K/V buffers at slab offset M_A9, bit 1 = keys/values already there (the window's queries attend to all M_A2 key tokens); M_NIMG = image slabs inside the window, M_NVALID = valid key tokens (global) */
    const int fl50 = meta[M_A4], onlyKV = fl50 & 1, haveKV = (fl50 >> 1) & 1, nkall = (onlyKV || haveKV) ? meta[M_A2] : ntok; u16 *k16w = k16 + (size_t)meta[M_A9] * 224 * T16, *v16w = v16 + (size_t)meta[M_A9] * 224 * T16;
    P.M1 = LM1; P.N1 = LN1; P.ns0 = nimg; P.eps = 1e-5f; P.M2 = LM2; P.N2 = LN2; P.ns2 = rs2; P.ns3 = rs3;
    k_rms_prep(LM, LN, gn + 256, mk[0], mk[0] + 5376); k_rms_prep(LM1, LN1, gn + 256, mk[1], mk[1] + 5376); if (rs3 > rs2) k_rms_prep(LM2, LN2, gn + 256, mkR, mkR + 5376); pool_run(p_ln2, &P); TICK();
    rot_run(h16, hr, ns, 168, hnd, vt); TICK();
    { const u16 *LW = (const u16 *)(wts + H3_TOTAL); u16 *D = (u16 *)BUF(M_DLT), *Lq = (u16 *)BUF(M_LQ), *Lk = Lq + (size_t)ns * 4 * T16, *Lv = Lk + (size_t)ns * 4 * T16;
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = gn; e.cosT = cosT; e.sinT = sinT; if (h3lora) { e.dlt = D; e.dnt = 224; }
      if (!onlyKV) {
      if (h3lora) { gemm_set_q8(0); gemm_run_geo(hr, ntok, 5376, (const u16 *)((const uint8_t *)LW + H3L_AQ), 128, Lq, vt, 0, 0); gemm_run_geo(Lq, ntok, 128, (const u16 *)((const uint8_t *)LW + H3L_BQ), 7168, D, vt, 0, 0); gemm_set_q8(3); }
      gemm_run_geo(hr, ntok, 5376, WB(H3_Q), 7168, q16, vt, &e, 0); }
      e.gain = gn + 128;
      if (!haveKV) {
      if (h3lora) { gemm_set_q8(0); gemm_run_geo(hr, ntok, 5376, (const u16 *)((const uint8_t *)LW + H3L_AK), 128, Lk, vt, 0, 0); gemm_run_geo(Lk, ntok, 128, (const u16 *)((const uint8_t *)LW + H3L_BK), 7168, D, vt, 0, 0); gemm_set_q8(3); }
      gemm_run_geo(hr, ntok, 5376, WB(H3_K), 7168, onlyKV ? k16w : k16, vt, &e, 0);
      { gemm_epi_t ev; memset(&ev, 0, sizeof ev);
        if (h3lora) { gemm_set_q8(0); gemm_run_geo(hr, ntok, 5376, (const u16 *)((const uint8_t *)LW + H3L_AV), 128, Lv, vt, 0, 0); gemm_run_geo(Lv, ntok, 128, (const u16 *)((const uint8_t *)LW + H3L_BV), 7168, D, vt, 0, 0); gemm_set_q8(3); ev.dlt = D; ev.dnt = 224; }
        gemm_run_geo(hr, ntok, 5376, WB(H3_V), 7168, onlyKV ? v16w : v16, vt, h3lora ? &ev : 0, 0); } } } TICK();
    if (onlyKV) { g_rope_eps = 1e-6f; gemm_set_q8(1); goto done; }
    attn_run(q16, k16, v16, o16, ns, nkall, meta[M_NVALID], 0, 0, 224, 224, 1, vt); TICK();
    rot_run(o16, hr, ns, 224, hnd, vt);
    { const u16 *LW = (const u16 *)(wts + H3_TOTAL); u16 *D = (u16 *)BUF(M_DLT), *Lo = (u16 *)BUF(M_LO);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = mk[0] + 2 * 5376; e.gate2 = mk[1] + 2 * 5376; e.ns0 = nimg; if (rs3 > rs2) { e.gate3 = mkR + 2 * 5376; e.ns2 = rs2; e.ns3 = rs3; }
      if (h3lora) { gemm_set_q8(0); gemm_run_geo(hr, ntok, 7168, (const u16 *)((const uint8_t *)LW + H3L_AO), 128, Lo, vt, 0, 0); gemm_run_geo(Lo, ntok, 128, (const u16 *)((const uint8_t *)LW + H3L_BO), 5376, D, vt, 0, 0); gemm_set_q8(3); e.dlt = D; e.dnt = 168; }
      gemm_run_geo(hr, ntok, 7168, WB(H3_O), 5376, t16, vt, &e, 0); } TICK();
    k_rms_prep(LM, LN, gn + 256 + 5376, mk[0] + 3 * 5376, mk[0] + 4 * 5376); k_rms_prep(LM1, LN1, gn + 256 + 5376, mk[1] + 3 * 5376, mk[1] + 4 * 5376); if (rs3 > rs2) k_rms_prep(LM2, LN2, gn + 256 + 5376, mkR + 3 * 5376, mkR + 4 * 5376); pool_run(p_ln2, &P);
    rot_run(h16, hr, ns, 168, hnd, vt); TICK();
    { const u16 *LW = (const u16 *)(wts + H3_TOTAL); u16 *D = (u16 *)BUF(M_DLT), *Lu = (u16 *)BUF(M_LU);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr;   /* silu(gate_j) * up_j, n-blocks interleaved (gate, up) */
      if (h3lora) { gemm_set_q8(0); gemm_run_geo(hr, ntok, 5376, (const u16 *)((const uint8_t *)LW + H3L_AGU), 128, Lu, vt, 0, 0); gemm_run_geo(Lu, ntok, 128, (const u16 *)((const uint8_t *)LW + H3L_BGU), 28672, D, vt, 0, 0); gemm_set_q8(3); e.dlt = D; e.dnt = 896; }
      gemm_run_geo(hr, ntok, 5376, WB(H3_GU), 28672, mid, vt, &e, 0); } TICK();
    rot_run(mid, hr, ns, 448, hnd, vt);
    { const u16 *LW = (const u16 *)(wts + H3_TOTAL); u16 *D = (u16 *)BUF(M_DLT), *Ld = (u16 *)BUF(M_LD);
      const gemm_geo_t d0 = {448, 448, 0, 0, 0, 0}, d1 = {448, 448, 224, 0, 0, 0};
      if (h3lora) { gemm_set_q8(0); gemm_run_geo(hr, ntok, 14336, (const u16 *)((const uint8_t *)LW + H3L_ADN), 128, Ld, vt, 0, 0); gemm_run_geo(Ld, ntok, 128, (const u16 *)((const uint8_t *)LW + H3L_BDN), 5376, D, vt, 0, 0); gemm_set_q8(3); }
      gemm_run_geo(hr, ntok, 7168, WB(H3_DN), 5376, t16, vt, 0, &d0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = mk[0] + 5 * 5376; e.gate2 = mk[1] + 5 * 5376; e.ns0 = nimg; if (rs3 > rs2) { e.gate3 = mkR + 5 * 5376; e.ns2 = rs2; e.ns3 = rs3; } e.prev = t16; if (h3lora) { e.dlt = D; e.dnt = 168; }
      gemm_run_geo(hr, ntok, 7168, WB(H3_DN), 5376, t16, vt, &e, &d1); } TICK();
    g_rope_eps = 1e-6f; gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 70) {   /* FLUX.3 Klein DiT block (anima/flux3): PARALLEL block, x += gate * (O(attn(LNmod(x))) + DN(silu(g) * u)), hidden 3072 (96 tiles), 24 heads x 128, ffn 9216, plain int8 (no ConvRot).
                                 slabs [0, M_NIMG) kind 0 rows, the rest kind 1 (padding at the very end masked through M_NVALID); mod = 2 kinds x [shift, scale, gate] x 3072 f32 */
    const int nimg = meta[M_NIMG]; const float *mk[2] = {mod, mod + 3 * 3072}; const float *gn = (const float *)(wts + KL_GAIN);
    static HVX_Vector LM1[MAXNCB] __attribute__((aligned(128))), LN1[MAXNCB] __attribute__((aligned(128)));
    gemm_set_q8(3);
    P.M1 = LM1; P.N1 = LN1; P.ns0 = nimg; P.eps = 1e-6f;
    k_ln_prep(LM, LN, mk[0], mk[0] + 3072); k_ln_prep(LM1, LN1, mk[1], mk[1] + 3072); pool_run(p_lnn2, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = gn; e.cosT = cosT; e.sinT = sinT;
      gemm_run_geo(h16, ntok, 3072, WB(KL_Q), 3072, q16, vt, &e, 0);
      e.gain = gn + 128; gemm_run_geo(h16, ntok, 3072, WB(KL_K), 3072, k16, vt, &e, 0);
      gemm_run_geo(h16, ntok, 3072, WB(KL_V), 3072, v16, vt, 0, 0); } TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr;   /* silu(gate_j) * up_j, n-blocks interleaved (gate, up) */
      gemm_run_geo(h16, ntok, 3072, WB(KL_GU), 18432, mid, vt, &e, 0); } TICK();
    attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 0, 0, 96, 96, 1, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = mk[0] + 2 * 3072; e.gate2 = mk[1] + 2 * 3072; e.ns0 = nimg;
      gemm_run_geo(o16, ntok, 3072, WB(KL_O), 3072, t16, vt, &e, 0); } TICK();
    { const gemm_geo_t d0 = {288, 288, 0, 0, 0, 0}, d1 = {288, 288, 144, 0, 0, 0};   /* K = 9216 in two halves of 4608 */
      gemm_run_geo(mid, ntok, 4608, WB(KL_DN), 3072, t16, vt, 0, &d0);
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = mk[0] + 2 * 3072; e.gate2 = mk[1] + 2 * 3072; e.ns0 = nimg; e.prev = t16;
      gemm_run_geo(mid, ntok, 4608, WB(KL_DN), 3072, t16, vt, &e, &d1); } TICK();
    gemm_set_q8(1);
    goto done;
  }
  if (meta[M_MODE] == 53) {   /* Mage-Flow double-stream block (mage/export_mage.py). rows = [img slabs (M_BA of them) | txt slabs]; keys = all rows (txt pad is trailing -> M_NVALID mask). mod = [img: shift1,scale1,gate1,shift2,scale2,gate2][txt: same] x DIM f32.
                                 Biases ride in the 4 extra K tiles (host pre-fills the L buffers with the bias-tile pattern), q/k per-head RMSNorm in the GEMM epilogue, rope on the img rows only.
                                 Windowed runs (like MiniMax-H3 and LTX): M_A4 bit 0 = only project this window's keys/values into the GLOBAL K/V buffers at slab offset M_A9, bit 1 = keys/values already there (the window's queries attend to all M_A2 key tokens); M_BA = image slabs inside the window, M_NVALID = valid key tokens (global). Without flags: the plain block over all rows. */
    const int32_t *h4 = (const int32_t *)wts; uint32_t po[14]; for (int i = 0; i < 14; i++) po[i] = (uint32_t)h4[1 + i];
    const int ni = meta[M_BA], nt = ns - ni; const int sl0[2] = {0, ni}, sln[2] = {ni, nt};
    const int fl53 = meta[M_A4], onlyKV = fl53 & 1, haveKV = (fl53 >> 1) & 1, nkall = (onlyKV || haveKV) ? meta[M_A2] : ntok; u16 *k16w = k16 + (size_t)meta[M_A9] * NCB * T16, *v16w = v16 + (size_t)meta[M_A9] * NCB * T16;
    g_hdt = 4; gemm_set_q8(1);
    for (int s = 0; s < 2; s++) {   /* stage 1: LN + q,k,v */
      const int n = sln[s], s0 = sl0[s]; if (!n) continue; const float *m = mod + (size_t)s * 8 * DIM; const uint32_t *o = po + 7 * s;
      uint64_t cc = PC(); k_ln_prep(LM, LN, m, m + DIM); P.x = x + (size_t)s0 * NCB * 1024; P.h = h16 + (size_t)s0 * NCB * T16; P.ns = n; pool_run(p_lnn, &P); g_acc[8] += PC() - cc; cc = PC();
      const gemm_geo_t g3[3] = {{NCB, NCB + 4, 0, Lq + (size_t)s0 * 12 * T16, 12, 4}, {NCB, NCB + 4, 0, Lq + (size_t)s0 * 12 * T16 + 4 * T16, 12, 4}, {NCB, NCB + 4, 0, Lq + (size_t)s0 * 12 * T16 + 8 * T16, 12, 4}};
      const float *gn = (const float *)(wts + o[6]);
      u16 *hs = h16 + (size_t)s0 * NCB * T16, *qs = q16 + (size_t)s0 * NCB * T16, *ks = (onlyKV ? k16w : k16) + (size_t)s0 * NCB * T16, *vs = (onlyKV ? v16w : v16) + (size_t)s0 * NCB * T16;
      gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = s == 0; e.gain = gn; e.cosT = cosT; e.sinT = sinT;
      if (!onlyKV) gemm_run_geo(hs, n * 32, DIM + 128, WB(o[0]), DIM, qs, vt, &e, &g3[0]);
      g_acc[9] += PC() - cc; cc = PC();
      if (haveKV) continue;
      e.gain = gn + 128; gemm_run_geo(hs, n * 32, DIM + 128, WB(o[1]), DIM, ks, vt, &e, &g3[1]); g_acc[10] += PC() - cc; cc = PC();
      k_ln_prep(LM, LN, m + 2 * DIM, m + 3 * DIM); pool_run(p_lnn, &P); g_acc[11] += PC() - cc; cc = PC();   /* v input at 1/4096 (its own scale) */
      gemm_run_geo(hs, n * 32, DIM + 128, WB(o[2]), DIM, vs, vt, 0, &g3[2]); g_acc[12] += PC() - cc;
    } TICK();
    if (onlyKV) goto done;
    attn_run(q16, k16, v16, o16, ns, nkall, meta[M_NVALID], 0, 0, NCB, NCB, 1, vt); TICK();
    for (int s = 0; s < 2; s++) {   /* stage 2: o-proj + residual, LN2, fc1(+GELU), fc2 + residual */
      const int n = sln[s], s0 = sl0[s]; if (!n) continue; const float *m = mod + (size_t)s * 8 * DIM; const uint32_t *o = po + 7 * s;
      u16 *os = o16 + (size_t)s0 * NCB * T16, *ts = t16 + (size_t)s0 * NCB * T16, *ms = mid + (size_t)s0 * 384 * T16, *hs = h16 + (size_t)s0 * NCB * T16;
      float *xs = x + (size_t)s0 * NCB * 1024;
      uint64_t cc = PC();
      { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = xs; e.gate = m + 4 * DIM; const gemm_geo_t go = {NCB, NCB + 4, 0, Lo + (size_t)s0 * 4 * T16, 4, 4};
        gemm_run_geo(os, n * 32, DIM + 128, WB(o[3]), DIM, ts, vt, &e, &go); } g_acc[13] += PC() - cc; cc = PC();
      k_ln_prep(LM, LN, m + 5 * DIM, m + 6 * DIM); P.x = xs; P.h = hs; P.ns = n; pool_run(p_lnn, &P); g_acc[14] += PC() - cc; cc = PC();
      { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 2; e.t2 = g_t2s; e.scr = g_scr; const gemm_geo_t gu = {NCB, NCB + 4, 0, Lu + (size_t)s0 * 4 * T16, 4, 4};
        gemm_run_geo(hs, n * 32, DIM + 128, WB(o[4]), 12288, ms, vt, &e, &gu); } g_acc[15] += PC() - cc; cc = PC();
      if (meta[M_FMT] == 4) {   /* fc2 in 4 K parts of 96 tiles: smaller weight panels -> more rows per VTCM row group (fewer weight re-reads); every part adds gate*partial to the fp32 stream */
        gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = xs; e.gate = m + 7 * DIM;
        for (int kp = 0; kp < 4; kp++) { const gemm_geo_t gp = kp == 3 ? (gemm_geo_t){384, 388, 288, Ld + (size_t)s0 * 4 * T16, 4, 4} : (gemm_geo_t){384, 388, kp * 96, 0, 0, 0};
          gemm_run_geo(ms, n * 32, kp == 3 ? 3072 + 128 : 3072, WB(o[5]), DIM, ts, vt, &e, &gp); }
      } else
      { const gemm_geo_t g0 = {384, 388, 0, 0, 0, 0}, g1_ = {384, 388, 192, Ld + (size_t)s0 * 4 * T16, 4, 4};
        gemm_run_geo(ms, n * 32, 6144, WB(o[5]), DIM, ts, vt, 0, &g0);
        gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = xs; e.gate = m + 7 * DIM; e.prev = ts;
        gemm_run_geo(ms, n * 32, 6144 + 128, WB(o[5]), DIM, ts, vt, &e, &g1_); } g_acc[16] += PC() - cc;
    } TICK();
    goto done;
  }
  if (meta[M_MODE] == 61) {   /* generic GEMM for the CPU-side VAE: t16 = [gelu](h16 @ W + bias). M_NCB = K tiles, M_BA = N (output columns), M_FMT: 0 plain, 2 tanh-GELU epilogue; W at M_WOFF = [N/32][K/32+4] q8 tiles (bias in the 4 extra K tiles), L buffer M_LQ = bias tile pattern */
    const int kt = meta[M_NCB], N = meta[M_BA]; gemm_set_q8(1);
    gemm_epi_t e; memset(&e, 0, sizeof e); if (meta[M_FMT] == 2) { e.mode = 2; e.t2 = g_t2; e.scr = g_scr; }
    const gemm_geo_t g = {kt, kt + 4, 0, Lq, 4, 4};
    gemm_run_geo(h16, ntok, kt * 32 + 128, WB(0), N, t16, vt, &e, &g); TICK();
    goto done;
  }
  if (meta[M_MODE] == 55) {   /* Qwen3-VL-4B vision layer (mage/export_mvit.py): like MODE 23 with hidden 1024, ffn 4096 (no padding) */
    const float *gw = (const float *)(wts + QM_GAIN), *ones = mod;
    for (int cb = 0; cb < NCB; cb++) { LM[cb] = *(const HVX_Vector *)(gw + cb * 32); LN[cb] = *(const HVX_Vector *)(gw + 1024 + cb * 32); }
    pool_run(p_lnn, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = 0; e.cosT = cosT; e.sinT = sinT;
      gemm_run_geo(h16, ntok, DIM + 128, WB(QM_Q), 2048, q16, vt, &e, &gq);
      gemm_run_geo(h16, ntok, DIM + 128, WB(QM_K), 2048, k16, vt, &e, &gk);
      gemm_run_geo(h16, ntok, DIM + 128, WB(QM_V), 2048, v16, vt, 0, &gv); } TICK();
    g_hdt = 4; attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], 0, 0, 64, 64, 1, vt); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = ones; const gemm_geo_t go = {64, 68, 0, Lo, 4, 4};
      gemm_run_geo(o16, ntok, 2048 + 128, WB(QM_O), DIM, t16, vt, &e, &go); } TICK();
    for (int cb = 0; cb < NCB; cb++) { LM[cb] = *(const HVX_Vector *)(gw + 2048 + cb * 32); LN[cb] = *(const HVX_Vector *)(gw + 3072 + cb * 32); }
    pool_run(p_lnn, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 2; e.t2 = g_t2; e.scr = g_scr; const gemm_geo_t gu = {NCB, NCB + 4, 0, Lu, 4, 4};
      gemm_run_geo(h16, ntok, DIM + 128, WB(QM_FC1), 4096, mid, vt, &e, &gu); } TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = ones; const gemm_geo_t gd = {128, 132, 0, Ld, 4, 4};
      gemm_run_geo(mid, ntok, 4096 + 128, WB(QM_FC2), DIM, t16, vt, &e, &gd); } TICK();
    goto done;
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
  if (meta[M_REL] == 1) { pool_quit(); HAP_compute_res_release(g_ctx); g_vt = 0; }   /* M_REL 2: only flush x to memory (windowed runs) */
  return 0;
}

static void cache_sync(void *p, size_t bytes) { for (char *c = (char *)p; c < (char *)p + bytes; c += 32) Q6_dccleaninva_A(c); }
int mega_block(remote_handle64 hd, const int *meta, int nmeta, int wfd, int bfd, const uint8_t *mod, int modlen, int *times, int ntimes, int *rc) {
  void *bp = 0; uint64 bpa = 0;
  g_ncb = meta[M_NCB] ? meta[M_NCB] : 96;
  if (HAP_mmap_get(bfd, &bp, &bpa) || !bp) { *rc = -7; return 0; }
  g_blob = (uint8_t *)bp;
  const size_t xb = (size_t)meta[M_NTOK] * DIM * 4;
  if (meta[M_FIRST] && meta[M_MODE] != 61) { cache_sync(g_blob + meta[M_X], xb); cache_sync(g_blob + meta[M_COS], (size_t)meta[M_NTOK] / 32 * 8192); cache_sync(g_blob + meta[M_SIN], (size_t)meta[M_NTOK] / 32 * 8192); }
  if ((meta[M_MODE] == 50 || meta[M_MODE] == 51 || meta[M_MODE] == 80 || meta[M_MODE] == 81 || meta[M_MODE] == 82 || meta[M_MODE] == 90 || meta[M_MODE] == 91 || meta[M_MODE] == 92) && meta[M_FIRST]) cache_sync(g_blob + meta[M_HN], 131072);
  if (meta[M_NPRE]) { const size_t own = (size_t)meta[M_NTOK] * DIM * 2, pre = (size_t)meta[M_NPRE] * 32 * DIM * 2; cache_sync(g_blob + meta[M_K] + own, pre); cache_sync(g_blob + meta[M_V] + own, pre); }
  if (meta[M_MODE] == 61) { cache_sync(g_blob + meta[M_H], (size_t)meta[M_NTOK] / 32 * meta[M_NCB] * 2048); cache_sync(g_blob + meta[M_LQ], (size_t)meta[M_NTOK] / 32 * 4 * 2048); }   /* Mage VAE GEMM: host-written A tiles + bias pattern */
  int r = mega_block_impl(hd, meta, nmeta, wfd, mod, modlen, times, ntimes, rc);
  if (meta[M_MODE] == 61) cache_sync(g_blob + meta[M_T], (size_t)meta[M_NTOK] / 32 * (meta[M_BA] / 32) * 2048);   /* DSP-written result */
  if (meta[M_REL]) cache_sync(g_blob + meta[M_X], xb);
  HAP_mmap_put(bfd);
  if (*rc != -5) HAP_mmap_put(wfd);
  return r;
}
