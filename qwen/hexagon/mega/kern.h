#pragma once
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
/* Activation streams live in "crouton" tiles of 32 rows x 32 cols.
   h16 tile (2 KB, fp16): halfword (r,c) = (r/2)*64 + (c%32)*2 + (r&1)  -> 16 HVX vectors, vector p = rows 2p (even lanes), 2p+1 (odd lanes)
   x32 tile (4 KB, fp32): float (r,c) = r*32 + c%32                      -> 32 HVX vectors, vector r = row r
   A matrix is [row slab][col block] tiles, DIM=2048 -> 64 col blocks. */
extern int g_ncb;           /* runtime hidden size in 32-col tiles: 96 (DiT 3072) or 128 (Qwen3.5 4096) */
#define NCB g_ncb
#define DIM (g_ncb * 32)
#define MAXNCB 128
#define NHEAD 24
void k_ln_prep(HVX_Vector *M, HVX_Vector *N, const float *shift, const float *scale);
void k_lnmod_s(const float *x32, uint16_t *h16, int nslab, const HVX_Vector *M, const HVX_Vector *N, float eps);
void k_lnmod(const float *x32, uint16_t *h16, int nslab, const float *shift, const float *scale, float eps);
void k_resadd(float *x32, const uint16_t *t16, int nslab, const float *gate);
void k_gelu(uint16_t *t16, size_t nvec);                               /* in place, tanh-form via exp */
/* softmax numerators for one 32-query slab: S tiles [nkb] -> P=exp(s-max) tiles (fp16), R[r] = splat(1/sum_r) (32 vectors) */
void k_softmax(const uint16_t *S, uint16_t *P, int nkb, float *R);
void k_rowscale(uint16_t *O, int ntile, const float *R);               /* O tiles *= R[row] */
/* per-head RMSNorm(gain[128]) + split-half RoPE on one slab of q or k: x = 64 tiles (cb) in place; cs = [2 cb][cos|sin] fp32 tiles for this slab */
void k_rmsrope(uint16_t *x, const float *gain, const float *cosT, const float *sinT, int rope);
/* table kernels: tables live in VTCM (64 KB each, indexed by 15 magnitude bits of an fp16 value) */
#define TAB_BYTES 65536
void tab_init(uint16_t *exp_tab, uint16_t *t2_tab);   /* exp_tab[b]=exp(-|x|), t2_tab[b]=|x|*Phi(-|x|) as fp16 */
void k_gelu2(uint16_t *t16, size_t nvec, const uint16_t *t2_tab, HVX_Vector *scratch);
void k_softmax2p(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *exp_tab, HVX_Vector *scratch, int p0, int p1, const HVX_Vector *kmask, int kst);   /* kst=1: kmask is 16 vectors (one per row pair), 0: one vector */   /* kmask: lanes of invalid keys of the LAST key tile = 0xFFFF (or NULL) */
void k_softmax2(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *exp_tab, HVX_Vector *scratch);
void k_rowscale_sum(uint16_t *O, int ntile, const uint16_t *sumtile);   /* O tiles *= 1/sumtile (rows) */
/* transpose one 32x32 fp16 tile between the paired-row layouts: dst[(d/2)*64+key*2+(d&1)] = src[(key/2)*64+d*2+(key&1)] */
void k_transpose_tile(const uint16_t *src, uint16_t *dst);
void k_resadd_tile(float *x32tile, const uint16_t *t16tile, const float *gate32);   /* x += gate*t for one 32x32 tile */
void k_rmsrope_head(uint16_t *x4, const float *gain, const float *cosT, const float *sinT, int rope);   /* one head: 4 contiguous tiles in place */
void k_resadd2_tile(float *x32tile, const uint16_t *t16tile, const uint16_t *prev16tile, const float *gate32);

/* MOSS-Image additions */
void k_rms_prep(HVX_Vector *M, HVX_Vector *N, const float *w, const float *shift, const float *scale);   /* M = w*(1+scale), N = shift */
void k_rmsmod_s(const float *x32, uint16_t *h16, int nslab, const HVX_Vector *M, const HVX_Vector *N, float eps);   /* y = x*rsqrt(mean x^2+eps)*M + N */
void tab_init_situ(uint16_t *tabs);   /* 3 tables of 64 KB: T25[a]=25tanh(a/25), Gp[a]=4tanh(a/4)sig(a), Gn[a]=4tanh(a/4)sig(-a), a = |fp16| */
void k_situ_tile(const uint16_t *up, const uint16_t *gate, uint16_t *out, const uint16_t *tabs, HVX_Vector *scr);   /* scr: 12 vectors */

/* Qwen3.5 text encoder additions */
void tab_init_swiglu(uint16_t *tabs);   /* 2 tables of 64 KB: Sg1[a]=a*sig(a), Sg2[a]=a*sig(-a) */
void k_swiglu_tile(const uint16_t *gate, const uint16_t *up, uint16_t *out, const uint16_t *tabs, HVX_Vector *scr);   /* silu(gate)*up; scr: 8 vectors */
void te_t2r(const uint16_t *t, float *rows, int ns, int nct, int idx, int n);
void te_r2t(const float *rows, uint16_t *t, int ns, int nct, int idx, int n);
void te_conv_silu(const float *in, float *out, const float *w, int T, int C, int idx, int n);
void te_delta(const float *qkv, const float *ab, const float *z, float *out, const float *A, const float *dtb, const float *nw, int T, int idx, int n);
void te_attn_prep(float *q, float *k, const float *wq, const float *wk, const float *cs, int T, int idx, int n);
void te_attn(const float *q, const float *kv, float *out, int T, int idx, int n);
void te_attn_prep3(float *q, float *k, const float *wq, const float *wk, const float *cs, int T, int idx, int n);   /* Qwen3-VL: cs rows [T][128] = cos64|sin64 */
void te_attn3(const float *q, const float *kv, float *out, float *ktbase, int T, int idx, int n);
void k_softmax2pm(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *exp_tab, HVX_Vector *scratch, int p0, int p1, const int *pk, const int *pmi, const HVX_Vector *pm, int np);
void tab_init_gelut(uint16_t *t2);   /* tanh-GELU variant of the t2 table */
