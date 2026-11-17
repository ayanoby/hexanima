#pragma once
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
/* Activation streams live in "crouton" tiles of 32 rows x 32 cols.
   h16 tile (2 KB, fp16): halfword (r,c) = (r/2)*64 + (c%32)*2 + (r&1)  -> 16 HVX vectors, vector p = rows 2p (even lanes), 2p+1 (odd lanes)
   x32 tile (4 KB, fp32): float (r,c) = r*32 + c%32                      -> 32 HVX vectors, vector r = row r
   A matrix is [row slab][col block] tiles, DIM=2048 -> 64 col blocks. */
#define DIM 2048
#define NCB (DIM / 32)
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
void k_softmax2p(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *exp_tab, HVX_Vector *scratch, int p0, int p1);
void k_softmax2(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *exp_tab, HVX_Vector *scratch);
void k_rowscale_sum(uint16_t *O, int ntile, const uint16_t *sumtile);   /* O tiles *= 1/sumtile (rows) */
/* transpose one 32x32 fp16 tile between the paired-row layouts: dst[(d/2)*64+key*2+(d&1)] = src[(key/2)*64+d*2+(key&1)] */
void k_transpose_tile(const uint16_t *src, uint16_t *dst);
void k_resadd_tile(float *x32tile, const uint16_t *t16tile, const float *gate32);   /* x += gate*t for one 32x32 tile */
void k_rmsrope_head(uint16_t *x4, const float *gain, const float *cosT, const float *sinT, int rope);   /* one head: 4 contiguous tiles in place */
void k_resadd2_tile(float *x32tile, const uint16_t *t16tile, const uint16_t *prev16tile, const float *gate32);
/* chunked attention (no limit on the number of tokens) */
void k_rowmax_chunk(const uint16_t *S, int nkb, int p0, int p1, const HVX_Vector *kmask, HVX_Vector *mx);
void k_softmax_fixed(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *exp_tab, HVX_Vector *scratch, int p0, int p1, const HVX_Vector *kmask, const HVX_Vector *mx);
void k_acc_tile(HVX_Vector *acc, const uint16_t *t16, int first);
void k_final_acc(uint16_t *O, const HVX_Vector *acc, int nt);
void k_nag_slab(uint16_t *z, const uint16_t *n, float s, float tau, float alpha, HVX_Vector *scr);
