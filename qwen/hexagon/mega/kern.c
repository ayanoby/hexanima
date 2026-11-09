#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>
#include "kern.h"
int g_ncb = 96;

#define QF2SF(x) Q6_Vsf_equals_Vqf32(x)
#define MULSF(a, b) Q6_Vqf32_vmpy_VsfVsf((a), (b))
static inline HVX_Vector splat_f(float f) { union { float f; int32_t i; } c; c.f = f; return Q6_V_vsplat_R(c.i); }
static inline void pf(const void *p, size_t bytes) {   /* l2fetch bytes (multiple of 128) in <=32 KB pieces */
  const char *c = (const char *)p;
  while (bytes) { size_t n = bytes > 32768 ? 32768 : bytes; Q6_l2fetch_AP((void *)c, ((uint64_t)128 << 32) | ((uint64_t)128 << 16) | (n / 128)); c += n; bytes -= n; }
}
static inline HVX_Vector lane_sum(HVX_Vector s) {  /* qf32 32-lane vector -> every lane holds the total (qf32) */
  for (int r = 64; r >= 4; r >>= 1) s = Q6_Vqf32_vadd_Vqf32Vqf32(s, Q6_V_vror_VR(s, r));
  return s;
}
static inline HVX_Vector rsqrt_sf(HVX_Vector v) {  /* sf in -> sf 1/sqrt(v), magic constant + 3 Newton steps */
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5f3759df), Q6_Vw_vasr_VwR(v, 1));
  const HVX_Vector hv = QF2SF(MULSF(v, splat_f(0.5f))), th = splat_f(1.5f);
  for (int it = 0; it < 3; it++) {
    const HVX_Vector yy = QF2SF(MULSF(y, y));
    y = QF2SF(MULSF(y, QF2SF(Q6_Vqf32_vsub_VsfVsf(th, QF2SF(MULSF(hv, yy))))));
  }
  return y;
}
static inline HVX_VectorPair widen_hf(HVX_Vector hf) {  /* hf vector -> qf32 pair; lo = even halfwords, hi = odd */
  return Q6_Wqf32_vmpy_VhfVhf(hf, Q6_Vh_vsplat_R(0x3c00));
}

/* y = (x-mean)*rstd*(1+scale)+shift per token row, x32 -> h16. M = 1+scale, N = shift as NCB vectors each (shared, read-only) */
void k_ln_prep(HVX_Vector *M, HVX_Vector *N, const float *shift, const float *scale) {
  const HVX_Vector vone = splat_f(1.0f);
  for (int cb = 0; cb < NCB; cb++) {
    M[cb] = QF2SF(Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)(scale + cb * 32), vone));
    N[cb] = *(const HVX_Vector *)(shift + cb * 32);
  }
}
void k_lnmod_s(const float *x32, uint16_t *h16, int nslab, const HVX_Vector *M, const HVX_Vector *N, float eps) {
  const HVX_Vector vinv = splat_f(1.0f / DIM), veps = splat_f(eps), vone = splat_f(1.0f);
  for (int s = 0; s < nslab; s++) {
    const float *xs = x32 + (size_t)s * NCB * 1024;
    pf(xs, (size_t)NCB * 4096);
    HVX_Vector *hs = (HVX_Vector *)(h16 + (size_t)s * NCB * 1024);
    for (int p = 0; p < 16; p++) {
      HVX_Vector sa = Q6_V_vzero(), sb = Q6_V_vzero(), qa = Q6_V_vzero(), qb = Q6_V_vzero();
      for (int cb = 0; cb < NCB; cb++) {
        const HVX_Vector a = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p) * 32), b = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p + 1) * 32);
        sa = Q6_Vqf32_vadd_Vqf32Vsf(sa, a); sb = Q6_Vqf32_vadd_Vqf32Vsf(sb, b);
        qa = Q6_Vqf32_vadd_Vqf32Vqf32(qa, MULSF(a, a)); qb = Q6_Vqf32_vadd_Vqf32Vqf32(qb, MULSF(b, b));
      }
      HVX_Vector ma = QF2SF(MULSF(QF2SF(lane_sum(sa)), vinv)), mb = QF2SF(MULSF(QF2SF(lane_sum(sb)), vinv));
      HVX_Vector ea = QF2SF(MULSF(QF2SF(lane_sum(qa)), vinv)), eb = QF2SF(MULSF(QF2SF(lane_sum(qb)), vinv));
      HVX_Vector va = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(ea, vone), MULSF(ma, ma)), veps));
      HVX_Vector vb = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(eb, vone), MULSF(mb, mb)), veps));
      const HVX_Vector ra = rsqrt_sf(va), rb = rsqrt_sf(vb);
      for (int cb = 0; cb < NCB; cb++) {
        const HVX_Vector a = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p) * 32), b = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p + 1) * 32);
        const HVX_Vector da = QF2SF(Q6_Vqf32_vsub_VsfVsf(a, ma)), db = QF2SF(Q6_Vqf32_vsub_VsfVsf(b, mb));
        const HVX_Vector ya = QF2SF(MULSF(da, ra)), yb = QF2SF(MULSF(db, rb));
        const HVX_Vector oa = Q6_Vqf32_vadd_Vqf32Vsf(MULSF(ya, M[cb]), N[cb]), ob = Q6_Vqf32_vadd_Vqf32Vsf(MULSF(yb, M[cb]), N[cb]);
        hs[cb * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(ob, oa));
      }
    }
  }
}


void k_lnmod(const float *x32, uint16_t *h16, int nslab, const float *shift, const float *scale, float eps) {
  static HVX_Vector M[MAXNCB] __attribute__((aligned(128))), N[MAXNCB] __attribute__((aligned(128)));
  k_ln_prep(M, N, shift, scale); k_lnmod_s(x32, h16, nslab, M, N, eps);
}

/* x += gate * t (t fp16 tile, x fp32 tile, gate per column) */
void k_resadd(float *x32, const uint16_t *t16, int nslab, const float *gate) {
  for (int s = 0; s < nslab; s++) {
    float *xs = x32 + (size_t)s * NCB * 1024;
    pf(xs, (size_t)NCB * 4096); pf(t16 + (size_t)s * NCB * 1024, (size_t)NCB * 2048);
    const HVX_Vector *ts = (const HVX_Vector *)(t16 + (size_t)s * NCB * 1024);
    for (int cb = 0; cb < NCB; cb++) {
      const HVX_Vector g = *(const HVX_Vector *)(gate + cb * 32);
      for (int p = 0; p < 16; p++) {
        const HVX_VectorPair w = widen_hf(ts[cb * 16 + p]);
        HVX_Vector *xa = (HVX_Vector *)(xs + cb * 1024 + (2 * p) * 32), *xb = (HVX_Vector *)(xs + cb * 1024 + (2 * p + 1) * 32);
        *xa = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(Q6_V_lo_W(w)), g), *xa));
        *xb = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(Q6_V_hi_W(w)), g), *xb));
      }
    }
  }
}

static inline HVX_Vector exp_sf(HVX_Vector x) {  /* sf e^x for x <= ~80 (x clamped below at -80) */
  x = Q6_Vsf_vmax_VsfVsf(x, splat_f(-80.f));
  const HVX_Vector t0 = QF2SF(MULSF(x, splat_f(1.4426950408889634f)));
  const HVX_Vector big = splat_f(12582912.0f);
  const HVX_Vector t = QF2SF(Q6_Vqf32_vadd_VsfVsf(t0, big));                  /* low mantissa bits = round(t0) */
  const HVX_Vector n_f = QF2SF(Q6_Vqf32_vsub_VsfVsf(t, big));
  const HVX_Vector f = QF2SF(Q6_Vqf32_vsub_VsfVsf(t0, n_f));                  /* [-.5,.5] */
  const HVX_Vector z = QF2SF(MULSF(f, splat_f(0.6931471805599453f)));
  HVX_Vector p = splat_f(1.0f / 720.f);                                        /* Taylor e^z, degree 6 */
  p = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(MULSF(p, z)), splat_f(1.0f / 120.f)));
  p = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(MULSF(p, z)), splat_f(1.0f / 24.f)));
  p = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(MULSF(p, z)), splat_f(1.0f / 6.f)));
  p = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(MULSF(p, z)), splat_f(0.5f)));
  p = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(MULSF(p, z)), splat_f(1.0f)));
  p = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(MULSF(p, z)), splat_f(1.0f)));
  const HVX_Vector ni = Q6_Vw_vsub_VwVw(t, Q6_V_vsplat_R(0x4B400000));          /* integer n */
  return Q6_Vw_vadd_VwVw(p, Q6_Vw_vasl_VwR(ni, 23));
}
static inline HVX_Vector recip_sf(HVX_Vector d) {  /* d > 0 */
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C7), d);
  const HVX_Vector two = splat_f(2.0f);
  for (int it = 0; it < 3; it++) y = QF2SF(MULSF(y, QF2SF(Q6_Vqf32_vsub_VsfVsf(two, QF2SF(MULSF(d, y))))));
  return y;
}

void k_gelu(uint16_t *t16, size_t nvec) {
  HVX_Vector *v = (HVX_Vector *)t16;
  const HVX_Vector c3 = splat_f(0.044715f), k = splat_f(-1.5957691216057308f /* -2*sqrt(2/pi) */), one = splat_f(1.0f);
  for (size_t i = 0; i < nvec; i++) {
    HVX_VectorPair w = widen_hf(v[i]); HVX_Vector r[2];
    for (int h = 0; h < 2; h++) {
      const HVX_Vector x = QF2SF(h ? Q6_V_hi_W(w) : Q6_V_lo_W(w));
      const HVX_Vector x2 = QF2SF(MULSF(x, x));
      const HVX_Vector inner = QF2SF(MULSF(x, QF2SF(Q6_Vqf32_vadd_VsfVsf(one, QF2SF(MULSF(x2, c3))))));  /* x + .044715 x^3 */
      const HVX_Vector a = QF2SF(MULSF(inner, k));                                                           /* -2u */
      const HVX_Vector e = exp_sf(Q6_Vsf_vmin_VsfVsf(a, splat_f(30.f)));
      const HVX_Vector d = QF2SF(Q6_Vqf32_vadd_VsfVsf(one, e)) ;
      r[h] = MULSF(x, recip_sf(d));
    }
    v[i] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(r[1], r[0]));
  }
}

void k_softmax(const uint16_t *S, uint16_t *P, int nkb, float *R) {
  const HVX_Vector *sv = (const HVX_Vector *)S; HVX_Vector *pv = (HVX_Vector *)P;
  for (int p = 0; p < 16; p++) {
    HVX_Vector ma = splat_f(-3e38f), mb = ma;
    for (int cb = 0; cb < nkb; cb++) {
      HVX_VectorPair w = widen_hf(sv[cb * 16 + p]);
      ma = Q6_Vsf_vmax_VsfVsf(ma, QF2SF(Q6_V_lo_W(w))); mb = Q6_Vsf_vmax_VsfVsf(mb, QF2SF(Q6_V_hi_W(w)));
    }
    for (int r = 64; r >= 4; r >>= 1) { ma = Q6_Vsf_vmax_VsfVsf(ma, Q6_V_vror_VR(ma, r)); mb = Q6_Vsf_vmax_VsfVsf(mb, Q6_V_vror_VR(mb, r)); }
    HVX_Vector sa = Q6_V_vzero(), sb = Q6_V_vzero();
    for (int cb = 0; cb < nkb; cb++) {
      HVX_VectorPair w = widen_hf(sv[cb * 16 + p]);
      const HVX_Vector ea = exp_sf(QF2SF(Q6_Vqf32_vsub_VsfVsf(QF2SF(Q6_V_lo_W(w)), ma))), eb = exp_sf(QF2SF(Q6_Vqf32_vsub_VsfVsf(QF2SF(Q6_V_hi_W(w)), mb)));
      sa = Q6_Vqf32_vadd_Vqf32Vsf(sa, ea); sb = Q6_Vqf32_vadd_Vqf32Vsf(sb, eb);
      pv[cb * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(eb, Q6_V_vzero()), Q6_Vqf32_vadd_VsfVsf(ea, Q6_V_vzero())));
    }
    ((HVX_Vector *)R)[2 * p] = recip_sf(QF2SF(lane_sum(sa)));
    ((HVX_Vector *)R)[2 * p + 1] = recip_sf(QF2SF(lane_sum(sb)));
  }
}

void k_rowscale(uint16_t *O, int ntile, const float *R) {
  HVX_Vector *v = (HVX_Vector *)O;
  for (int t = 0; t < ntile; t++) for (int p = 0; p < 16; p++) {
    HVX_VectorPair w = widen_hf(v[t * 16 + p]);
    v[t * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(MULSF(QF2SF(Q6_V_hi_W(w)), ((const HVX_Vector *)R)[2 * p + 1]), MULSF(QF2SF(Q6_V_lo_W(w)), ((const HVX_Vector *)R)[2 * p])));
  }
}

void k_rmsrope(uint16_t *x, const float *gain, const float *cosT, const float *sinT, int rope) {
  /* x: 64 tiles (cb) of one slab; heads = 4 consecutive cb. cosT/sinT: 2 tiles (cb j of the first half) of 32 rows x 32 fp32 */
  const HVX_Vector inv = splat_f(1.0f / 128.f), eps = splat_f(1e-6f);
  HVX_Vector *xv = (HVX_Vector *)x;
  pf(x, (size_t)NCB * 2048);
  for (int hd = 0; hd < NHEAD; hd++) {
    for (int p = 0; p < 16; p++) {
      HVX_Vector a[4], b[4], qa = Q6_V_vzero(), qb = Q6_V_vzero();
      for (int j = 0; j < 4; j++) {
        HVX_VectorPair w = widen_hf(xv[(hd * 4 + j) * 16 + p]);
        a[j] = QF2SF(Q6_V_lo_W(w)); b[j] = QF2SF(Q6_V_hi_W(w));
        qa = Q6_Vqf32_vadd_Vqf32Vqf32(qa, MULSF(a[j], a[j])); qb = Q6_Vqf32_vadd_Vqf32Vqf32(qb, MULSF(b[j], b[j]));
      }
      const HVX_Vector ra = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(qa)), inv), eps))), rb = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(qb)), inv), eps)));
      HVX_Vector na[4], nb[4];
      for (int j = 0; j < 4; j++) {
        const HVX_Vector g = *(const HVX_Vector *)(gain + j * 32);
        na[j] = QF2SF(MULSF(QF2SF(MULSF(a[j], ra)), g)); nb[j] = QF2SF(MULSF(QF2SF(MULSF(b[j], rb)), g));
      }
      if (rope) {
        for (int j = 0; j < 2; j++) {
          const HVX_Vector ca = *(const HVX_Vector *)(cosT + j * 1024 + (2 * p) * 32), cb2 = *(const HVX_Vector *)(cosT + j * 1024 + (2 * p + 1) * 32);
          const HVX_Vector sa = *(const HVX_Vector *)(sinT + j * 1024 + (2 * p) * 32), sb = *(const HVX_Vector *)(sinT + j * 1024 + (2 * p + 1) * 32);
          /* row a: lo = x[d]*c - x[d+64]*s ; hi = x[d]*s + x[d+64]*c   (x[d]=na[j], x[d+64]=na[j+2]) */
          const HVX_Vector u = na[j], v = na[j + 2], u2 = nb[j], v2 = nb[j + 2];
          na[j] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(u, ca), MULSF(v, sa)));
          na[j + 2] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(u, sa), MULSF(v, ca)));
          nb[j] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(u2, cb2), MULSF(v2, sb)));
          nb[j + 2] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(u2, sb), MULSF(v2, cb2)));
        }
      }
      for (int j = 0; j < 4; j++) xv[(hd * 4 + j) * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(nb[j], Q6_V_vzero()), Q6_Vqf32_vadd_VsfVsf(na[j], Q6_V_vzero())));
    }
  }
}

void k_rmsrope_head(uint16_t *x4, const float *gain, const float *cosT, const float *sinT, int rope) {   /* one head = 4 contiguous tiles */
  /* x: 64 tiles (cb) of one slab; heads = 4 consecutive cb. cosT/sinT: 2 tiles (cb j of the first half) of 32 rows x 32 fp32 */
  const HVX_Vector inv = splat_f(1.0f / 128.f), eps = splat_f(1e-6f);
  HVX_Vector *xv = (HVX_Vector *)x4;
  for (int hd = 0; hd < 1; hd++) {
    for (int p = 0; p < 16; p++) {
      HVX_Vector a[4], b[4], qa = Q6_V_vzero(), qb = Q6_V_vzero();
      for (int j = 0; j < 4; j++) {
        HVX_VectorPair w = widen_hf(xv[(hd * 4 + j) * 16 + p]);
        a[j] = QF2SF(Q6_V_lo_W(w)); b[j] = QF2SF(Q6_V_hi_W(w));
        qa = Q6_Vqf32_vadd_Vqf32Vqf32(qa, MULSF(a[j], a[j])); qb = Q6_Vqf32_vadd_Vqf32Vqf32(qb, MULSF(b[j], b[j]));
      }
      HVX_Vector na[4], nb[4];
      if (gain) {   /* gain == NULL: rope only (ViT q/k have no norm) */
        const HVX_Vector ra = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(qa)), inv), eps))), rb = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(qb)), inv), eps)));
        for (int j2 = 0; j2 < 4; j2++) {
          const HVX_Vector g = *(const HVX_Vector *)(gain + j2 * 32);
          na[j2] = QF2SF(MULSF(QF2SF(MULSF(a[j2], ra)), g)); nb[j2] = QF2SF(MULSF(QF2SF(MULSF(b[j2], rb)), g));
        }
      } else for (int j2 = 0; j2 < 4; j2++) { na[j2] = a[j2]; nb[j2] = b[j2]; }
      if (rope) {
        for (int j = 0; j < 2; j++) {
          const HVX_Vector ca = *(const HVX_Vector *)(cosT + j * 1024 + (2 * p) * 32), cb2 = *(const HVX_Vector *)(cosT + j * 1024 + (2 * p + 1) * 32);
          const HVX_Vector sa = *(const HVX_Vector *)(sinT + j * 1024 + (2 * p) * 32), sb = *(const HVX_Vector *)(sinT + j * 1024 + (2 * p + 1) * 32);
          /* row a: lo = x[d]*c - x[d+64]*s ; hi = x[d]*s + x[d+64]*c   (x[d]=na[j], x[d+64]=na[j+2]) */
          const HVX_Vector u = na[j], v = na[j + 2], u2 = nb[j], v2 = nb[j + 2];
          na[j] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(u, ca), MULSF(v, sa)));
          na[j + 2] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(u, sa), MULSF(v, ca)));
          nb[j] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(u2, cb2), MULSF(v2, sb)));
          nb[j + 2] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(u2, sb), MULSF(v2, cb2)));
        }
      }
      for (int j = 0; j < 4; j++) xv[(hd * 4 + j) * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(nb[j], Q6_V_vzero()), Q6_Vqf32_vadd_VsfVsf(na[j], Q6_V_vzero())));
    }
  }
}

#include <math.h>
void tab_init(uint16_t *et, uint16_t *t2) {
  for (int b = 0; b < 32768; b++) {
    uint16_t hb = (uint16_t)b; __fp16 hx; memcpy(&hx, &hb, 2);
    float a = (float)hx; __fp16 e, t;
    if (b >= 0x7C00) { e = 0; t = 0; }
    else { e = (__fp16)expf(-a); t = (__fp16)(a * 0.5 * erfc(a / sqrt(2.0))); }
    memcpy(&et[b], &e, 2); memcpy(&t2[b], &t, 2);
  }
}
void tab_init_gelut(uint16_t *t2) {   /* t2[b] = |x| * Phi_tanh(-|x|), Phi_tanh(-a) = 0.5 (1 - tanh(c (a + 0.044715 a^3))): GELU(x) = max(x,0) - t2[|x|] for the tanh GELU (ViT) */
  for (int b = 0; b < 32768; b++) {
    uint16_t hb = (uint16_t)b; __fp16 hx; memcpy(&hx, &hb, 2); float a = (float)hx; __fp16 t;
    if (b >= 0x7C00) t = 0; else t = (__fp16)(a * 0.5 * (1.0 - tanh(0.7978845608028654 * (a + 0.044715 * a * a * a))));
    memcpy(&t2[b], &t, 2);
  }
}
static inline HVX_Vector gather_tab(const uint16_t *tab, HVX_Vector offs, HVX_Vector *scratch) {
  Q6_vgather_ARMVh(scratch, (uint32_t)tab, TAB_BYTES - 1, offs);
  return *scratch;
}
void k_gelu2(uint16_t *t16, size_t nvec, const uint16_t *t2, HVX_Vector *scratch) {   /* scratch: 8 vectors */
  HVX_Vector *v = (HVX_Vector *)t16;
  const HVX_Vector mask = Q6_Vh_vsplat_R(0x7FFF), zero = Q6_V_vzero();
  size_t i = 0;
  for (; i + 8 <= nvec; i += 8) {
    for (int u = 0; u < 8; u++) Q6_vgather_ARMVh(scratch + u, (uint32_t)t2, TAB_BYTES - 1, Q6_Vh_vasl_VhR(Q6_V_vand_VV(v[i + u], mask), 1));
    for (int u = 0; u < 8; u++) v[i + u] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(Q6_Vhf_vmax_VhfVhf(v[i + u], zero), scratch[u]));
  }
  for (; i < nvec; i++) {
    const HVX_Vector x = v[i], t = gather_tab(t2, Q6_Vh_vasl_VhR(Q6_V_vand_VV(x, mask), 1), scratch);
    v[i] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(Q6_Vhf_vmax_VhfVhf(x, zero), t));
  }
}
void k_softmax2p(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *et, HVX_Vector *scratch, int p0, int p1, const HVX_Vector *kmask, int kst) {
  /* scratch: 8 gather destination vectors per caller; gathers are issued 8 deep before the results are read back */
  const HVX_Vector *sv = (const HVX_Vector *)S; HVX_Vector *pv = (HVX_Vector *)P;
  const HVX_Vector mask = Q6_Vh_vsplat_R(0x7FFF);
  const HVX_Vector neginf = Q6_Vh_vsplat_R(0xFBFF);
  for (int p = p0; p < p1; p++) {
    if (kmask) { HVX_Vector *lv = (HVX_Vector *)&sv[(size_t)(nkb - 1) * 16 + p]; const HVX_Vector km_ = kmask[kst ? p : 0]; *lv = Q6_V_vor_VV(Q6_V_vand_VV(*lv, Q6_V_vnot_V(km_)), Q6_V_vand_VV(neginf, km_)); }
    HVX_Vector m = sv[p];
    for (int cb = 1; cb < nkb; cb++) m = Q6_Vhf_vmax_VhfVhf(m, sv[cb * 16 + p]);
    for (int r = 64; r >= 4; r >>= 1) m = Q6_Vhf_vmax_VhfVhf(m, Q6_V_vror_VR(m, r));
    int cb = 0;
    for (; cb + 8 <= nkb; cb += 8) {
      for (int u = 0; u < 8; u++) {
        const HVX_Vector d = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(sv[(cb + u) * 16 + p], m));
        Q6_vgather_ARMVh(scratch + u, (uint32_t)et, TAB_BYTES - 1, Q6_Vh_vasl_VhR(Q6_V_vand_VV(d, mask), 1));
      }
      for (int u = 0; u < 8; u++) pv[(cb + u) * 16 + p] = scratch[u];
    }
    for (; cb < nkb; cb++) {
      const HVX_Vector d = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(sv[cb * 16 + p], m));
      pv[cb * 16 + p] = gather_tab(et, Q6_Vh_vasl_VhR(Q6_V_vand_VV(d, mask), 1), scratch);
    }
  }
}
/* like k_softmax2p, but np partially visible key tiles pk[i] get the per-row-pair mask set pm[pmi[i]*16+p] (0xFFFF lanes = masked) */
void k_softmax2pm(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *et, HVX_Vector *scratch, int p0, int p1, const int *pk, const int *pmi, const HVX_Vector *pm, int np) {
  /* scratch: 8 gather destination vectors per caller; gathers are issued 8 deep before the results are read back */
  const HVX_Vector *sv = (const HVX_Vector *)S; HVX_Vector *pv = (HVX_Vector *)P;
  const HVX_Vector mask = Q6_Vh_vsplat_R(0x7FFF);
  const HVX_Vector neginf = Q6_Vh_vsplat_R(0xFBFF);
  for (int p = p0; p < p1; p++) {
    for (int pi = 0; pi < np; pi++) { HVX_Vector *lv = (HVX_Vector *)&sv[(size_t)pk[pi] * 16 + p]; const HVX_Vector km_ = pm[pmi[pi] * 16 + p]; *lv = Q6_V_vor_VV(Q6_V_vand_VV(*lv, Q6_V_vnot_V(km_)), Q6_V_vand_VV(neginf, km_)); }
    HVX_Vector m = sv[p];
    for (int cb = 1; cb < nkb; cb++) m = Q6_Vhf_vmax_VhfVhf(m, sv[cb * 16 + p]);
    for (int r = 64; r >= 4; r >>= 1) m = Q6_Vhf_vmax_VhfVhf(m, Q6_V_vror_VR(m, r));
    int cb = 0;
    for (; cb + 8 <= nkb; cb += 8) {
      for (int u = 0; u < 8; u++) {
        const HVX_Vector d = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(sv[(cb + u) * 16 + p], m));
        Q6_vgather_ARMVh(scratch + u, (uint32_t)et, TAB_BYTES - 1, Q6_Vh_vasl_VhR(Q6_V_vand_VV(d, mask), 1));
      }
      for (int u = 0; u < 8; u++) pv[(cb + u) * 16 + p] = scratch[u];
    }
    for (; cb < nkb; cb++) {
      const HVX_Vector d = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(sv[cb * 16 + p], m));
      pv[cb * 16 + p] = gather_tab(et, Q6_Vh_vasl_VhR(Q6_V_vand_VV(d, mask), 1), scratch);
    }
  }
}
void k_softmax2(const uint16_t *S, uint16_t *P, int nkb, const uint16_t *et, HVX_Vector *scratch) { k_softmax2p(S, P, nkb, et, scratch, 0, 16, 0, 0); }

void k_rowscale_sum(uint16_t *O, int ntile, const uint16_t *sumtile) {
  HVX_Vector *v = (HVX_Vector *)O; const HVX_Vector *sm = (const HVX_Vector *)sumtile;
  for (int p = 0; p < 16; p++) {
    HVX_VectorPair ws = widen_hf(sm[p]);
    const HVX_Vector ra = recip_sf(QF2SF(Q6_V_lo_W(ws))), rb = recip_sf(QF2SF(Q6_V_hi_W(ws)));
    for (int t = 0; t < ntile; t++) {
      HVX_VectorPair w = widen_hf(v[t * 16 + p]);
      v[t * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(MULSF(QF2SF(Q6_V_hi_W(w)), rb), MULSF(QF2SF(Q6_V_lo_W(w)), ra)));
    }
  }
}

void k_transpose_tile(const uint16_t *src, uint16_t *dst) {
  static const uint16_t m1a[64] __attribute__((aligned(128))) = {0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0,
                                                                  0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0, 0,0xFFFF,0,0};
  static const uint16_t m2a[64] __attribute__((aligned(128))) = {0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0,
                                                                  0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0, 0,0,0xFFFF,0};
  const HVX_Vector m1 = *(const HVX_Vector *)m1a, m2 = *(const HVX_Vector *)m2a, keep = Q6_V_vnot_V(Q6_V_vor_VV(m1, m2));
  const HVX_Vector *sv = (const HVX_Vector *)src; HVX_Vector v[16];
  for (int p = 0; p < 16; p++) {
    const HVX_Vector x = sv[p];
    v[p] = Q6_V_vor_VV(Q6_V_vand_VV(x, keep), Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(x, 2), m1), Q6_V_vand_VV(Q6_V_vror_VR(x, 126), m2)));
  }
  for (int st = 0; st < 4; st++) {
    const int d = 1 << st, R = 8 << st;
    for (int i = 0; i < 16; i++) if (!(i & d)) {
      HVX_VectorPair w = Q6_W_vshuff_VVR(v[i + d], v[i], -R);
      v[i] = Q6_V_lo_W(w); v[i + d] = Q6_V_hi_W(w);
    }
  }
  static const uint8_t br[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  HVX_Vector *dv = (HVX_Vector *)dst;
  for (int j = 0; j < 16; j++) dv[br[j]] = v[j];
}

void k_resadd_tile(float *xs, const uint16_t *t16, const float *gate) {
  const HVX_Vector g = *(const HVX_Vector *)gate; const HVX_Vector *ts = (const HVX_Vector *)t16;
  for (int p = 0; p < 16; p++) {
    const HVX_VectorPair w = widen_hf(ts[p]);
    HVX_Vector *xa = (HVX_Vector *)(xs + (2 * p) * 32), *xb = (HVX_Vector *)(xs + (2 * p + 1) * 32);
    *xa = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(Q6_V_lo_W(w)), g), *xa));
    *xb = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(Q6_V_hi_W(w)), g), *xb));
  }
}

void k_resadd2_tile(float *xs, const uint16_t *t16, const uint16_t *prev, const float *gate) {   /* x += gate*(t + prev) */
  const HVX_Vector g = *(const HVX_Vector *)gate; const HVX_Vector *ts = (const HVX_Vector *)t16, *pv = (const HVX_Vector *)prev;
  for (int p = 0; p < 16; p++) {
    const HVX_VectorPair w = widen_hf(ts[p]), q = widen_hf(pv[p]);
    HVX_Vector *xa = (HVX_Vector *)(xs + (2 * p) * 32), *xb = (HVX_Vector *)(xs + (2 * p + 1) * 32);
    const HVX_Vector sa = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(Q6_V_lo_W(w)), QF2SF(Q6_V_lo_W(q)))), sb = QF2SF(Q6_Vqf32_vadd_VsfVsf(QF2SF(Q6_V_hi_W(w)), QF2SF(Q6_V_hi_W(q))));
    *xa = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(sa, g), *xa));
    *xb = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(sb, g), *xb));
  }
}

/* ---------------- MOSS-Image ---------------- */
void k_rms_prep(HVX_Vector *M, HVX_Vector *N, const float *w, const float *shift, const float *scale) {
  const HVX_Vector vone = splat_f(1.0f);
  for (int cb = 0; cb < NCB; cb++) {
    const HVX_Vector s1 = QF2SF(Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)(scale + cb * 32), vone));
    M[cb] = QF2SF(MULSF(s1, *(const HVX_Vector *)(w + cb * 32)));
    N[cb] = *(const HVX_Vector *)(shift + cb * 32);
  }
}
void k_rmsmod_s(const float *x32, uint16_t *h16, int nslab, const HVX_Vector *M, const HVX_Vector *N, float eps) {
  const HVX_Vector vinv = splat_f(1.0f / DIM), veps = splat_f(eps);
  for (int s = 0; s < nslab; s++) {
    const float *xs = x32 + (size_t)s * NCB * 1024;
    pf(xs, (size_t)NCB * 4096);
    HVX_Vector *hs = (HVX_Vector *)(h16 + (size_t)s * NCB * 1024);
    for (int p = 0; p < 16; p++) {
      HVX_Vector qa = Q6_V_vzero(), qb = Q6_V_vzero();
      for (int cb = 0; cb < NCB; cb++) {
        const HVX_Vector a = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p) * 32), b = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p + 1) * 32);
        qa = Q6_Vqf32_vadd_Vqf32Vqf32(qa, MULSF(a, a)); qb = Q6_Vqf32_vadd_Vqf32Vqf32(qb, MULSF(b, b));
      }
      const HVX_Vector va = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(qa)), vinv), veps)), vb = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(qb)), vinv), veps));
      const HVX_Vector ra = rsqrt_sf(va), rb = rsqrt_sf(vb);
      for (int cb = 0; cb < NCB; cb++) {
        const HVX_Vector a = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p) * 32), b = *(const HVX_Vector *)(xs + cb * 1024 + (2 * p + 1) * 32);
        const HVX_Vector ya = QF2SF(MULSF(a, ra)), yb = QF2SF(MULSF(b, rb));
        const HVX_Vector oa = Q6_Vqf32_vadd_Vqf32Vsf(MULSF(ya, M[cb]), N[cb]), ob = Q6_Vqf32_vadd_Vqf32Vsf(MULSF(yb, M[cb]), N[cb]);
        hs[cb * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(ob, oa));
      }
    }
  }
}
void tab_init_situ(uint16_t *tabs) {
  for (int t = 0; t < 3; t++) for (int b = 0; b < 32768; b++) {
    uint16_t hb = (uint16_t)b; __fp16 hx; memcpy(&hx, &hb, 2); double a = (double)(float)hx, v;
    if (b >= 0x7C00) v = 0;
    else if (t == 0) v = 25.0 * tanh(a / 25.0);
    else if (t == 1) v = 4.0 * tanh(a / 4.0) / (1.0 + exp(-a));
    else v = 4.0 * tanh(a / 4.0) / (1.0 + exp(a));
    __fp16 h = (__fp16)v; memcpy(&tabs[t * 32768 + b], &h, 2);
  }
}
void k_situ_tile(const uint16_t *up, const uint16_t *gate, uint16_t *out, const uint16_t *tabs, HVX_Vector *scr) {
  const HVX_Vector mask = Q6_Vh_vsplat_R(0x7FFF), sgn = Q6_Vh_vsplat_R(0x8000);
  const uint16_t *T25 = tabs, *Gp = tabs + 32768, *Gn = tabs + 65536;
  const HVX_Vector *uv = (const HVX_Vector *)up, *gv = (const HVX_Vector *)gate; HVX_Vector *ov = (HVX_Vector *)out;
  for (int p = 0; p < 16; p += 4) {
    for (int u = 0; u < 4; u++) {
      const HVX_Vector ou = Q6_Vh_vasl_VhR(Q6_V_vand_VV(uv[p + u], mask), 1), og = Q6_Vh_vasl_VhR(Q6_V_vand_VV(gv[p + u], mask), 1);
      Q6_vgather_ARMVh(scr + 3 * u, (uint32_t)T25, TAB_BYTES - 1, ou);
      Q6_vgather_ARMVh(scr + 3 * u + 1, (uint32_t)Gp, TAB_BYTES - 1, og);
      Q6_vgather_ARMVh(scr + 3 * u + 2, (uint32_t)Gn, TAB_BYTES - 1, og);
    }
    for (int u = 0; u < 4; u++) {
      const HVX_Vector tu = Q6_V_vor_VV(scr[3 * u], Q6_V_vand_VV(uv[p + u], sgn));
      const HVX_Vector neg = Q6_Vh_vasr_VhR(gv[p + u], 15);
      const HVX_Vector gs = Q6_V_vor_VV(Q6_V_vand_VV(neg, Q6_V_vor_VV(scr[3 * u + 2], sgn)), Q6_V_vand_VV(Q6_V_vnot_V(neg), scr[3 * u + 1]));
      ov[p + u] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(tu, gs));
    }
  }
}

void tab_init_swiglu(uint16_t *tabs) {
  for (int t = 0; t < 2; t++) for (int b = 0; b < 32768; b++) {
    uint16_t hb = (uint16_t)b; __fp16 hx; memcpy(&hx, &hb, 2); double a = (double)(float)hx, v;
    if (b >= 0x7C00) v = 0; else v = t == 0 ? a / (1.0 + exp(-a)) : a / (1.0 + exp(a));
    __fp16 h = (__fp16)v; memcpy(&tabs[t * 32768 + b], &h, 2);
  }
}
void k_swiglu_tile(const uint16_t *gate, const uint16_t *up, uint16_t *out, const uint16_t *tabs, HVX_Vector *scr) {
  const HVX_Vector mask = Q6_Vh_vsplat_R(0x7FFF), sgn = Q6_Vh_vsplat_R(0x8000);
  const uint16_t *S1 = tabs, *S2 = tabs + 32768;
  const HVX_Vector *gv = (const HVX_Vector *)gate, *uv = (const HVX_Vector *)up; HVX_Vector *ov = (HVX_Vector *)out;
  for (int p = 0; p < 16; p += 4) {
    for (int u = 0; u < 4; u++) {
      const HVX_Vector og = Q6_Vh_vasl_VhR(Q6_V_vand_VV(gv[p + u], mask), 1);
      Q6_vgather_ARMVh(scr + 2 * u, (uint32_t)S1, TAB_BYTES - 1, og);
      Q6_vgather_ARMVh(scr + 2 * u + 1, (uint32_t)S2, TAB_BYTES - 1, og);
    }
    for (int u = 0; u < 4; u++) {
      const HVX_Vector neg = Q6_Vh_vasr_VhR(gv[p + u], 15);
      const HVX_Vector sl = Q6_V_vor_VV(Q6_V_vand_VV(neg, Q6_V_vor_VV(scr[2 * u + 1], sgn)), Q6_V_vand_VV(Q6_V_vnot_V(neg), scr[2 * u]));
      ov[p + u] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sl, uv[p + u]));
    }
  }
}

/* ===================== Qwen3.5 text-encoder helpers (row-major fp32 domain) ===================== */
#include "pool.h"
void te_t2r(const uint16_t *t, float *rows, int ns, int nct, int idx, int n) {   /* tile layout -> rows[token][nct*32] fp32 */
  const size_t stride = (size_t)nct * 32;
  for (int s = idx; s < ns; s += n) for (int cb = 0; cb < nct; cb++) for (int p = 0; p < 16; p++) {
    const HVX_VectorPair w = widen_hf(((const HVX_Vector *)t)[((size_t)s * nct + cb) * 16 + p]);
    float *r0 = rows + ((size_t)s * 32 + 2 * p) * stride + cb * 32;
    *(HVX_Vector *)r0 = QF2SF(Q6_V_lo_W(w)); *(HVX_Vector *)(r0 + stride) = QF2SF(Q6_V_hi_W(w));
  }
}
void te_r2t(const float *rows, uint16_t *t, int ns, int nct, int idx, int n) {
  const size_t stride = (size_t)nct * 32; const HVX_Vector z = Q6_V_vzero();
  for (int s = idx; s < ns; s += n) for (int cb = 0; cb < nct; cb++) for (int p = 0; p < 16; p++) {
    const float *r0 = rows + ((size_t)s * 32 + 2 * p) * stride + cb * 32;
    const HVX_Vector lo = Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)r0, z), hi = Q6_Vqf32_vadd_VsfVsf(*(const HVX_Vector *)(r0 + stride), z);
    ((HVX_Vector *)t)[((size_t)s * nct + cb) * 16 + p] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
  }
}
static inline HVX_Vector silu_sf(HVX_Vector x) {   /* x * sigmoid(x) */
  const HVX_Vector e = exp_sf(Q6_Vsf_vmin_VsfVsf(QF2SF(MULSF(x, splat_f(-1.0f))), splat_f(30.f)));
  return QF2SF(MULSF(x, recip_sf(QF2SF(Q6_Vqf32_vadd_VsfVsf(e, splat_f(1.0f))))));
}
/* causal depthwise conv (taps w[tap*C + c], tap 3 multiplies x[t]) + silu: in rows[T][C] -> out rows[T][C]; channel vectors split over workers */
void te_conv_silu(const float *in, float *out, const float *w, int T, int C, int idx, int n) {
  for (int cv = idx; cv < C / 32; cv += n) {
    const HVX_Vector w0 = *(const HVX_Vector *)(w + cv * 32), w1 = *(const HVX_Vector *)(w + C + cv * 32), w2 = *(const HVX_Vector *)(w + 2 * C + cv * 32), w3 = *(const HVX_Vector *)(w + 3 * C + cv * 32);
    HVX_Vector x1 = Q6_V_vzero(), x2 = x1, x3 = x1;
    for (int t = 0; t < T; t++) {
      const HVX_Vector x0 = *(const HVX_Vector *)(in + (size_t)t * C + cv * 32);
      HVX_Vector acc = MULSF(w3, x0);
      acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, MULSF(w2, x1)); acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, MULSF(w1, x2)); acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, MULSF(w0, x3));
      *(HVX_Vector *)(out + (size_t)t * C + cv * 32) = silu_sf(QF2SF(acc));
      x3 = x2; x2 = x1; x1 = x0;
    }
  }
}
static HVX_Vector g_S[8][512] __attribute__((aligned(128)));   /* delta-net state of one head: [128 k][4 vectors of 32 v] per worker */
static inline HVX_Vector splat_of(float f) { return splat_f(f); }
/* Gated delta net for all value heads (worker idx handles heads idx, idx+n, ...). qkv: rows[T][8192] after conv+silu; ab: rows[T][64] (alpha 0..31, beta 32..63); z: rows[T][4096]; out: rows[T][4096] (norm*silu(z) applied) */
void te_delta(const float *qkv, const float *ab, const float *z, float *out, const float *A, const float *dtb, const float *nw, int T, int idx, int n) {
  HVX_Vector *S = g_S[idx];
  for (int h = idx; h < 32; h += n) {
    const int kh = h & 15;
    for (int i = 0; i < 512; i++) S[i] = Q6_V_vzero();
    for (int t = 0; t < T; t++) {
      const float *qp = qkv + (size_t)t * 8192 + kh * 128, *kp = qkv + (size_t)t * 8192 + 2048 + kh * 128, *vp = qkv + (size_t)t * 8192 + 4096 + h * 128;
      HVX_Vector qv[4], kv4[4], vv[4];
      HVX_Vector sq = Q6_V_vzero(), sk = Q6_V_vzero();
      for (int j = 0; j < 4; j++) {
        qv[j] = *(const HVX_Vector *)(qp + 32 * j); kv4[j] = *(const HVX_Vector *)(kp + 32 * j); vv[j] = *(const HVX_Vector *)(vp + 32 * j);
        sq = Q6_Vqf32_vadd_Vqf32Vqf32(sq, MULSF(qv[j], qv[j])); sk = Q6_Vqf32_vadd_Vqf32Vqf32(sk, MULSF(kv4[j], kv4[j]));
      }
      const HVX_Vector eps = splat_f(1e-6f);
      const HVX_Vector rq = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(lane_sum(sq), eps))), rk = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(lane_sum(sk), eps)));
      const HVX_Vector rqs = QF2SF(MULSF(rq, splat_f(0.08838834764831845f)));   /* 1/sqrt(128) */
      float kn[128] __attribute__((aligned(128))), qn[128] __attribute__((aligned(128)));
      for (int j = 0; j < 4; j++) { *(HVX_Vector *)(kn + 32 * j) = QF2SF(MULSF(kv4[j], rk)); *(HVX_Vector *)(qn + 32 * j) = QF2SF(MULSF(qv[j], rqs)); }
      const float a = ab[(size_t)t * 64 + h], b = ab[(size_t)t * 64 + 32 + h];
      const float sp = a + dtb[h]; const float spl = sp > 20.f ? sp : logf(1.f + expf(sp));
      const float decay = expf(spl * A[h]), beta = 1.f / (1.f + expf(-b));
      const HVX_Vector vdec = splat_f(decay);
      HVX_Vector kvm[4] = {Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero()};
      for (int k = 0; k < 128; k++) {
        const HVX_Vector ks = splat_f(kn[k]);
        for (int j = 0; j < 4; j++) { const HVX_Vector s = QF2SF(MULSF(S[k * 4 + j], vdec)); S[k * 4 + j] = s; kvm[j] = Q6_Vqf32_vadd_Vqf32Vqf32(kvm[j], MULSF(s, ks)); }
      }
      HVX_Vector d[4], o[4];
      const HVX_Vector vb = splat_f(beta);
      for (int j = 0; j < 4; j++) { d[j] = QF2SF(MULSF(QF2SF(Q6_Vqf32_vsub_VsfVsf(vv[j], QF2SF(kvm[j]))), vb)); o[j] = Q6_V_vzero(); }
      for (int k = 0; k < 128; k++) {
        const HVX_Vector ks = splat_f(kn[k]), qs = splat_f(qn[k]);
        for (int j = 0; j < 4; j++) { const HVX_Vector s = QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(ks, d[j]), S[k * 4 + j])); S[k * 4 + j] = s; o[j] = Q6_Vqf32_vadd_Vqf32Vqf32(o[j], MULSF(s, qs)); }
      }
      /* gated rms norm over the 128 values, then * silu(z) */
      HVX_Vector os[4], ss = Q6_V_vzero();
      for (int j = 0; j < 4; j++) { os[j] = QF2SF(o[j]); ss = Q6_Vqf32_vadd_Vqf32Vqf32(ss, MULSF(os[j], os[j])); }
      const HVX_Vector rn = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(ss)), splat_f(1.0f / 128)), eps)));
      for (int j = 0; j < 4; j++) {
        const HVX_Vector zz = *(const HVX_Vector *)(z + (size_t)t * 4096 + h * 128 + 32 * j), w = *(const HVX_Vector *)(nw + 32 * j);
        const HVX_Vector y = QF2SF(MULSF(QF2SF(MULSF(os[j], rn)), w));
        *(HVX_Vector *)(out + (size_t)t * 4096 + h * 128 + 32 * j) = QF2SF(MULSF(y, silu_sf(zz)));
      }
    }
  }
}

static float g_sc[8][4096] __attribute__((aligned(128)));
/* Qwen3.5 full attention, row domain. q: rows[T][16*512] (per head q256|gate256), k: rows[T][2048] (k 0..1023 | v 1024..2047). phase 0 (parallel over tokens): q/k head rms (weight wq/wk, 256) + NEOX rope on 64 dims (cs rows[T][64]: cos 32, sin 32), in place.
   phase 1 (parallel over heads): causal softmax(q.k/16) v, times sigmoid(gate) -> out rows[T][4096] */
void te_attn_prep(float *q, float *k, const float *wq, const float *wk, const float *cs, int T, int idx, int n) {
  const HVX_Vector inv = splat_f(1.0f / 256), eps = splat_f(1e-6f);
  for (int t = idx; t < T; t += n) {
    for (int hh = 0; hh < 20; hh++) {   /* 16 q heads then 4 k heads */
      float *p = hh < 16 ? q + (size_t)t * 8192 + hh * 512 : k + (size_t)t * 2048 + (hh - 16) * 256; const float *w = hh < 16 ? wq : wk;
      HVX_Vector x[8], ss = Q6_V_vzero();
      for (int j = 0; j < 8; j++) { x[j] = *(const HVX_Vector *)(p + 32 * j); ss = Q6_Vqf32_vadd_Vqf32Vqf32(ss, MULSF(x[j], x[j])); }
      const HVX_Vector r = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(ss)), inv), eps)));
      for (int j = 0; j < 8; j++) x[j] = QF2SF(MULSF(QF2SF(MULSF(x[j], r)), *(const HVX_Vector *)(w + 32 * j)));
      const HVX_Vector c = *(const HVX_Vector *)(cs + (size_t)t * 64), s = *(const HVX_Vector *)(cs + (size_t)t * 64 + 32);
      const HVX_Vector a = x[0], b = x[1];
      x[0] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(a, c), MULSF(b, s))); x[1] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(b, c), MULSF(a, s)));
      for (int j = 0; j < 8; j++) *(HVX_Vector *)(p + 32 * j) = x[j];
    }
  }
}
void te_attn(const float *q, const float *kv, float *out, int T, int idx, int n) {
  float *sc = g_sc[idx]; float tmp[32] __attribute__((aligned(128)));
  for (int h = idx; h < 16; h += n) {
    const int g = h >> 2;
    for (int t = 0; t < T; t++) {
      const float *qp = q + (size_t)t * 8192 + h * 512; HVX_Vector qv[8];
      for (int j = 0; j < 8; j++) qv[j] = *(const HVX_Vector *)(qp + 32 * j);
      float m = -1e30f;
      for (int j = 0; j <= t; j++) {
        const float *kp = kv + (size_t)j * 2048 + g * 256; HVX_Vector acc = Q6_V_vzero();
        for (int i = 0; i < 8; i++) acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, MULSF(qv[i], *(const HVX_Vector *)(kp + 32 * i)));
        *(HVX_Vector *)tmp = QF2SF(lane_sum(acc)); const float s = tmp[0] * 0.0625f; sc[j] = s; if (s > m) m = s;
      }
      float sum = 0.f; for (int j = 0; j <= t; j++) { sc[j] = expf(sc[j] - m); sum += sc[j]; }
      const float rs = 1.f / sum;
      HVX_Vector o[8]; for (int i = 0; i < 8; i++) o[i] = Q6_V_vzero();
      for (int j = 0; j <= t; j++) {
        const HVX_Vector ps = splat_f(sc[j] * rs); const float *vp = kv + (size_t)j * 2048 + 1024 + g * 256;
        for (int i = 0; i < 8; i++) o[i] = Q6_Vqf32_vadd_Vqf32Vqf32(o[i], MULSF(*(const HVX_Vector *)(vp + 32 * i), ps));
      }
      for (int i = 0; i < 8; i++) {
        const HVX_Vector gt = *(const HVX_Vector *)(qp + 256 + 32 * i);
        const HVX_Vector e = exp_sf(Q6_Vsf_vmin_VsfVsf(QF2SF(MULSF(gt, splat_f(-1.0f))), splat_f(30.f)));
        const HVX_Vector sg = recip_sf(QF2SF(Q6_Vqf32_vadd_VsfVsf(e, splat_f(1.0f))));
        *(HVX_Vector *)(out + (size_t)t * 4096 + h * 256 + 32 * i) = QF2SF(MULSF(QF2SF(o[i]), sg));
      }
    }
  }
}

/* ===== Qwen3-VL text encoder attention (hd128, 32 q heads / 8 kv heads, full NEOX rope over 128 dims), row domain ===== */
/* q rows[T][4096], kv rows[T][2048] (k 0..1023 | v 1024..2047). phase 0 (parallel over tokens): per-head RMSNorm(w[128]) + rotate-half rope, cs rows[T][128] = cos[64] | sin[64] */
void te_attn_prep3(float *q, float *k, const float *wq, const float *wk, const float *cs, int T, int idx, int n) {
  const HVX_Vector inv = splat_f(1.0f / 128), eps = splat_f(1e-6f);
  for (int t = idx; t < T; t += n) {
    const HVX_Vector c0 = *(const HVX_Vector *)(cs + (size_t)t * 128), c1 = *(const HVX_Vector *)(cs + (size_t)t * 128 + 32), s0 = *(const HVX_Vector *)(cs + (size_t)t * 128 + 64), s1 = *(const HVX_Vector *)(cs + (size_t)t * 128 + 96);
    for (int hh = 0; hh < 40; hh++) {   /* 32 q heads then 8 k heads */
      float *p = hh < 32 ? q + (size_t)t * 4096 + hh * 128 : k + (size_t)t * 2048 + (hh - 32) * 128; const float *w = hh < 32 ? wq : wk;
      HVX_Vector x[4], ss = Q6_V_vzero();
      for (int j = 0; j < 4; j++) { x[j] = *(const HVX_Vector *)(p + 32 * j); ss = Q6_Vqf32_vadd_Vqf32Vqf32(ss, MULSF(x[j], x[j])); }
      const HVX_Vector r = rsqrt_sf(QF2SF(Q6_Vqf32_vadd_Vqf32Vsf(MULSF(QF2SF(lane_sum(ss)), inv), eps)));
      for (int j = 0; j < 4; j++) x[j] = QF2SF(MULSF(QF2SF(MULSF(x[j], r)), *(const HVX_Vector *)(w + 32 * j)));
      const HVX_Vector a0 = x[0], a1 = x[1], b0 = x[2], b1 = x[3];
      x[0] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(a0, c0), MULSF(b0, s0))); x[1] = QF2SF(Q6_Vqf32_vsub_Vqf32Vqf32(MULSF(a1, c1), MULSF(b1, s1)));
      x[2] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(b0, c0), MULSF(a0, s0))); x[3] = QF2SF(Q6_Vqf32_vadd_Vqf32Vqf32(MULSF(b1, c1), MULSF(a1, s1)));
      for (int j = 0; j < 4; j++) *(HVX_Vector *)(p + 32 * j) = x[j];
    }
  }
}
/* causal GQA attention in fp32 (scores reach hundreds in this encoder's attention sinks, fp16 score tiles are not precise enough): out rows[T][4096]; workers split the 32 q heads.
   K of the kv head is transposed into kt ([key block][128 dims][32 keys], per worker scratch of 128*ceil32(T) floats) so one HVX vector holds the scores of 32 keys, and exp runs on vectors. */
void te_attn3(const float *q, const float *kv, float *out, float *ktbase, int T, int idx, int n) {
  const int NB = (T + 31) / 32; float *kt = ktbase + (size_t)idx * 128 * NB * 32; float *sc = g_sc[idx];
  const HVX_Vector vscale = splat_f(0.08838834764831845f), vlo = splat_f(-80.f);
  int cur = -1;
  for (int h = idx; h < 32; h += n) {
    const int g = h >> 2;
    if (g != cur) {
      for (int j = 0; j < NB * 32; j++) { const float *kp = kv + (size_t)j * 2048 + g * 128; float *d = kt + (size_t)(j >> 5) * 4096 + (j & 31);
        if (j < T) for (int c = 0; c < 128; c++) d[c * 32] = kp[c]; else for (int c = 0; c < 128; c++) d[c * 32] = 0.f; }
      cur = g;
    }
    for (int t = 0; t < T; t++) {
      const float *qp = q + (size_t)t * 4096 + h * 128; const int nbk = t / 32 + 1;
      for (int jb = 0; jb < nbk; jb++) {
        HVX_Vector acc = Q6_V_vzero(); const float *kb = kt + (size_t)jb * 4096;
        for (int d = 0; d < 128; d++) acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, MULSF(splat_f(qp[d]), *(const HVX_Vector *)(kb + d * 32)));
        *(HVX_Vector *)(sc + jb * 32) = QF2SF(MULSF(QF2SF(acc), vscale));
      }
      for (int j = t + 1; j < nbk * 32; j++) sc[j] = -1e30f;
      float m = -1e30f; for (int j = 0; j <= t; j++) if (sc[j] > m) m = sc[j];
      const HVX_Vector vm = splat_f(m);
      for (int jb = 0; jb < nbk; jb++) {
        HVX_Vector d = QF2SF(Q6_Vqf32_vsub_VsfVsf(*(const HVX_Vector *)(sc + jb * 32), vm)); d = Q6_Vsf_vmax_VsfVsf(d, vlo);
        *(HVX_Vector *)(sc + jb * 32) = exp_sf(d);
      }
      for (int j = t + 1; j < nbk * 32; j++) sc[j] = 0.f;
      float sum = 0.f; for (int j = 0; j <= t; j++) sum += sc[j];
      const float rs = 1.f / sum; HVX_Vector o[4]; for (int i = 0; i < 4; i++) o[i] = Q6_V_vzero();
      for (int j = 0; j <= t; j++) {
        const HVX_Vector ps = splat_f(sc[j] * rs); const float *vp = kv + (size_t)j * 2048 + 1024 + g * 128;
        for (int i = 0; i < 4; i++) o[i] = Q6_Vqf32_vadd_Vqf32Vqf32(o[i], MULSF(*(const HVX_Vector *)(vp + 32 * i), ps));
      }
      for (int i = 0; i < 4; i++) *(HVX_Vector *)(out + (size_t)t * 4096 + h * 128 + 32 * i) = QF2SF(o[i]);
    }
  }
}
