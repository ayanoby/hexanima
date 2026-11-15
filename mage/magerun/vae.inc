// ------------------------------------------------------------------ Mage-VAE on the CPU (NEON): encoder + decoder at t=0 (adaLN folded), weights in vae.bin (export_vae.py)
// activations are fp32 channel-last [pos][C]; 1x1 convs = gemm_f16, 3x3 convs = im2col + gemm_f16.
static Glue g_vae;
static void vlin(const std::string& nm, const f32* X, int n, int in, int out, f32* Y) { gemm_f16(g_vae.h((nm + ".w").c_str()), X, Y, n, out, in); add_bias(Y, n, out, g_vae.f((nm + ".b").c_str())); }
static void ln_gen(f32* x, int n, int C, const f32* M, const f32* N) {   // LayerNorm over channels (no affine) * M + N, eps 1e-6
  par_for(n, [&](int t) { f32* r = x + (size_t)t * C; double mu = 0, ss = 0; for (int i = 0; i < C; i++) mu += r[i]; mu /= C; for (int i = 0; i < C; i++) { double q = r[i] - mu; ss += q * q; }
    const float rs = 1.f / sqrtf((float)(ss / C) + 1e-6f), m = (float)mu; for (int i = 0; i < C; i++) r[i] = (r[i] - m) * rs * M[i] + N[i]; });
}
static inline float gelu_erf(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118f)); }
static void im2col3(const f32* X, int h, int w, int C, f32* col) {   // [L][C] -> [L][9*C], tap order (ky,kx,c), zero padded
  par_for(h, [&](int y) { for (int x = 0; x < w; x++) { f32* d = col + ((size_t)y * w + x) * 9 * C;
    for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) { const int yy = y + ky - 1, xx = x + kx - 1; f32* o = d + (ky * 3 + kx) * C;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) memset(o, 0, (size_t)C * 4); else memcpy(o, X + ((size_t)yy * w + xx) * C, (size_t)C * 4); } } });
}
static void vconv3(const std::string& nm, const f32* X, int h, int w, int Cin, int Cout, f32* Y) { std::vector<f32> col((size_t)h * w * 9 * Cin); im2col3(X, h, w, Cin, col.data()); vlin(nm, col.data(), h * w, 9 * Cin, Cout, Y); }
static void dw3(const f32* X, f32* Y, int h, int w, int C, const f32* W /*[C][9]*/, const f32* b) {
  std::vector<f32> wt((size_t)9 * C); for (int c = 0; c < C; c++) for (int k = 0; k < 9; k++) wt[(size_t)k * C + c] = W[c * 9 + k];
  par_for(h, [&](int y) { for (int x = 0; x < w; x++) { f32* o = Y + ((size_t)y * w + x) * C; for (int c = 0; c < C; c++) o[c] = b[c];
    for (int ky = 0; ky < 3; ky++) { const int yy = y + ky - 1; if (yy < 0 || yy >= h) continue; for (int kx = 0; kx < 3; kx++) { const int xx = x + kx - 1; if (xx < 0 || xx >= w) continue;
      const f32* s = X + ((size_t)yy * w + xx) * C; const f32* k = &wt[(size_t)(ky * 3 + kx) * C]; for (int c = 0; c < C; c += 4) vst1q_f32(o + c, vfmaq_f32(vld1q_f32(o + c), vld1q_f32(s + c), vld1q_f32(k + c))); } } } });
}
// DiCo block on s [L][C]; mod = [6][C] folded adaLN (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) or null (encoder head block: affine LayerNorms, no gates)
static void dico(const std::string& nm, f32* s, int h, int w, int C, const f32* mod) {
  const int L = h * w, F = 4 * C; std::vector<f32> a((size_t)L * C), b((size_t)L * C), t((size_t)L * std::max(C, F)), M(C), N(C);
  auto prep = [&](int which) { if (mod) { for (int i = 0; i < C; i++) { M[i] = 1.f + mod[(which * 3 + 1) * C + i]; N[i] = mod[(which * 3) * C + i]; } }
    else { const f32* nw = g_vae.f((nm + (which ? ".norm2.w" : ".norm1.w")).c_str()); const f32* nb = g_vae.f((nm + (which ? ".norm2.b" : ".norm1.b")).c_str()); for (int i = 0; i < C; i++) { M[i] = nw[i]; N[i] = nb[i]; } } };
  memcpy(a.data(), s, (size_t)L * C * 4); prep(0); ln_gen(a.data(), L, C, M.data(), N.data());
  vlin(nm + ".c1", a.data(), L, C, C, b.data()); dw3(b.data(), a.data(), h, w, C, g_vae.f((nm + ".dw").c_str()), g_vae.f((nm + ".dwb").c_str()));
  par_for(L, [&](int i) { for (int c = 0; c < C; c++) a[(size_t)i * C + c] = gelu_erf(a[(size_t)i * C + c]); });
  { std::vector<double> m(C, 0.0); for (int i = 0; i < L; i++) for (int c = 0; c < C; c++) m[c] += a[(size_t)i * C + c]; std::vector<f32> p(C), q(C); for (int c = 0; c < C; c++) p[c] = (float)(m[c] / L);
    vlin(nm + ".ca", p.data(), 1, C, C, q.data()); for (int c = 0; c < C; c++) q[c] = 1.f / (1.f + expf(-q[c]));
    par_for(L, [&](int i) { for (int c = 0; c < C; c++) a[(size_t)i * C + c] *= q[c]; }); }
  vlin(nm + ".c3", a.data(), L, C, C, b.data());
  { const f32* g = mod ? mod + 2 * C : nullptr; par_for(L, [&](int i) { for (int c = 0; c < C; c++) s[(size_t)i * C + c] += (g ? g[c] : 1.f) * b[(size_t)i * C + c]; }); }
  memcpy(a.data(), s, (size_t)L * C * 4); prep(1); ln_gen(a.data(), L, C, M.data(), N.data());
  vlin(nm + ".c4", a.data(), L, C, F, t.data()); par_for(L, [&](int i) { for (int c = 0; c < F; c++) t[(size_t)i * F + c] = gelu_erf(t[(size_t)i * F + c]); });
  vlin(nm + ".c5", t.data(), L, F, C, b.data());
  { const f32* g = mod ? mod + 5 * C : nullptr; par_for(L, [&](int i) { for (int c = 0; c < C; c++) s[(size_t)i * C + c] += (g ? g[c] : 1.f) * b[(size_t)i * C + c]; }); }
}
static void gnorm(f32* x, int L, int C, const f32* w, const f32* b, bool silu) {   // GroupNorm(32) (eps 1e-6) on [L][C], optional SiLU after
  const int G = 32, cg = C / G;
  par_for(G, [&](int g) { double mu = 0, ss = 0; for (int i = 0; i < L; i++) for (int c = 0; c < cg; c++) mu += x[(size_t)i * C + g * cg + c]; mu /= (double)L * cg;
    for (int i = 0; i < L; i++) for (int c = 0; c < cg; c++) { double q = x[(size_t)i * C + g * cg + c] - mu; ss += q * q; }
    const float rs = 1.f / sqrtf((float)(ss / ((double)L * cg)) + 1e-6f), m = (float)mu;
    for (int i = 0; i < L; i++) for (int c = 0; c < cg; c++) { f32& v = x[(size_t)i * C + g * cg + c]; v = (v - m) * rs * w[g * cg + c] + b[g * cg + c]; if (silu) v = v / (1.f + expf(-v)); } });
}
static void resnet(const std::string& nm, f32* x, int h, int w, int C) {
  const int L = h * w; std::vector<f32> a((size_t)L * C), b((size_t)L * C); memcpy(a.data(), x, (size_t)L * C * 4);
  gnorm(a.data(), L, C, g_vae.f((nm + ".n1.w").c_str()), g_vae.f((nm + ".n1.b").c_str()), true); vconv3(nm + ".c1", a.data(), h, w, C, C, b.data());
  gnorm(b.data(), L, C, g_vae.f((nm + ".n2.w").c_str()), g_vae.f((nm + ".n2.b").c_str()), true); vconv3(nm + ".c2", b.data(), h, w, C, C, a.data());
  par_for(L, [&](int i) { for (int c = 0; c < C; c++) x[(size_t)i * C + c] += a[(size_t)i * C + c]; });
}
static void attnblk(const std::string& nm, f32* x, int h, int w, int C) {   // GroupNorm + q/k/v 1x1, single head attention inside 32x32 windows (replicate padded), proj + residual
  const int L = h * w; std::vector<f32> hn((size_t)L * C), q((size_t)L * C), k((size_t)L * C), v((size_t)L * C), o((size_t)L * C); memcpy(hn.data(), x, (size_t)L * C * 4);
  gnorm(hn.data(), L, C, g_vae.f((nm + ".n.w").c_str()), g_vae.f((nm + ".n.b").c_str()), false);
  vlin(nm + ".q", hn.data(), L, C, C, q.data()); vlin(nm + ".k", hn.data(), L, C, C, k.data()); vlin(nm + ".v", hn.data(), L, C, C, v.data());
  const int nwy = (h + 31) / 32, nwx = (w + 31) / 32; const float sc = 1.f / sqrtf((float)C);
  par_for(nwy * nwx * 32, [&](int job) {   // job = window * 32 + row-group of 32 query rows
    const int win = job / 32, rg = job % 32, wy = win / nwx, wx = win % nwx; if (rg >= 32) return;
    auto idx = [&](int p) { const int yy = std::min(h - 1, wy * 32 + p / 32), xx = std::min(w - 1, wx * 32 + p % 32); return (size_t)yy * w + xx; };
    std::vector<f32> sbuf(1024);
    for (int qi = rg * 32; qi < rg * 32 + 32; qi++) {   // one query row against the 1024 keys of the window (rows of the padded window map to clamped positions)
      const int qy = wy * 32 + qi / 32, qx = wx * 32 + qi % 32; if (qy >= h || qx >= w) continue;
      const f32* qr = &q[idx(qi) * C]; float mx = -1e30f;
      for (int j = 0; j < 1024; j++) { const f32* kr = &k[idx(j) * C]; float32x4_t a0 = vdupq_n_f32(0), a1 = a0; for (int c = 0; c < C; c += 8) { a0 = vfmaq_f32(a0, vld1q_f32(qr + c), vld1q_f32(kr + c)); a1 = vfmaq_f32(a1, vld1q_f32(qr + c + 4), vld1q_f32(kr + c + 4)); } const float s = vaddvq_f32(vaddq_f32(a0, a1)) * sc; sbuf[j] = s; mx = std::max(mx, s); }
      float sum = 0; for (int j = 0; j < 1024; j++) { sbuf[j] = expf(sbuf[j] - mx); sum += sbuf[j]; } const float inv = 1.f / sum;
      f32* orow = &o[(size_t)((size_t)qy * w + qx) * C]; for (int c = 0; c < C; c++) orow[c] = 0;
      for (int j = 0; j < 1024; j++) { const f32* vr = &v[idx(j) * C]; const float p = sbuf[j] * inv; const float32x4_t pv = vdupq_n_f32(p); for (int c = 0; c < C; c += 4) vst1q_f32(orow + c, vfmaq_f32(vld1q_f32(orow + c), pv, vld1q_f32(vr + c))); } } });
  vlin(nm + ".proj_out", o.data(), L, C, C, hn.data()); par_for(L, [&](int i) { for (int c = 0; c < C; c++) x[(size_t)i * C + c] += hn[(size_t)i * C + c]; });
}
// tokens [L][128] -> rgb [H][W][3] in [-1,1]
static void vae_decode(const f32* tok, int h, int w, f32* rgb) {
  const int L = h * w; double t0 = now(); std::vector<f32> a((size_t)L * 384), cond((size_t)L * 384), s((size_t)L * 384);
  vconv3("d.conv_in", tok, h, w, 128, 384, a.data());
  resnet("d.r0", a.data(), h, w, 384); attnblk("d.a0", a.data(), h, w, 384); resnet("d.r1", a.data(), h, w, 384); attnblk("d.a1", a.data(), h, w, 384); resnet("d.r2", a.data(), h, w, 384);
  gnorm(a.data(), L, 384, g_vae.f("d.norm_out.w"), g_vae.f("d.norm_out.b"), true); vconv3("d.conv_out", a.data(), h, w, 384, 384, cond.data());
  fprintf(stderr, "vae dec cond path %.2fs\n", now() - t0);
  vlin("s", cond.data(), L, 384, 384, s.data());
  for (int i = 0; i < 21; i++) dico("db" + std::to_string(i), s.data(), h, w, 384, g_vae.f(("db" + std::to_string(i) + ".mod").c_str()));
  fprintf(stderr, "vae dec blocks %.2fs\n", now() - t0);
  // pixel stage in chunks of positions
  const f32 *xec = g_vae.f("xe.c"), *flw = g_vae.f("fl.n"); const int CH = 64; const int W = w * 16;
  std::vector<f32> wf(32 * 32), win(32 * 32), wm0[3], wm2[3], wad[3], lnw[3], lnb[3], b_in(32), b_m0[3], b_m2[3], b_ad[3], wl(3 * 32), bl(3);
  { const __fp16* h16 = g_vae.h("xe.wf"); for (int i = 0; i < 1024; i++) wf[i] = (float)h16[i]; h16 = g_vae.h("dn.in.w"); for (int i = 0; i < 1024; i++) win[i] = (float)h16[i]; memcpy(b_in.data(), g_vae.f("dn.in.b"), 128);
    for (int r = 0; r < 3; r++) { const std::string p = "dn" + std::to_string(r); wm0[r].resize(1024); wm2[r].resize(1024); wad[r].resize(96 * 32); b_m0[r].resize(32); b_m2[r].resize(32); b_ad[r].resize(96);
      const __fp16 *a0 = g_vae.h((p + ".m0.w").c_str()), *a2 = g_vae.h((p + ".m2.w").c_str()), *aa = g_vae.h((p + ".ad.w").c_str()); for (int i = 0; i < 1024; i++) { wm0[r][i] = (float)a0[i]; wm2[r][i] = (float)a2[i]; } for (int i = 0; i < 3072; i++) wad[r][i] = (float)aa[i];
      memcpy(b_m0[r].data(), g_vae.f((p + ".m0.b").c_str()), 128); memcpy(b_m2[r].data(), g_vae.f((p + ".m2.b").c_str()), 128); memcpy(b_ad[r].data(), g_vae.f((p + ".ad.b").c_str()), 384);
      lnw[r].assign(g_vae.f((p + ".ln.w").c_str()), g_vae.f((p + ".ln.w").c_str()) + 32); lnb[r].assign(g_vae.f((p + ".ln.b").c_str()), g_vae.f((p + ".ln.b").c_str()) + 32); }
    const __fp16* hl = g_vae.h("fl.l.w"); for (int i = 0; i < 96; i++) wl[i] = (float)hl[i]; memcpy(bl.data(), g_vae.f("fl.l.b"), 12); }
  for (int p0 = 0; p0 < L; p0 += CH) {
    const int n = std::min(CH, L - p0); std::vector<f32> yf((size_t)n * 8192), ce((size_t)n * 8192);
    vlin("yx", cond.data() + (size_t)p0 * 384, n, 384, 8192, yf.data()); vlin("dn.ce", s.data() + (size_t)p0 * 384, n, 384, 8192, ce.data());
    par_for(n, [&](int ii) { const int pos = p0 + ii, py = pos / w, px = pos % w; const f32* Y = &yf[(size_t)ii * 8192]; const f32* Cc = &ce[(size_t)ii * 8192];
      for (int p = 0; p < 256; p++) {
        float xin[32], x[32], t1[32], hh[32], md[96], cs[32];
        for (int f = 0; f < 32; f++) xin[f] = Y[f * 256 + p];
        for (int o = 0; o < 32; o++) { float a = xec[p * 32 + o]; const f32* wr = &wf[o * 32]; for (int i = 0; i < 32; i++) a += wr[i] * xin[i]; t1[o] = a; }
        for (int o = 0; o < 32; o++) { float a = b_in[o]; const f32* wr = &win[o * 32]; for (int i = 0; i < 32; i++) a += wr[i] * t1[i]; x[o] = a; }
        for (int r = 0; r < 3; r++) {
          for (int i = 0; i < 32; i++) { const float cv = Cc[p * 32 + i]; cs[i] = cv / (1.f + expf(-cv)); }
          for (int o = 0; o < 96; o++) { float a = b_ad[r][o]; const f32* wr = &wad[r][o * 32]; for (int i = 0; i < 32; i++) a += wr[i] * cs[i]; md[o] = a; }
          float mu = 0, var = 0; for (int i = 0; i < 32; i++) mu += x[i]; mu /= 32; for (int i = 0; i < 32; i++) var += (x[i] - mu) * (x[i] - mu); const float rs = 1.f / sqrtf(var / 32 + 1e-6f);
          for (int i = 0; i < 32; i++) hh[i] = ((x[i] - mu) * rs * lnw[r][i] + lnb[r][i]) * (1.f + md[32 + i]) + md[i];
          for (int o = 0; o < 32; o++) { float a = b_m0[r][o]; const f32* wr = &wm0[r][o * 32]; for (int i = 0; i < 32; i++) a += wr[i] * hh[i]; t1[o] = a / (1.f + expf(-a)); }
          for (int o = 0; o < 32; o++) { float a = b_m2[r][o]; const f32* wr = &wm2[r][o * 32]; for (int i = 0; i < 32; i++) a += wr[i] * t1[i]; x[o] += md[64 + o] * a; }
        }
        float ss = 0; for (int i = 0; i < 32; i++) ss += x[i] * x[i]; const float rs = 1.f / sqrtf(ss / 32 + 1e-6f); for (int i = 0; i < 32; i++) x[i] = x[i] * rs * flw[i];
        const int Y0 = py * 16 + p / 16, X0 = px * 16 + p % 16;
        for (int c = 0; c < 3; c++) { float a = bl[c]; for (int i = 0; i < 32; i++) a += wl[c * 32 + i] * x[i]; rgb[((size_t)Y0 * W + X0) * 3 + c] = std::max(-1.f, std::min(1.f, a)); }
      } });
  }
  fprintf(stderr, "vae dec total %.2fs\n", now() - t0);
}
// rgb [H][W][3] in [-1,1] -> tokens [L][128] (posterior mean)
static void vae_encode(const f32* rgb, int h, int w, f32* tok) {
  const int L = h * w, W = w * 16; double t0 = now(); std::vector<f32> patch((size_t)L * 768), cond((size_t)L * 768), s((size_t)L * 384), c2((size_t)L * 384);
  par_for(L, [&](int i) { const int py = i / w, px = i % w; f32* d = &patch[(size_t)i * 768]; for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) memcpy(d + (y * 16 + x) * 3, rgb + ((size_t)(py * 16 + y) * W + px * 16 + x) * 3, 12); });
  vlin("pe", patch.data(), L, 768, 768, cond.data());
  dico("eh0", cond.data(), h, w, 768, nullptr); dico("eh1", cond.data(), h, w, 768, nullptr);
  vlin("e.pd", cond.data(), L, 768, 384, c2.data());
  gemm_f16(g_vae.h("e.fw.w"), c2.data(), s.data(), L, 384, 384); { const f32* fc = g_vae.f("e.fc"); par_for(L, [&](int i) { for (int c = 0; c < 384; c++) s[(size_t)i * 384 + c] += fc[c]; }); }
  for (int i = 0; i < 21; i++) dico("eb" + std::to_string(i), s.data(), h, w, 384, g_vae.f(("eb" + std::to_string(i) + ".mod").c_str()));
  { std::vector<f32> M(g_vae.f("e.no.w"), g_vae.f("e.no.w") + 384), N(g_vae.f("e.no.b"), g_vae.f("e.no.b") + 384); ln_gen(s.data(), L, 384, M.data(), N.data()); }
  vlin("e.po", s.data(), L, 384, 128, tok);
  fprintf(stderr, "vae enc total %.2fs\n", now() - t0);
}
