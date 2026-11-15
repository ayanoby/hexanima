// ------------------------------------------------------------------ vision tower (Qwen3-VL-4B ViT, MODE 51) + edit inputs
static Glue g_vglue; static Buf g_blobV; static Layout g_LV; static int g_V_ntok = -1;
static const int NLAYER_VIT = 24;
static Layout make_layout_vit(int ntok) {
  Layout L; memset(&L, 0, sizeof L); const size_t ns = ntok / 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, (size_t)ntok * 1024 * 4}, {M_COS, ns * 8192}, {M_SIN, ns * 8192}, {M_H, (size_t)ntok * 1024 * 2}, {M_Q, (size_t)ntok * 2048 * 2}, {M_K, (size_t)ntok * 2048 * 2}, {M_V, (size_t)ntok * 2048 * 2},
    {M_O, (size_t)ntok * 2048 * 2}, {M_T, (size_t)ntok * 1024 * 2}, {M_MID, (size_t)ntok * 4096 * 2}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LQ, ns * 12 * 2048}, {M_LO, ns * 4 * 2048}, {M_LU, ns * 4 * 2048}, {M_LD, ns * 4 * 2048}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
static void layer_norm_rows(f32* x, int n, int d, const f32* w, const f32* b) { par_for(n, [&](int t) { f32* r = x + (size_t)t * d; double mu = 0, ss = 0; for (int i = 0; i < d; i++) mu += r[i]; mu /= d; for (int i = 0; i < d; i++) ss += (r[i] - mu) * (r[i] - mu); const float rs = 1.f / sqrtf((float)(ss / d) + 1e-6f); for (int i = 0; i < d; i++) r[i] = (float)((r[i] - mu) * rs) * w[i] + b[i]; }); }
static void merger(const f32* rows, int nm, bool postshuffle, const char* pre, std::vector<f32>& out) {   // rows [4*nm][1024] (merge order) -> out [nm][2560]
  std::vector<f32> x(rows, rows + (size_t)nm * 4096); const std::string p = pre;
  if (postshuffle) layer_norm_rows(x.data(), nm, 4096, g_vglue.f((p + ".norm.weight").c_str()), g_vglue.f((p + ".norm.bias").c_str()));
  else layer_norm_rows(x.data(), nm * 4, 1024, g_vglue.f((p + ".norm.weight").c_str()), g_vglue.f((p + ".norm.bias").c_str()));
  std::vector<f32> a((size_t)nm * 4096); gemm_f16(g_vglue.h((p + ".fc1.weight").c_str()), x.data(), a.data(), nm, 4096, 4096);
  const f32* b1 = g_vglue.f((p + ".fc1.bias").c_str()); for (int t = 0; t < nm; t++) for (int i = 0; i < 4096; i++) { float v = a[(size_t)t * 4096 + i] + b1[i]; a[(size_t)t * 4096 + i] = 0.5f * v * (1.f + erff(v * 0.70710678f)); }
  out.assign((size_t)nm * TD, 0.f); gemm_f16(g_vglue.h((p + ".fc2.weight").c_str()), a.data(), out.data(), nm, TD, 4096);
  const f32* b2 = g_vglue.f((p + ".fc2.bias").c_str()); for (int t = 0; t < nm; t++) for (int i = 0; i < TD; i++) out[(size_t)t * TD + i] += b2[i];
}
// px [Np][1536] (normalised, merge-ordered patches) on a gh x gw patch grid -> pooled [nm][2560] and 3 deepstack feature sets
static bool vit_features(const std::vector<f32>& px, int gh, int gw, std::vector<f32>& pooled, std::vector<f32> ds[3]) {
  const int Np = gh * gw, nm = Np / 4, ns = (Np + 31) / 32, ntok = ns * 32; double t0 = now();
  std::vector<f32> rows((size_t)ntok * 1024, 0.f);
  gemm_f16(g_vglue.h("patch.weight"), px.data(), rows.data(), Np, 1024, 1536);
  { const f32* pb = g_vglue.f("patch.bias"); const __fp16* pe = g_vglue.h("pos_embed"); const int side = 48;
    auto taps = [&](int idx, int size, int* t, float* wgt) { const float src = size > 1 ? idx * (float)(side - 1) / (size - 1) : 0.f; const int fl = (int)floorf(src);
      for (int k = 0; k < 2; k++) { t[k] = std::min(std::max(fl + k, 0), side - 1); wgt[k] = std::max(0.f, 1.f - fabsf(src - fl - k)); } };
    par_for(Np, [&](int p) {   // merge-order patch p -> (row, col)
      const int blk = p / 4, ir = (p / 2) % 2, ic = p % 2, bw = gw / 2; const int row = (blk / bw) * 2 + ir, col = (blk % bw) * 2 + ic;
      int th[2], tw[2]; float wh[2], ww[2]; taps(row, gh, th, wh); taps(col, gw, tw, ww); f32* r = &rows[(size_t)p * 1024];
      for (int i = 0; i < 1024; i++) { float a = pb[i]; for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) a += wh[u] * ww[v] * (float)pe[(size_t)(th[u] * side + tw[v]) * 1024 + i]; r[i] += a; } }); }
  if (g_V_ntok != ntok) { if (g_blobV.p) { unmap_buf(g_blobV); rpcmem_free(g_blobV.p); } g_LV = make_layout_vit(ntok); g_blobV = make_buf(g_LV.total, true); if (!g_blobV.p) return false; g_V_ntok = ntok; }
  uint8_t* blob = g_blobV.p; memset(blob, 0, g_LV.total); fill_bias_tiles(blob, g_LV, ns);
  rows_to_x32(rows.data(), Np, 1024, (f32*)(blob + g_LV.meta[M_X]), ns);
  { f32* cs = (f32*)(blob + g_LV.meta[M_COS]); f32* sn = (f32*)(blob + g_LV.meta[M_SIN]);
    par_for(Np, [&](int p) { const int blk = p / 4, ir = (p / 2) % 2, ic = p % 2, bw = gw / 2; const int row = (blk / bw) * 2 + ir, col = (blk % bw) * 2 + ic;
      for (int d = 0; d < 32; d++) { const double inv = pow(10000.0, -2.0 * (d % 16) / 32.0); const double a = (d < 16 ? row : col) * inv; const size_t idx = ((size_t)(p / 32) * 2 * 1024) + (p % 32) * 32 + d; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } }); }
  std::vector<f32> mod(18432, 0.f); for (int i = 0; i < 1024; i++) mod[i] = 1.f;
  sync_clean(g_blobV);
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = g_LV.meta[i];
  meta[M_NTOK] = ntok; meta[M_NVALID] = Np; meta[M_NW] = 3; meta[M_NCB] = 32; meta[M_MODE] = 51; meta[M_NPRE] = 0; meta[M_CAUS] = 0; meta[M_BA] = meta[M_BB] = 0;
  std::vector<std::string> vf; for (int i = 0; i < NLAYER_VIT; i++) { char p[512]; snprintf(p, sizeof p, "%s/vit/vit%02d.bin", g_dir.c_str(), i); vf.push_back(p); }
  std::vector<f32> xr((size_t)Np * 1024); bool ok = true;
  ok = stream_files(vf, g_te, [&](int k, Buf& b) {
    const bool rd = k == 5 || k == 11 || k == 17 || k == NLAYER_VIT - 1; meta[M_FIRST] = k == 0; meta[M_REL] = rd;
    if (call(g_LV, meta, b, 0, g_blobV, mod.data(), 18432)) return false;
    if (rd) { sync_inval(g_blobV); x32_to_rows((const f32*)(blob + g_LV.meta[M_X]), Np, 1024, xr.data());
      const int di = k == 5 ? 0 : k == 11 ? 1 : k == 17 ? 2 : -1;
      if (di >= 0) merger(xr.data(), nm, true, (std::string("ds") + char('0' + di)).c_str(), ds[di]);
      else merger(xr.data(), nm, false, "merger", pooled); }
    return true; }, NLAYER_VIT);
  fprintf(stderr, "vit: %.2fs (%d patches)\n", now() - t0, Np); return ok;
}
struct EditIn { bool on = false; int nm = 0, k0 = 0, gh = 0, gw = 0; std::vector<f32> pooled, ds[3]; };
static void add_rows_x32(f32* X, int row0, int nrows, const f32* feat) {   // X x32 tiles [slab][80 cb][32][32]: rows row0.. += feat[r][2560]
  for (int r = 0; r < nrows; r++) { const int row = row0 + r; const int s = row / 32, rr = row % 32;
    for (int cb = 0; cb < 80; cb++) { f32* t = X + ((size_t)s * 80 + cb) * 1024 + (size_t)rr * 32; const f32* a = feat + (size_t)r * TD + cb * 32; for (int c = 0; c < 32; c++) t[c] += a[c]; } }
}
