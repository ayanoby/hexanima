// ------------------------------------------------------------------ Mage-Flow DiT (MODE 50)
static bool readraw(const char* p, std::vector<f32>& v, size_t n) { FILE* f = fopen(p, "rb"); if (!f) return false; v.resize(n); bool ok = fread(v.data(), 4, n, f) == n; fclose(f); return ok; }
static const int NB = 12, D = 3072, TD = 2560;
static std::vector<std::string> g_files; static Buf g_blobM; static Layout g_LM; static int g_M_ns = -1, g_M_wns = -1;
static Layout make_layout_m(int ns) {
  Layout L; memset(&L, 0, sizeof L); const size_t ntok = (size_t)ns * 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * D * 4}, {M_COS, (size_t)ns * 8192}, {M_SIN, (size_t)ns * 8192}, {M_H, ntok * D * 2}, {M_Q, ntok * D * 2}, {M_K, ntok * D * 2}, {M_V, ntok * D * 2},
    {M_O, ntok * D * 2}, {M_T, ntok * D * 2}, {M_MID, ntok * 12288 * 2}, {M_ROWS, 4096}, {M_CS, 4096},
    {M_LQ, (size_t)ns * 12 * 2048}, {M_LO, (size_t)ns * 4 * 2048}, {M_LU, (size_t)ns * 4 * 2048}, {M_LD, (size_t)ns * 4 * 2048}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
// windowed DiT (like LTX / MiniMax-H3): the activations X and the rope tables of all rows stay in phone memory, the DSP sees a window of wns slabs and the keys / values of ALL ns slabs (global K / V buffers)
static Layout make_layout_mw(int wns, int ns) {
  Layout L; memset(&L, 0, sizeof L); const size_t wtok = (size_t)wns * 32, ntok = (size_t)ns * 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, wtok * D * 4}, {M_COS, (size_t)wns * 8192}, {M_SIN, (size_t)wns * 8192}, {M_H, wtok * D * 2}, {M_Q, wtok * D * 2}, {M_K, ntok * D * 2}, {M_V, ntok * D * 2},
    {M_O, wtok * D * 2}, {M_T, wtok * D * 2}, {M_MID, wtok * 12288 * 2}, {M_ROWS, 4096}, {M_CS, 4096},
    {M_LQ, (size_t)wns * 12 * 2048}, {M_LO, (size_t)wns * 4 * 2048}, {M_LU, (size_t)wns * 4 * 2048}, {M_LD, (size_t)wns * 4 * 2048}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
static void fill_bias_tiles(uint8_t* blob, const Layout& L, int ns) {   // L buffers: fp16 1.0 in column 0 of every 4th tile (selects row 0 of the 4 extra K tiles = the bias)
  auto fill = [&](int id, size_t ntiles) { __fp16* t = (__fp16*)(blob + L.meta[id]); memset(t, 0, ntiles * 2048);
    for (size_t i = 0; i < ntiles; i += 4) for (int r = 0; r < 32; r++) t[i * 1024 + (r / 2) * 64 + (r & 1)] = (__fp16)1.0f; };
  fill(M_LQ, (size_t)ns * 12); fill(M_LO, (size_t)ns * 4); fill(M_LU, (size_t)ns * 4); fill(M_LD, (size_t)ns * 4);
}
static void rope_rows(const double (*pos)[3], int nrow, int ntok, f32* cs, f32* sn) {   // tiles [slab][2][32][32]; axes (frame,h,w) pairs 8/28/28, theta 10000
  memset(cs, 0, (size_t)ntok / 32 * 8192); memset(sn, 0, (size_t)ntok / 32 * 8192);
  par_for(nrow, [&](int tk) { for (int d = 0; d < 64; d++) {
    const int ax = d < 8 ? 0 : d < 36 ? 1 : 2, j = ax == 0 ? d : ax == 1 ? d - 8 : d - 36, np = ax == 0 ? 8 : 28;
    const double a = pos[tk][ax] * pow(10000.0, -(double)j / np); const size_t idx = (((size_t)(tk / 32) * 2 + d / 32) * 1024) + (tk % 32) * 32 + d % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } });
}
static void ln_rows(f32* r, int n, int d, const f32* scale1, const f32* shift) {   // LayerNorm (no affine, eps 1e-6) * (1 + scale) + shift, in place
  par_for(n, [&](int t) { f32* x = r + (size_t)t * d; double mu = 0, ss = 0; for (int i = 0; i < d; i++) mu += x[i]; mu /= d; for (int i = 0; i < d; i++) { double q = x[i] - mu; ss += q * q; }
    const float rs = 1.f / sqrtf((float)(ss / d) + 1e-6f); for (int i = 0; i < d; i++) x[i] = (float)(x[i] - mu) * rs * (1.f + scale1[i]) + shift[i]; });
}
static void add_bias(f32* y, int n, int out, const f32* b) { par_for(n, [&](int t) { for (int o = 0; o < out; o++) y[(size_t)t * out + o] += b[o]; }); }
// txt [nt][2560] (final-normed text-encoder output) -> text stream rows [nt][3072]
static void txt_in(const f32* tx, int nt, f32* out) {
  std::vector<f32> a((size_t)nt * TD); const f32* nw = g_glue.f("txt_norm.weight");
  for (int t = 0; t < nt; t++) { const f32* r = tx + (size_t)t * TD; double ss = 0; for (int i = 0; i < TD; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / TD) + 1e-6f); for (int i = 0; i < TD; i++) a[(size_t)t * TD + i] = r[i] * rs * nw[i]; }
  gemm_f16(g_glue.h("txt_in.weight"), a.data(), out, nt, D, TD); add_bias(out, nt, D, g_glue.f("txt_in.bias"));
}
// ------------------------------------------------------------------ text encoder (Qwen3-VL-4B language model, text only): ids -> [T-drop][2560]
static const int NL_TE = 36, MAXT = 1024;
static std::vector<std::string> g_tefiles; static std::vector<f32> g_tefinal;
static int g_T_ns = -1;
static Layout make_layout_te(int ns) {
  Layout L; memset(&L, 0, sizeof L); const size_t ntok = (size_t)ns * 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * TD * 4}, {M_COS, (size_t)ns * 8192}, {M_SIN, (size_t)ns * 8192}, {M_H, ntok * 4096 * 2}, {M_Q, ntok * 4096 * 2}, {M_K, ntok * 1024 * 2}, {M_V, ntok * 1024 * 2},
    {M_O, ntok * 4096 * 2}, {M_T, ntok * TD * 2}, {M_MID, ntok * 9728 * 2}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LQ, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
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
  meta[M_NTOK] = ntok; meta[M_NVALID] = Np; meta[M_NW] = 3; meta[M_NCB] = 32; meta[M_MODE] = 55; meta[M_NPRE] = 0; meta[M_CAUS] = 0; meta[M_BA] = meta[M_BB] = 0;
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

static bool encode_text(const std::vector<int>& ids, int drop, std::vector<f32>& tx, const std::vector<EditIn>* eds = nullptr) {   // eds: one entry per reference picture (up to 3), in the order of their <|image_pad|> runs
  const int T = (int)ids.size(), ns = (T + 31) / 32, ntok = ns * 32; if (T > MAXT) return false;
  double t0 = now();
  if (g_T_ns != ns) { if (g_blobTE.p) { unmap_buf(g_blobTE); rpcmem_free(g_blobTE.p); } g_LT = make_layout_te(ns); g_blobTE = make_buf(g_LT.total, true); if (!g_blobTE.p) return false; g_T_ns = ns; }
  uint8_t* blob = g_blobTE.p; memset(blob, 0, g_LT.total);
  { std::vector<f32> rows((size_t)ntok * TD, 0.f); const f32* sc = (const f32*)g_embd; const int8_t* q = (const int8_t*)(g_embd + (size_t)151936 * 4);
    for (int t = 0; t < T; t++) { const int id = ids[t]; const f32 s = sc[id]; const int8_t* r = q + (size_t)id * TD; f32* o = &rows[(size_t)t * TD]; for (int i = 0; i < TD; i++) o[i] = s * r[i]; }
    if (eds) for (const EditIn& e : *eds) memcpy(&rows[(size_t)e.k0 * TD], e.pooled.data(), (size_t)e.nm * TD * 4);   // image slots carry the ViT features
    rows_to_x32(rows.data(), ntok, TD, (f32*)(blob + g_LT.meta[M_X]), ns); }
  { std::vector<double> flat((size_t)T * 3); for (int i = 0; i < T; i++) { flat[i * 3] = flat[i * 3 + 1] = flat[i * 3 + 2] = i; }
    if (eds && !eds->empty()) {   // interleaved mrope: the merged grid of every picture (gh/2 x gw/2) carries (t, h, w) = (p, p+row, p+col), p = the running text position at its start; the text after it continues at p + max(gh, gw)
      int p = 0, t = 0; size_t ei = 0;
      while (t < T) {
        if (ei < eds->size() && t == (*eds)[ei].k0) {
          const EditIn& e = (*eds)[ei++]; const int gh = e.gh / 2, gw = e.gw / 2;
          for (int r = 0; r < gh; r++) for (int c = 0; c < gw; c++) { const int j = t + r * gw + c; flat[j * 3] = p; flat[j * 3 + 1] = p + r; flat[j * 3 + 2] = p + c; }
          t += e.nm; p += std::max(gh, gw);
        } else { flat[t * 3] = flat[t * 3 + 1] = flat[t * 3 + 2] = p; t++; p++; }
      } }
    f32* cs = (f32*)(blob + g_LT.meta[M_COS]); f32* sn = (f32*)(blob + g_LT.meta[M_SIN]); memset(cs, 0, (size_t)ns * 8192); memset(sn, 0, (size_t)ns * 8192);
    par_for(T, [&](int t) { for (int d = 0; d < 64; d++) { const int ax = (d % 3 == 1 && d < 60) ? 1 : (d % 3 == 2 && d < 60) ? 2 : 0; const double a = flat[t * 3 + ax] * pow(5e6, -(double)d / 64.0); const size_t idx = (((size_t)(t / 32) * 2 + d / 32) * 1024) + (t % 32) * 32 + d % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } }); }
  sync_clean(g_blobTE);
  std::vector<f32> mod(18432, 0.f); for (int i = 8192; i < 12288; i++) mod[i] = 1.f;
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = g_LT.meta[i];
  meta[M_NW] = 3; meta[M_NCB] = 80; meta[M_MODE] = 22; meta[M_FMT] = 1; meta[M_NTOK] = ntok; meta[M_NVALID] = T; meta[M_STOP] = 0;
  const bool dsk = eds && !eds->empty();
  if (!stream_files(g_tefiles, g_te, [&](int l, Buf& w) {
    meta[M_FIRST] = l == 0 || (dsk && l >= 1 && l <= 3); meta[M_REL] = l == NL_TE - 1 || (dsk && l < 3);
    if (call(g_LT, meta, w, 0, g_blobTE, mod.data(), (int)mod.size())) return false;
    if (dsk && l < 3) { sync_inval(g_blobTE); for (const EditIn& e : *eds) add_rows_x32((f32*)(blob + g_LT.meta[M_X]), e.k0, e.nm, e.ds[l].data()); sync_clean(g_blobTE); }   // DeepStack: every picture's rows
    return true; }, getenv("MAGE_KEEPTE") ? NL_TE : 0)) return false;
  sync_inval(g_blobTE);
  std::vector<f32> rows((size_t)T * TD); x32_to_rows((const f32*)(blob + g_LT.meta[M_X]), T, TD, rows.data());
  tx.resize((size_t)(T - drop) * TD);
  par_for(T - drop, [&](int t) { const f32* r = &rows[(size_t)(t + drop) * TD]; double ss = 0; for (int i = 0; i < TD; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / TD) + 1e-6f); f32* o = &tx[(size_t)t * TD]; for (int i = 0; i < TD; i++) o[i] = r[i] * rs * g_tefinal[i]; });
  fprintf(stderr, "te: %.2fs (%d tokens, waited %.2fs for io)\n", now() - t0, T, g_waitio); g_waitio = 0; return true;
}
// run the DiT. refs (optional): clean reference-image latent tokens [Nr][128] (edit), rh x rw tokens each, appended one after the other to the image stream after the target tokens; picture k sits at RoPE frame index k + 1
struct RefImg { std::vector<f32> tok; int h = 0, w = 0; };
static bool run_dit(const std::vector<f32>& tx, int nt, int h, int w, unsigned seed, int steps, const std::string& outz, const std::vector<RefImg>* refs, std::vector<f32>* ztok = nullptr) {
  const double T0 = now(); const int N = h * w; int Nr = 0; if (refs) for (const RefImg& r : *refs) Nr += r.h * r.w; const int NI = N + Nr;
  if (NI % 32 || N % 32) { fprintf(stderr, "image tokens must be a multiple of 32\n"); return false; }
  const int nsI = NI / 32, nsT = (nt + 31) / 32, ns = nsI + nsT, ntok = ns * 32;
  const int WSL = getenv("MAGE_WIN") ? atoi(getenv("MAGE_WIN")) : 32; const bool win = getenv("MAGE_WIN") ? WSL > 0 : ntok > 6000;   // windows (slabs of 32 rows): automatic above 6000 rows, MAGE_WIN=0 forces the plain path
  const int wns = win ? std::min(std::max(WSL, 1), ns) : ns; const int wkey = win ? wns : 0;
  if (g_M_ns != ns || g_M_wns != wkey) { if (g_blobM.p) { unmap_buf(g_blobM); rpcmem_free(g_blobM.p); } g_blobM.p = nullptr; g_LM = win ? make_layout_mw(wns, ns) : make_layout_m(ns); g_blobM = make_buf(g_LM.total, true); if (!g_blobM.p) return false; g_M_ns = ns; g_M_wns = wkey; }
  uint8_t* blob = g_blobM.p; memset(blob, 0, g_LM.total); fill_bias_tiles(blob, g_LM, wns);
  const size_t XS = (size_t)D / 32 * 1024, CS = (size_t)2 * 1024; std::vector<f32> Xh, Ch, Sh; if (win) { Xh.assign((size_t)ns * XS, 0.f); Ch.assign((size_t)ns * CS, 1.f); Sh.assign((size_t)ns * CS, 0.f); }
  f32* Xp = win ? Xh.data() : (f32*)(blob + g_LM.meta[M_X]); f32* Cp = win ? Ch.data() : (f32*)(blob + g_LM.meta[M_COS]); f32* Sp = win ? Sh.data() : (f32*)(blob + g_LM.meta[M_SIN]);
  { std::vector<double> flat((size_t)NI * 3); for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { const int i = y * w + x; flat[i * 3] = 0; flat[i * 3 + 1] = y - (h - h / 2); flat[i * 3 + 2] = x - (w - w / 2); }
    { int base = N; int fr = 1; if (refs) for (const RefImg& r : *refs) { for (int y = 0; y < r.h; y++) for (int x = 0; x < r.w; x++) { const int i = base + y * r.w + x; flat[i * 3] = fr; flat[i * 3 + 1] = y - (r.h - r.h / 2); flat[i * 3 + 2] = x - (r.w - r.w / 2); } base += r.h * r.w; fr++; } }
    rope_rows((const double(*)[3])flat.data(), NI, ntok, Cp, Sp); }
  std::vector<f32> rows((size_t)ntok * D, 0.f), ttx((size_t)nt * D);
  double t1 = now(); txt_in(tx.data(), nt, ttx.data()); fprintf(stderr, "txt_in: %.2fs\n", now() - t1);
  std::vector<f32> z((size_t)N * 128);
  if (getenv("MAGE_Z0")) { if (!readraw(getenv("MAGE_Z0"), z, z.size())) { fprintf(stderr, "MAGE_Z0 unreadable\n"); return false; } }
  else { std::mt19937 gen(seed); std::normal_distribution<float> nd(0.f, 1.f); for (auto& a : z) a = nd(gen); }
  std::vector<f32> refimg; if (refs && Nr) { refimg.resize((size_t)Nr * D); size_t o = 0; for (const RefImg& r : *refs) { const int n = r.h * r.w; gemm_f16(g_glue.h("img_in.weight"), r.tok.data(), refimg.data() + o * D, n, D, 128); add_bias(refimg.data() + o * D, n, D, g_glue.f("img_in.bias")); o += n; } }
  const f32* sig = g_glue.f("sigmas"); const f32* mods = g_glue.f("mods"); const f32* nmods = g_glue.f("nmods");
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = g_LM.meta[i];
  meta[M_NW] = getenv("MAGE_NW") ? atoi(getenv("MAGE_NW")) : 3; meta[M_NCB] = 96; meta[M_MODE] = 53; meta[M_NTOK] = ntok; meta[M_NVALID] = NI + nt; meta[M_BA] = nsI; meta[M_STOP] = 0;
  std::vector<f32> v((size_t)N * 128);
  for (int s = 0; s < steps; s++) {
    const double ts = now();
    gemm_f16(g_glue.h("img_in.weight"), z.data(), rows.data(), N, D, 128); add_bias(rows.data(), N, D, g_glue.f("img_in.bias"));
    if (Nr) memcpy(rows.data() + (size_t)N * D, refimg.data(), (size_t)Nr * D * 4);
    memcpy(rows.data() + (size_t)NI * D, ttx.data(), (size_t)nt * D * 4);
    rows_to_x32(rows.data(), ntok, D, Xp, ns); if (!win) sync_clean(g_blobM);
    if (!stream_files(g_files, g_te, [&](int b, Buf& w) {
      if (win) {   // per block: pass 0 = keys / values of every window into the global K / V buffers, pass 1 = the full block per window (attention over all keys)
        const int nwin = (ns + wns - 1) / wns; const f32* mb = mods + ((size_t)(s * NB + b) * 16) * D;
        for (int pass = 0; pass < 2; pass++) for (int wn = 0; wn < nwin; wn++) {
          const int w0 = wn * wns, nws = std::min(wns, ns - w0); const int ni0 = std::max(0, std::min(nsI - w0, nws));
          memcpy(blob + g_LM.meta[M_X], &Xh[(size_t)w0 * XS], (size_t)nws * XS * 4); memcpy(blob + g_LM.meta[M_COS], &Ch[(size_t)w0 * CS], (size_t)nws * CS * 4); memcpy(blob + g_LM.meta[M_SIN], &Sh[(size_t)w0 * CS], (size_t)nws * CS * 4);
          sync_clean(g_blobM);
          meta[M_NTOK] = nws * 32; meta[M_BA] = ni0; meta[M_A2] = ntok; meta[M_A4] = pass == 0 ? 1 : 2; meta[M_A9] = w0; meta[M_FIRST] = (b == 0 && pass == 0 && wn == 0) ? 1 : 2; meta[M_REL] = (b == NB - 1 && pass == 1 && wn == nwin - 1) ? 1 : 2; meta[M_FMT] = 0;
          if (call(g_LM, meta, w, 0, g_blobM, mb, 16 * D)) return false;
          if (pass == 1) { sync_inval(g_blobM); memcpy(&Xh[(size_t)w0 * XS], blob + g_LM.meta[M_X], (size_t)nws * XS * 4); }
        }
        meta[M_A2] = meta[M_A4] = meta[M_A9] = 0; return true; }
      meta[M_FIRST] = b == 0; meta[M_REL] = b == NB - 1; meta[M_FMT] = getenv("MAGE_FC2") ? atoi(getenv("MAGE_FC2")) : 0;
      return call(g_LM, meta, w, 0, g_blobM, mods + ((size_t)(s * NB + b) * 16) * D, 16 * D) == 0; }, NB)) return false;
    if (!win) sync_inval(g_blobM);
    x32_to_rows((const f32*)Xp, N, D, rows.data());
    ln_rows(rows.data(), N, D, nmods + (size_t)(s * 2) * D, nmods + (size_t)(s * 2 + 1) * D);
    gemm_f16(g_glue.h("proj_out.weight"), rows.data(), v.data(), N, 128, D); add_bias(v.data(), N, 128, g_glue.f("proj_out.bias"));
    if (const char* pv = getenv("MAGE_PREVIEW")) if (*pv && s < steps - 1) {   // preview: x0 estimate z - sigma * v as [128][N] (like the final latent file), written atomically
      std::vector<f32> xo((size_t)128 * N); for (int i = 0; i < N; i++) for (int c = 0; c < 128; c++) xo[(size_t)c * N + i] = z[(size_t)i * 128 + c] - sig[s] * v[(size_t)i * 128 + c];
      const std::string tmp = std::string(pv) + ".tmp"; FILE* pf = fopen(tmp.c_str(), "wb");
      if (pf) { int hdr[4] = {h, w, 128, s}; fwrite(hdr, 4, 4, pf); fwrite(xo.data(), 4, xo.size(), pf); fclose(pf); rename(tmp.c_str(), pv); } }
    const float dt = sig[s + 1] - sig[s]; for (size_t i = 0; i < z.size(); i++) z[i] += dt * v[i];
    fprintf(stderr, "step %d: %.2fs (io wait %.2fs)\n", s, now() - ts, g_waitio); g_waitio = 0;
  }
  { size_t bad = 0; for (float v : z) if (!std::isfinite(v)) bad++; if (bad) fprintf(stderr, "WARNING: %zu non-finite latent values\n", bad); }
  if (ztok) *ztok = z;
  { std::vector<f32> zt((size_t)128 * N); for (int i = 0; i < N; i++) for (int c = 0; c < 128; c++) zt[(size_t)c * N + i] = z[(size_t)i * 128 + c];
    FILE* f = fopen(outz.c_str(), "wb"); if (!f) return false; fwrite(zt.data(), 4, zt.size(), f); fclose(f); }
  fprintf(stderr, "dit total %.2fs\n", now() - T0); return true;
}
// ------------------------------------------------------------------ Mage-VAE on the CPU (NEON): encoder + decoder at t=0 (adaLN folded), weights in vae.bin (export_vae.py)
// activations are fp32 channel-last [pos][C]; 1x1 convs = gemm_f16, 3x3 convs = im2col + gemm_f16.
static Glue g_vae;
// ---- DSP offload of the VAE GEMMs (MODE 60): fp32 rows <-> fp16 crouton tiles on the host, W = q8 tiles with bias in vaepack.bin
struct VPm { size_t off; int K, N; };
static std::map<std::string, VPm> g_vp; static Buf g_vpack, g_blobG; static int g_G_ns = -1; static Layout g_LG; static bool g_vdsp = true;
static void rows_to_h16(const f32* X, int n, int K, uint16_t* h, int ns) {   // [n][K] fp32 -> tiles [slab][K/32][1024] (halfword (r,c) = (r/2)*64 + c*2 + (r&1))
  const int kt = K / 32;
  par_for(ns, [&](int s) { for (int cb = 0; cb < kt; cb++) { __fp16* t = (__fp16*)h + ((size_t)s * kt + cb) * 1024;
    for (int p = 0; p < 16; p++) { const int r0 = s * 32 + 2 * p, r1 = r0 + 1;
      for (int c = 0; c < 32; c += 8) { float16x8_t a = vdupq_n_f16(0), b = a;
        if (r0 < n) a = vcombine_f16(vcvt_f16_f32(vld1q_f32(X + (size_t)r0 * K + cb * 32 + c)), vcvt_f16_f32(vld1q_f32(X + (size_t)r0 * K + cb * 32 + c + 4)));
        if (r1 < n) b = vcombine_f16(vcvt_f16_f32(vld1q_f32(X + (size_t)r1 * K + cb * 32 + c)), vcvt_f16_f32(vld1q_f32(X + (size_t)r1 * K + cb * 32 + c + 4)));
        const float16x8x2_t z = vzipq_f16(a, b); vst1q_f16(t + p * 64 + c * 2, z.val[0]); vst1q_f16(t + p * 64 + c * 2 + 8, z.val[1]); } } } });
}
static void h16_to_rows(const uint16_t* h, int n, int N, f32* Y) {
  const int nt = N / 32, ns = (n + 31) / 32;
  par_for(ns, [&](int s) { for (int cb = 0; cb < nt; cb++) { const __fp16* t = (const __fp16*)h + ((size_t)s * nt + cb) * 1024;
    for (int p = 0; p < 16; p++) { const int r0 = s * 32 + 2 * p, r1 = r0 + 1;
      for (int c = 0; c < 32; c += 8) { const float16x8x2_t z = vuzpq_f16(vld1q_f16(t + p * 64 + c * 2), vld1q_f16(t + p * 64 + c * 2 + 8));
        if (r0 < n) { vst1q_f32(Y + (size_t)r0 * N + cb * 32 + c, vcvt_f32_f16(vget_low_f16(z.val[0]))); vst1q_f32(Y + (size_t)r0 * N + cb * 32 + c + 4, vcvt_f32_f16(vget_high_f16(z.val[0]))); }
        if (r1 < n) { vst1q_f32(Y + (size_t)r1 * N + cb * 32 + c, vcvt_f32_f16(vget_low_f16(z.val[1]))); vst1q_f32(Y + (size_t)r1 * N + cb * 32 + c + 4, vcvt_f32_f16(vget_high_f16(z.val[1]))); } } } } });
}
static thread_local bool tl_cpu_only = false;   // set on helper threads that must not touch the DSP while the main thread does
static bool g_vdsp_ok(int n) { return g_vdsp && !tl_cpu_only && g_vpack.p && n >= 128; }
static bool vgemm(const std::string& nm, const f32* X, int n, f32* Y, bool gelu) {
  auto it = g_vp.find(nm); if (it == g_vp.end()) return false; const VPm& P = it->second; const int kt = P.K / 32, ns = (n + 31) / 32, ntok = ns * 32;
  if (g_G_ns < ns) {   // blob sized for the widest layer (K up to 3456+, N up to 8192) at this slab count
    if (g_blobG.p) { unmap_buf(g_blobG); rpcmem_free(g_blobG.p); } memset(&g_LG, 0, sizeof g_LG); size_t off = 0;
    struct E { int id; size_t b; } ent[] = {{M_X, 4096}, {M_H, (size_t)ns * 112 * 2048}, {M_T, (size_t)ns * 256 * 2048}, {M_LQ, (size_t)ns * 4 * 2048}};
    for (auto& e : ent) { g_LG.meta[e.id] = (int)off; off += up4k(e.b); } g_LG.total = off; g_blobG = make_buf(g_LG.total, false); if (!g_blobG.p) return false; g_G_ns = ns;
    __fp16* l = (__fp16*)(g_blobG.p + g_LG.meta[M_LQ]); memset(l, 0, (size_t)ns * 4 * 2048); for (size_t i = 0; i < (size_t)ns; i++) for (int r = 0; r < 32; r++) l[i * 4 * 1024 + (r / 2) * 64 + (r & 1)] = (__fp16)1.0f;
  }
  if (!map_buf(g_blobG)) return false;
  rows_to_h16(X, n, P.K, (uint16_t*)(g_blobG.p + g_LG.meta[M_H]), ns); sync_clean(g_blobG);
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = g_LG.meta[i];
  meta[M_MODE] = 61; meta[M_NTOK] = ntok; meta[M_NCB] = kt; meta[M_BA] = P.N; meta[M_FMT] = gelu ? 2 : 0; meta[M_NW] = 3; static bool first = true; meta[M_FIRST] = first; first = false; meta[M_REL] = 0; meta[M_STOP] = 0; meta[M_NVALID] = n;
  static float dummy[8]; if (call(g_LG, meta, g_vpack, P.off, g_blobG, dummy, 8)) return false;
  sync_inval(g_blobG); h16_to_rows((const uint16_t*)(g_blobG.p + g_LG.meta[M_T]), n, P.N, Y); return true;
}
static void vlin(const std::string& nm, const f32* X, int n, int in, int out, f32* Y, bool gelu = false) {
  if (g_vdsp_ok(n) && vgemm(nm, X, n, Y, gelu)) return;
  gemm_f16(g_vae.h((nm + ".w").c_str()), X, Y, n, out, in); add_bias(Y, n, out, g_vae.f((nm + ".b").c_str()));
  if (gelu) par_for(n, [&](int i) { for (int c = 0; c < out; c++) Y[(size_t)i * out + c] = 0.5f * Y[(size_t)i * out + c] * (1.f + erff(Y[(size_t)i * out + c] * 0.70710678118f)); }); }
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
  vlin(nm + ".c4", a.data(), L, C, F, t.data(), true);
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
static inline float32x4_t exp4(float32x4_t x) {   // e^x, rel err ~1e-7
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-80.f)), vdupq_n_f32(80.f));
  const float32x4_t t = vmulq_n_f32(x, 1.44269504f), fl = vrndmq_f32(t), f = vsubq_f32(t, fl); const int32x4_t i = vcvtq_s32_f32(fl);
  float32x4_t p = vdupq_n_f32(1.5403530e-4f); p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), p, f); p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, f); p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f); p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f); p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f); p = vfmaq_f32(vdupq_n_f32(1.f), p, f);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(i, 23)));
}
static inline float32x4_t silu4(float32x4_t x) { return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.f), exp4(vnegq_f32(x)))); }
// out[o][p] = bias[o] + sum_i W[o][i] * in[i][p], p = 0..255 (SoA pixel vectors)
static void soa_mm(const f32* W, const f32* bias, const f32* in, f32* out, int no, int ni) {
  for (int pc = 0; pc < 256; pc += 64) for (int o = 0; o < no; o++) {
    float32x4_t acc[16]; const float32x4_t b = vdupq_n_f32(bias ? bias[o] : 0.f); for (int v = 0; v < 16; v++) acc[v] = b;
    for (int i = 0; i < ni; i++) { const float wv = W[o * ni + i]; const f32* x = in + i * 256 + pc; for (int v = 0; v < 16; v++) acc[v] = vfmaq_n_f32(acc[v], vld1q_f32(x + 4 * v), wv); }
    for (int v = 0; v < 16; v++) vst1q_f32(out + o * 256 + pc + 4 * v, acc[v]); }
}
template <class F> static void par_for8(int n, F f) {
  int nt = std::min(n, std::max(2, (int)std::thread::hardware_concurrency())); nt = std::min(nt, 8); std::atomic<int> next(0); std::vector<std::thread> th;
  auto work = [&] { for (;;) { int i = next.fetch_add(1); if (i >= n) break; f(i); } }; for (int t = 1; t < nt; t++) th.emplace_back(work); work(); for (auto& x : th) x.join();
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
  // pixel stage: per position 256 pixels x 32 features in SoA form [feature][pixel] (the yx / cond_embed GEMM outputs are already f*256+p), vectorised over pixels
  const f32 *xec = g_vae.f("xe.c"), *flw = g_vae.f("fl.n"); const int CH = 64; const int W = w * 16;
  std::vector<f32> xecT(32 * 256), wf(32 * 32), win(32 * 32), wm0[3], wm2[3], wad[3], lnw[3], lnb[3], b_in(32), b_m0[3], b_m2[3], b_ad[3], wl(3 * 32), bl(3);
  for (int p = 0; p < 256; p++) for (int o = 0; o < 32; o++) xecT[o * 256 + p] = xec[p * 32 + o];
  { const __fp16* h16 = g_vae.h("xe.wf"); for (int i = 0; i < 1024; i++) wf[i] = (float)h16[i]; h16 = g_vae.h("dn.in.w"); for (int i = 0; i < 1024; i++) win[i] = (float)h16[i]; memcpy(b_in.data(), g_vae.f("dn.in.b"), 128);
    for (int r = 0; r < 3; r++) { const std::string p = "dn" + std::to_string(r); wm0[r].resize(1024); wm2[r].resize(1024); wad[r].resize(96 * 32); b_m0[r].resize(32); b_m2[r].resize(32); b_ad[r].resize(96);
      const __fp16 *a0 = g_vae.h((p + ".m0.w").c_str()), *a2 = g_vae.h((p + ".m2.w").c_str()), *aa = g_vae.h((p + ".ad.w").c_str()); for (int i = 0; i < 1024; i++) { wm0[r][i] = (float)a0[i]; wm2[r][i] = (float)a2[i]; } for (int i = 0; i < 3072; i++) wad[r][i] = (float)aa[i];
      memcpy(b_m0[r].data(), g_vae.f((p + ".m0.b").c_str()), 128); memcpy(b_m2[r].data(), g_vae.f((p + ".m2.b").c_str()), 128); memcpy(b_ad[r].data(), g_vae.f((p + ".ad.b").c_str()), 384);
      lnw[r].assign(g_vae.f((p + ".ln.w").c_str()), g_vae.f((p + ".ln.w").c_str()) + 32); lnb[r].assign(g_vae.f((p + ".ln.b").c_str()), g_vae.f((p + ".ln.b").c_str()) + 32); }
    const __fp16* hl = g_vae.h("fl.l.w"); for (int i = 0; i < 96; i++) wl[i] = (float)hl[i]; memcpy(bl.data(), g_vae.f("fl.l.b"), 12); }
  for (int p0 = 0; p0 < L; p0 += CH) {
    const int n = std::min(CH, L - p0); std::vector<f32> yf((size_t)n * 8192), ce((size_t)n * 8192);
    vlin("yx", cond.data() + (size_t)p0 * 384, n, 384, 8192, yf.data()); vlin("dn.ce", s.data() + (size_t)p0 * 384, n, 384, 8192, ce.data());
    par_for8(n, [&](int ii) { const int pos = p0 + ii, py = pos / w, px = pos % w; const f32* Y = &yf[(size_t)ii * 8192]; const f32* Cc = &ce[(size_t)ii * 8192];
      thread_local static f32 T1[32 * 256], X[32 * 256], MD[96 * 256], Hh[32 * 256], CS[32 * 256];
      // T1 = wf @ Y + xec ; X = win @ T1 + b
      soa_mm(wf.data(), nullptr, Y, T1, 32, 32); for (int i = 0; i < 32 * 256; i++) T1[i] += xecT[i];
      soa_mm(win.data(), b_in.data(), T1, X, 32, 32);
      for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 32 * 256; i += 4) vst1q_f32(CS + i, silu4(vld1q_f32(Cc + i)));
        soa_mm(wad[r].data(), b_ad[r].data(), CS, MD, 96, 32);
        for (int p = 0; p < 256; p += 4) {   // LayerNorm over the 32 features of every pixel + modulation
          float32x4_t mu = vdupq_n_f32(0); for (int i = 0; i < 32; i++) mu = vaddq_f32(mu, vld1q_f32(X + i * 256 + p)); mu = vmulq_n_f32(mu, 1.f / 32);
          float32x4_t var = vdupq_n_f32(0); for (int i = 0; i < 32; i++) { const float32x4_t d = vsubq_f32(vld1q_f32(X + i * 256 + p), mu); var = vfmaq_f32(var, d, d); }
          const float32x4_t rs = vdivq_f32(vdupq_n_f32(1.f), vsqrtq_f32(vfmaq_n_f32(vdupq_n_f32(1e-6f), var, 1.f / 32)));
          for (int i = 0; i < 32; i++) { const float32x4_t xn = vmulq_f32(vsubq_f32(vld1q_f32(X + i * 256 + p), mu), rs);
            float32x4_t hv = vfmaq_n_f32(vdupq_n_f32(lnb[r][i]), xn, lnw[r][i]); hv = vfmaq_f32(vld1q_f32(MD + i * 256 + p), hv, vaddq_f32(vdupq_n_f32(1.f), vld1q_f32(MD + (32 + i) * 256 + p))); vst1q_f32(Hh + i * 256 + p, hv); } }
        soa_mm(wm0[r].data(), b_m0[r].data(), Hh, T1, 32, 32); for (int i = 0; i < 32 * 256; i += 4) vst1q_f32(T1 + i, silu4(vld1q_f32(T1 + i)));
        soa_mm(wm2[r].data(), b_m2[r].data(), T1, Hh, 32, 32);
        for (int i = 0; i < 32 * 256; i += 4) vst1q_f32(X + i, vfmaq_f32(vld1q_f32(X + i), vld1q_f32(MD + 64 * 256 + i), vld1q_f32(Hh + i)));
      }
      for (int p = 0; p < 256; p += 4) {   // RMSNorm + final linear to rgb
        float32x4_t ss = vdupq_n_f32(0); for (int i = 0; i < 32; i++) { const float32x4_t v = vld1q_f32(X + i * 256 + p); ss = vfmaq_f32(ss, v, v); }
        const float32x4_t rs = vdivq_f32(vdupq_n_f32(1.f), vsqrtq_f32(vfmaq_n_f32(vdupq_n_f32(1e-6f), ss, 1.f / 32)));
        float32x4_t o0 = vdupq_n_f32(bl[0]), o1 = vdupq_n_f32(bl[1]), o2 = vdupq_n_f32(bl[2]);
        for (int i = 0; i < 32; i++) { const float32x4_t v = vmulq_n_f32(vmulq_f32(vld1q_f32(X + i * 256 + p), rs), flw[i]); o0 = vfmaq_n_f32(o0, v, wl[i]); o1 = vfmaq_n_f32(o1, v, wl[32 + i]); o2 = vfmaq_n_f32(o2, v, wl[64 + i]); }
        float r0[4], r1[4], r2[4]; vst1q_f32(r0, o0); vst1q_f32(r1, o1); vst1q_f32(r2, o2);
        for (int u = 0; u < 4; u++) { const int pp = p + u, Y0 = py * 16 + pp / 16, X0 = px * 16 + pp % 16; f32* d = rgb + ((size_t)Y0 * W + X0) * 3; d[0] = std::max(-1.f, std::min(1.f, r0[u])); d[1] = std::max(-1.f, std::min(1.f, r1[u])); d[2] = std::max(-1.f, std::min(1.f, r2[u])); } } });
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
  if (!(g_vdsp_ok(L) && vgemm("e.fw", c2.data(), L, s.data(), false))) gemm_f16(g_vae.h("e.fw"), c2.data(), s.data(), L, 384, 384); { const f32* fc = g_vae.f("e.fc"); par_for(L, [&](int i) { for (int c = 0; c < 384; c++) s[(size_t)i * 384 + c] += fc[c]; }); }
  for (int i = 0; i < 21; i++) dico("eb" + std::to_string(i), s.data(), h, w, 384, g_vae.f(("eb" + std::to_string(i) + ".mod").c_str()));
  { std::vector<f32> M(g_vae.f("e.no.w"), g_vae.f("e.no.w") + 384), N(g_vae.f("e.no.b"), g_vae.f("e.no.b") + 384); ln_gen(s.data(), L, 384, M.data(), N.data()); }
  vlin("e.po", s.data(), L, 384, 128, tok);
  fprintf(stderr, "vae enc total %.2fs\n", now() - t0);
}

// text-encoder output cache (MAGE_CACHE=dir): the encoder output depends only on the token ids (+ the photo's ViT input when editing), so a new seed / "again" skips the ViT and the 36 layers
static uint64_t fnv(const void* p, size_t n, uint64_t h) { const uint8_t* b = (const uint8_t*)p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; } return h; }
static std::string tx_cache_path(const std::vector<int>& ids, int drop, const std::vector<std::vector<f32>>* pxs) {
  const char* d = getenv("MAGE_CACHE"); if (!d || !*d) return ""; uint64_t k = fnv(ids.data(), ids.size() * 4, 1469598103934665603ull); k = fnv(&drop, 4, k); if (pxs) for (const auto& px : *pxs) k = fnv(px.data(), px.size() * 4, k);
  char b[64]; snprintf(b, sizeof b, "/tx_%016llx.f32", (unsigned long long)k); return std::string(d) + b;
}
static bool tx_cache_load(const std::string& p, size_t n, std::vector<f32>& tx) { if (p.empty()) return false; FILE* f = fopen(p.c_str(), "rb"); if (!f) return false; tx.resize(n); bool ok = fread(tx.data(), 4, n, f) == n; fclose(f); return ok; }
static void tx_cache_save(const std::string& p, const std::vector<f32>& tx) { if (p.empty()) return; FILE* f = fopen(p.c_str(), "wb"); if (!f) return; fwrite(tx.data(), 4, tx.size(), f); fclose(f); }
static void finish_rgb(const std::vector<f32>& z, int h, int w) {   // MAGE_RGB=path: decode the latent tokens with the VAE and write fp32 rgb [H][W][3]
  const char* p = getenv("MAGE_RGB"); if (!p || !*p) return; std::vector<f32> rgb((size_t)h * w * 256 * 3); vae_decode(z.data(), h, w, rgb.data());
  FILE* f = fopen(p, "wb"); fwrite(rgb.data(), 4, rgb.size(), f); fclose(f);
}
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  rpcmem_init();
  struct remote_rpc_control_unsigned_module ud = {CDSP_DOMAIN_ID, 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof ud);
  if (mega_open(mega_URI CDSP_DOMAIN, &g_h)) { fprintf(stderr, "mega_open failed\n"); return 1; }
  printf("ready\n");
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd; ss >> cmd;
    if (cmd == "quit") break;
    if (cmd == "mem") { FILE* f = fopen("/proc/self/status", "r"); char l[256]; while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmHWM", 5) || !strncmp(l, "VmRSS", 5)) fputs(l, stdout); if (f) fclose(f); continue; }
    if (cmd == "load") {
      ss >> g_dir; double t0 = now();
      if (!g_glue.load(g_dir + "/glue.bin")) { printf("error glue\n"); continue; }
      bool ok = true; size_t mx = 0;
      for (int b = 0; b < NB && ok; b++) { char p[512]; snprintf(p, sizeof p, "%s/dit/blk%02d.bin", g_dir.c_str(), b); struct stat st; if (stat(p, &st)) ok = false; else mx = std::max<size_t>(mx, st.st_size); g_files.push_back(p); }
      for (int l = 0; l < NL_TE; l++) { char p[512]; snprintf(p, sizeof p, "%s/%s/te%02d.bin", g_dir.c_str(), getenv("MAGE_TE") ? getenv("MAGE_TE") : "te", l); struct stat st; if (stat(p, &st)) ok = false; else mx = std::max<size_t>(mx, st.st_size); g_tefiles.push_back(p); }
      { int fd = open((g_dir + "/" + (getenv("MAGE_TE") ? getenv("MAGE_TE") : "te") + "/embd.q8").c_str(), O_RDONLY); if (fd < 0) ok = false; else { struct stat st; fstat(fd, &st); g_embd_len = st.st_size; g_embd = (uint8_t*)mmap(nullptr, g_embd_len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); } }
      { std::vector<f32> v; if (!readraw((g_dir + "/" + (getenv("MAGE_TE") ? getenv("MAGE_TE") : "te") + "/tefinal.bin").c_str(), v, TD)) ok = false; g_tefinal = v; }
      if (!ok) { printf("error blocks\n"); continue; }
      g_te[0] = make_buf(mx); g_te[1] = make_buf(mx); g_all.push_back(&g_te[0]); g_all.push_back(&g_te[1]); if (!g_te[0].p || !g_te[1].p) { printf("error bufs\n"); continue; }
      if (!g_vae.load(g_dir + "/vae.bin")) { printf("error vae\n"); continue; }
      { struct stat st; if (!getenv("MAGE_NODSPVAE") && !stat((g_dir + "/vaepack.bin").c_str(), &st)) { g_vpack = make_buf(st.st_size); if (g_vpack.p && read_into(g_dir + "/vaepack.bin", g_vpack.p, g_vpack.sz, true) == (size_t)st.st_size) {
          sync_clean(g_vpack); FILE* f = fopen((g_dir + "/vaepack.idx").c_str(), "r"); char nm[64]; unsigned long long off; int K, N; while (f && fscanf(f, "%63s %llu %d %d", nm, &off, &K, &N) == 4) g_vp[nm] = {(size_t)off, K, N}; if (f) fclose(f); } } }
      printf("loaded %.1f\n", now() - t0); continue;
    }
    if (cmd == "gen") {   // gen TX.f32 NT H W SEED STEPS OUT.z    (TX: fp32 [NT][2560] text-encoder output after the final norm)
      std::string txf, outz; int nt, H, W, steps; unsigned seed; ss >> txf >> nt >> H >> W >> seed >> steps >> outz;
      std::vector<f32> tx; if (!readraw(txf.c_str(), tx, (size_t)nt * TD)) { printf("error tx\n"); continue; }
      if (H % 16 || W % 16 || ((H / 16) * (W / 16)) % 32 || steps < 1 || steps > 4) { printf("error args\n"); continue; }
      std::vector<f32> zt; if (!run_dit(tx, nt, H / 16, W / 16, seed, steps, outz, nullptr, &zt)) printf("error gen\n"); else { finish_rgb(zt, H / 16, W / 16); printf("done\n"); }
    }
    if (cmd == "vdec") {   // vdec Z.f32 H W OUT.rgb   (tokens [h*w][128] -> rgb fp32 [H][W][3])
      std::string zf, of; int h, w; ss >> zf >> h >> w >> of; std::vector<f32> z, rgb((size_t)h * w * 256 * 3);
      if (!readraw(zf.c_str(), z, (size_t)h * w * 128)) { printf("error z\n"); continue; } vae_decode(z.data(), h, w, rgb.data());
      FILE* f = fopen(of.c_str(), "wb"); fwrite(rgb.data(), 4, rgb.size(), f); fclose(f); printf("done\n"); continue;
    }
    if (cmd == "venc") {   // venc IN.rgb h w OUT.tok   (rgb fp32 [H][W][3] in [-1,1], h,w in latent units)
      std::string inf, of; int h, w; ss >> inf >> h >> w >> of; std::vector<f32> rgb, tok((size_t)h * w * 128);
      if (!readraw(inf.c_str(), rgb, (size_t)h * w * 256 * 3)) { printf("error rgb\n"); continue; } vae_encode(rgb.data(), h, w, tok.data());
      FILE* f = fopen(of.c_str(), "wb"); fwrite(tok.data(), 4, tok.size(), f); fclose(f); printf("done\n"); continue;
    }
    if (cmd == "genp") {   // genp IDS.i32 DROP H W SEED STEPS OUT.z    (token ids of the templated prompt)
      std::string idsf, outz; int drop, H, W, steps; unsigned seed; ss >> idsf >> drop >> H >> W >> seed >> steps >> outz;
      std::vector<int> ids; { FILE* f = fopen(idsf.c_str(), "rb"); if (!f) { printf("error ids\n"); continue; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(n / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) {} fclose(f); }
      if (ids.size() <= (size_t)drop || H % 16 || W % 16 || ((H / 16) * (W / 16)) % 32 || steps < 1 || steps > 4) { printf("error args\n"); continue; }
      std::vector<f32> tx; { const std::string cp = tx_cache_path(ids, drop, nullptr); if (tx_cache_load(cp, (ids.size() - drop) * TD, tx)) fprintf(stderr, "te: cached\n"); else { if (!encode_text(ids, drop, tx)) { printf("error te\n"); continue; } tx_cache_save(cp, tx); } }
      if (getenv("MAGE_DUMP_TX")) { FILE* f = fopen(getenv("MAGE_DUMP_TX"), "wb"); fwrite(tx.data(), 4, tx.size(), f); fclose(f); }
      std::vector<f32> zt; if (!run_dit(tx, (int)tx.size() / TD, H / 16, W / 16, seed, steps, outz, nullptr, &zt)) printf("error gen\n"); else { finish_rgb(zt, H / 16, W / 16); printf("done\n"); }
    }
    if (cmd == "edit") {   // edit DIR DROP SEED STEPS OUT.z : DIR/{meta.txt: one line "gh gw h w" per picture (1 to 3; the first is the picture to edit, the others are extra references), ids.i32 (image pads expanded, one run per picture), px.f32 [gh*gw][1536] and ref.rgb fp32 [h*16][w*16][3] in [-1,1] for the first picture, px2.f32 / ref2.rgb, px3.f32 / ref3.rgb for the others}
      std::string dir, outz; int drop, steps; unsigned seed; ss >> dir >> drop >> seed >> steps >> outz;
      std::vector<std::array<int, 4>> mt; { FILE* mf = fopen((dir + "/meta.txt").c_str(), "r"); int a, b, c, d; while (mf && mt.size() < 3 && fscanf(mf, "%d %d %d %d", &a, &b, &c, &d) == 4) mt.push_back({a, b, c, d}); if (mf) fclose(mf); }
      if (mt.empty()) { printf("error meta\n"); continue; }
      std::vector<int> ids; { FILE* f = fopen((dir + "/ids.i32").c_str(), "rb"); if (!f) { printf("error ids\n"); continue; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(n / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) {} fclose(f); }
      std::vector<EditIn> eds(mt.size()); bool bad = false;
      { size_t ei = 0; for (size_t i = 0; i < ids.size(); i++) if (ids[i] == 151655 && (i == 0 || ids[i - 1] != 151655)) { if (ei >= eds.size()) { bad = true; break; } eds[ei].k0 = (int)i; size_t j = i; while (j < ids.size() && ids[j] == 151655) j++; if ((int)(j - i) != mt[ei][0] * mt[ei][1] / 4) bad = true; ei++; }
        if (ei != eds.size()) bad = true; }
      if (bad) { printf("error pads\n"); continue; }
      for (size_t i = 0; i < eds.size(); i++) { eds[i].on = true; eds[i].gh = mt[i][0]; eds[i].gw = mt[i][1]; eds[i].nm = mt[i][0] * mt[i][1] / 4; }
      const int h = mt[0][2], w = mt[0][3]; long NIall = (long)h * w; for (auto& m : mt) NIall += (long)m[2] * m[3];
      if (((h * w) % 32) || NIall > 60000) { printf("error size\n"); continue; }
      double t0 = now(); std::vector<RefImg> refs(mt.size()); for (size_t i = 0; i < mt.size(); i++) { refs[i].h = mt[i][2]; refs[i].w = mt[i][3]; refs[i].tok.resize((size_t)refs[i].h * refs[i].w * 128); }
      const std::string sfx[3] = {"", "2", "3"};
      std::thread enc_thread; bool ref_ok = true;   // the VAE encoder runs on the CPU in parallel with the (flash-bound) ViT / text encoder
      { auto* rgbs = new std::vector<std::vector<f32>>(mt.size()); for (size_t i = 0; i < mt.size(); i++) if (!readraw((dir + "/ref" + sfx[i] + ".rgb").c_str(), (*rgbs)[i], (size_t)refs[i].h * refs[i].w * 256 * 3)) ref_ok = false;
        if (!ref_ok) { printf("error ref\n"); delete rgbs; continue; }
        enc_thread = std::thread([&, rgbs] { tl_cpu_only = true; for (size_t i = 0; i < refs.size(); i++) vae_encode((*rgbs)[i].data(), refs[i].h, refs[i].w, refs[i].tok.data()); delete rgbs; }); }
      std::vector<std::vector<f32>> pxs(mt.size()); bool pxok = true; for (size_t i = 0; i < mt.size(); i++) if (!readraw((dir + "/px" + sfx[i] + ".f32").c_str(), pxs[i], (size_t)mt[i][0] * mt[i][1] * 1536)) pxok = false;
      if (!pxok) { printf("error px\n"); enc_thread.join(); continue; }
      std::vector<f32> tx; const std::string cp = tx_cache_path(ids, drop, &pxs);
      if (tx_cache_load(cp, (ids.size() - drop) * TD, tx)) fprintf(stderr, "te: cached (no ViT)\n");
      else {
        if (!g_vglue.base && !g_vglue.load(g_dir + "/vit/vit_glue.bin")) { printf("error vitglue\n"); enc_thread.join(); continue; }
        bool vok = true; for (size_t i = 0; i < eds.size() && vok; i++) vok = vit_features(pxs[i], eds[i].gh, eds[i].gw, eds[i].pooled, eds[i].ds);
        if (!vok) { printf("error vit\n"); enc_thread.join(); continue; }
        if (!encode_text(ids, drop, tx, &eds)) { printf("error te\n"); enc_thread.join(); continue; }
        tx_cache_save(cp, tx); }
      enc_thread.join();
      std::vector<f32> zt; if (!run_dit(tx, (int)tx.size() / TD, h, w, seed, steps, outz, &refs, &zt)) { printf("error gen\n"); continue; }
      finish_rgb(zt, h, w); fprintf(stderr, "edit total %.2fs\n", now() - t0); printf("done\n");
    }
  }
  mega_close(g_h);
  return 0;
}
