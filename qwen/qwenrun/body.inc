
// ------------------------------------------------------------------ activation blob layout
struct Layout { int meta[M_NMETA]; size_t total; };
static Layout make_layout(int ntok, int extra_slabs, bool rows) {   // hid 4096; K/V hold extra_slabs of cached prefix after the own rows
  Layout L; memset(&L, 0, sizeof L); const size_t ns = ntok / 32, hid = 4096; size_t off = 0, kvb = ((size_t)ntok + 32 * extra_slabs) * hid * 2;
  struct E { int id; size_t b; } ent[] = {
    {M_X, (size_t)ntok * hid * 4}, {M_COS, ns * 8192}, {M_SIN, ns * 8192}, {M_H, (size_t)ntok * hid * 2}, {M_Q, (size_t)ntok * hid * 2}, {M_K, kvb}, {M_V, kvb},
    {M_O, (size_t)ntok * hid * 2}, {M_T, (size_t)ntok * hid * 2}, {M_MID, (size_t)ntok * 12288 * 2}, {M_ROWS, rows ? (size_t)ntok * (10240 + 384) * 4 : 4096}, {M_CS, (size_t)ntok * 128 * 4},
    {M_LQ, ns * 12 * 2048}, {M_LO, ns * 4 * 2048}, {M_LU, ns * 4 * 2048}, {M_LD, ns * 4 * 2048}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}

// ------------------------------------------------------------------ prefix K/V file helpers (the prefix cache is read layer by layer instead of living in RAM)
static void free_buf(Buf& b) { if (b.p) { unmap_buf(b); rpcmem_free(b.p); b.p = nullptr; b.mapped = false; } }
static bool kv_write(int fd, off_t off, const uint8_t* p, size_t n) { size_t d = 0; while (d < n) { ssize_t r = pwrite(fd, p + d, n - d, off + d); if (r <= 0) return false; d += r; } return true; }
static bool kv_read(int fd, off_t off, uint8_t* p, size_t n) {   // 4 parallel preads (8 MB chunks)
  const size_t chunk = 8u << 20; std::atomic<size_t> next(0); std::atomic<bool> bad(false);
  auto work = [&] { for (;;) { size_t o = next.fetch_add(chunk); if (o >= n) break; size_t len = std::min(chunk, n - o), d = 0; while (d < len) { ssize_t r = pread(fd, p + o + d, len - d, off + o + d); if (r <= 0) { bad = true; return; } d += r; } } };
  std::vector<std::thread> th; for (int i = 0; i < 3; i++) th.emplace_back(work); work(); for (auto& t : th) t.join();
  return !bad.load();
}

// ------------------------------------------------------------------ state
static remote_handle64 g_h = 0;
static Glue g_glue; static uint8_t* g_embd = nullptr; static size_t g_embd_len = 0; static std::string g_dir;
static Buf g_te[2];                       // streaming pack buffers (TE layers and DiT blocks)
static Buf g_blobTE; static Layout g_LT;  // TE blob (MAXT tokens)
static Buf g_blobP, g_blobD; static Layout g_LP, g_LD; static int g_P_ntok = -1, g_D_ntok = -1, g_D_extra = -1;

static int g_profn = 0;
static int call(const Layout& L, int* meta, Buf& w, size_t woff, const Buf& blob, const f32* mod, int nmod) {
  int times[32]; int rc = 0; meta[M_WOFF] = (int)woff;
  if (!map_buf(w)) return -1;
  int r = mega_block(g_h, meta, M_NMETA, w.fd, blob.fd, (const uint8_t*)mod, nmod * 4, times, 32, &rc);
  if (getenv("QWEN_PROF") && (meta[M_MODE] == 22 || meta[M_MODE] == 40 || meta[M_MODE] == 23) && (g_profn++ % 12) == 5) { fprintf(stderr, "prof mode %d:", meta[M_MODE]); for (int i = 0; i < 12; i++) fprintf(stderr, " %d", times[i]); fprintf(stderr, "\n"); }
  if (r || rc) { fprintf(stderr, "mega_block r=0x%x rc=%d mode=%d\n", r, rc, meta[M_MODE]); return -1; }
  return 0;
}

// Y[n][out] = X[n][in] * W[out][in]^T (W fp16, X/Y fp32, in % 4 == 0). 4 weight rows are widened to fp32 once, then a 4-token x 4-output register tile runs 16 FMAs per 8 loads; blocks of outputs are handed out dynamically to all cores (big and little).
static void gemm_f16(const __fp16* W, const f32* X, f32* Y, int n, int out, int in) {
  const int OB = 4, nblk = (out + OB - 1) / OB; std::atomic<int> next(0);
  auto worker = [&] {
    std::vector<f32> w((size_t)OB * in + 16);
    for (;;) {
      const int blk = next.fetch_add(1); if (blk >= nblk) break;
      const int o0 = blk * OB, ob = std::min(OB, out - o0);
      for (int r = 0; r < OB; r++) { f32* wr = &w[(size_t)r * in]; if (r < ob) { const __fp16* src = W + (size_t)(o0 + r) * in; for (int k = 0; k < in; k += 4) vst1q_f32(wr + k, vcvt_f32_f16(vld1_f16(src + k))); } else memset(wr, 0, (size_t)in * 4); }
      const f32 *w0 = &w[0], *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in; int t = 0;
      for (; t + 4 <= n; t += 4) {
        const f32 *x0 = X + (size_t)t * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
        float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00, a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
        for (int k = 0; k < in; k += 4) {
          const float32x4_t v0 = vld1q_f32(w0 + k), v1 = vld1q_f32(w1 + k), v2 = vld1q_f32(w2 + k), v3 = vld1q_f32(w3 + k);
          float32x4_t xv = vld1q_f32(x0 + k); a00 = vfmaq_f32(a00, xv, v0); a01 = vfmaq_f32(a01, xv, v1); a02 = vfmaq_f32(a02, xv, v2); a03 = vfmaq_f32(a03, xv, v3);
          xv = vld1q_f32(x1 + k); a10 = vfmaq_f32(a10, xv, v0); a11 = vfmaq_f32(a11, xv, v1); a12 = vfmaq_f32(a12, xv, v2); a13 = vfmaq_f32(a13, xv, v3);
          xv = vld1q_f32(x2 + k); a20 = vfmaq_f32(a20, xv, v0); a21 = vfmaq_f32(a21, xv, v1); a22 = vfmaq_f32(a22, xv, v2); a23 = vfmaq_f32(a23, xv, v3);
          xv = vld1q_f32(x3 + k); a30 = vfmaq_f32(a30, xv, v0); a31 = vfmaq_f32(a31, xv, v1); a32 = vfmaq_f32(a32, xv, v2); a33 = vfmaq_f32(a33, xv, v3);
        }
        const float32x4_t acc[4][4] = {{a00, a01, a02, a03}, {a10, a11, a12, a13}, {a20, a21, a22, a23}, {a30, a31, a32, a33}};
        for (int i = 0; i < 4; i++) for (int r = 0; r < ob; r++) Y[(size_t)(t + i) * out + o0 + r] = vaddvq_f32(acc[i][r]);
      }
      for (; t < n; t++) {
        const f32* x0 = X + (size_t)t * in; float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int k = 0; k < in; k += 4) { const float32x4_t xv = vld1q_f32(x0 + k); a0 = vfmaq_f32(a0, xv, vld1q_f32(w0 + k)); a1 = vfmaq_f32(a1, xv, vld1q_f32(w1 + k)); a2 = vfmaq_f32(a2, xv, vld1q_f32(w2 + k)); a3 = vfmaq_f32(a3, xv, vld1q_f32(w3 + k)); }
        const float32x4_t acc[4] = {a0, a1, a2, a3}; for (int r = 0; r < ob; r++) Y[(size_t)t * out + o0 + r] = vaddvq_f32(acc[r]);
      }
    }
  };
  const int nt = std::max(1, std::min<int>(8, (int)std::thread::hardware_concurrency())); std::vector<std::thread> th;
  for (int i = 1; i < nt; i++) th.emplace_back(worker); worker(); for (auto& x : th) x.join();
}
static bool readraw(const char* p, std::vector<f32>& v, size_t n) { FILE* f = fopen(p, "rb"); if (!f) return false; v.resize(n); bool ok = fread(v.data(), 4, n, f) == n; fclose(f); return ok; }

// ------------------------------------------------------------------ vision tower (Qwen3-VL ViT, 27 layers) on the NPU: MODE 23 layers, host patch-embed / position embed / mergers
static Glue g_vglue; static Buf g_blobV; static Layout g_LV; static int g_V_ntok = -1;
static const int NLAYER_VIT = 27;
static Layout make_layout_vit(int ntok) {
  Layout L; memset(&L, 0, sizeof L); const size_t ns = ntok / 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, (size_t)ntok * 1152 * 4}, {M_COS, ns * 8192}, {M_SIN, ns * 8192}, {M_H, (size_t)ntok * 1152 * 2}, {M_Q, (size_t)ntok * 2048 * 2}, {M_K, (size_t)ntok * 2048 * 2}, {M_V, (size_t)ntok * 2048 * 2},
    {M_O, (size_t)ntok * 2048 * 2}, {M_T, (size_t)ntok * 1152 * 2}, {M_MID, (size_t)ntok * 4352 * 2}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LQ, ns * 12 * 2048}, {M_LO, ns * 4 * 2048}, {M_LU, ns * 4 * 2048}, {M_LD, ns * 4 * 2048}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
static void fill_bias_tiles(uint8_t* blob, const Layout& L, int ns) {   // L buffers: fp16 1.0 in column 0 of every 4th tile (selects row 0 of the 4 extra K tiles = the bias)
  auto fill = [&](int id, size_t ntiles) { __fp16* t = (__fp16*)(blob + L.meta[id]); memset(t, 0, ntiles * 2048);
    for (size_t i = 0; i < ntiles; i += 4) for (int r = 0; r < 32; r++) t[i * 1024 + (r / 2) * 64 + (r & 1)] = (__fp16)1.0f; };
  fill(M_LQ, (size_t)ns * 12); fill(M_LO, (size_t)ns * 4); fill(M_LU, (size_t)ns * 4); fill(M_LD, (size_t)ns * 4);
}
static void layer_norm_rows(f32* x, int n, int d, const f32* w, const f32* b) { par_for(n, [&](int t) { f32* r = x + (size_t)t * d; double mu = 0, ss = 0; for (int i = 0; i < d; i++) mu += r[i]; mu /= d; for (int i = 0; i < d; i++) ss += (r[i] - mu) * (r[i] - mu); const float rs = 1.f / sqrtf((float)(ss / d) + 1e-6f); for (int i = 0; i < d; i++) r[i] = (float)((r[i] - mu) * rs) * w[i] + b[i]; }); }
static void merger(const f32* rows, int nm, bool postshuffle, const char* pre, std::vector<f32>& out) {   // rows [4*nm][1152] (merge order) -> out [nm][4096]
  std::vector<f32> x(rows, rows + (size_t)nm * 4608); const std::string p = pre;
  if (postshuffle) layer_norm_rows(x.data(), nm, 4608, g_vglue.f((p + ".norm.weight").c_str()), g_vglue.f((p + ".norm.bias").c_str()));
  else { layer_norm_rows(x.data(), nm * 4, 1152, g_vglue.f((p + ".norm.weight").c_str()), g_vglue.f((p + ".norm.bias").c_str())); }
  std::vector<f32> a((size_t)nm * 4608); gemm_f16(g_vglue.h((p + ".fc1.weight").c_str()), x.data(), a.data(), nm, 4608, 4608);
  const f32* b1 = g_vglue.f((p + ".fc1.bias").c_str()); for (int t = 0; t < nm; t++) for (int i = 0; i < 4608; i++) { float v = a[(size_t)t * 4608 + i] + b1[i]; a[(size_t)t * 4608 + i] = 0.5f * v * (1.f + erff(v * 0.70710678f)); }
  out.assign((size_t)nm * 4096, 0.f); gemm_f16(g_vglue.h((p + ".fc2.weight").c_str()), a.data(), out.data(), nm, 4096, 4608);
  const f32* b2 = g_vglue.f((p + ".fc2.bias").c_str()); for (int t = 0; t < nm; t++) for (int i = 0; i < 4096; i++) out[(size_t)t * 4096 + i] += b2[i];
}
// px [Np][1536] (normalised, merge-ordered patches, see vit_ref.prep) on a gh x gw patch grid -> pooled [nm][4096] and 3 deepstack feature sets
static bool vit_features(const std::vector<f32>& px, int gh, int gw, std::vector<f32>& pooled, std::vector<f32> ds[3]) {
  const int Np = gh * gw, nm = Np / 4, ns = (Np + 31) / 32, ntok = ns * 32; double t0 = now();
  std::vector<f32> rows((size_t)ntok * 1152, 0.f);
  gemm_f16(g_vglue.h("patch.weight"), px.data(), rows.data(), Np, 1152, 1536);
  { const f32* pb = g_vglue.f("patch.bias"); const __fp16* pe = g_vglue.h("pos_embed"); const int side = 48;
    auto taps = [&](int idx, int size, int* t, float* wgt) { const float src = size > 1 ? idx * (float)(side - 1) / (size - 1) : 0.f; const int fl = (int)floorf(src);
      for (int k = 0; k < 2; k++) { t[k] = std::min(std::max(fl + k, 0), side - 1); wgt[k] = std::max(0.f, 1.f - fabsf(src - fl - k)); } };
    par_for(Np, [&](int p) {   // merge-order patch p -> (row, col)
      const int blk = p / 4, ir = (p / 2) % 2, ic = p % 2, bw = gw / 2; const int row = (blk / bw) * 2 + ir, col = (blk % bw) * 2 + ic;
      int th[2], tw[2]; float wh[2], ww[2]; taps(row, gh, th, wh); taps(col, gw, tw, ww); f32* r = &rows[(size_t)p * 1152];
      for (int i = 0; i < 1152; i++) { float a = pb[i]; for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) a += wh[u] * ww[v] * (float)pe[(size_t)(th[u] * side + tw[v]) * 1152 + i]; r[i] += a; } }); }
  if (g_V_ntok != ntok) { if (g_blobV.p) { unmap_buf(g_blobV); rpcmem_free(g_blobV.p); } g_LV = make_layout_vit(ntok); g_blobV = make_buf(g_LV.total, true); if (!g_blobV.p) return false; g_V_ntok = ntok; }
  uint8_t* blob = g_blobV.p; memset(blob, 0, g_LV.total); fill_bias_tiles(blob, g_LV, ns);
  rows_to_x32(rows.data(), Np, 1152, (f32*)(blob + g_LV.meta[M_X]), ns);
  { f32* cs = (f32*)(blob + g_LV.meta[M_COS]); f32* sn = (f32*)(blob + g_LV.meta[M_SIN]);
    par_for(Np, [&](int p) { const int blk = p / 4, ir = (p / 2) % 2, ic = p % 2, bw = gw / 2; const int row = (blk / bw) * 2 + ir, col = (blk % bw) * 2 + ic;
      for (int d = 0; d < 36; d++) { const double inv = pow(10000.0, -2.0 * (d % 18) / 36.0); const double a = (d < 18 ? row : col) * inv; const size_t idx = (((size_t)(p / 32) * 2 + d / 32) * 1024) + (p % 32) * 32 + d % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } }); }
  std::vector<f32> mod(18432, 0.f); for (int i = 0; i < 1152; i++) mod[i] = 1.f;
  sync_clean(g_blobV);
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = g_LV.meta[i];
  meta[M_NTOK] = ntok; meta[M_NVALID] = Np; meta[M_NW] = 3; meta[M_NCB] = 36; meta[M_MODE] = 23; meta[M_NPRE] = 0; meta[M_CAUS] = 0; meta[M_BA] = meta[M_BB] = meta[M_B2A] = meta[M_B2B] = meta[M_B3A] = meta[M_B3B] = 0;
  std::vector<std::string> vf; for (int i = 0; i < NLAYER_VIT; i++) { char p[512]; snprintf(p, sizeof p, "%s/vit/vit%02d.bin", g_dir.c_str(), i); vf.push_back(p); }
  std::vector<f32> xr((size_t)Np * 1152); bool ok = true;
  auto tap = [&](int k) -> bool { sync_inval(g_blobV); x32_to_rows((const f32*)(blob + g_LV.meta[M_X]), Np, 1152, xr.data()); return true; };
  ok = stream_files(vf, g_te, [&](int k, Buf& b) {
    const bool rd = k == 8 || k == 16 || k == 24 || k == NLAYER_VIT - 1; meta[M_FIRST] = k == 0; meta[M_REL] = rd;
    if (call(g_LV, meta, b, 0, g_blobV, mod.data(), 18432)) return false;
    if (rd) { tap(k); const int di = k == 8 ? 0 : k == 16 ? 1 : k == 24 ? 2 : -1;
      if (di >= 0) merger(xr.data(), nm, true, (std::string("ds") + char('0' + di)).c_str(), ds[di]);
      else merger(xr.data(), nm, false, "merger", pooled); }
    return true; });
  fprintf(stderr, "vit: %.2fs (%d patches)\n", now() - t0, Np); return ok;
}

// ------------------------------------------------------------------ editing inputs (QWEN_EDIT=dir: up to 3 condition images; image 0 lives in dir, images 1 and 2 in dir/1 and dir/2;
// each has meta.txt "CW CH", cond.f32 [Nc][64] normalised latent tokens or cond_raw.f32 (vaeenc), vit.f32 = pooled | ds0 | ds1 | ds2 (each [nm][4096]) or px.f32 (ViT patches))
struct EditImg { int CW = 0, CH = 0, hc = 0, wc = 0, Nc = 0, nm = 0, k0 = 0; std::vector<f32> cond, pooled, ds[3]; };
struct EditIn { bool on = false; std::vector<EditImg> im; };
static std::string edit_dir(int j) { const char* d = getenv("QWEN_EDIT"); return j == 0 ? std::string(d) : std::string(d) + "/" + std::to_string(j); }
static bool load_edit(const std::vector<int>& ids, EditIn& e, bool need_vit = true) {
  const char* d = getenv("QWEN_EDIT"); if (!d || !*d) return true;
  for (int j = 0; j < 3; j++) {
    const std::string dir = edit_dir(j); FILE* f = fopen((dir + "/meta.txt").c_str(), "r"); if (!f) { if (j == 0) fprintf(stderr, "no meta.txt\n"); break; }
    EditImg m; if (fscanf(f, "%d %d", &m.CW, &m.CH) != 2) { fclose(f); return false; } fclose(f);
    m.hc = m.CH / 16; m.wc = m.CW / 16; m.Nc = m.hc * m.wc; m.nm = m.Nc / 4; e.im.push_back(m);
  }
  if (e.im.empty()) return false;
  e.on = true;
  { int j = 0; for (size_t i = 0; i < ids.size(); ) { if (ids[i] != 151655) { i++; continue; }
      size_t n = 0; const size_t i0 = i; while (i < ids.size() && ids[i] == 151655) { i++; n++; }
      if (j >= (int)e.im.size() || (int)n != e.im[j].nm) { fprintf(stderr, "image pad run %d: %d != expected\n", j, (int)n); return false; }
      e.im[j++].k0 = (int)i0; }
    if (j != (int)e.im.size()) { fprintf(stderr, "image pad runs %d != images %d\n", j, (int)e.im.size()); return false; } }
  for (size_t j = 0; j < e.im.size(); j++) { EditImg& m = e.im[j]; const std::string dir = edit_dir((int)j);
    if (!readraw((dir + "/cond.f32").c_str(), m.cond, (size_t)m.Nc * 64)) {   // raw encoder mean [64][hc][wc] (vaeenc) -> normalised tokens (z-mean)/std
      std::vector<f32> raw; if (!readraw((dir + "/cond_raw.f32").c_str(), raw, (size_t)m.Nc * 64)) { fprintf(stderr, "need cond.f32 or cond_raw.f32\n"); return false; }
      const f32* mean = g_glue.f("vae.mean"); const f32* sd = g_glue.f("vae.std"); m.cond.assign((size_t)m.Nc * 64, 0.f);
      for (int t = 0; t < m.Nc; t++) for (int c = 0; c < 64; c++) m.cond[(size_t)t * 64 + c] = (raw[(size_t)c * m.Nc + t] - mean[c]) / sd[c];
    }
  }
  if (!need_vit) return true;   // the text encoder output is cached: no ViT
  for (size_t j = 0; j < e.im.size(); j++) { EditImg& m = e.im[j]; const std::string dir = edit_dir((int)j); std::vector<f32> v;
    if (readraw((dir + "/vit.f32").c_str(), v, (size_t)4 * m.nm * 4096)) {   // precomputed features (laptop reference)
      m.pooled.assign(v.begin(), v.begin() + (size_t)m.nm * 4096); for (int i = 0; i < 3; i++) m.ds[i].assign(v.begin() + (size_t)(i + 1) * m.nm * 4096, v.begin() + (size_t)(i + 2) * m.nm * 4096);
      continue; }
    std::vector<f32> px; if (!readraw((dir + "/px.f32").c_str(), px, (size_t)m.Nc * 1536)) { fprintf(stderr, "need vit.f32 or px.f32\n"); return false; }
    if (!g_vglue.base && !g_vglue.load(g_dir + "/vit/vit_glue.bin")) { fprintf(stderr, "no vit_glue.bin\n"); return false; }
    if (!vit_features(px, m.hc, m.wc, m.pooled, m.ds)) return false;
    if (j == 0 && getenv("QWEN_DUMP_VIT")) { FILE* g = fopen(getenv("QWEN_DUMP_VIT"), "wb"); fwrite(m.pooled.data(), 4, m.pooled.size(), g); for (int i = 0; i < 3; i++) fwrite(m.ds[i].data(), 4, m.ds[i].size(), g); fclose(g); }
  }
  free_buf(g_blobV); g_V_ntok = -1;   // the ViT blob is not needed any more
  return true;
}
static void add_rows_x32(f32* X, int row0, int nrows, const f32* feat) {   // X x32 tiles [slab][128 cb][32][32]: rows row0.. += feat[r][4096]
  for (int r = 0; r < nrows; r++) { const int row = row0 + r; const int s = row / 32, rr = row % 32;
    for (int cb = 0; cb < 128; cb++) { f32* t = X + ((size_t)s * 128 + cb) * 1024 + (size_t)rr * 32; const f32* a = feat + (size_t)r * 4096 + cb * 32; for (int c = 0; c < 32; c++) t[c] += a[c]; } }
}

// ------------------------------------------------------------------ text encoder (Qwen3-VL-8B, 36 layers) on the NPU
static bool encode_text(const std::vector<int>& ids, int drop, std::vector<f32>& tx, const EditIn* ed = nullptr) {   // tx [T-drop][4096] = last layer output before the final norm
  if (!g_blobTE.p) { g_blobTE = make_buf(g_LT.total, true); if (!g_blobTE.p) return false; memset(g_blobTE.p, 0, g_LT.total); }   // allocated for the text phase only
  const int T = (int)ids.size(); const int ntokT = (T + 31) / 32 * 32, ns = ntokT / 32; uint8_t* blob = g_blobTE.p; const Layout& L = g_LT;
  double t0 = now();
  std::vector<f32> emb((size_t)T * 4096);
  par_for(T, [&](int i) { q4k_row(g_embd + (size_t)ids[i] * 2304, &emb[(size_t)i * 4096]); });
  if (ed && ed->on) for (const EditImg& m : ed->im) memcpy(&emb[(size_t)m.k0 * 4096], m.pooled.data(), (size_t)m.nm * 4096 * 4);   // image slots carry the ViT features
  rows_to_x32(emb.data(), T, 4096, (f32*)(blob + L.meta[M_X]), ns);
  { f32* cs = (f32*)(blob + L.meta[M_CS]); memset(cs, 0, (size_t)ntokT * 128 * 4);
    std::vector<double> pos((size_t)T * 3);
    for (int t = 0; t < T; t++) pos[t * 3] = pos[t * 3 + 1] = pos[t * 3 + 2] = t;
    if (ed && ed->on) {   // interleaved mrope: text advances all axes; a merged image grid (hc/2 x wc/2) carries (t, h, w) = (cur, cur+row, cur+col), then cur += max(gh, gw)
      int t = 0; double cur = 0; size_t nxt = 0;
      while (t < T) {
        if (nxt < ed->im.size() && t == ed->im[nxt].k0) { const EditImg& m = ed->im[nxt++]; const int gh = m.hc / 2, gw = m.wc / 2;
          for (int r = 0; r < gh; r++) for (int c = 0; c < gw; c++) { const int j = t + r * gw + c; pos[j * 3] = cur; pos[j * 3 + 1] = cur + r; pos[j * 3 + 2] = cur + c; }
          t += m.nm; cur += std::max(gh, gw); }
        else { pos[t * 3] = pos[t * 3 + 1] = pos[t * 3 + 2] = cur; t++; cur += 1; }
      }
    }
    for (int t = 0; t < T; t++) for (int i = 0; i < 64; i++) { const int ax = (i % 3 == 1 && i < 60) ? 1 : (i % 3 == 2 && i < 60) ? 2 : 0; const double a = pos[t * 3 + ax] * pow(5e6, -(double)i / 64); cs[(size_t)t * 128 + i] = (f32)cos(a); cs[(size_t)t * 128 + 64 + i] = (f32)sin(a); }
    f32* ct = (f32*)(blob + L.meta[M_COS]); f32* st = (f32*)(blob + L.meta[M_SIN]); memset(ct, 0, (size_t)ns * 8192); memset(st, 0, (size_t)ns * 8192);   // tile tables for the epilogue rope: [slab][2][32][32]
    par_for(T, [&](int t) { for (int i = 0; i < 64; i++) { const int ax = (i % 3 == 1 && i < 60) ? 1 : (i % 3 == 2 && i < 60) ? 2 : 0; const double a = pos[t * 3 + ax] * pow(5e6, -(double)i / 64);
      const size_t idx = (((size_t)(t / 32) * 2 + i / 32) * 1024) + (t % 32) * 32 + i % 32; ct[idx] = (f32)cos(a); st[idx] = (f32)sin(a); } }); }
  std::vector<f32> mod(18432, 0.f); for (int i = 0; i < 4096; i++) mod[8192 + i] = 1.f;   // residual gate = 1
  memset(blob + L.meta[M_H], 0, (size_t)ntokT * 4096 * 2);
  sync_clean(g_blobTE);
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = L.meta[i];
  meta[M_NTOK] = ntokT; meta[M_NVALID] = T; meta[M_NW] = 3; meta[M_NCB] = 128; meta[M_NPRE] = 0; meta[M_CAUS] = 0; meta[M_FMT] = getenv("QWEN_TE_ROWS") ? 0 : 1;   // 1 = HMX tile-domain attention (fast); QWEN_TE_ROWS=1 selects the fp32 row-domain one
  std::vector<std::string> tef; for (int i = 0; i < NLAYER_TE; i++) { char p[512]; snprintf(p, sizeof p, "%s/te/te%02d.bin", g_dir.c_str(), i); tef.push_back(p); }
  const bool ds = ed && ed->on;
  bool ok = stream_files(tef, g_te, [&](int i, Buf& b) {
    meta[M_MODE] = 22; meta[M_FIRST] = i == 0 || (ds && i >= 1 && i <= 3); meta[M_REL] = i == NLAYER_TE - 1 || (ds && i < 3);
    if (call(L, meta, b, 0, g_blobTE, mod.data(), 18432)) return false;
    if (ds && i < 3) { sync_inval(g_blobTE); for (const EditImg& m : ed->im) add_rows_x32((f32*)(blob + L.meta[M_X]), m.k0, m.nm, m.ds[i].data()); sync_clean(g_blobTE); }   // DeepStack
    return true; });
  if (!ok) return false;
  sync_inval(g_blobTE);
  std::vector<f32> rows((size_t)T * 4096); x32_to_rows((const f32*)(blob + L.meta[M_X]), T, 4096, rows.data());
  tx.assign(rows.begin() + (size_t)drop * 4096, rows.end());
  free_buf(g_blobTE);
  fprintf(stderr, "te: %.2fs (%d tokens, %d kept)\n", now() - t0, T, T - drop);
  return true;
}

// ------------------------------------------------------------------ CPU glue
static void matvec_f32(const f32* W, const f32* x, int out, int in, f32* y) { par_for(out, [&](int o) { const f32* w = W + (size_t)o * in; float s = 0; for (int i = 0; i < in; i++) s += w[i] * x[i]; y[o] = s; }); }
static void txt_in(std::vector<f32>& tx, int n) {   // RMSNorm(w+1) -> Linear -> GELU(tanh) -> Linear, in place on tx [n][4096]
  const f32* nw = g_glue.f("txt_norm.weight");
  for (int t = 0; t < n; t++) { f32* r = &tx[(size_t)t * 4096]; double ss = 0; for (int i = 0; i < 4096; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / 4096) + 1e-6f); for (int i = 0; i < 4096; i++) r[i] = r[i] * rs * nw[i]; }
  std::vector<f32> a((size_t)n * 4096);
  gemm_f16(g_glue.h("txt_in.weight"), tx.data(), a.data(), n, 4096, 4096);
  for (auto& v : a) v = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
  gemm_f16(g_glue.h("txt_out.weight"), a.data(), tx.data(), n, 4096, 4096);
}
static void time_mod(float sigma, std::vector<f32>& mod, std::vector<f32>* nscale) {   // mod = [scale1, tanh(gate1), scale2, tanh(gate2)] x 4096 (+pad to 18432); nscale = norm_out scale
  f32 e[256]; for (int i = 0; i < 128; i++) { const double a = 1000.0 * sigma * exp(-log(10000.0) * i / 128); e[i] = (f32)cos(a); e[128 + i] = (f32)sin(a); }
  std::vector<f32> a(4096), b(4096), st(4096), m(16384);
  matvec_f32(g_glue.f("time.linear_1.weight"), e, 4096, 256, a.data()); for (auto& v : a) v = v / (1.f + expf(-v));
  gemm_f16(g_glue.h("time.linear_2.weight"), a.data(), b.data(), 1, 4096, 4096);
  for (int i = 0; i < 4096; i++) st[i] = b[i] / (1.f + expf(-b[i]));
  gemm_f16(g_glue.h("modulation.weight"), st.data(), m.data(), 1, 16384, 4096);
  mod.assign(18432, 0.f);
  for (int i = 0; i < 4096; i++) { mod[i] = m[i]; mod[4096 + i] = tanhf(m[4096 + i]); mod[8192 + i] = m[8192 + i]; mod[12288 + i] = tanhf(m[12288 + i]); }
  if (nscale) { nscale->resize(4096); gemm_f16(g_glue.h("norm_out.weight"), st.data(), nscale->data(), 1, 4096, 4096); }
}
static void rope_rows(const double (*pos)[3], int nrow, int ntok, f32* cs, f32* sn) {   // tiles [slab][2][32][32]; axes (frame,h,w) pairs 8/28/28, theta 10000
  memset(cs, 0, (size_t)ntok / 32 * 8192); memset(sn, 0, (size_t)ntok / 32 * 8192);
  par_for(nrow, [&](int tk) { for (int d = 0; d < 64; d++) {
    const int ax = d < 8 ? 0 : d < 36 ? 1 : 2, j = ax == 0 ? d : ax == 1 ? d - 8 : d - 36, np = ax == 0 ? 8 : 28;
    const double a = pos[tk][ax] * pow(10000.0, -(double)j / np); const size_t idx = (((size_t)(tk / 32) * 2 + d / 32) * 1024) + (tk % 32) * 32 + d % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } });
}

// ------------------------------------------------------------------ generation
// text-encoder cache (QWEN_CACHE=dir): the encoder output depends on the tokens (and, when editing, the photo) only, so another seed / "again" skips the ViT and the 36 encoder layers
static uint64_t fnv(const void* p, size_t n, uint64_t h) { const uint8_t* b = (const uint8_t*)p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; } return h; }
static uint64_t tx_key(const std::vector<int>& ids, int drop) {
  uint64_t h = fnv("qwen-te-v2", 10, 1469598103934665603ull); h = fnv(ids.data(), ids.size() * 4, h); h = fnv(&drop, 4, h);
  const char* d = getenv("QWEN_EDIT"); if (d && *d) for (int j = 0; j < 3; j++) { FILE* f = fopen((edit_dir(j) + "/px.f32").c_str(), "rb"); if (!f) break; std::vector<uint8_t> b(1 << 20); size_t r; while ((r = fread(b.data(), 1, b.size(), f)) > 0) h = fnv(b.data(), r, h); fclose(f); h = fnv("img", 3, h); }
  return h;
}
static bool tx_cache_load(uint64_t key, std::vector<f32>& tx) {
  const char* d = getenv("QWEN_CACHE"); if (!d || !*d) return false;
  FILE* f = fopen((std::string(d) + "/tx.cache").c_str(), "rb"); if (!f) return false;
  uint64_t k = 0; long n = 0; bool ok = fread(&k, 8, 1, f) == 1 && fread(&n, 8, 1, f) == 1 && k == key && n > 0;
  if (ok) { tx.resize((size_t)n); ok = fread(tx.data(), 4, tx.size(), f) == tx.size(); }
  fclose(f); return ok;
}
static void tx_cache_save(uint64_t key, const std::vector<f32>& tx) {
  const char* d = getenv("QWEN_CACHE"); if (!d || !*d) return; std::string p = std::string(d) + "/tx.cache";
  FILE* f = fopen(p.c_str(), "wb"); if (!f) return; long n = (long)tx.size(); fwrite(&key, 8, 1, f); fwrite(&n, 8, 1, f); fwrite(tx.data(), 4, tx.size(), f); fclose(f);
}
static bool generate(const std::vector<int>& ids, int drop, int Hpx, int Wpx, unsigned seed, int steps, const std::string& outz) {
  double tall = now(); std::vector<f32> tx;
  static const f32 SIG4[5] = {1.0f, 0.95418f, 0.89508f, 0.704534f, 0.0f}, SIG8[9] = {1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f, 0.0f};   // 4: with the Lightning LoRA, 8: the schedule shipped with plain Turbo
  // sigma schedule: the 4-step (Lightning) or 8-step (plain Turbo) table, resampled to the number of steps asked for (QWEN_SCHED = the table of the LoRA in use; steps = table length gives it back exactly)
  const char* sc = getenv("QWEN_SCHED"); const int sched = sc && atoi(sc) > 0 ? (atoi(sc) >= 8 ? 8 : 4) : (steps == 8 ? 8 : 4); const f32* tab = sched == 8 ? SIG8 : SIG4; const int tn = sched;
  std::vector<f32> sigv(steps + 1);
  for (int k = 0; k <= steps; k++) { double x = (double)k * tn / steps; int i0 = (int)x; if (i0 >= tn) { sigv[k] = tab[tn]; continue; } double fr = x - i0; sigv[k] = (f32)(tab[i0] * (1 - fr) + tab[i0 + 1] * fr); }
  const f32* SIG = sigv.data();
  const int h = Hpx / 16, w = Wpx / 16, N = h * w;
  if (getenv("QWEN_TX")) { const int Tt0 = (int)ids.size() - drop; if (!readraw(getenv("QWEN_TX"), tx, (size_t)Tt0 * 4096)) { fprintf(stderr, "QWEN_TX unreadable\n"); return false; } }
  const uint64_t key = tx_key(ids, drop); const bool hit = !getenv("QWEN_TX") && tx_cache_load(key, tx);
  EditIn ed; if (!load_edit(ids, ed, !hit)) return false;
  if (hit) fprintf(stderr, "te: cached\n");
  else if (!getenv("QWEN_TX")) { if (!encode_text(ids, drop, tx, &ed)) return false; tx_cache_save(key, tx); }
  if (getenv("QWEN_DUMP_TX")) { FILE* f = fopen(getenv("QWEN_DUMP_TX"), "wb"); fwrite(tx.data(), 4, tx.size(), f); fclose(f); }
  const int Tt = (int)tx.size() / 4096;
  double t1 = now(); txt_in(tx, Tt); fprintf(stderr, "txt_in: %.2fs\n", now() - t1);
  // prefix rows: text with every image slot (nm rows) replaced by the Nc rows img_in(latent tokens) of its condition image; plain text for T2I
  const f32* Wimg = g_glue.f("img_in.weight"); const f32* Wout = g_glue.f("proj_out.weight");
  const int nimg = ed.on ? (int)ed.im.size() : 0; int Lp = Tt; int pst[3] = {0, 0, 0};   // pst[j]: first prefix row of image j
  for (int j = 0; j < nimg; j++) { const EditImg& m = ed.im[j]; pst[j] = m.k0 - drop + (Lp - Tt); if (m.k0 - drop < 0) { fprintf(stderr, "bad edit layout\n"); return false; } Lp += m.Nc - m.nm; }
  // prefix K/V cache file (QWEN_CACHE): the prefix (text + condition photos) does not depend on the seed or the target size, so a repeat skips the whole prefill pass.
  // The K/V of the 32 layers (0.5 MB per prefix token) stay on disk and are read one layer at a time while the DSP works: they used to be ~1.5-2.5 GB of RAM.
  const int ntokP0 = (Lp + 31) / 32 * 32; const size_t kvPre = (size_t)ntokP0 * 4096 * 2; const off_t KVH = 16;
  const uint64_t pkey = fnv(&Lp, 4, fnv("qwen-prefix-v1", 14, tx_key(ids, drop))); const char* cdir = getenv("QWEN_CACHE"); const std::string pfile = cdir && *cdir ? std::string(cdir) + "/prefix.cache" : std::string(), ptmp = pfile + ".tmp";
  bool pre_hit = false, kvram = false; int kvfd = -1; std::vector<std::vector<uint8_t>> cK(NLAYER_DIT), cV(NLAYER_DIT);   // cK/cV: only when there is no room for the file
  auto kvoff = [&](int layer, int vk) { return KVH + ((off_t)layer * 2 + vk) * (off_t)kvPre; };
  if (!pfile.empty()) {
    int fd = open(pfile.c_str(), O_RDONLY);
    if (fd >= 0) { uint64_t k = 0; int l = 0, nt = 0; struct stat st; pre_hit = pread(fd, &k, 8, 0) == 8 && pread(fd, &l, 4, 8) == 4 && pread(fd, &nt, 4, 12) == 4 && k == pkey && l == Lp && nt == ntokP0 && fstat(fd, &st) == 0 && (size_t)st.st_size == (size_t)KVH + (size_t)NLAYER_DIT * 2 * kvPre;
      if (pre_hit) kvfd = fd; else close(fd); }
    if (!pre_hit) {
      struct statvfs sv; const bool room = statvfs(cdir, &sv) == 0 && (double)sv.f_bavail * sv.f_frsize > (double)NLAYER_DIT * 2 * kvPre + (512 << 20);
      if (room) { kvfd = open(ptmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600); if (kvfd >= 0) { if (!kv_write(kvfd, 0, (const uint8_t*)&pkey, 8) || !kv_write(kvfd, 8, (const uint8_t*)&Lp, 4) || !kv_write(kvfd, 12, (const uint8_t*)&ntokP0, 4)) { close(kvfd); kvfd = -1; } } }
    }
  }
  if (kvfd < 0) kvram = true;   // no cache dir / disk full: the old behaviour, K/V in RAM (and no cache file)
  if (pre_hit) fprintf(stderr, "prefix: cached\n");
  std::vector<f32> pre;
  if (ed.on && !pre_hit) {
    pre.assign((size_t)Lp * 4096, 0.f); int src = 0, dst = 0;   // src: row of tx, dst: row of pre
    for (int j = 0; j < nimg; j++) { const EditImg& m = ed.im[j]; const int tb = m.k0 - drop - src;
      memcpy(&pre[(size_t)dst * 4096], &tx[(size_t)src * 4096], (size_t)tb * 4096 * 4); dst += tb; src += tb + m.nm;
      par_for(m.Nc / 16 + 1, [&](int blk) { for (int t = blk * 16; t < std::min(m.Nc, blk * 16 + 16); t++) { const f32* zt = &m.cond[(size_t)t * 64]; f32* r = &pre[(size_t)(pst[j] + t) * 4096];
        for (int o = 0; o < 4096; o++) { const f32* wr = Wimg + (size_t)o * 64; float a = 0; for (int c = 0; c < 64; c++) a += wr[c] * zt[c]; r[o] = a; } } });
      dst += m.Nc; }
    memcpy(&pre[(size_t)dst * 4096], &tx[(size_t)src * 4096], (size_t)(Tt - src) * 4096 * 4);
  }
  const int nsP = (Lp + 31) / 32, ntokP = nsP * 32; double posAfter = Tt;
  const int nsI = N / 32;
  g_LP = make_layout(ntokP, 0, false);
  if (pre_hit) { free_buf(g_blobP); g_P_ntok = -1; }   // a cached prefix needs no prefill blob
  else if (g_P_ntok != ntokP) { free_buf(g_blobP); g_blobP = make_buf(g_LP.total, true); if (!g_blobP.p) return false; g_P_ntok = ntokP; }
  if (g_D_ntok != N || g_D_extra != nsP) { if (g_blobD.p) { unmap_buf(g_blobD); rpcmem_free(g_blobD.p); } g_LD = make_layout(N, nsP, false); g_blobD = make_buf(g_LD.total, true); if (!g_blobD.p) return false; g_D_ntok = N; g_D_extra = nsP; }
  if (!pre_hit) memset(g_blobP.p, 0, g_LP.total);
  memset(g_blobD.p, 0, g_LD.total);
  { std::vector<double> flat((size_t)std::max(Lp, N) * 3); double cur = 0; int i = 0;   // text rows advance (p, p, p); an image block freezes the frame axis at cur, lays h/w out centred, and cur += max(hc, wc)
    for (int j = 0; j < nimg; j++) { const EditImg& m = ed.im[j];
      for (; i < pst[j]; i++) { flat[i * 3] = flat[i * 3 + 1] = flat[i * 3 + 2] = cur; cur += 1; }
      for (int y = 0; y < m.hc; y++) for (int x = 0; x < m.wc; x++, i++) { flat[i * 3] = cur; flat[i * 3 + 1] = y - (m.hc - m.hc / 2); flat[i * 3 + 2] = x - (m.wc - m.wc / 2); }
      cur += std::max(m.hc, m.wc); }
    for (; i < Lp; i++) { flat[i * 3] = flat[i * 3 + 1] = flat[i * 3 + 2] = cur; cur += 1; }
    posAfter = cur;
    if (!pre_hit) rope_rows((const double(*)[3])flat.data(), Lp, ntokP, (f32*)(g_blobP.p + g_LP.meta[M_COS]), (f32*)(g_blobP.p + g_LP.meta[M_SIN]));
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { const int i2 = y * w + x; flat[i2 * 3] = posAfter; flat[i2 * 3 + 1] = y - (h - h / 2); flat[i2 * 3 + 2] = x - (w - w / 2); }
    rope_rows((const double(*)[3])flat.data(), N, N, (f32*)(g_blobD.p + g_LD.meta[M_COS]), (f32*)(g_blobD.p + g_LD.meta[M_SIN])); }
  std::vector<f32> z((size_t)N * 64);   // tokens [N][64]
  if (getenv("QWEN_Z0")) { if (!readraw(getenv("QWEN_Z0"), z, z.size())) return false; }
  else { std::mt19937 gen(seed); std::normal_distribution<float> nd(0.f, 1.f); for (auto& a : z) a = nd(gen); }
  // text rows -> prefix blob
  if (!pre_hit) { std::vector<f32> pad((size_t)ntokP * 4096, 0.f); memcpy(pad.data(), ed.on ? pre.data() : tx.data(), (size_t)Lp * 4096 * 4); rows_to_x32(pad.data(), ntokP, 4096, (f32*)(g_blobP.p + g_LP.meta[M_X]), nsP); }
  std::vector<f32>().swap(pre);
  std::vector<f32> mod0, modv, nscale; time_mod(0.f, mod0, nullptr);
  if (!pre_hit) sync_clean(g_blobP);
  const size_t kvOwnD = (size_t)N * 4096 * 2;
  int metaP[M_NMETA], metaD[M_NMETA]; for (int i = 0; i < M_NMETA; i++) { metaP[i] = g_LP.meta[i]; metaD[i] = g_LD.meta[i]; }
  metaP[M_NW] = metaD[M_NW] = 3; metaP[M_NCB] = metaD[M_NCB] = 128; metaP[M_MODE] = metaD[M_MODE] = 40;
  metaP[M_NTOK] = ntokP; metaP[M_NVALID] = Lp; metaP[M_NPRE] = 0; metaP[M_CAUS] = ed.on ? 2 : 1;
  { int* bm[3][2] = {{&metaP[M_BA], &metaP[M_BB]}, {&metaP[M_B2A], &metaP[M_B2B]}, {&metaP[M_B3A], &metaP[M_B3B]}}; for (int j = 0; j < 3; j++) { *bm[j][0] = j < nimg ? pst[j] : 0; *bm[j][1] = j < nimg ? pst[j] + ed.im[j].Nc : 0; } }
  metaD[M_NTOK] = N; metaD[M_NVALID] = N + Lp; metaD[M_NPRE] = nsP; metaD[M_CAUS] = 0; metaD[M_BA] = metaD[M_BB] = metaD[M_B2A] = metaD[M_B2B] = metaD[M_B3A] = metaD[M_B3B] = 0;
  std::vector<std::string> df; for (int k = 0; k < NLAYER_DIT; k++) { char p[512]; snprintf(p, sizeof p, "%s/dit/blk%02d.bin", g_dir.c_str(), k); df.push_back(p); }
  std::vector<f32> rows((size_t)N * 4096), v((size_t)N * 64);
  // K/V of the next layer is read from the file by a helper thread while the DSP runs the current one (two staging buffers of one layer each)
  std::vector<uint8_t> kvSt[2]; std::thread kvTh; int kvPendLayer = -1, kvPendSlot = 0, kvSeq = 0; bool kvOk[2] = {true, true};
  auto kv_start = [&](int layer, int slot) { kvPendLayer = layer; kvPendSlot = slot; kvOk[slot] = true;
    kvTh = std::thread([&, layer, slot] { kvOk[slot] = kv_read(kvfd, kvoff(layer, 0), kvSt[slot].data(), 2 * kvPre); posix_fadvise(kvfd, kvoff(layer, 0), 2 * kvPre, POSIX_FADV_DONTNEED); }); };
  if (!kvram) { kvSt[0].resize(2 * kvPre); kvSt[1].resize(2 * kvPre); }
  for (int s = 0; s < steps; s++) {
    double ts = now(); const float sg = SIG[s], sgn = SIG[s + 1];
    time_mod(sg, modv, &nscale);
    par_for(N / 16 + 1, [&](int blk) { for (int t = blk * 16; t < std::min(N, blk * 16 + 16); t++) { const f32* zt = &z[(size_t)t * 64]; f32* r = &rows[(size_t)t * 4096];
      for (int o = 0; o < 4096; o++) { const f32* wr = Wimg + (size_t)o * 64; float a = 0; for (int c = 0; c < 64; c++) a += wr[c] * zt[c]; r[o] = a; } } });
    rows_to_x32(rows.data(), N, 4096, (f32*)(g_blobD.p + g_LD.meta[M_X]), nsI);
    sync_clean(g_blobD); double tA = now();
    bool ok = stream_files(df, g_te, [&](int k, Buf& b) {
      uint8_t* dK = g_blobD.p + g_LD.meta[M_K] + kvOwnD; uint8_t* dV = g_blobD.p + g_LD.meta[M_V] + kvOwnD;
      if (s == 0 && !pre_hit) {
        metaP[M_FIRST] = k == 0; metaP[M_REL] = 0;
        if (call(g_LP, metaP, b, 0, g_blobP, mod0.data(), 18432)) return false;
        sync_inval(g_blobP);
        const uint8_t* sK = g_blobP.p + g_LP.meta[M_K]; const uint8_t* sV = g_blobP.p + g_LP.meta[M_V];
        if (kvram) { cK[k].assign(sK, sK + kvPre); cV[k].assign(sV, sV + kvPre); memcpy(dK, cK[k].data(), kvPre); memcpy(dV, cV[k].data(), kvPre); }
        else {
          if (!kv_write(kvfd, kvoff(k, 0), sK, kvPre) || !kv_write(kvfd, kvoff(k, 1), sV, kvPre)) { fprintf(stderr, "prefix cache write failed\n"); return false; }
          memcpy(dK, sK, kvPre); memcpy(dV, sV, kvPre);
          sync_file_range(kvfd, kvoff(k, 0), 2 * kvPre, SYNC_FILE_RANGE_WRITE);   // start the write-back; older layers are dropped from the page cache below
          if (k >= 2) { sync_file_range(kvfd, kvoff(k - 2, 0), 2 * kvPre, SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER); posix_fadvise(kvfd, kvoff(k - 2, 0), 2 * kvPre, POSIX_FADV_DONTNEED); }
          if (k == NLAYER_DIT - 1 && steps > 1) kv_start(0, kvSeq++ & 1);   // layer 0 for the next step
        }
      } else if (kvram) { memcpy(dK, cK[k].data(), kvPre); memcpy(dV, cV[k].data(), kvPre); }
      else {
        if (kvPendLayer != k) { if (kvTh.joinable()) kvTh.join(); kv_start(k, kvSeq++ & 1); }   // not prefetched (first layer of a cached run)
        kvTh.join(); kvPendLayer = -1; const int cur = kvPendSlot;
        if (!kvOk[cur]) { fprintf(stderr, "prefix cache read failed\n"); return false; }
        if (!(s == steps - 1 && k == NLAYER_DIT - 1)) kv_start((k + 1) % NLAYER_DIT, cur ^ 1);   // the next layer is read while this one runs on the DSP
        memcpy(dK, kvSt[cur].data(), kvPre); memcpy(dV, kvSt[cur].data() + kvPre, kvPre);
      }
      sync_clean(g_blobD);
      metaD[M_FIRST] = k == 0; metaD[M_REL] = k == NLAYER_DIT - 1;
      return call(g_LD, metaD, b, 0, g_blobD, modv.data(), 18432) == 0; });
    if (!ok) { if (kvTh.joinable()) kvTh.join(); if (kvfd >= 0) { close(kvfd); if (!pre_hit) remove(ptmp.c_str()); } return false; }
    if (s == 0 && !pre_hit) {
      free_buf(g_blobP); g_P_ntok = -1;   // the prefill blob is done
      if (!kvram) { for (int k = std::max(0, NLAYER_DIT - 2); k < NLAYER_DIT; k++) { sync_file_range(kvfd, kvoff(k, 0), 2 * kvPre, SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER); posix_fadvise(kvfd, kvoff(k, 0), 2 * kvPre, POSIX_FADV_DONTNEED); } }
    }
    sync_inval(g_blobD); double tB = now();
    x32_to_rows((const f32*)(g_blobD.p + g_LD.meta[M_X]), N, 4096, rows.data());
    par_for(N / 16 + 1, [&](int blk) { for (int t = blk * 16; t < std::min(N, blk * 16 + 16); t++) {
      f32* r = &rows[(size_t)t * 4096]; double mu = 0, ss = 0; for (int i = 0; i < 4096; i++) mu += r[i]; mu /= 4096; for (int i = 0; i < 4096; i++) ss += (r[i] - mu) * (r[i] - mu);
      const float rs = 1.f / sqrtf((float)(ss / 4096) + 1e-6f); for (int i = 0; i < 4096; i++) r[i] = (float)((r[i] - mu) * rs) * (1.f + nscale[i]);
      for (int o = 0; o < 64; o++) { const f32* wr = Wout + (size_t)o * 4096; float a = 0; for (int i = 0; i < 4096; i++) a += wr[i] * r[i]; v[(size_t)t * 64 + o] = a; } } });
    if (const char* pv = getenv("QWEN_PREVIEW")) if (*pv && s < steps - 1) {   // preview: the current x0 estimate z - sigma * v, de-normalised, [64][h][w] like the final latent, written atomically
      const f32* pm = g_glue.f("vae.mean"); const f32* ps = g_glue.f("vae.std"); std::vector<f32> xo((size_t)64 * N);
      for (int t = 0; t < N; t++) for (int c = 0; c < 64; c++) xo[(size_t)c * N + t] = (z[(size_t)t * 64 + c] - sg * v[(size_t)t * 64 + c]) * ps[c] + pm[c];
      const std::string tmp = std::string(pv) + ".tmp"; FILE* pf = fopen(tmp.c_str(), "wb");
      if (pf) { int hdr[4] = {h, w, 64, s}; fwrite(hdr, 4, 4, pf); fwrite(xo.data(), 4, xo.size(), pf); fclose(pf); rename(tmp.c_str(), pv); } }
    for (size_t i = 0; i < z.size(); i++) z[i] += (sgn - sg) * v[i];
    fprintf(stderr, "  step %d: DSP blocks %.2fs, total %.2fs\n", s, tB - tA, now() - ts);
    printf("step %d %.0f\n", s, (now() - ts) * 1000); fflush(stdout);
  }
  const f32* mean = g_glue.f("vae.mean"); const f32* sd = g_glue.f("vae.std");
  if (kvTh.joinable()) kvTh.join();
  if (kvfd >= 0) { close(kvfd); if (!pre_hit && rename(ptmp.c_str(), pfile.c_str())) remove(ptmp.c_str()); }   // a finished cache file replaces the old one
  std::vector<f32> zo((size_t)64 * N);   // [64][h][w], de-normalised
  for (int t = 0; t < N; t++) for (int c = 0; c < 64; c++) zo[(size_t)c * N + t] = z[(size_t)t * 64 + c] * sd[c] + mean[c];
  FILE* f = fopen(outz.c_str(), "wb"); fwrite(zo.data(), 4, zo.size(), f); fclose(f);
  if (getenv("QWEN_DUMP_Z")) { FILE* g = fopen(getenv("QWEN_DUMP_Z"), "wb"); fwrite(z.data(), 4, z.size(), g); fclose(g); }
  printf("done %.0f\n", (now() - tall) * 1000); fflush(stdout);
  return true;
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
    if (cmd == "load") {
      ss >> g_dir; double t0 = now();
      if (!g_glue.load(g_dir + "/glue.bin")) { printf("error glue\n"); continue; }
      if (const char* ql = getenv("QWEN_LORA")) if (*ql) {   // a LoRA on the plain pack: slot overlays (blkNN.ovl) + the host layers it touches (glueL.bin overrides tensors of glue.bin)
        g_ovl = ql; const std::string gl = g_ovl + "/glueL.bin"; if (access(gl.c_str(), R_OK) == 0 && !g_glue.load(gl)) { printf("error glueL\n"); continue; } }
      { int fd = open((g_dir + "/embd.q4k").c_str(), O_RDONLY); if (fd < 0) { printf("error embd\n"); continue; } struct stat st; fstat(fd, &st); g_embd_len = st.st_size; g_embd = (uint8_t*)mmap(nullptr, g_embd_len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); }
      g_te[0] = make_buf(TE_BUF); g_te[1] = make_buf(TE_BUF); g_all.push_back(&g_te[0]); g_all.push_back(&g_te[1]); if (!g_te[0].p || !g_te[1].p) { printf("error tebuf\n"); continue; }
      g_LT = make_layout(MAXT, 0, true);
      printf("loaded %.1f\n", now() - t0); continue;
    }
    if (cmd == "gen") {   // gen IDS.i32 DROP H W SEED STEPS OUT.z
      std::string idsf, outz; int drop, H, W, steps; unsigned seed; ss >> idsf >> drop >> H >> W >> seed >> steps >> outz;
      std::vector<int> ids; { FILE* f = fopen(idsf.c_str(), "rb"); if (!f) { printf("error ids\n"); continue; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(n / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) {} fclose(f); }
      if (ids.size() <= (size_t)drop || (int)ids.size() > MAXT || H % 16 || W % 16 || ((H / 16) * (W / 16)) % 32 || (H / 16) * (W / 16) > 6144 || steps < 1 || steps > 50) { printf("error args\n"); continue; }
      if (!generate(ids, drop, H, W, seed, steps, outz)) printf("error gen\n");
    }
  }
  mega_close(g_h);
  return 0;
}
