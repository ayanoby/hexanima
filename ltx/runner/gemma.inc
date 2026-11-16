// ltxrun: LTX-2.5 text encoder on the Hexagon NPU = Gemma 4 12B text model (48 layers, MODE 80 of the megakernel) + the two aggregate-embed projections (49 normalised states -> video 4096 / audio 2048).
// DIR: te/gm00..gm47.bin (export_gemma.py), glue/embd.f16 final_norm.f32 vproj.f16 vproj_b.f32 aproj.f16 aproj_b.f32. Commands: load DIR | te IDS.i32 OUTPREFIX (-> OUTPREFIX.v.f32 [T][4096], OUTPREFIX.a.f32 [T][2048]; GM_DUMP=1 also OUTPREFIX.states.f32 [49][T][3840])
static const int GD = 3840, NLG = 48, NST = 49, MAXTG = 1024;
static std::vector<std::string> g_gm; static const uint16_t* g_embf = nullptr; static size_t g_embfl = 0;
static bool g_audio = false;   // the audio projection and connector are only used when their files are installed (video-only runs do not need them)
static std::vector<f32> g_fnorm, g_vb, g_ab; static const __fp16 *g_vw = nullptr, *g_aw = nullptr; static size_t g_vwl = 0, g_awl = 0;
static Layout g_LG; static int g_G_ns = -1; static size_t g_offCosF = 0, g_offSinF = 0; static Buf g_blobG;
static bool is_full(int l) { return l % 6 == 5; }
static Layout make_layout_gm(int ns, size_t* cf, size_t* sf) {
  Layout L; memset(&L, 0, sizeof L); const size_t ntok = (size_t)ns * 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * GD * 4}, {M_COS, (size_t)ns * 16384}, {M_SIN, (size_t)ns * 16384}, {M_HN, 131072}, {M_H, ntok * GD * 2}, {M_HR, ntok * 15360 * 2}, {M_Q, ntok * 8192 * 2}, {M_K, ntok * 8192 * 2},
    {M_V, ntok * 8192 * 2}, {M_O, ntok * 8192 * 2}, {M_T, ntok * GD * 2}, {M_MID, ntok * 15360 * 2}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LQ, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  *cf = off; off += up4k((size_t)ns * 32768); *sf = off; off += up4k((size_t)ns * 32768); L.total = off; return L;
}
static void gm_rope(f32* cs, f32* sn, int T, int ns, int hd, double theta, double frac) {   // tables [slab][hd/64 tiles][32 rows][32]: token t, angle d (rotate-half, pairs d / d + hd/2); frac < 1: only the first frac*hd/2 angles rotate
  const int nang = hd / 2, nrot = (int)(frac * hd / 2); memset(cs, 0, (size_t)ns * (nang / 32) * 4096); memset(sn, 0, (size_t)ns * (nang / 32) * 4096);
  par_for(T, [&](int t) { for (int d = 0; d < nang; d++) {
    const double a = d < nrot ? t * pow(theta, -2.0 * d / hd) : 0.0; const size_t idx = (((size_t)(t / 32) * (nang / 32) + d / 32) * 1024) + (t % 32) * 32 + d % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } });
}
static void gm_hn(uint16_t* out) {   // Hn = kron(h4,h4,h4,h4)/16 as the engine's fp16 weight tiles: in[j0][j1][j2][j3][j4] (rows j0*32 + j1*2 + j2, cols j3*32 + j4) -> out[j3][j0][j1][j4][j2]
  static const int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
  auto hv = [&](int r, int c) { int s = 1; for (int k = 0; k < 4; k++) { s *= h4[(r >> (2 * k)) & 3][(c >> (2 * k)) & 3]; } return s; };
  const __fp16 p = (__fp16)0.0625f, n = (__fp16)-0.0625f;
  for (int j0 = 0; j0 < 8; j0++) for (int j1 = 0; j1 < 16; j1++) for (int j2 = 0; j2 < 2; j2++) for (int j3 = 0; j3 < 8; j3++) for (int j4 = 0; j4 < 32; j4++) {
    const int r = j0 * 32 + j1 * 2 + j2, c = j3 * 32 + j4; const __fp16 v = hv(r, c) > 0 ? p : n;
    memcpy(&out[((((size_t)j3 * 8 + j0) * 16 + j1) * 32 + j4) * 2 + j2], &v, 2); }
}

// ---------------------------------------------------------------- embeddings connectors (MODE 81): 8 blocks per modality over 1024 rows = [real tokens | learnable registers tiled]
static std::vector<std::string> g_cf[2]; static std::vector<f32> g_reg[2]; static Layout g_LC[2]; static Buf g_blobC[2]; static bool g_conn = false;
static Layout make_layout_conn(int D, int hh) {
  Layout L; memset(&L, 0, sizeof L); const size_t ntok = 1024, ns = 32, tbl = (size_t)32 * hh * 4096; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * D * 4}, {M_COS, ns * tbl}, {M_SIN, ns * tbl}, {M_HN, 131072}, {M_H, ntok * D * 2}, {M_HR, ntok * 4 * D * 2}, {M_Q, ntok * D * 2}, {M_K, ntok * D * 2}, {M_V, ntok * D * 2},
    {M_O, ntok * D * 2}, {M_T, ntok * D * 2}, {M_MID, ntok * 4 * D * 2}, {M_LQ, ns * 4096}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
static void conn_rope(f32* cs, f32* sn, int D, int hh) {   // LTX 1D rope (rope_type split, float64 frequency grid): tables [slab][head][tile][32 rows][32]; head h uses frequencies [h*hd/2, (h+1)*hd/2) of the D/2 values
  const int nf = D / 2, hd2 = hh * 32; std::vector<double> idx(nf);
  for (int i = 0; i < nf; i++) idx[i] = pow(10000.0, (double)i / (nf - 1)) * M_PI / 2;
  par_for(1024, [&](int t) { const float frac = (float)t / 4096.0f; for (int i = 0; i < nf; i++) { const float f = (float)idx[i] * (frac * 2.0f - 1.0f); const int h = i / hd2, d = i % hd2;
    const size_t o = ((((size_t)(t / 32) * 32 + h) * hh + d / 32) * 1024) + (size_t)(t % 32) * 32 + d % 32; cs[o] = cosf(f); sn[o] = sinf(f); } });
}
static bool run_conn(int kind, const std::vector<f32>& feat, int T, std::vector<f32>& out) {   // kind 0 video (D 4096, head_dim 128), 1 audio (D 2048, head_dim 64)
  const int D = kind ? 2048 : 4096, hh = kind ? 1 : 2, ns = 32, ntok = 1024; double t0 = now();
  if (!g_blobC[kind].p) { g_LC[kind] = make_layout_conn(D, hh); g_blobC[kind] = make_buf(g_LC[kind].total, true); if (!g_blobC[kind].p) return false; }
  uint8_t* blob = g_blobC[kind].p; const Layout& L = g_LC[kind]; memset(blob, 0, L.total);
  { std::vector<f32> rows((size_t)ntok * D); memcpy(rows.data(), feat.data(), (size_t)T * D * 4);
    for (int i = T; i < ntok; i++) memcpy(&rows[(size_t)i * D], &g_reg[kind][(size_t)(i % 128) * D], (size_t)D * 4);
    rows_to_x32(rows.data(), ntok, D, (f32*)(blob + L.meta[M_X]), ns); }
  conn_rope((f32*)(blob + L.meta[M_COS]), (f32*)(blob + L.meta[M_SIN]), D, hh); gm_hn((uint16_t*)(blob + L.meta[M_HN])); sync_clean(g_blobC[kind]);
  std::vector<f32> mod(64, 0.f); int meta[M_NMETA];
  if (!stream_files(g_cf[kind], g_te, [&](int l, Buf& w) {
    for (int i = 0; i < M_NMETA; i++) meta[i] = L.meta[i];
    meta[M_NW] = 3; meta[M_NCB] = D / 32; meta[M_MODE] = 81; meta[M_FMT] = kind; meta[M_NTOK] = ntok; meta[M_NVALID] = ntok; meta[M_STOP] = 0; meta[M_FIRST] = l == 0; meta[M_REL] = l == 7;
    return call(L, meta, w, 0, g_blobC[kind], mod.data(), (int)mod.size()) == 0; }, 0)) return false;
  sync_inval(g_blobC[kind]); out.resize((size_t)ntok * D); x32_to_rows((const f32*)(blob + L.meta[M_X]), ntok, D, out.data());
  par_for(ntok, [&](int t) { f32* r = &out[(size_t)t * D]; double ss = 0; for (int i = 0; i < D; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / D) + 1e-6f); for (int i = 0; i < D; i++) r[i] *= rs; });   // final rms_norm (no weight)
  fprintf(stderr, "connector %s: %.2fs\n", kind ? "audio" : "video", now() - t0); return true;
}
static bool encode_gm(const std::vector<int>& ids, const std::string& outp) {
  const int T = (int)ids.size(), ns = (T + 31) / 32, ntok = ns * 32; if (T > MAXTG || T < 1) return false;
  double t0 = now(); g_waitio = 0;
  if (g_G_ns != ns) { if (g_blobG.p) { unmap_buf(g_blobG); rpcmem_free(g_blobG.p); } g_LG = make_layout_gm(ns, &g_offCosF, &g_offSinF); g_blobG = make_buf(g_LG.total, true); if (!g_blobG.p) return false; g_G_ns = ns; }
  uint8_t* blob = g_blobG.p; memset(blob, 0, g_LG.total);
  std::vector<f32> st((size_t)NST * T * GD);   // the 49 hidden states [state][token][3840]
  { std::vector<f32> rows((size_t)ntok * GD, 0.f); const float sc = sqrtf((float)GD);
    for (int t = 0; t < T; t++) { const uint16_t* e = g_embf + (size_t)ids[t] * GD; f32* o = &rows[(size_t)t * GD]; for (int i = 0; i < GD; i++) { __fp16 h; memcpy(&h, &e[i], 2); o[i] = (f32)h * sc; } }
    rows_to_x32(rows.data(), ntok, GD, (f32*)(blob + g_LG.meta[M_X]), ns); memcpy(st.data(), rows.data(), (size_t)T * GD * 4); }
  gm_rope((f32*)(blob + g_LG.meta[M_COS]), (f32*)(blob + g_LG.meta[M_SIN]), T, ns, 256, 1e4, 1.0);
  gm_rope((f32*)(blob + g_offCosF), (f32*)(blob + g_offSinF), T, ns, 512, 1e6, 0.25);
  gm_hn((uint16_t*)(blob + g_LG.meta[M_HN])); sync_clean(g_blobG);
  std::vector<f32> mod(64, 0.f); int meta[M_NMETA]; std::vector<f32> rows((size_t)T * GD);
  if (!stream_files(g_gm, g_te, [&](int l, Buf& w) {
    for (int i = 0; i < M_NMETA; i++) meta[i] = g_LG.meta[i];
    if (is_full(l)) { meta[M_COS] = (int)g_offCosF; meta[M_SIN] = (int)g_offSinF; }
    meta[M_NW] = 3; meta[M_NCB] = 120; meta[M_MODE] = 80; meta[M_FMT] = is_full(l) ? 1 : 0; meta[M_NTOK] = ntok; meta[M_NVALID] = T; meta[M_STOP] = 0; meta[M_FIRST] = l == 0; meta[M_REL] = 1;
    if (call(g_LG, meta, w, 0, g_blobG, mod.data(), (int)mod.size())) return false;
    sync_inval(g_blobG); x32_to_rows((const f32*)(blob + g_LG.meta[M_X]), T, GD, rows.data()); memcpy(&st[(size_t)(l + 1) * T * GD], rows.data(), (size_t)T * GD * 4);
    return true; }, 0)) return false;
  double t1 = now();
  // state 48 = final norm of the last layer output
  par_for(T, [&](int t) { f32* r = &st[((size_t)(NST - 1) * T + t) * GD]; double ss = 0; for (int i = 0; i < GD; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / GD) + 1e-6f); for (int i = 0; i < GD; i++) r[i] *= rs * g_fnorm[i]; });
  // LTX feature extractor V2: per-token RMS norm of every state (no weight), concat -> [T][49*3840]
  const size_t F = (size_t)NST * GD; std::vector<f32> feat((size_t)T * F);
  par_for(T, [&](int t) { for (int s = 0; s < NST; s++) { const f32* r = &st[((size_t)s * T + t) * GD]; f32* o = &feat[(size_t)t * F]; double ss = 0; for (int i = 0; i < GD; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / GD) + 1e-6f); for (int i = 0; i < GD; i++) o[(size_t)i * NST + s] = r[i] * rs; } });   // torch.stack(states, dim=-1) -> [T, D, L] flattened: index = d * 49 + layer (layer is the fastest axis)
  if (getenv("GM_DUMP")) { FILE* f = fopen((outp + ".states.f32").c_str(), "wb"); fwrite(st.data(), 4, st.size(), f); fclose(f); }
  { std::vector<f32> v((size_t)T * 4096), a((size_t)T * 2048);
    gemm_f16(g_vw, feat.data(), v.data(), T, 4096, (int)F); { const float k = sqrtf(4096.f / GD); for (int t = 0; t < T; t++) for (int i = 0; i < 4096; i++) v[(size_t)t * 4096 + i] = v[(size_t)t * 4096 + i] * k + g_vb[i]; }
    if (g_audio) { gemm_f16(g_aw, feat.data(), a.data(), T, 2048, (int)F); const float k = sqrtf(2048.f / GD); for (int t = 0; t < T; t++) for (int i = 0; i < 2048; i++) a[(size_t)t * 2048 + i] = a[(size_t)t * 2048 + i] * k + g_ab[i]; }
    FILE* f = fopen((outp + ".v.f32").c_str(), "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f); f = fopen((outp + ".a.f32").c_str(), "wb"); fwrite(a.data(), 4, a.size(), f); fclose(f);
    if (g_conn) { std::vector<f32> cv, ca; if (!run_conn(0, v, T, cv) || (g_audio && !run_conn(1, a, T, ca))) return false;
      f = fopen((outp + ".vc.f32").c_str(), "wb"); fwrite(cv.data(), 4, cv.size(), f); fclose(f); if (g_audio) { f = fopen((outp + ".ac.f32").c_str(), "wb"); fwrite(ca.data(), 4, ca.size(), f); fclose(f); } } }
  fprintf(stderr, "te: layers %.2fs (%d tokens, waited %.2fs for io), projections %.2fs\n", t1 - t0, T, g_waitio, now() - t1); return true;
}

// ---------------------------------------------------------------- LTX DiT, video stream only (MODE 91 modulated norm, 90 attention, 92 feed-forward): 48 blocks, block file = [attn1][attn2][ff] 269 MB
static const int VD = 4096, VFF = 16384, CTXN = 1024; static const size_t ATTN_PACK = 67535360u;   // bytes of one attention pack (Dq = Dk = I = 4096), see blk.c / export_dit.py
struct DL { Layout L; size_t offCtx = 0, offKin2 = 0, offCosK = 0, offK2 = 0, offV2 = 0; };
static DL make_layout_dit(int nv) {
  DL d; Layout& L = d.L; memset(&L, 0, sizeof L); const size_t ns = nv / 32, ntok = nv, tbl = (size_t)32 * 2 * 4096, mx = std::max<size_t>(ntok, CTXN); size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * VD * 4}, {M_COS, ns * tbl}, {M_SIN, ns * tbl}, {M_HN, 131072}, {M_H, ntok * VD * 2}, {M_HR, mx * VFF * 2}, {M_Q, ntok * VD * 2}, {M_K, mx * VD * 2}, {M_V, mx * VD * 2},
    {M_O, ntok * VD * 2}, {M_T, ntok * VD * 2}, {M_MID, ntok * VFF * 2}, {M_LQ, ns * 4096}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  d.offCtx = off; off += up4k((size_t)CTXN * VD * 2); d.offKin2 = off; off += up4k((size_t)CTXN * VD * 2); L.total = off; return d;
}
static void rows_to_h16(const f32* rows, int nrows, int cols, uint16_t* dst, int ns) {   // fp16 tiles [slab][cols/32][1024]: (r/2)*64 + c*2 + (r&1)
  const int ncb = cols / 32;
  par_for(ns, [&](int s) { for (int cb = 0; cb < ncb; cb++) { uint16_t* t = dst + ((size_t)s * ncb + cb) * 1024;
    for (int r = 0; r < 32; r++) { const int row = s * 32 + r; for (int c = 0; c < 32; c++) { __fp16 h = row < nrows ? (__fp16)rows[(size_t)row * cols + cb * 32 + c] : (__fp16)0.f; memcpy(&t[(r >> 1) * 64 + c * 2 + (r & 1)], &h, 2); } } } });
}
// one block of the video stream. mods: 12 vectors of 4096: M1 N1 g1 | M2 N2 g2 | M3 N3 g3 | shift_kv scale_kv
/** mods0/ns0: image-to-video, the first ns0 slabs (latent frame 0, timestep 0) use the modulation set mods0, the others mods (the prompt entries 9, 10 are shared) */
static bool run_dit_block(const DL& d, Buf& blob, Buf& w, const f32* ctx, const f32* mods, int nv, bool first, bool last, const f32* mods0 = nullptr, int ns0 = 0) {
  const Layout& L = d.L; uint8_t* b = blob.p; const int ns = nv / 32; int meta[M_NMETA]; std::vector<f32> mm(4 * VD), gb(2 * VD);
  if (!mods0) ns0 = 0;
  auto gate = [&](int k, int* n) -> const f32* { if (!ns0) { *n = VD; return mods + (size_t)k * VD; } memcpy(gb.data(), mods0 + (size_t)k * VD, VD * 4); memcpy(gb.data() + VD, mods + (size_t)k * VD, VD * 4); *n = 2 * VD; return gb.data(); };
  static const int fc = getenv("ATTN_CHUNK") ? atoi(getenv("ATTN_CHUNK")) : 0, fq = getenv("ATTN_QB") ? atoi(getenv("ATTN_QB")) : 0;   // tests: force the chunked attention
  auto base = [&](int mode) { for (int i = 0; i < M_NMETA; i++) meta[i] = L.meta[i]; meta[M_A10] = fc; meta[M_A11] = fq; meta[M_NW] = 3; meta[M_NCB] = 128; meta[M_MODE] = mode; meta[M_NTOK] = nv; meta[M_NVALID] = nv; meta[M_STOP] = 0; meta[M_FIRST] = 0; meta[M_REL] = 0; };
  auto ln = [&](int k, bool fst) { base(91); meta[M_A0] = L.meta[M_H]; meta[M_FIRST] = fst; meta[M_A8] = ns0; size_t o = 0;
    if (ns0) { memcpy(mm.data(), mods0 + (size_t)(3 * k) * VD, VD * 4); memcpy(mm.data() + VD, mods0 + (size_t)(3 * k + 1) * VD, VD * 4); o = 2 * VD; }
    memcpy(mm.data() + o, mods + (size_t)(3 * k) * VD, VD * 4); memcpy(mm.data() + o + VD, mods + (size_t)(3 * k + 1) * VD, VD * 4); return call(L, meta, w, 0, blob, mm.data(), (int)(o + 2 * VD)) == 0; };
  // modulated context for the cross attention: ctx * (1 + scale_kv) + shift_kv, fp16 tiles
  { std::vector<f32> cm((size_t)CTXN * VD); const f32* sk = mods + 9 * VD; const f32* ck = mods + 10 * VD;
    par_for(CTXN, [&](int t) { for (int i = 0; i < VD; i++) cm[(size_t)t * VD + i] = ctx[(size_t)t * VD + i] * (1.f + ck[i]) + sk[i]; });
    rows_to_h16(cm.data(), CTXN, VD, (uint16_t*)(b + d.offCtx), CTXN / 32); }
  sync_clean(blob);
  if (!ln(0, first)) return false;
  { base(90); meta[M_A0] = meta[M_A1] = L.meta[M_H]; meta[M_A2] = 128; meta[M_A3] = 128; meta[M_A4] = 1 | 2 | 4; meta[M_A5] = L.meta[M_COS]; meta[M_A6] = L.meta[M_SIN]; meta[M_A7] = (int)d.offKin2;
    meta[M_A8] = ns0; int gn; const f32* gp = gate(2, &gn); if (call(L, meta, w, 0, blob, gp, gn)) return false; }
  if (!ln(1, false)) return false;
  { base(90); meta[M_A0] = L.meta[M_H]; meta[M_A1] = (int)d.offCtx; meta[M_NVALID] = CTXN; meta[M_A2] = 128; meta[M_A3] = 128; meta[M_A4] = 4; meta[M_A5] = L.meta[M_COS]; meta[M_A6] = L.meta[M_SIN]; meta[M_A7] = (int)d.offKin2;
    meta[M_A8] = ns0; int gn; const f32* gp = gate(5, &gn); if (call(L, meta, w, ATTN_PACK, blob, gp, gn)) return false; }
  if (!ln(2, false)) return false;
  { base(92); meta[M_A0] = L.meta[M_H]; meta[M_A1] = VFF / 32; meta[M_REL] = last; meta[M_A8] = ns0; int gn; const f32* gp = gate(8, &gn); if (call(L, meta, w, 2 * ATTN_PACK, blob, gp, gn)) return false; }
  return true;
}
// ---- windowed DiT block: the activations of one step live in HOST memory (Xh: residual stream, Ch/Sh: rope tables); the DSP-mapped blob holds only ONE window of WS slabs of them plus the keys/values of ALL tokens (16 KB per token),
// so the token count is no longer limited by what the DSP can map (~1.6 GB). Per block: pass A (per window: LN1 -> K,V projections into the global K/V buffers), the cross-attention K/V of the context once, pass B (per window: LN1 -> self attention of the window against all keys -> LN2 -> cross attention -> LN3 -> FFN).
static DL make_layout_dit_w(int nv, int WS) {
  DL d; Layout& L = d.L; memset(&L, 0, sizeof L); const size_t nwt = (size_t)WS * 32, tbl = (size_t)32 * 2 * 4096, mxk = std::max<size_t>(nv, CTXN); size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, nwt * VD * 4}, {M_COS, (size_t)WS * tbl}, {M_SIN, (size_t)WS * tbl}, {M_HN, 131072}, {M_H, nwt * VD * 2}, {M_HR, nwt * VFF * 2}, {M_Q, nwt * VD * 2}, {M_K, mxk * VD * 2}, {M_V, mxk * VD * 2},
    {M_O, nwt * VD * 2}, {M_T, nwt * VD * 2}, {M_MID, nwt * VFF * 2}, {M_LQ, (size_t)WS * 4096}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  d.offCtx = off; off += up4k((size_t)CTXN * VD * 2); d.offKin2 = off; off += up4k((size_t)CTXN * VD * 2); d.offK2 = off; off += up4k((size_t)CTXN * VD * 2); d.offV2 = off; off += up4k((size_t)CTXN * VD * 2); L.total = off; return d;
}
static bool run_dit_block_w(const DL& d, Buf& blob, Buf& w, const f32* ctx, const f32* mods, int nv, int WS, uint8_t* Xh, const uint8_t* Ch, const uint8_t* Sh, bool first, bool last, const f32* mods0, int ns0) {
  const Layout& L = d.L; uint8_t* bb = blob.p; const int ns = nv / 32, nwin = (ns + WS - 1) / WS; int meta[M_NMETA]; std::vector<f32> mm(4 * VD), gb(2 * VD);
  const size_t xsl = (size_t)128 * 1024 * 4, tsl = (size_t)32 * 2 * 4096; if (!mods0) ns0 = 0;
  static const int fc = getenv("ATTN_CHUNK") ? atoi(getenv("ATTN_CHUNK")) : 0, fq = getenv("ATTN_QB") ? atoi(getenv("ATTN_QB")) : 0;
  auto base = [&](int mode, int wtok, int nvalid) { for (int i = 0; i < M_NMETA; i++) meta[i] = L.meta[i]; meta[M_A10] = fc; meta[M_A11] = fq; meta[M_NW] = 3; meta[M_NCB] = 128; meta[M_MODE] = mode; meta[M_NTOK] = wtok; meta[M_NVALID] = nvalid; meta[M_STOP] = 0; meta[M_FIRST] = 0; meta[M_REL] = 0; };
  auto gate = [&](int k, int ns0w, int* n) -> const f32* { if (!ns0w) { *n = VD; return mods + (size_t)k * VD; } memcpy(gb.data(), mods0 + (size_t)k * VD, VD * 4); memcpy(gb.data() + VD, mods + (size_t)k * VD, VD * 4); *n = 2 * VD; return gb.data(); };
  auto ln = [&](int k, bool fst, int ns0w, int wtok) { base(91, wtok, wtok); meta[M_A0] = L.meta[M_H]; meta[M_FIRST] = fst; meta[M_A8] = ns0w; size_t o = 0;
    if (ns0w) { memcpy(mm.data(), mods0 + (size_t)(3 * k) * VD, VD * 4); memcpy(mm.data() + VD, mods0 + (size_t)(3 * k + 1) * VD, VD * 4); o = 2 * VD; }
    memcpy(mm.data() + o, mods + (size_t)(3 * k) * VD, VD * 4); memcpy(mm.data() + o + VD, mods + (size_t)(3 * k + 1) * VD, VD * 4); return call(L, meta, w, 0, blob, mm.data(), (int)(o + 2 * VD)) == 0; };
  // modulated context, then its K/V once into the second K/V buffers
  { std::vector<f32> cm((size_t)CTXN * VD); const f32* sk = mods + 9 * VD; const f32* ck = mods + 10 * VD;
    par_for(CTXN, [&](int t) { for (int i = 0; i < VD; i++) cm[(size_t)t * VD + i] = ctx[(size_t)t * VD + i] * (1.f + ck[i]) + sk[i]; });
    rows_to_h16(cm.data(), CTXN, VD, (uint16_t*)(bb + d.offCtx), CTXN / 32); }
  sync_clean(blob);
  auto crossKV = [&]() -> bool {
    { base(90, CTXN, CTXN); meta[M_A0] = L.meta[M_H]; meta[M_A1] = (int)d.offCtx; meta[M_A2] = 128; meta[M_A3] = 128; meta[M_A4] = 16; meta[M_A7] = (int)d.offKin2; meta[M_A9] = 0; meta[M_K] = (int)d.offK2; meta[M_V] = (int)d.offV2;
    if (call(L, meta, w, ATTN_PACK, blob, mods, VD)) return false; return true; }
  };
  auto load = [&](int w0, int nws) { memcpy(bb + L.meta[M_X], Xh + (size_t)w0 * xsl, (size_t)nws * xsl); memcpy(bb + L.meta[M_COS], Ch + (size_t)w0 * tsl, (size_t)nws * tsl); memcpy(bb + L.meta[M_SIN], Sh + (size_t)w0 * tsl, (size_t)nws * tsl); sync_clean(blob); };
  for (int pass = 0; pass < 2; pass++)
    for (int wi = 0; wi < nwin; wi++) {
      const int w0 = wi * WS, nws = std::min(WS, ns - w0), wtok = nws * 32; const int ns0w = ns0 > w0 ? std::min(ns0 - w0, nws) : 0;
      if (getenv("DIT_TRACE")) fprintf(stderr, "  win pass %d w0 %d nws %d t=%.2f\n", pass, w0, nws, now());
      load(w0, nws);
      if (!ln(0, first && pass == 0 && wi == 0, ns0w, wtok)) return false;
      if (pass == 0 && wi == 0 && !crossKV()) return false;   // after the first op of the process (it powers up HMX): K/V of the context, once per block
      if (pass == 0) {   // keys / values of this window into the global K/V buffers
        base(90, wtok, wtok); meta[M_A0] = meta[M_A1] = L.meta[M_H]; meta[M_A2] = 128; meta[M_A3] = 128; meta[M_A4] = 16 | 2; meta[M_A5] = L.meta[M_COS]; meta[M_A6] = L.meta[M_SIN]; meta[M_A7] = L.meta[M_HR]; meta[M_A9] = w0;
        if (call(L, meta, w, 0, blob, mods, VD)) return false;
        continue;
      }
      { base(90, wtok, nv); meta[M_A0] = meta[M_A1] = L.meta[M_H]; meta[M_A2] = 128; meta[M_A3] = 128; meta[M_A4] = 1 | 4 | 8; meta[M_A8] = ns0w; int gn; const f32* gp = gate(2, ns0w, &gn); if (call(L, meta, w, 0, blob, gp, gn)) return false; }
      if (!ln(1, false, ns0w, wtok)) return false;
      { base(90, wtok, CTXN); meta[M_A0] = L.meta[M_H]; meta[M_A1] = (int)d.offCtx; meta[M_A2] = 128; meta[M_A3] = 128; meta[M_A4] = 4 | 8; meta[M_A8] = ns0w; meta[M_K] = (int)d.offK2; meta[M_V] = (int)d.offV2; int gn; const f32* gp = gate(5, ns0w, &gn); if (call(L, meta, w, ATTN_PACK, blob, gp, gn)) return false; }
      if (!ln(2, false, ns0w, wtok)) return false;
      { base(92, wtok, wtok); meta[M_A0] = L.meta[M_H]; meta[M_A1] = VFF / 32; meta[M_REL] = (last && wi == nwin - 1) ? 1 : 2; meta[M_A8] = ns0w; int gn; const f32* gp = gate(8, ns0w, &gn); if (call(L, meta, w, 2 * ATTN_PACK, blob, gp, gn)) return false; }
      sync_inval(blob); memcpy(Xh + (size_t)w0 * xsl, bb + L.meta[M_X], (size_t)nws * xsl);
    }
  return true;
}
static bool read_all(const std::string& p, std::vector<f32>& v, size_t n) { FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); return false; } v.resize(n); bool ok = fread(v.data(), 4, n, f) == n; fclose(f); return ok; }
static bool dittest(const std::string& dir) {   // one block from a case dir (mkdit_case.py)
  std::vector<f32> x0, ctx, mods, ref, cs, sn; 
  struct stat s0; if (stat((dir + "/x0.f32").c_str(), &s0)) return false; const int nv = (int)(s0.st_size / (VD * 4)), ns = nv / 32;
  if (!read_all(dir + "/x0.f32", x0, (size_t)nv * VD) || !read_all(dir + "/ctx.f32", ctx, (size_t)CTXN * VD) || !read_all(dir + "/mods.f32", mods, 11 * VD) || !read_all(dir + "/ref.f32", ref, (size_t)nv * VD)
      || !read_all(dir + "/cos.bin", cs, (size_t)ns * 32 * 2 * 1024) || !read_all(dir + "/sin.bin", sn, (size_t)ns * 32 * 2 * 1024)) return false;
  DL d = make_layout_dit(nv); Buf blob = make_buf(d.L.total, true); if (!blob.p) return false; memset(blob.p, 0, d.L.total);
  struct stat sb; if (stat((dir + "/blk.bin").c_str(), &sb)) return false; Buf w = make_buf(sb.st_size); if (!w.p) return false; g_all.push_back(&w);
  if (!read_into(dir + "/blk.bin", w.p, w.sz)) return false; sync_clean(w);
  rows_to_x32(x0.data(), nv, VD, (f32*)(blob.p + d.L.meta[M_X]), ns); memcpy(blob.p + d.L.meta[M_COS], cs.data(), cs.size() * 4); memcpy(blob.p + d.L.meta[M_SIN], sn.data(), sn.size() * 4); gm_hn((uint16_t*)(blob.p + d.L.meta[M_HN]));
  double t0 = now(); if (!run_dit_block(d, blob, w, ctx.data(), mods.data(), nv, true, true)) return false;
  sync_inval(blob); std::vector<f32> rows((size_t)nv * VD); x32_to_rows((const f32*)(blob.p + d.L.meta[M_X]), nv, VD, rows.data());
  double e = 0, n = 0; for (size_t i = 0; i < rows.size(); i++) { const double q = rows[i] - ref[i]; e += q * q; n += (double)ref[i] * ref[i]; }
  fprintf(stderr, "dittest: %d tokens, block %.3fs, rel err vs reference %.3e\n", nv, now() - t0, sqrt(e / n));
  FILE* f = fopen((dir + "/out.f32").c_str(), "wb"); fwrite(rows.data(), 4, rows.size(), f); fclose(f); return true;
}
static bool map_file_decl_placeholder;
static bool map_file(const std::string& p, const void** out, size_t* len) { int fd = open(p.c_str(), O_RDONLY); if (fd < 0) return false; struct stat s; fstat(fd, &s); *len = s.st_size; void* m = mmap(nullptr, *len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); if (m == MAP_FAILED) return false; *out = m; return true; }
static bool read_f32(const std::string& p, std::vector<f32>& v, size_t n) { FILE* f = fopen(p.c_str(), "rb"); if (!f) return false; v.resize(n); bool ok = fread(v.data(), 4, n, f) == n; fclose(f); return ok; }

// ---------------------------------------------------------------- DiT driver: glue (CPU), 48 streamed blocks, sampler
struct DG { const __fp16 *pw = nullptr, *a1 = nullptr, *a2 = nullptr, *al = nullptr, *p1 = nullptr, *p2 = nullptr, *pl = nullptr, *projw = nullptr; std::vector<f32> pb, a1b, a2b, alb, p1b, p2b, plb, sst, osst, projb; bool ok = false; };
static DG g_dg; static std::vector<std::string> g_df;
static bool load_dit_glue(const std::string& d) {
  DG& g = g_dg; size_t l; auto mm = [&](const char* n, const __fp16** p) { return map_file(d + "/dit/" + n, (const void**)p, &l); }; auto rf = [&](const char* n, std::vector<f32>& v, size_t c) { return read_f32(d + "/dit/" + n, v, c); };
  g.ok = mm("patch_w.f16", &g.pw) && mm("a1_w.f16", &g.a1) && mm("a2_w.f16", &g.a2) && mm("al_w.f16", &g.al) && mm("p1_w.f16", &g.p1) && mm("p2_w.f16", &g.p2) && mm("pl_w.f16", &g.pl) && mm("proj_w.f16", &g.projw)
      && rf("patch_b.f32", g.pb, 4096) && rf("a1_b.f32", g.a1b, 4096) && rf("a2_b.f32", g.a2b, 4096) && rf("al_b.f32", g.alb, 36864) && rf("p1_b.f32", g.p1b, 4096) && rf("p2_b.f32", g.p2b, 4096) && rf("pl_b.f32", g.plb, 8192)
      && rf("sst.f32", g.sst, (size_t)48 * 11 * VD) && rf("out_sst.f32", g.osst, 2 * VD) && rf("proj_b.f32", g.projb, 128);
  return g.ok;
}
static void ada_chain(double t, const __fp16* w1, const std::vector<f32>& b1, const __fp16* w2, const std::vector<f32>& b2, const __fp16* wl, const std::vector<f32>& bl, int outl, std::vector<f32>& embedded, std::vector<f32>& temb) {
  f32 e[256]; for (int i = 0; i < 128; i++) { const double a = t * exp(-log(10000.0) * i / 128.0); e[i] = (f32)cos(a); e[128 + i] = (f32)sin(a); }   // flip_sin_to_cos: [cos | sin]
  std::vector<f32> y(VD); gemm_f16(w1, e, y.data(), 1, VD, 256); for (int i = 0; i < VD; i++) { const f32 v = y[i] + b1[i]; y[i] = v / (1.f + expf(-v)); }
  embedded.resize(VD); gemm_f16(w2, y.data(), embedded.data(), 1, VD, VD); for (int i = 0; i < VD; i++) embedded[i] += b2[i];
  for (int i = 0; i < VD; i++) y[i] = embedded[i] / (1.f + expf(-embedded[i]));
  temb.resize(outl); gemm_f16(wl, y.data(), temb.data(), 1, outl, VD); for (int i = 0; i < outl; i++) temb[i] += bl[i];
}
static void dit_rope(f32* cs, f32* sn, int Fl, int Hl, int Wl, double fps) {   // video 3D rope (split, float64 grid): per-head tables [slab][head][tile][32 rows][32]; 2046 = 682 freqs x 3 axes interleaved, 2 padding entries (cos 1, sin 0) in front
  const int nv = Fl * Hl * Wl; std::vector<float> idx(682); for (int i = 0; i < 682; i++) idx[i] = (float)(pow(10000.0, (double)i / 681.0) * M_PI / 2);
  const float mx[3] = {20.f, 2048.f, 2048.f};
  par_for(nv, [&](int t) {
    const int f = t / (Hl * Wl), h = (t / Wl) % Hl, w = t % Wl;
    auto pf = [&](int p) { return (float)(std::max(p * 8 - 7, 0)); };   // causal_fix on the pixel-space frame index (scale 8): (p*8 + 1 - 8).clamp(0)
    const float mid[3] = {(float)((pf(f) + pf(f + 1)) / 2.0 / fps), (float)(h * 32 + 16), (float)(w * 32 + 16)};
    for (int j = 0; j < 2048; j++) {
      float c = 1.f, s = 0.f;
      if (j >= 2) { const int k = j - 2, i = k / 3, ax = k % 3; const float fr = mid[ax] / mx[ax]; const float a = idx[i] * (fr * 2.f - 1.f); c = cosf(a); s = sinf(a); }
      const int head = j / 64, d = j % 64; const size_t o = ((((size_t)(t / 32) * 32 + head) * 2 + d / 32) * 1024) + (size_t)(t % 32) * 32 + d % 32; cs[o] = c; sn[o] = s; } });
}
static bool run_dit(const std::vector<f32>& ctx, int Fl, int Hl, int Wl, unsigned seed, int steps, const std::string& outf, const char* z0f, const char* condf = nullptr) {
  const int nv = Fl * Hl * Wl, ns = nv / 32; if (nv % 32 || !g_dg.ok) return false; double T0 = now();
  static const double SIG9[9] = {1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0}; if (steps < 1 || steps > 8) { fprintf(stderr, "steps 1..8\n"); return false; }
  double SIG[9]; for (int i = 0; i <= steps; i++) SIG[i] = SIG9[(i * 8 + steps / 2) / steps];   // fewer steps: an even subset of the distilled schedule (ends at 0)
  std::vector<f32> z((size_t)nv * 128); if (z0f) { if (!read_f32(z0f, z, z.size())) return false; } else { std::mt19937_64 rng(seed); std::normal_distribution<float> nd(0.f, 1.f); for (auto& v : z) v = nd(rng); }
  std::vector<f32> cond; const size_t nc = (size_t)Hl * Wl * 128; const int ns0 = (condf && (Hl * Wl) % 32 == 0) ? Hl * Wl / 32 : 0;   // per-token timestep: latent frame 0 runs with timestep 0 (needs whole slabs)
  if (condf && !ns0) fprintf(stderr, "I2V: %d tokens per frame is not a multiple of 32, no per-token timestep\n", Hl * Wl);
     // image-to-video: the clean latent of the image replaces the tokens of latent frame 0 (at the start and after every step)
  if (condf) { if (!read_f32(condf, cond, nc)) return false; memcpy(z.data(), cond.data(), nc * 4); }
  const bool win = getenv("DIT_WIN") != nullptr || nv > 6000; const int WS = getenv("DIT_WIN") ? std::max(1, atoi(getenv("DIT_WIN"))) : 64;   // windowed activations (no token limit) above 6000 tokens, or when forced for tests
  DL d = win ? make_layout_dit_w(nv, WS) : make_layout_dit(nv); Buf blob = make_buf(d.L.total, true); if (!blob.p) { fprintf(stderr, "cannot allocate %.0f MB\n", d.L.total / 1048576.0); return false; } memset(blob.p, 0, d.L.total);
  std::vector<uint8_t> Xh, Ch, Sh;
  if (win) { Xh.resize((size_t)ns * 128 * 1024 * 4); Ch.resize((size_t)ns * 32 * 2 * 4096); Sh.resize((size_t)ns * 32 * 2 * 4096); dit_rope((f32*)Ch.data(), (f32*)Sh.data(), Fl, Hl, Wl, 24.0); }
  else dit_rope((f32*)(blob.p + d.L.meta[M_COS]), (f32*)(blob.p + d.L.meta[M_SIN]), Fl, Hl, Wl, 24.0);
  gm_hn((uint16_t*)(blob.p + d.L.meta[M_HN]));
  for (int s = 0; s < steps; s++) {
    const double sg = SIG[s], sn = SIG[s + 1], ts = now();
    std::vector<f32> emb, temb, pemb, ptemb, emb0, temb0, pe0, pt0; if (ns0) ada_chain(0.0, g_dg.a1, g_dg.a1b, g_dg.a2, g_dg.a2b, g_dg.al, g_dg.alb, 36864, emb0, temb0);
    ada_chain(sg * 1000.0, g_dg.a1, g_dg.a1b, g_dg.a2, g_dg.a2b, g_dg.al, g_dg.alb, 36864, emb, temb); ada_chain(sg * 1000.0, g_dg.p1, g_dg.p1b, g_dg.p2, g_dg.p2b, g_dg.pl, g_dg.plb, 8192, pemb, ptemb);
    { std::vector<f32> x((size_t)nv * VD); gemm_f16(g_dg.pw, z.data(), x.data(), nv, VD, 128); par_for(nv, [&](int t) { for (int i = 0; i < VD; i++) x[(size_t)t * VD + i] += g_dg.pb[i]; });
      rows_to_x32(x.data(), nv, VD, win ? (f32*)Xh.data() : (f32*)(blob.p + d.L.meta[M_X]), ns); }
    std::vector<f32> mods(11 * VD), mods0(11 * VD);
    if (!stream_files(g_df, g_te, [&](int b, Buf& w) {
      const f32* sst = &g_dg.sst[(size_t)b * 11 * VD]; auto v = [&](int k, int i) { return sst[(size_t)k * VD + i] + temb[(size_t)k * VD + i]; };
      for (int i = 0; i < VD; i++) { mods[0 * VD + i] = 1.f + v(1, i); mods[1 * VD + i] = v(0, i); mods[2 * VD + i] = v(2, i); mods[3 * VD + i] = 1.f + v(7, i); mods[4 * VD + i] = v(6, i); mods[5 * VD + i] = v(8, i);
        mods[6 * VD + i] = 1.f + v(4, i); mods[7 * VD + i] = v(3, i); mods[8 * VD + i] = v(5, i); mods[9 * VD + i] = sst[(size_t)9 * VD + i] + ptemb[i]; mods[10 * VD + i] = sst[(size_t)10 * VD + i] + ptemb[VD + i]; }
      if (ns0) { auto v0 = [&](int k, int i) { return sst[(size_t)k * VD + i] + temb0[(size_t)k * VD + i]; };
        for (int i = 0; i < VD; i++) { mods0[0 * VD + i] = 1.f + v0(1, i); mods0[1 * VD + i] = v0(0, i); mods0[2 * VD + i] = v0(2, i); mods0[3 * VD + i] = 1.f + v0(7, i); mods0[4 * VD + i] = v0(6, i); mods0[5 * VD + i] = v0(8, i);
          mods0[6 * VD + i] = 1.f + v0(4, i); mods0[7 * VD + i] = v0(3, i); mods0[8 * VD + i] = v0(5, i); } }
      if (win) return run_dit_block_w(d, blob, w, ctx.data(), mods.data(), nv, WS, Xh.data(), Ch.data(), Sh.data(), b == 0, b == 47, ns0 ? mods0.data() : nullptr, ns0);
      return run_dit_block(d, blob, w, ctx.data(), mods.data(), nv, b == 0, b == 47, ns0 ? mods0.data() : nullptr, ns0); }, 0)) return false;
    if (!win) sync_inval(blob); std::vector<f32> rows((size_t)nv * VD); x32_to_rows(win ? (const f32*)Xh.data() : (const f32*)(blob.p + d.L.meta[M_X]), nv, VD, rows.data());
    std::vector<f32> vel((size_t)nv * 128);
    { std::vector<f32> sh(VD), sc(VD), sh0(VD), sc0(VD); for (int i = 0; i < VD; i++) { sh[i] = g_dg.osst[i] + emb[i]; sc[i] = 1.f + g_dg.osst[VD + i] + emb[i]; if (ns0) { sh0[i] = g_dg.osst[i] + emb0[i]; sc0[i] = 1.f + g_dg.osst[VD + i] + emb0[i]; } }
      par_for(nv, [&](int t) { f32* r = &rows[(size_t)t * VD]; const bool c0 = t < ns0 * 32; const f32* shp = c0 ? sh0.data() : sh.data(); const f32* scp = c0 ? sc0.data() : sc.data(); double mu = 0, ss = 0; for (int i = 0; i < VD; i++) mu += r[i]; mu /= VD; for (int i = 0; i < VD; i++) { const double q = r[i] - mu; ss += q * q; }
        const float rs = 1.f / sqrtf((float)(ss / VD) + 1e-6f); for (int i = 0; i < VD; i++) r[i] = (float)(r[i] - mu) * rs * scp[i] + shp[i]; });
      gemm_f16(g_dg.projw, rows.data(), vel.data(), nv, 128, VD); for (int t = 0; t < nv; t++) for (int i = 0; i < 128; i++) vel[(size_t)t * 128 + i] += g_dg.projb[i]; }
    if (getenv("DIT_DUMPV")) { FILE* f = fopen((outf + ".v" + std::to_string(s)).c_str(), "wb"); fwrite(vel.data(), 4, vel.size(), f); fclose(f); }
    for (size_t i = 0; i < z.size(); i++) z[i] += (f32)(sn - sg) * vel[i];
    if (condf) memcpy(z.data(), cond.data(), nc * 4);
    double sv = 0, sx = 0; for (size_t i = 0; i < vel.size(); i++) { sv += (double)vel[i] * vel[i]; sx += (double)z[i] * z[i]; }
    fprintf(stderr, "step %d sigma=%.4f |v|=%.3f |x|=%.3f  %.2fs (io wait %.2fs)\n", s, sg, sqrt(sv / vel.size()), sqrt(sx / z.size()), now() - ts, g_waitio);
  }
  FILE* f = fopen(outf.c_str(), "wb"); fwrite(z.data(), 4, z.size(), f); fclose(f); fprintf(stderr, "dit total %.2fs\n", now() - T0);
  unmap_buf(blob); rpcmem_free(blob.p); return true;
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
    if (cmd == "load") {   // load DIR
      ss >> g_dir; double t0 = now(); bool ok = true; size_t mx = 0; g_gm.clear();
      for (int l = 0; l < NLG; l++) { char p[512]; snprintf(p, sizeof p, "%s/te/gm%02d.bin", g_dir.c_str(), l); struct stat s; if (stat(p, &s)) ok = false; else mx = std::max<size_t>(mx, s.st_size); g_gm.push_back(p); }
      { struct stat sa; g_audio = !stat((g_dir + "/glue/aproj.f16").c_str(), &sa); }
      ok = ok && map_file(g_dir + "/glue/embd.f16", (const void**)&g_embf, &g_embfl) && map_file(g_dir + "/glue/vproj.f16", (const void**)&g_vw, &g_vwl)
         && read_f32(g_dir + "/glue/final_norm.f32", g_fnorm, GD) && read_f32(g_dir + "/glue/vproj_b.f32", g_vb, 4096);
      if (g_audio) ok = ok && map_file(g_dir + "/glue/aproj.f16", (const void**)&g_aw, &g_awl) && read_f32(g_dir + "/glue/aproj_b.f32", g_ab, 2048);
      g_df.clear(); { bool dok = true; for (int l = 0; l < 48; l++) { char p[512]; snprintf(p, sizeof p, "%s/dit/vb%02d.bin", g_dir.c_str(), l); struct stat s3; if (stat(p, &s3)) dok = false; else mx = std::max<size_t>(mx, s3.st_size); g_df.push_back(p); } if (dok) load_dit_glue(g_dir); else g_dg.ok = false; }
      g_conn = true; for (int k = 0; k < (g_audio ? 2 : 1); k++) { g_cf[k].clear(); for (int l = 0; l < 8; l++) { char p[512]; snprintf(p, sizeof p, "%s/conn/%cc%d.bin", g_dir.c_str(), k ? 'a' : 'v', l); struct stat s2; if (stat(p, &s2)) g_conn = false; else mx = std::max<size_t>(mx, s2.st_size); g_cf[k].push_back(p); }
        if (!read_f32(g_dir + (k ? "/glue/areg.f32" : "/glue/vreg.f32"), g_reg[k], (size_t)128 * (k ? 2048 : 4096))) g_conn = false; }
      if (!ok) { printf("error files\n"); continue; }
      g_te[0] = make_buf(mx); g_te[1] = make_buf(mx); g_all.push_back(&g_te[0]); g_all.push_back(&g_te[1]); if (!g_te[0].p || !g_te[1].p) { printf("error bufs\n"); continue; }
      printf("loaded %.1f connectors %d\n", now() - t0, (int)g_conn); continue;
    }
    if (cmd == "dit") {   // dit CTX.f32 FL HL WL SEED STEPS OUT.f32 [Z0.f32|-] [COND.f32: clean latent tokens [HL*WL][128] of the first frame]
      std::string cf, outz, z0, cond; int Fl, Hl, Wl, steps; unsigned seed; ss >> cf >> Fl >> Hl >> Wl >> seed >> steps >> outz >> z0 >> cond; std::vector<f32> ctx; if (z0 == "-") z0.clear();
      if (!read_f32(cf, ctx, (size_t)CTXN * VD)) { printf("error ctx\n"); continue; } printf(run_dit(ctx, Fl, Hl, Wl, seed, steps, outz, z0.empty() ? nullptr : z0.c_str(), cond.empty() ? nullptr : cond.c_str()) ? "done\n" : "error dit\n"); continue; }
    if (cmd == "dittest") { std::string dir; ss >> dir; printf(dittest(dir) ? "done\n" : "error dittest\n"); continue; }
    if (cmd == "te") {   // te IDS.i32 OUTPREFIX
      std::string idsf, outp; ss >> idsf >> outp; std::vector<int> ids;
      { FILE* f = fopen(idsf.c_str(), "rb"); if (!f) { printf("error ids\n"); continue; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(n / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) {} fclose(f); }
      if (!encode_gm(ids, outp)) printf("error te\n"); else printf("done %zu\n", ids.size()); continue;
    }
  }
  mega_close(g_h);
  return 0;
}
