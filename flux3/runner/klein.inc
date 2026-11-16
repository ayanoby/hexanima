// ------------------------------------------------------------------ FLUX.3 Klein: text encoder (Qwen3-VL-4B language model, layers 0..31, MODE 22, hidden 2560)
// features for the DiT: residual stream after layers 3,7,...,31 (= HF hidden_states[4,8,...,32], before the final norm) concatenated per token: [T][8*2560]
static const int TD = 2560, NL_TE = 32, MAXT = 1024, NTAP = 8;
static const int TAPS[NTAP] = {3, 7, 11, 15, 19, 23, 27, 31};
static std::vector<std::string> g_tefiles;
static int g_T_ns = -1;
static Layout make_layout_te(int ns) {
  Layout L; memset(&L, 0, sizeof L); const size_t ntok = (size_t)ns * 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * TD * 4}, {M_COS, (size_t)ns * 8192}, {M_SIN, (size_t)ns * 8192}, {M_H, ntok * 4096 * 2}, {M_Q, ntok * 4096 * 2}, {M_K, ntok * 1024 * 2}, {M_V, ntok * 1024 * 2},
    {M_O, ntok * 4096 * 2}, {M_T, ntok * TD * 2}, {M_MID, ntok * 9728 * 2}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LQ, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
static bool readraw(const char* p, std::vector<f32>& v, size_t n) { FILE* f = fopen(p, "rb"); if (!f) return false; v.resize(n); bool ok = fread(v.data(), 4, n, f) == n; fclose(f); return ok; }
static bool encode_text(const std::vector<int>& ids, std::vector<f32>& feat) {
  const int T = (int)ids.size(), ns = (T + 31) / 32, ntok = ns * 32; if (T > MAXT) return false;
  double t0 = now(); g_waitio = 0;
  if (g_T_ns != ns) { if (g_blobTE.p) { unmap_buf(g_blobTE); rpcmem_free(g_blobTE.p); } g_LT = make_layout_te(ns); g_blobTE = make_buf(g_LT.total, true); if (!g_blobTE.p) return false; g_T_ns = ns; }
  uint8_t* blob = g_blobTE.p; memset(blob, 0, g_LT.total);
  { std::vector<f32> rows((size_t)ntok * TD, 0.f); const f32* sc = (const f32*)g_embd; const int8_t* q = (const int8_t*)(g_embd + (size_t)151936 * 4);
    for (int t = 0; t < T; t++) { const int id = ids[t]; const f32 s = sc[id]; const int8_t* r = q + (size_t)id * TD; f32* o = &rows[(size_t)t * TD]; for (int i = 0; i < TD; i++) o[i] = s * r[i]; }
    rows_to_x32(rows.data(), ntok, TD, (f32*)(blob + g_LT.meta[M_X]), ns); }
  { f32* cs = (f32*)(blob + g_LT.meta[M_COS]); f32* sn = (f32*)(blob + g_LT.meta[M_SIN]); memset(cs, 0, (size_t)ns * 8192); memset(sn, 0, (size_t)ns * 8192);   // text only: all three mrope axes carry the position
    par_for(T, [&](int t) { for (int d = 0; d < 64; d++) { const double a = t * pow(5e6, -(double)d / 64.0); const size_t idx = (((size_t)(t / 32) * 2 + d / 32) * 1024) + (t % 32) * 32 + d % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } }); }
  sync_clean(g_blobTE);
  std::vector<f32> mod(18432, 0.f); for (int i = 8192; i < 12288; i++) mod[i] = 1.f;
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = g_LT.meta[i];
  meta[M_NW] = 3; meta[M_NCB] = 80; meta[M_MODE] = 22; meta[M_FMT] = 1; meta[M_NTOK] = ntok; meta[M_NVALID] = T; meta[M_STOP] = 0;
  feat.assign((size_t)T * NTAP * TD, 0.f); std::vector<f32> rows((size_t)T * TD);
  auto tap = [](int l) { for (int j = 0; j < NTAP; j++) if (TAPS[j] == l) return j; return -1; };
  if (!stream_files(g_tefiles, g_te, [&](int l, Buf& w) {
    meta[M_FIRST] = l == 0 || tap(l - 1) >= 0; meta[M_REL] = tap(l) >= 0;
    if (call(g_LT, meta, w, 0, g_blobTE, mod.data(), (int)mod.size())) return false;
    const int j = tap(l);
    if (j >= 0) { sync_inval(g_blobTE); x32_to_rows((const f32*)(blob + g_LT.meta[M_X]), T, TD, rows.data());
      par_for(T, [&](int t) { memcpy(&feat[((size_t)t * NTAP + j) * TD], &rows[(size_t)t * TD], TD * 4); }); }
    return true; }, 0)) return false;
  fprintf(stderr, "te: %.2fs (%d tokens, waited %.2fs for io)\n", now() - t0, T, g_waitio); return true;
}

// ------------------------------------------------------------------ FLUX.3 Klein DiT (MODE 70): 5 per-mode blocks (image rows, text rows), then 28 shared blocks, 16 steps (flow map), no CFG
static const int HD = 3072, NB_I = 5, NB_S = 28;
struct KG { std::vector<__fp16> xi, ctx, t1, t2, r1, r2, ei, et, si, st, nout, pout; bool ok = false; };   // glue, all fp16 [out][in]
static KG g_kg;
static bool load_h(const std::string& p, std::vector<__fp16>& v, size_t n, bool isf32) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); return false; } v.resize(n);
  if (!isf32) { if (fread(v.data(), 2, n, f) != n) { fclose(f); return false; } }
  else { std::vector<f32> t(n); if (fread(t.data(), 4, n, f) != n) { fclose(f); return false; } for (size_t i = 0; i < n; i++) v[i] = (__fp16)t[i]; }
  fclose(f); return true;
}
static bool load_glue(const std::string& d) {
  KG& g = g_kg;
  g.ok = load_h(d + "/x_img.f32", g.xi, (size_t)HD * 128, true) && load_h(d + "/ctx.f16", g.ctx, (size_t)HD * 20480, false)
      && load_h(d + "/time_1.f32", g.t1, (size_t)HD * 256, true) && load_h(d + "/time_2.f32", g.t2, (size_t)HD * HD, true) && load_h(d + "/ref_1.f32", g.r1, (size_t)HD * 256, true) && load_h(d + "/ref_2.f32", g.r2, (size_t)HD * HD, true)
      && load_h(d + "/early_image.f16", g.ei, (size_t)3 * HD * HD, false) && load_h(d + "/early_txt.f16", g.et, (size_t)3 * HD * HD, false)
      && load_h(d + "/single_image.f16", g.si, (size_t)3 * HD * HD, false) && load_h(d + "/single_txt.f16", g.st, (size_t)3 * HD * HD, false)
      && load_h(d + "/nout.f32", g.nout, (size_t)2 * HD * HD, true) && load_h(d + "/proj_out.f32", g.pout, (size_t)128 * HD, true);
  return g.ok;
}
static void tembed(double t, f32* out) { for (int i = 0; i < 128; i++) { const double a = 1000.0 * t * exp(-log(10000.0) * i / 128.0); out[i] = (f32)cos(a); out[128 + i] = (f32)sin(a); } }
static void silu_v(f32* v, int n) { for (int i = 0; i < n; i++) v[i] = v[i] / (1.f + expf(-v[i])); }
static void tmlp(const std::vector<__fp16>& w1, const std::vector<__fp16>& w2, const f32* emb, f32* out) { std::vector<f32> h(HD); gemm_f16(w1.data(), emb, h.data(), 1, HD, 256); silu_v(h.data(), HD); gemm_f16(w2.data(), h.data(), out, 1, HD, HD); }
static Layout make_layout_dit(int ns) {
  Layout L; memset(&L, 0, sizeof L); const size_t ntok = (size_t)ns * 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, ntok * HD * 4}, {M_COS, (size_t)ns * 8192}, {M_SIN, (size_t)ns * 8192}, {M_H, ntok * HD * 2}, {M_Q, ntok * HD * 2}, {M_K, ntok * HD * 2}, {M_V, ntok * HD * 2},
    {M_O, ntok * HD * 2}, {M_T, ntok * HD * 2}, {M_MID, ntok * 9216 * 2}, {M_HN, 131072}, {M_HR, 4096}, {M_ROWS, 4096}, {M_CS, 4096}, {M_LQ, 4096}, {M_LO, 4096}, {M_LU, 4096}, {M_LD, 4096}, {M_DLT, 4096}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}
// rope tables for tokens with 4-axis positions pos[t][4]: pair i -> axis i/16, angle = pos * 10000^-((i%16)/16); tiles [slab][2][32][32]
static void rope_tab(const std::vector<f32>& pos, int ntok_valid, int ns, f32* cs, f32* sn) {
  memset(cs, 0, (size_t)ns * 8192); memset(sn, 0, (size_t)ns * 8192);
  par_for(ntok_valid, [&](int t) { for (int i = 0; i < 64; i++) { const double a = pos[(size_t)t * 4 + i / 16] * pow(10000.0, -(double)(i % 16) / 16.0); const size_t idx = (((size_t)(t / 32) * 2 + i / 32) * 1024) + (t % 32) * 32 + i % 32; cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a); } });
}
static bool run_dit(const std::vector<f32>& feat, int R, int lh, int lw, unsigned seed, int steps, const std::string& outz, const char* z0file) {
  const double T0 = now(); const int N = lh * lw; if (N % 32) { fprintf(stderr, "image tokens must be a multiple of 32\n"); return false; }
  const int nsI = N / 32, nsT = (R + 31) / 32, nsM = nsI + nsT; const KG& G = g_kg; if (!G.ok) { fprintf(stderr, "no glue\n"); return false; }
  Buf bI, bT, bM; Layout LI = make_layout_dit(nsI), LT = make_layout_dit(nsT), LM = make_layout_dit(nsM);
  bI = make_buf(LI.total, true); bT = make_buf(LT.total, true); bM = make_buf(LM.total, true); if (!bI.p || !bT.p || !bM.p) return false;
  memset(bI.p, 0, LI.total); memset(bT.p, 0, LT.total); memset(bM.p, 0, LM.total);
  { std::vector<f32> pi((size_t)N * 4, 0.f), pt((size_t)R * 4, 0.f);
    for (int r = 0; r < lh; r++) for (int c = 0; c < lw; c++) { pi[(size_t)(r * lw + c) * 4 + 1] = r * 0.5f; pi[(size_t)(r * lw + c) * 4 + 2] = c * 0.5f; }
    for (int t = 0; t < R; t++) pt[(size_t)t * 4 + 3] = (f32)t;
    rope_tab(pi, N, nsI, (f32*)(bI.p + LI.meta[M_COS]), (f32*)(bI.p + LI.meta[M_SIN])); rope_tab(pt, R, nsT, (f32*)(bT.p + LT.meta[M_COS]), (f32*)(bT.p + LT.meta[M_SIN]));
    memcpy(bM.p + LM.meta[M_COS], bI.p + LI.meta[M_COS], (size_t)nsI * 8192); memcpy(bM.p + LM.meta[M_COS] + (size_t)nsI * 8192, bT.p + LT.meta[M_COS], (size_t)nsT * 8192);
    memcpy(bM.p + LM.meta[M_SIN], bI.p + LI.meta[M_SIN], (size_t)nsI * 8192); memcpy(bM.p + LM.meta[M_SIN] + (size_t)nsI * 8192, bT.p + LT.meta[M_SIN], (size_t)nsT * 8192); }
  std::vector<f32> ttx((size_t)R * HD);   // text rows once per prompt
  { double t1 = now(); gemm_f16(G.ctx.data(), feat.data(), ttx.data(), R, HD, 20480); fprintf(stderr, "ctx embed: %.2fs\n", now() - t1);
    std::vector<f32> rows((size_t)nsT * 32 * HD, 0.f); memcpy(rows.data(), ttx.data(), (size_t)R * HD * 4); rows_to_x32(rows.data(), nsT * 32, HD, (f32*)(bT.p + LT.meta[M_X]), nsT); }
  std::vector<f32> z((size_t)N * 128);
  if (z0file) { if (!readraw(z0file, z, z.size())) { fprintf(stderr, "z0 unreadable\n"); return false; } }
  else { std::mt19937 gen(seed); std::normal_distribution<float> nd(0.f, 1.f); for (auto& a : z) a = nd(gen); }
  const double shift = getenv("KLEIN_SHIFT") ? std::max(1.0, atof(getenv("KLEIN_SHIFT"))) : 16.0;   // the timestep schedule: t' = shift*t / (1 + (shift-1)*t), 16 = trained, 1 = linear
  std::vector<double> sch(steps + 1); for (int i = 0; i <= steps; i++) { const double t = 1.0 - (double)i / steps; sch[i] = shift * t / (1.0 + (shift - 1.0) * t); }
  std::vector<f32> rowsI((size_t)N * HD), emb(256), tt(HD), tr(HD), v_i(HD), v_t(HD), modE(18432), modS(18432), nm(2 * HD), tnt(HD), ev(256);
  tembed(0.0, ev.data()); tmlp(G.t1, G.t2, ev.data(), tnt.data());   // time embedding of the text rows (t = 0)
  int meta[M_NMETA];
  for (int s = 0; s < steps; s++) {
    const double ts = now(), t = sch[s], tn = sch[s + 1], dt = tn - t;
    tembed(t, emb.data()); tmlp(G.t1, G.t2, emb.data(), tt.data()); tembed(tn, emb.data()); tmlp(G.r1, G.r2, emb.data(), tr.data());
    for (int i = 0; i < HD; i++) { v_i[i] = tt[i] + tr[i]; v_t[i] = tnt[i] + tr[i]; } silu_v(v_i.data(), HD); silu_v(v_t.data(), HD);
    gemm_f16(G.ei.data(), v_i.data(), &modE[0], 1, 3 * HD, HD); gemm_f16(G.et.data(), v_t.data(), &modE[9216], 1, 3 * HD, HD);
    gemm_f16(G.si.data(), v_i.data(), &modS[0], 1, 3 * HD, HD); gemm_f16(G.st.data(), v_t.data(), &modS[9216], 1, 3 * HD, HD);
    gemm_f16(G.nout.data(), v_i.data(), nm.data(), 1, 2 * HD, HD);   // shift, scale of the output norm
    gemm_f16(G.xi.data(), z.data(), rowsI.data(), N, HD, 128);
    rows_to_x32(rowsI.data(), N, HD, (f32*)(bI.p + LI.meta[M_X]), nsI); sync_clean(bI);
    { std::vector<f32> rt((size_t)nsT * 32 * HD, 0.f); memcpy(rt.data(), ttx.data(), (size_t)R * HD * 4); rows_to_x32(rt.data(), nsT * 32, HD, (f32*)(bT.p + LT.meta[M_X]), nsT); sync_clean(bT); }
    std::vector<std::string> files; char p[512];
    for (int d = 0; d < NB_I; d++) { snprintf(p, sizeof p, "%s/img%02d.bin", g_dir.c_str(), d); files.push_back(p); snprintf(p, sizeof p, "%s/txt%02d.bin", g_dir.c_str(), d); files.push_back(p); }
    for (int d = 0; d < NB_S; d++) { snprintf(p, sizeof p, "%s/sgl%02d.bin", g_dir.c_str(), d); files.push_back(p); }
    const int nfile = (int)files.size();
    if (!stream_files(files, g_te, [&](int i, Buf& w) {
      Buf* blob; Layout* L; int ns_, nimg, nvalid;
      if (i < 2 * NB_I) { if (i & 1) { blob = &bT; L = &LT; ns_ = nsT; nimg = 0; nvalid = R; } else { blob = &bI; L = &LI; ns_ = nsI; nimg = nsI; nvalid = N; } }
      else {
        if (i == 2 * NB_I) { sync_inval(bI); sync_inval(bT); memcpy(bM.p + LM.meta[M_X], bI.p + LI.meta[M_X], (size_t)nsI * 32 * HD * 4); memcpy(bM.p + LM.meta[M_X] + (size_t)nsI * 32 * HD * 4, bT.p + LT.meta[M_X], (size_t)nsT * 32 * HD * 4); sync_clean(bM); }
        blob = &bM; L = &LM; ns_ = nsM; nimg = nsI; nvalid = N + R; }
      for (int k = 0; k < M_NMETA; k++) meta[k] = L->meta[k];
      meta[M_NW] = 3; meta[M_NCB] = 96; meta[M_MODE] = 70; meta[M_NTOK] = ns_ * 32; meta[M_NVALID] = nvalid; meta[M_NIMG] = nimg; meta[M_STOP] = 0; meta[M_FIRST] = 1; meta[M_REL] = 1;
      meta[M_FIRST] = i == 0; meta[M_REL] = i == nfile - 1 || i == 2 * NB_I - 1;   // release the HMX lock at the end of the step (and before the host copies)
      if (i == 2 * NB_I) meta[M_FIRST] = 1;
      return call(*L, meta, w, 0, *blob, i < 2 * NB_I ? modE.data() : modS.data(), 18432) == 0; }, nfile)) return false;
    sync_inval(bM);
    { std::vector<f32> rows((size_t)N * HD); x32_to_rows((const f32*)(bM.p + LM.meta[M_X]), N, HD, rows.data());
      par_for(N, [&](int t) { f32* r = &rows[(size_t)t * HD]; double mu = 0, ss = 0; for (int i = 0; i < HD; i++) mu += r[i]; mu /= HD; for (int i = 0; i < HD; i++) { const double q = r[i] - mu; ss += q * q; }
        const float rs = 1.f / sqrtf((float)(ss / HD) + 1e-6f); for (int i = 0; i < HD; i++) r[i] = (float)(r[i] - mu) * rs * (1.f + nm[HD + i]) + nm[i]; });
      std::vector<f32> v((size_t)N * 128); gemm_f16(G.pout.data(), rows.data(), v.data(), N, 128, HD);
      for (size_t i = 0; i < z.size(); i++) z[i] += (f32)dt * v[i];
      double sv = 0, sx = 0; for (size_t i = 0; i < v.size(); i++) { sv += (double)v[i] * v[i]; sx += (double)z[i] * z[i]; }
      fprintf(stderr, "step %d t=%.4f |v|=%.3f |x|=%.3f  %.2fs\n", s, t, sqrt(sv / v.size()), sqrt(sx / z.size()), now() - ts);
      if (const char* pf = getenv("KLEIN_PREVIEW")) if (s + 1 < steps) {   // live preview: the clean-latent estimate z - tn*v after every step but the last, [128][lh][lw], header {lh, lw, 128, step}, atomic rename
        std::vector<f32> x0((size_t)N * 128); for (int n = 0; n < N; n++) for (int c = 0; c < 128; c++) x0[(size_t)c * N + n] = z[(size_t)n * 128 + c] - (f32)tn * v[(size_t)n * 128 + c];
        const std::string tmp = std::string(pf) + ".tmp"; FILE* pfp = fopen(tmp.c_str(), "wb"); if (pfp) { int hd[4] = {lh, lw, 128, s}; fwrite(hd, 4, 4, pfp); fwrite(x0.data(), 4, x0.size(), pfp); fclose(pfp); rename(tmp.c_str(), pf); } }
    }
  }
  FILE* f = fopen(outz.c_str(), "wb"); fwrite(z.data(), 4, z.size(), f); fclose(f);
  fprintf(stderr, "dit total %.2fs\n", now() - T0);
  unmap_buf(bI); unmap_buf(bT); unmap_buf(bM); rpcmem_free(bI.p); rpcmem_free(bT.p); rpcmem_free(bM.p); return true;
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
    if (cmd == "load") {   // load DIR : DIR/te/te00..te31.bin + embd.q8
      ss >> g_dir; double t0 = now(); bool ok = true; size_t mx = 0;
      for (int l = 0; l < NL_TE; l++) { char p[512]; snprintf(p, sizeof p, "%s/te/te%02d.bin", g_dir.c_str(), l); struct stat st; if (stat(p, &st)) ok = false; else mx = std::max<size_t>(mx, st.st_size); g_tefiles.push_back(p); }
      for (const char* nm : {"img00", "txt00", "sgl00"}) { struct stat st; if (!stat((g_dir + "/" + nm + ".bin").c_str(), &st)) mx = std::max<size_t>(mx, st.st_size); }   // DiT block packs share the two streaming buffers
      { int fd = open((g_dir + "/te/embd.q8").c_str(), O_RDONLY); if (fd < 0) ok = false; else { struct stat st; fstat(fd, &st); g_embd_len = st.st_size; g_embd = (uint8_t*)mmap(nullptr, g_embd_len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); } }
      if (!ok) { printf("error files\n"); continue; }
      g_te[0] = make_buf(mx); g_te[1] = make_buf(mx); g_all.push_back(&g_te[0]); g_all.push_back(&g_te[1]); if (!g_te[0].p || !g_te[1].p) { printf("error bufs\n"); continue; }
      printf("loaded %.1f\n", now() - t0); continue;
    }
    if (cmd == "glue") { std::string d; ss >> d; double t0 = now(); if (!load_glue(d)) { printf("error glue\n"); continue; } printf("glue %.1f\n", now() - t0); continue; }
    if (cmd == "dit") {   // dit FEAT.f32 R H W SEED STEPS OUT.z [Z0.f32] : text features [R][8*2560], latent grid H x W tokens (picture 16H x 16W), z0 = optional initial tokens [H*W][128]
      std::string ff, outz, z0; int R, H, W, steps; unsigned seed; ss >> ff >> R >> H >> W >> seed >> steps >> outz >> z0; std::vector<f32> feat;
      if (!readraw(ff.c_str(), feat, (size_t)R * NTAP * TD)) { printf("error feat\n"); continue; }
      if (!run_dit(feat, R, H, W, seed, steps, outz, z0.empty() ? nullptr : z0.c_str())) printf("error dit\n"); else printf("done\n"); continue; }
    if (cmd == "te") {   // te IDS.i32 OUT.f32 : token ids of the chat-templated prompt (no padding) -> features [T][8*2560] fp32
      std::string idsf, outf; ss >> idsf >> outf; std::vector<int> ids;
      { FILE* f = fopen(idsf.c_str(), "rb"); if (!f) { printf("error ids\n"); continue; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(n / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) {} fclose(f); }
      std::vector<f32> feat; if (!encode_text(ids, feat)) { printf("error te\n"); continue; }
      FILE* f = fopen(outf.c_str(), "wb"); fwrite(feat.data(), 4, feat.size(), f); fclose(f); printf("done %zu\n", ids.size()); continue;
    }
  }
  mega_close(g_h);
  return 0;
}
