#pragma once
// Anima LoRA folded into the fp16 crouton block weights right after megarun reads them (no copy of the model on disk, the cost is paid at load).
// MEGA_LORA=<file*strength|...>: kohya ("lora_unet_blocks_<b>_<module>.lora_down/lora_up/alpha"), diffusers ("diffusion_model.blocks.<b>.<module>.lora_A/lora_B"),
// LyCORIS LoKr / LoHa; the 10 matrices of a block and its adaLN modulation. W' = W + strength * scale * delta.
// The LLM-adapter and text-encoder parts are applied by animad (phone/lora_side.h). q8 blocks (2.9B models) are dequantised, updated, requantised.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <unistd.h>
#include <vector>
#include "blk_wlayout.h"

struct LoraFile {
  struct T { std::string dtype; std::vector<int64_t> shape; size_t off = 0, end = 0; };
  std::map<std::string, T> t; const uint8_t* data = nullptr; size_t size = 0;
  LoraFile() = default; LoraFile(const LoraFile&) = delete; LoraFile& operator=(const LoraFile&) = delete;
  ~LoraFile() { if (data) munmap((void*)data, size); }   // megarun lives on: do not keep the (touched) file pages around
  bool open(const char* path) {
    int fd = ::open(path, O_RDONLY); if (fd < 0) return false;
    struct stat st; fstat(fd, &st); size = st.st_size;
    void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); ::close(fd);
    if (m == MAP_FAILED) { size = 0; return false; }
    data = (const uint8_t*)m; if (size < 8) return false; uint64_t n; memcpy(&n, data, 8); if (8 + n > size) return false;
    std::string h((const char*)data + 8, n); const uint8_t* base = data + 8 + n;
    // minimal parser for the safetensors header: {"name":{"dtype":"BF16","shape":[a,b],"data_offsets":[x,y]},...}; values are objects, skipped
    // with brace matching that ignores braces inside strings (checkpoint metadata often holds JSON text)
    size_t p = h.find('{'); if (p == std::string::npos) return false; p++;
    auto str = [&](std::string& out) {
      p = h.find('"', p); if (p == std::string::npos) return false;
      out.clear(); size_t q = p + 1; for (; q < h.size() && h[q] != '"'; q++) { if (h[q] == '\\' && q + 1 < h.size()) q++; out += h[q]; }
      p = q + 1; return q < h.size();
    };
    auto object_end = [&](size_t ob) {   // index just past the object that starts at ob
      int depth = 0; bool in = false;
      for (size_t i = ob; i < h.size(); i++) {
        char ch = h[i];
        if (in) { if (ch == '\\') i++; else if (ch == '"') in = false; continue; }
        if (ch == '"') in = true; else if (ch == '{') depth++; else if (ch == '}' && --depth == 0) return i + 1;
      }
      return h.size();
    };
    while (true) {
      std::string k; if (!str(k)) break;
      size_t ob = h.find_first_of("{\"", h.find(':', p)); if (ob == std::string::npos) break;
      if (h[ob] == '"') { std::string skip; p = ob; if (!str(skip)) break; continue; }   // a string value (not expected at top level)
      size_t ce = object_end(ob); std::string body = h.substr(ob, ce - ob); p = ce;
      if (k == "__metadata__") continue;
      T x; size_t d = body.find("\"dtype\""); if (d != std::string::npos) { size_t a = body.find('"', d + 7); size_t b = body.find('"', a + 1); x.dtype = body.substr(a + 1, b - a - 1); }
      size_t s = body.find("\"shape\""); if (s != std::string::npos) { size_t a = body.find('[', s), b = body.find(']', a); std::string v = body.substr(a + 1, b - a - 1); for (char* c = &v[0]; *c;) { x.shape.push_back(strtoll(c, &c, 10)); while (*c == ',' || *c == ' ') c++; } }
      size_t o = body.find("\"data_offsets\""); if (o != std::string::npos) { size_t a = body.find('[', o); char* c = &body[a + 1]; x.off = strtoull(c, &c, 10); while (*c == ',' || *c == ' ') c++; x.end = strtoull(c, &c, 10); }
      x.off += base - data; x.end += base - data; t[k] = x;
    }
    return !t.empty();
  }
  // tensor as fp32 (F32 / F16 / BF16)
  bool get(const std::string& k, std::vector<float>& out, std::vector<int64_t>* shape = nullptr) const {
    auto it = t.find(k); if (it == t.end() || it->second.end > size) return false;
    const T& x = it->second; size_t n = 1; for (auto d : x.shape) n *= d; out.resize(n);
    const uint8_t* s = data + x.off;
    if (x.dtype == "F32") memcpy(out.data(), s, n * 4);
    else if (x.dtype == "F16") for (size_t i = 0; i < n; i++) out[i] = (float)((const __fp16*)s)[i];
    else if (x.dtype == "BF16") for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)((const uint16_t*)s)[i] << 16; memcpy(&out[i], &u, 4); }
    else return false;
    if (shape) *shape = x.shape; return true;
  }
  bool has(const std::string& k) const { return t.count(k) != 0; }
};

// The 2.9B Anima models are the 2B ones with 12 blocks doubled (measured on Anima-2.9B-preview-v1 vs anima-base-v1.0: per-block norm weights identical):
// 2.9B block j comes from 2B block FROM2B[j]. A 28-block LoRA is applied to a 40-block model through this table, a 40-block LoRA to a 28-block model
// through the first 2.9B block of each 2B block.
static const int FROM2B[40] = {0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 27};
static int lora_blocks(const LoraFile& L) {   // highest block index + 1 found in the LoRA's DiT keys
  int n = 0;
  for (auto& kv : L.t) {
    const std::string& k = kv.first;
    for (const char* pre : {"lora_unet_blocks_", "diffusion_model.blocks.", "lycoris_blocks_", "transformer.blocks.", "blocks."})
      if (k.compare(0, strlen(pre), pre) == 0) { n = std::max(n, atoi(k.c_str() + strlen(pre)) + 1); break; }
  }
  return n;
}
/** LoRA block to read for model block mb (-1: none) */
static int lora_source(int mb, int model_blocks, int lora_n) {
  if (lora_n <= 28 && model_blocks == 40) return mb < 40 ? FROM2B[mb] : -1;
  if (lora_n > 28 && model_blocks == 28) { for (int j = 0; j < 40; j++) if (FROM2B[j] == mb) return j; return -1; }
  return mb;
}
// adds block b of the LoRA to one block's weights (fp16 crouton tiles, or q8 tiles when q8); returns the number of matrices changed (-1: shapes do not fit)
// One update of a weight W [N out][K in], in any of the formats Anima LoRAs come in (102 Civitai Anima LoRAs checked: kohya 70, diffusers 31, LoKr 1):
//   kohya / sd-scripts  <base>.lora_down.weight [r][K], .lora_up.weight [N][r], .alpha
//   diffusers / PEFT (diffusion-pipe, ComfyUI)  <base>.lora_A.weight / .lora_B.weight (.alpha optional)
//   LyCORIS LoKr  <base>.lokr_w1 | lokr_w1_a·lokr_w1_b, lokr_w2 | lokr_w2_a·lokr_w2_b  -> kron(W1, W2) [* alpha / rank when decomposed]
//   LyCORIS LoHa  <base>.hada_w1_a·hada_w1_b ⊙ hada_w2_a·hada_w2_b * alpha / rank
// operator() gives the final delta (strength and scale included).
struct Delta {
  int N = 0, K = 0, r = 0; float sc = 1;
  std::vector<float> B, At;     // low rank: B [N][r], A transposed [K][r]
  std::vector<float> full;      // dense delta [N][K] (LoHa), already scaled
  std::vector<float> W1, W2; int n2 = 0, k1 = 0, k2 = 0;   // LoKr: kron(W1 [N/n2][k1], W2 [n2][k2]) * sc
  int kind = 0;                 // 0 low rank, 1 dense, 2 kron
  inline float operator()(int n, int k) const {
    if (kind == 1) return full[(size_t)n * K + k];
    if (kind == 2) return sc * W1[(size_t)(n / n2) * k1 + k / k2] * W2[(size_t)(n % n2) * k2 + k % k2];
    const float* bn = &B[(size_t)n * r]; const float* ak = &At[(size_t)k * r]; float d = 0; for (int i = 0; i < r; i++) d += bn[i] * ak[i]; return sc * d;
  }
};
static std::vector<float> mm(const std::vector<float>& a, int n, int r, const std::vector<float>& b, int k) {   // [n][r] @ [r][k]
  std::vector<float> o((size_t)n * k, 0.f);
  for (int i = 0; i < n; i++) for (int j = 0; j < r; j++) { float x = a[(size_t)i * r + j]; if (x == 0) continue; const float* bj = &b[(size_t)j * k]; float* oi = &o[(size_t)i * k]; for (int q = 0; q < k; q++) oi[q] += x * bj[q]; }
  return o;
}
// 1: found, 0: not in the file, -1: found but the shapes do not fit this matrix
static int lora_delta(const LoraFile& L, const std::vector<std::string>& bases, int K, int N, float strength, Delta& d) {
  for (const std::string& bs : bases) {
    std::vector<float> al; const float alpha = L.get(bs + ".alpha", al) && !al.empty() ? al[0] : -1.f;
    std::string dn, up;
    if (L.has(bs + ".lora_down.weight")) { dn = bs + ".lora_down.weight"; up = bs + ".lora_up.weight"; }
    else if (L.has(bs + ".lora_A.weight")) { dn = bs + ".lora_A.weight"; up = bs + ".lora_B.weight"; }
    if (!dn.empty()) {
      std::vector<float> A; std::vector<int64_t> sa, sb;
      if (!L.get(dn, A, &sa) || !L.get(up, d.B, &sb) || sa.size() != 2 || sb.size() != 2) continue;
      d.r = (int)sa[0]; d.N = N; d.K = K; d.kind = 0;
      if (sa[1] != K || sb[0] != N || sb[1] != d.r) return -1;
      d.sc = strength * (alpha > 0 ? alpha / d.r : 1.f);
      d.At.assign((size_t)K * d.r, 0.f);
      for (int i = 0; i < d.r; i++) for (int k = 0; k < K; k++) d.At[(size_t)k * d.r + i] = A[(size_t)i * K + k];
      return 1;
    }
    if (L.has(bs + ".lokr_w1") || L.has(bs + ".lokr_w1_a")) {
      std::vector<float> a, b; std::vector<int64_t> s1, s2; int rank = 0;
      auto part = [&](const char* w, std::vector<float>& out, std::vector<int64_t>& sh) {   // full matrix or a·b
        if (L.get(bs + ".lokr_" + w, out, &sh)) return sh.size() == 2;
        std::vector<int64_t> sa, sb;
        if (!L.get(bs + ".lokr_" + w + "_a", a, &sa) || !L.get(bs + ".lokr_" + w + "_b", b, &sb) || sa.size() != 2 || sb.size() != 2) return false;
        rank = (int)sa[1]; out = mm(a, (int)sa[0], (int)sa[1], b, (int)sb[1]); sh = {sa[0], sb[1]}; return true;
      };
      if (!part("w1", d.W1, s1) || !part("w2", d.W2, s2)) continue;
      if (s1[0] * s2[0] != N || s1[1] * s2[1] != K) return -1;
      d.N = N; d.K = K; d.kind = 2; d.n2 = (int)s2[0]; d.k1 = (int)s1[1]; d.k2 = (int)s2[1];
      d.sc = strength * (rank > 0 && alpha > 0 ? alpha / rank : 1.f);
      return 1;
    }
    if (L.has(bs + ".hada_w1_a")) {
      std::vector<float> a1, b1, a2, b2; std::vector<int64_t> s1, t1, s2, t2;
      if (!L.get(bs + ".hada_w1_a", a1, &s1) || !L.get(bs + ".hada_w1_b", b1, &t1) || !L.get(bs + ".hada_w2_a", a2, &s2) || !L.get(bs + ".hada_w2_b", b2, &t2)) continue;
      if (s1.size() != 2 || t1.size() != 2 || s1[0] != N || t1[1] != K || s2[0] != N || t2[1] != K) return -1;
      const int rk = (int)s1[1];
      std::vector<float> m1 = mm(a1, N, rk, b1, K), m2 = mm(a2, N, (int)s2[1], b2, K);
      const float sc = strength * (alpha > 0 ? alpha / rk : 1.f);
      d.full.resize((size_t)N * K); for (size_t i = 0; i < d.full.size(); i++) d.full[i] = sc * m1[i] * m2[i];
      d.N = N; d.K = K; d.kind = 1; return 1;
    }
  }
  return 0;
}
static std::vector<std::string> lora_bases(int b, const std::string& us, const std::string& dot) {
  const std::string n = std::to_string(b);
  return {"lora_unet_blocks_" + n + "_" + us, "diffusion_model.blocks." + n + "." + dot, "lycoris_blocks_" + n + "_" + us, "transformer.blocks." + n + "." + dot, "blocks." + n + "." + dot};
}
// adds block b of the LoRA to one block's weights (fp16 crouton tiles, or q8 tiles when q8) and to its adaLN modulation (fp16 [out][in] after the
// tiles); returns the number of matrices changed (-1: shapes do not fit)
static int lora_fold_block(const LoraFile& L, float strength, int b, uint8_t* w, bool q8 = false) {
  struct M { const char* name; uint32_t off; int K, N; };   // K = in features, N = out features (the file holds W^T [K][N] as tiles)
  static const M mats[10] = {{"self_attn_q_proj", W_SQ, 2048, 2048}, {"self_attn_k_proj", W_SK, 2048, 2048}, {"self_attn_v_proj", W_SV, 2048, 2048}, {"self_attn_output_proj", W_SO, 2048, 2048},
                             {"cross_attn_q_proj", W_CQ, 2048, 2048}, {"cross_attn_k_proj", W_CK, 1024, 2048}, {"cross_attn_v_proj", W_CV, 1024, 2048}, {"cross_attn_output_proj", W_CO, 2048, 2048},
                             {"mlp_layer1", W_M1, 2048, 8192}, {"mlp_layer2", W_M2, 8192, 2048}};
  int done = 0;
  for (const M& m : mats) {
    std::string mod = m.name, dot; size_t u = mod.find('_', mod.find('_') + 1);   // self_attn_q_proj -> self_attn.q_proj, mlp_layer1 -> mlp.layer1
    if (mod.compare(0, 4, "mlp_") == 0) dot = "mlp." + mod.substr(4); else dot = mod.substr(0, u) + "." + mod.substr(u + 1);
    Delta dl; int fr = lora_delta(L, lora_bases(b, mod, dot), m.K, m.N, strength, dl);
    if (fr < 0) return -1;
    if (fr == 0) continue;
    const Delta& delta = dl;
    const int KB = m.K / 32;
    std::function<void(int, int)> job;
    if (!q8) {
      __fp16* W = (__fp16*)(w + m.off);
      job = [&, W](int nb0, int nb1) {   // W by value: it goes out of scope before the threads run
        for (int n = nb0 * 32; n < nb1 * 32; n++) { const int nb = n >> 5, c = n & 31;
          for (int k = 0; k < m.K; k++) {
            __fp16& e = W[((((size_t)nb * KB + (k >> 5)) * 16 + ((k & 31) >> 1)) * 32 + c) * 2 + (k & 1)];   // wt16 tile index: [nb][kb][k%32 / 2][c][k & 1]
            e = (__fp16)((float)e + delta(n, k));
          } }
      };
    } else {
      // q8 tiles (export_all.py q8tiles): 1152 B per [nb][kb] tile = 64 fp16 column scales (halfword l -> column l/2) + 1024 offset-binary int8 in the
      // fp16 tile's order; dequantise, add the delta, quantise again the same way (per-column max/127 as fp16, round to nearest even, clamp +-127)
      uint8_t* Wq = w + WQ(m.off);
      job = [&, Wq](int nb0, int nb1) {
        float a[1024];
        for (int nb = nb0; nb < nb1; nb++) for (int kb = 0; kb < KB; kb++) {
          uint8_t* rec = Wq + ((size_t)nb * KB + kb) * 1152; __fp16* sv = (__fp16*)rec; uint8_t* q = rec + 128;
          for (int r16 = 0; r16 < 16; r16++) for (int c = 0; c < 32; c++) for (int p = 0; p < 2; p++) {
            int i = (r16 * 32 + c) * 2 + p;
            a[i] = (float)((int)q[i] - 128) * (float)sv[2 * c] + delta(nb * 32 + c, kb * 32 + r16 * 2 + p);
          }
          for (int c = 0; c < 32; c++) {
            float mx = 0; for (int r16 = 0; r16 < 16; r16++) for (int p = 0; p < 2; p++) mx = std::max(mx, std::fabs((float)(__fp16)a[(r16 * 32 + c) * 2 + p]));
            float v = mx / 127.0f; __fp16 s = (__fp16)(v == 0 ? 1.0f : v); float sf = (float)s;
            sv[2 * c] = sv[2 * c + 1] = s;
            for (int r16 = 0; r16 < 16; r16++) for (int p = 0; p < 2; p++) {
              int i = (r16 * 32 + c) * 2 + p; float qv = std::nearbyint((float)(__fp16)a[i] / sf); qv = std::min(127.f, std::max(-127.f, qv)); q[i] = (uint8_t)((int)qv + 128);
            }
          }
        }
      };
    }
    const int nt = 4, NB = m.N / 32; std::vector<std::thread> th;
    for (int i = 0; i < nt; i++) th.emplace_back(job, NB * i / nt, NB * (i + 1) / nt);
    for (auto& x : th) x.join();
    done++;
  }
  // adaLN modulation (many Anima LoRAs train it): per site self_attn / cross_attn / mlp, Linear 2048->256 (".1") then 256->6144 (".2"), fp16 [out][in]
  __fp16* mw = (__fp16*)(w + (q8 ? WQ_MODW : W_MODW));
  const char* sites[3] = {"self_attn", "cross_attn", "mlp"};
  for (int s = 0; s < 3; s++) for (int j = 1; j <= 2; j++) {
    const int K = j == 1 ? 2048 : 256, N = j == 1 ? 256 : 6144;
    std::string site = sites[s], us = "adaln_modulation_" + site + "_" + std::to_string(j), dot = "adaln_modulation_" + site + "." + std::to_string(j);
    Delta dl; int fr = lora_delta(L, lora_bases(b, us, dot), K, N, strength, dl);
    if (fr < 0) return -1;
    if (fr == 0) continue;
    __fp16* W = mw + (size_t)s * (256 * 2048 + 6144 * 256) + (j == 2 ? 256 * 2048 : 0);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) W[(size_t)n * K + k] = (__fp16)((float)W[(size_t)n * K + k] + dl(n, k));
    done++;
  }
  return done;
}
