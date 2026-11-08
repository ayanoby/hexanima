#pragma once
// LoRA parts outside the DiT blocks (megarun folds the blocks, hexagon/mega/lora_fold.h), included by animad.cpp after W / par():
//  - the LLM adapter (runs here on the CPU): its fp16 weights in small.bin are updated in RAM at load
//  - the Qwen3 text encoder (lora_te_*): the NPU text encoder is a precompiled QNN graph that cannot take new weights, so animad writes a merged
//    copy of the text encoder (ANIMA_TE -> ANIMA_TE_CACHE/<key>.safetensors, only the last LoRA set is kept) and sd.cpp encodes the prompts with it on the CPU
// MEGA_LORA = "file*strength|file*strength|..." (a bare file takes MEGA_LORA_S, default 1), the same list megarun reads.
#include <dirent.h>
#include <memory>
#include <sys/statvfs.h>

static std::vector<std::pair<std::string, float>> lora_list() {
    std::vector<std::pair<std::string, float>> out;
    const char* lv = getenv("MEGA_LORA"); if (!lv || !*lv) return out;
    const float s0 = getenv("MEGA_LORA_S") ? (float)atof(getenv("MEGA_LORA_S")) : 1.f;
    std::string all = lv;
    for (size_t p = 0; p < all.size();) {
        size_t q = all.find('|', p); if (q == std::string::npos) q = all.size();
        std::string item = all.substr(p, q - p); p = q + 1; if (item.empty()) continue;
        float s = s0; size_t st = item.rfind('*'); if (st != std::string::npos) { s = (float)atof(item.c_str() + st + 1); item.resize(st); }
        out.push_back({item, s});
    }
    return out;
}
static std::string us_(std::string s) { for (char& c : s) if (c == '.') c = '_'; return s; }
static bool has_te_keys(const LoraFile& L) {
    for (auto& kv : L.t) if (kv.first.compare(0, 7, "lora_te") == 0 || kv.first.compare(0, 13, "text_encoder.") == 0) return true;
    return false;
}
// W += delta for a [N][K] matrix stored as float (dst) from one LoRA; true when the LoRA has this matrix
static int add_delta(const LoraFile& L, const std::vector<std::string>& bases, int N, int K, float s, float* dst) {
    Delta dl; int fr = lora_delta(L, bases, K, N, s, dl);
    if (fr < 0) return -1;
    if (fr > 0) { par(N, [&](int n0, int n1) { for (int n = n0; n < n1; n++) for (int k = 0; k < K; k++) dst[(size_t)n * K + k] += dl(n, k); }); return 1; }
    for (const std::string& b : bases) {   // LyCORIS "full" diff
        std::vector<float> df;
        if (L.get(b + ".diff", df) && df.size() == (size_t)N * K) { for (size_t i = 0; i < df.size(); i++) dst[i] += s * df[i]; return 1; }
    }
    return 0;
}

// LLM adapter: llm_adapter.blocks.N.{self_attn,cross_attn}.{q,k,v,o}_proj, .mlp.0 / .mlp.2 (+ bias diffs), out_proj. Keys as diffusers
// ("diffusion_model.llm_adapter...lora_A/B") or kohya ("lora_unet_llm_adapter_blocks_N_...lora_down/up").
static void fold_adapter_loras() {
    for (auto& [path, s] : lora_list()) {
        LoraFile L; if (!L.open(path.c_str())) continue;
        int n = 0; double t0 = now();
        for (auto& kv : W) {
            const std::string& name = kv.first;
            if (name.compare(0, 12, "llm_adapter.") != 0) continue;
            const bool isw = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0, isb = name.size() > 5 && name.compare(name.size() - 5, 5, ".bias") == 0;
            if (!isw && !isb) continue;
            const std::string stem = name.substr(0, name.size() - (isw ? 7 : 5));
            const std::vector<std::string> bases = {"diffusion_model." + stem, "lora_unet_" + us_(stem), "lycoris_" + us_(stem), "transformer." + stem, stem};
            __fp16* p = (__fp16*)kv.second.p; const auto& d = kv.second.d;
            if (isw && d.size() == 2) {
                const int N = (int)d[0], K = (int)d[1];
                std::vector<float> w((size_t)N * K); for (size_t i = 0; i < w.size(); i++) w[i] = (float)p[i];
                if (add_delta(L, bases, N, K, s, w.data()) == 1) { for (size_t i = 0; i < w.size(); i++) p[i] = (__fp16)w[i]; n++; }
            } else if (isb) {
                size_t el = 1; for (auto v : d) el *= v;
                for (const std::string& b : bases) {
                    std::vector<float> df;
                    if (L.get(b + ".diff_b", df) && df.size() == el) { for (size_t i = 0; i < el; i++) p[i] = (__fp16)((float)p[i] + s * df[i]); n++; break; }
                }
            }
        }
        if (n) fprintf(stderr, "animad: LoRA %s: %d LLM-adapter tensors, strength %.2f, %.1fs\n", path.c_str(), n, s, now() - t0);
    }
}

// text encoder: returns the merged copy ("" = no LoRA with text-encoder weights, or it could not be written)
static std::string te_merge() {
    const char* src = getenv("ANIMA_TE"); const char* root = getenv("ANIMA_TE_CACHE");
    if (!src || !root) return "";
    std::vector<std::pair<std::unique_ptr<LoraFile>, float>> te;
    std::string key = std::string(src) + "|f16";
    { struct stat st; if (!stat(src, &st)) key += "|" + std::to_string((long long)st.st_size) + ":" + std::to_string((long long)st.st_mtime); }
    for (auto& [path, s] : lora_list()) {
        std::unique_ptr<LoraFile> L(new LoraFile);
        if (!L->open(path.c_str()) || !has_te_keys(*L)) continue;
        struct stat st; stat(path.c_str(), &st);
        char sb[32]; snprintf(sb, sizeof sb, "%.3f", s);
        key += "|" + path + "*" + sb + ":" + std::to_string((long long)st.st_size) + ":" + std::to_string((long long)st.st_mtime);
        te.push_back({std::move(L), s});
    }
    if (te.empty()) return "";
    uint64_t h = 1469598103934665603ull; for (unsigned char c : key) { h ^= c; h *= 1099511628211ull; }
    char hx[24]; snprintf(hx, sizeof hx, "%016llx", (unsigned long long)h);
    const std::string out = std::string(root) + "/" + hx + ".safetensors";
    if (!access(out.c_str(), F_OK)) { fprintf(stderr, "animad: text encoder with LoRA from the cache\n"); return out; }
    mkdir(root, 0700);
    if (DIR* d = opendir(root)) {   // only the last LoRA set is kept
        for (struct dirent* e; (e = readdir(d));) if (e->d_name[0] != '.') unlink((std::string(root) + "/" + e->d_name).c_str());
        closedir(d);
    }
    LoraFile S; if (!S.open(src)) { fprintf(stderr, "animad: cannot read the text encoder %s\n", src); return ""; }
    struct statvfs vs; if (statvfs(root, &vs) || (double)vs.f_bavail * vs.f_frsize < (double)S.size + 1e9) { fprintf(stderr, "animad: text-encoder LoRA skipped (storage)\n"); return ""; }
    double t0 = now();
    const std::string tmp = out + ".tmp";
    int fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600); if (fd < 0) return "";
    // the copy is written as F16, not BF16: sd.cpp runs the text encoder on the CPU with it, and ARM cores have fp16 arithmetic but no bf16 (BF16: ~10 s per
    // picture for prompt + negative). Same tensor sizes, so the header only changes "BF16" into "F16 " (a space keeps its length) and the offsets stay.
    std::string hdr((const char*)S.data, (size_t)(S.t.empty() ? 0 : std::min_element(S.t.begin(), S.t.end(), [](auto& a, auto& b) { return a.second.off < b.second.off; })->second.off));
    for (size_t p = 0; (p = hdr.find("\"BF16\"", p)) != std::string::npos; p += 6) hdr.replace(p, 6, "\"F16\" ");
    for (size_t o = 0; o < S.size;) {   // copy of the file (new header), then every BF16 tensor is written over it as F16
        size_t n = std::min((size_t)64 << 20, S.size - o); ssize_t w = o < hdr.size() ? write(fd, hdr.data() + o, std::min(n, hdr.size() - o)) : write(fd, S.data + o, n);
        if (w <= 0) { close(fd); unlink(tmp.c_str()); return ""; }
        o += w;
    }
    int nm = 0;
    for (auto& kv : S.t) {
        const std::string& name = kv.first; const LoraFile::T& x = kv.second;
        if (x.dtype != "BF16") continue;
        std::vector<float> w; if (!S.get(name, w)) continue;
        if (name.compare(0, 13, "model.layers.") == 0 && x.shape.size() == 2 && name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
            const std::string stem = name.substr(0, name.size() - 7), s2 = stem.substr(6);   // model.layers.0.self_attn.q_proj / layers.0.self_attn.q_proj
            const std::vector<std::string> bases = {"lora_te_" + us_(s2), "lora_te_" + us_(stem), "lora_te1_" + us_(s2), "lora_te1_" + us_(stem), "text_encoder." + stem, "text_encoder." + s2};
            bool any = false;
            for (auto& [L, s] : te) if (add_delta(*L, bases, (int)x.shape[0], (int)x.shape[1], s, w.data()) == 1) any = true;
            if (any) nm++;
        }
        std::vector<__fp16> b(w.size());
        for (size_t i = 0; i < w.size(); i++) b[i] = (__fp16)std::min(65504.f, std::max(-65504.f, w[i]));
        if (pwrite(fd, b.data(), b.size() * 2, x.off) != (ssize_t)(b.size() * 2)) { close(fd); unlink(tmp.c_str()); return ""; }
    }
    fsync(fd); close(fd);
    if (!nm) { unlink(tmp.c_str()); fprintf(stderr, "animad: text-encoder LoRA: no matching layers\n"); return ""; }
    rename(tmp.c_str(), out.c_str());
    fprintf(stderr, "animad: text encoder with LoRA: %d matrices, %.1fs\n", nm, now() - t0);
    return out;
}
