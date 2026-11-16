package com.hexanima.npu;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** FLUX.3 Image Klein (Black Forest Labs, distilled flow map, 5.6B DiT + Qwen3-VL-4B text encoder, 16 steps, no CFG) on the NPU: files under files/klein (te/ img*.bin txt*.bin sgl*.bin glue/ vae/), one picture per klein.sh call.
 *  FLUX Kommunity licence (non-commercial): the model is in the download catalog (pack ayanoby/hexanima-klein-packs; made by flux3/export_flux3.py, the Mage TE conversion and flux3/vae/build_vae.sh). Text to image only. */
final class Klein {
    static final String ID = "flux3-klein";
    static boolean is(String id) { return id != null && id.startsWith("flux3-klein"); }
    static File dir(File home) { return new File(home, "klein"); }
    static boolean installed(File home) {
        File d = dir(home);
        return new File(d, ".installed").exists() && new File(d, "sgl27.bin").exists() && new File(d, "img04.bin").exists() && new File(d, "txt04.bin").exists() && new File(d, "te/te31.bin").exists()
            && new File(d, "te/embd.q8").exists() && new File(d, "glue/proj_out.f32").exists() && new File(d, "glue/vae_bn.f32").exists() && new File(d, "vae/" + vaeFile()).exists();
    }
        static String vaeFile() { return "kvae_t32.bin"; }
    static final int STEPS = 16, MAX_STEPS = 24;

    /** sizes: sides multiples of 16 (one token = 16x16 px), at least 256 (the VAE tile), the token count a multiple of 32 (whole 32-row slabs) and at most 6144 (the attention of one step holds all image + text tokens in the 8 MB VTCM) */
    static final int[][] PRESETS = {{512, 512}, {768, 768}, {1024, 1024}, {512, 768}, {768, 512}, {768, 1024}, {1024, 768}, {1024, 512}, {512, 1024}, {1536, 1024}, {1024, 1536}};
    /** MAX_KEYS: image patches + text tokens of one step must fit the attention in the 8 MB VTCM (measured: 6144 patches + 268 text tokens = 6412 keys work, the attention of Mage is known to fit 6244 or more) */
    static final int MAX_TOKENS = 6144, MAX_KEYS = 6500;
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        int t = (w / 16) * (h / 16);
        if (t % 32 != 0) return L.f("Число патчей (%d) должно делиться на 32, например 512×512, 768×1024, 1024×1024", "Patch count (%d) must be divisible by 32, e.g. 512×512, 768×1024, 1024×1024", t);
        if (t > MAX_TOKENS) return L.s("Слишком большой размер: максимум 6144 патчей (1536×1024)", "Too large: at most 6144 patches (1536×1024)");
        return null;
    }
    /** seconds per step, measured: 256 patches 1.05, 1024 patches 3.0, 4096 patches 13.5 (linear in between, the slope of the last interval beyond) */
    static double stepSec(double tok) {
        double[][] p = {{256, 1.05}, {1024, 3.0}, {4096, 13.5}};
        for (int i = 1; i < p.length; i++) if (tok <= p[i][0] || i == p.length - 1) return p[i - 1][1] + (p[i][1] - p[i - 1][1]) * (tok - p[i - 1][0]) / (p[i][0] - p[i - 1][0]);
        return 1;
    }
    /** rough seconds per picture: text encoder ~3 s + steps + VAE (1.7 s per 1024 patches) + start */
    static int estimate(int w, int h, int steps) { double tok = (w / 16.0) * (h / 16.0); return (int) Math.round(5 + steps * stepSec(tok) + 1.7 * tok / 1024); }

    /** the chat-templated prompt (user turn only, no system message) goes in without padding */
    static int writeIds(QwenTok t, String prompt, File out) throws Exception {
        int[] ids = t.encodeKlein(prompt == null || prompt.isEmpty() ? " " : prompt);
        if (ids.length > 480) ids = java.util.Arrays.copyOf(ids, 480);
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
        return ids.length;
    }
}
