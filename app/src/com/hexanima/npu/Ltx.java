package com.hexanima.npu;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** LTX-2.5 22B distilled (Lightricks) text-to-video on the NPU: Gemma 4 12B text encoder + embeddings connector, 48-block video DiT, 8 distilled steps (no CFG), QNN conv VAE decoder; 17 frames (3 latent frames) at 24 fps, no sound yet.
 *  Files under files/ltx: te/gm00..47.bin conn/vc0..7.bin glue/ dit/ (glue + vb00..47.bin) vae/vtile.bin tok.bin. LTX-2 Community License: the model is in the download catalog (pack ayanoby/hexanima-ltx-packs; made by ltx/export_*.py). */
final class Ltx {
    static final String ID = "ltx-2.5-video";
    static boolean is(String id) { return id != null && id.startsWith("ltx-2.5"); }
    static File dir(File home) { return new File(home, "ltx"); }
    static boolean installed(File home) {
        File d = dir(home);
        return new File(d, ".installed").exists() && new File(d, "te/gm47.bin").exists() && new File(d, "conn/vc7.bin").exists() && new File(d, "dit/vb47.bin").exists() && new File(d, "dit/sst.f32").exists()
            && new File(d, "glue/embd.f16").exists() && new File(d, "glue/vproj.f16").exists() && new File(d, "glue/vreg.f32").exists() && new File(d, "vae/vtile.bin").exists() && new File(d, "tok.bin").exists();
    }
    /** the image encoder (QNN context of one 256x256 tile) for image to video; optional */
    static boolean encoderInstalled(File home) { return new File(dir(home), "vae/etile.bin").exists(); }
    /** photo -> raw fp32 [3][h][w] in [-1,1] for the encoder: scaled to cover w x h, centre crop */
    static void writeImage(android.graphics.Bitmap src, int w, int h, File out) throws IOException {
        android.graphics.Bitmap bm = android.graphics.Bitmap.createBitmap(w, h, android.graphics.Bitmap.Config.ARGB_8888); android.graphics.Canvas c = new android.graphics.Canvas(bm);
        float sc = Math.max((float) w / src.getWidth(), (float) h / src.getHeight()); float dw = src.getWidth() * sc, dh = src.getHeight() * sc;
        c.drawBitmap(src, null, new android.graphics.RectF((w - dw) / 2, (h - dh) / 2, (w + dw) / 2, (h + dh) / 2), new android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG | android.graphics.Paint.ANTI_ALIAS_FLAG));
        int[] px = new int[w * h]; bm.getPixels(px, 0, w, 0, 0, w, h); bm.recycle();
        ByteBuffer b = ByteBuffer.allocate(w * h * 12).order(ByteOrder.LITTLE_ENDIAN);
        for (int ch = 0; ch < 3; ch++) { int sh = 16 - 8 * ch; for (int i = 0; i < w * h; i++) b.putFloat(((px[i] >> sh & 255) - 127.5f) / 127.5f); }
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
    }
    static final int STEPS = 8, LATENT_FRAMES = 3, FPS = 24;
    /** latent frame counts offered (frames = 8*(lf-1)+1: 17 25 33 41 49 65 97) and playback fps of the MP4 */
    static final int[] LF_OPTS = {3, 4, 5, 6, 7, 9, 13}, FPS_OPTS = {8, 12, 16, 24, 25, 30};
    static int frames(int lf) { return 8 * (lf - 1) + 1; }
    /** sizes: sides multiples of 32 (one latent token = 32x32 px), the token count 3*(W/32)*(H/32) a multiple of 32 (whole slabs), at most 24000 tokens (the keys stream through VTCM in chunks, the activations of one step live in phone memory in windows and only the keys/values of all tokens are mapped for the DSP: 16 KB per token; the limit is the phone RAM and the time, which grows with the square of the token count) */
    static final int[][] PRESETS = {{256, 256}, {384, 256}, {256, 384}, {512, 384}, {384, 512}, {512, 512}, {768, 512}, {512, 768}, {768, 768}, {1024, 576}, {576, 1024}, {1024, 1024}, {1280, 768}, {768, 1280}};
    static final int MAX_TOKENS = 24000;
    static String sizeError(int w, int h, int lf) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 32 != 0 || h % 32 != 0) return L.s("Ширина и высота должны быть кратны 32", "Width and height must be multiples of 32");
        int t = lf * (w / 32) * (h / 32);
        if (t % 32 != 0) return L.f("Число токенов (%d) должно делиться на 32, например 512×384, 384×512, 512×512", "Token count (%d) must be divisible by 32, e.g. 512×384, 384×512, 512×512", t);
        if (t > MAX_TOKENS) return L.s("Слишком большое видео: уменьшите размер или число кадров", "Video too large: reduce the size or the frame count");
        return null;
    }
    /** measured: text encoder 16 s, DiT 8 steps 71 s (flash bound: 8.9 s per step up to ~1700 tokens), VAE 0.7 s per 8x8-latent tile (overlap 2 latents) */
    static int estimate(int w, int h, int lf, int steps) {
        int hl = h / 32, wl = w / 32, ty = hl <= 8 ? 1 : Math.max(2, (hl - 2 + 5) / 6), tx = wl <= 8 ? 1 : Math.max(2, (wl - 2 + 5) / 6);
        int t = lf * hl * wl;
        double step = t <= 1152 ? 8.9 + 0.0038 * Math.max(0, t - 128) : t <= 1728 ? 12.8 + 0.0076 * (t - 1152) : t <= 7488 ? 17.2 + 0.0115 * (t - 1728) : t <= 12480 ? 83 + 0.0188 * (t - 7488) : 177 * Math.pow(t / 12480.0, 2); // measured on the phone: 128/1152/1728/7488/12480 tokens = 8.9/12.8/17.2/83/177 s per step
        return (int) Math.round(18 + steps * step + 0.7 * ty * tx * (lf - 2) + 3);
    }
    /** the Gemma prompt (BOS + BPE ids, no padding, at most 400 tokens) */
    static int writeIds(GemmaTok t, String prompt, File out) throws Exception {
        int[] ids = t.encode(prompt == null || prompt.isEmpty() ? " " : prompt);
        if (ids.length > 400) ids = java.util.Arrays.copyOf(ids, 400);
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
        return ids.length;
    }
}
