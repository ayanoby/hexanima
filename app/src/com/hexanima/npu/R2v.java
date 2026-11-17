package com.hexanima.npu;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** MiniMax-H3 Ref2VA (33B DiT, reference-to-video) on the NPU: text (+ one reference photo) to video, 4 distilled steps with the turbo LoRA (no CFG), no sound yet. Files under files/r2v (blk/adaln/lora packs, glue, vae/ = video VAE decoder, enc/henc.bin = image encoder);
 *  the Qwen3-VL text encoder is the one of MiniMax-H3-Image (files/h3/te). In the download catalog since 1.8.2 (catalog10.json, pack in ayanoby/hexanima-r2v-packs; MiniMax-H3-Image has to be installed first for its text encoder); the files can still be put into files/r2v by hand (anima/h3, phone/install_r2v.sh). MiniMax H3 Community License applies. */
final class R2v {
    static final String ID = "r2v-video";
    static boolean is(String id) { return id != null && id.startsWith("r2v-"); }
    static File dir(File home) { return new File(home, "r2v"); }
    static boolean installed(File home) {
        File d = dir(home);
        return new File(d, ".installed").exists() && new File(d, "blk49.bin").exists() && new File(d, "adaln49.bin").exists() && new File(d, "r1.bin").exists() && new File(d, "vpp_w.f32").exists() && new File(d, "lora/lora49.bin").exists() && new File(d, "lora/lorar1.bin").exists()
            && new File(d, "vae/vb35.bin").exists() && new File(d, "vae/glue/po_w.f16").exists() && new File(H3.dir(home), "te/lay49.bin").exists();
    }
    /** the text encoder of MiniMax-H3-Image (files/h3/te): all the video model needs from it */
    static boolean teInstalled(File home) { return new File(H3.dir(home), "te/lay49.bin").exists(); }
    static boolean encoderInstalled(File home) { return new File(dir(home), "enc/henc.bin").exists(); }
    static final int STEPS = 4, MAX_STEPS = 12, FPS = 24;   // 4 = the turbo LoRA's schedule; other counts use the same shifted sigma schedule (fewer: rougher, more: not trained for)
    /** latent frames T = 5n + 2 (n >= 1): frames = 17n + 5 (22, 39, 56, 73, 90) */
    static final int[] T_OPTS = {7, 12, 17, 22, 27}, FPS_OPTS = {16, 24, 25, 30};
    static int frames(int t) { return 17 * ((t - 2) / 5) + 5; }
    /** sizes: sides multiples of 32 (one DiT token = 32x32 px), at least 256 (the VAE tile), and the tokens of ONE frame (W/32 * H/32) a multiple of 32 (whole slabs; also needed for the reference photo rows) */
    static final int[][] PRESETS = {{512, 512}, {768, 512}, {512, 768}, {768, 768}, {1024, 576}, {576, 1024}, {1024, 768}, {768, 1024}, {1024, 1024}, {1280, 768}, {768, 1280}, {512, 384}, {384, 512}, {256, 256}};
    /** the DiT runs in windows (activations in phone memory, keys/values of all tokens mapped for the DSP: 28 KB per token): the limit is memory and time (attention grows with the square of the tokens) */
    static final int MAX_ROWS = 16000;
    static int rows(int w, int h, int t) { return t * (w / 32) * (h / 32); }
    static String sizeError(int w, int h, int t) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 32 != 0 || h % 32 != 0) return L.s("Ширина и высота должны быть кратны 32", "Width and height must be multiples of 32");
        int f = (w / 32) * (h / 32);
        if (f % 32 != 0) return L.f("Токенов в кадре (%d) должно быть кратно 32, например 512×512, 1024×576, 768×768", "Tokens per frame (%d) must be divisible by 32, e.g. 512×512, 1024×576, 768×768", f);
        if (rows(w, h, t) > MAX_ROWS) return L.s("Слишком большое видео: уменьшите размер или число кадров", "Video too large: reduce the size or the frame count");
        return null;
    }
    /** measured on the phone (seconds per DiT step by token count): 448 / 1792 / 3072 / 4032 / 4352 / 7168 / 9792 rows = 28 / 48 / 68 / 84 / 89 / 163 / 260 s */
    static final double[][] STEP = {{0, 22}, {448, 28}, {1792, 48}, {3072, 68}, {4032, 84}, {4352, 89}, {7168, 163}, {9792, 260}, {16000, 560}};
    static double step(int rows) { for (int i = 1; i < STEP.length; i++) if (rows <= STEP[i][0]) return STEP[i - 1][1] + (STEP[i][1] - STEP[i - 1][1]) * (rows - STEP[i - 1][0]) / (STEP[i][0] - STEP[i - 1][0]); return STEP[STEP.length - 1][1]; }
    static int tiles(int len) { if (len <= 256) return 1; int n = (len + 255) / 256; while (256 * n - 64 * (n - 1) < len) n++; return n; }
    /** text encoder 14 s + 4 steps + VAE (5.8 s per 256-px tile pass, n chunks of 7 latent frames) + the reference photo (about 2 s) */
    static int estimate(int w, int h, int t, boolean photo, int steps) {
        int n = (t - 2) / 5; double vae = 5.8 * tiles(w) * tiles(h) * n;
        return (int) Math.round(18 + steps * step(rows(w, h, t)) + vae + (photo ? 3 : 0));
    }
    /** the prompt verbatim (no chat template), at most 480 tokens */
    static int writeIds(QwenTok t, String prompt, File out) throws Exception {
        int[] ids = t.encode(prompt == null || prompt.isEmpty() ? " " : prompt);
        if (ids.length > 480) ids = java.util.Arrays.copyOf(ids, 480);
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
        return ids.length;
    }
    /** photo -> raw fp32 [3][h][w], ImageNet-normalised (the image encoder's input): scaled to cover w x h, centre crop */
    static void writeImage(android.graphics.Bitmap src, int w, int h, File out) throws IOException {
        android.graphics.Bitmap bm = android.graphics.Bitmap.createBitmap(w, h, android.graphics.Bitmap.Config.ARGB_8888); android.graphics.Canvas c = new android.graphics.Canvas(bm);
        float sc = Math.max((float) w / src.getWidth(), (float) h / src.getHeight()); float dw = src.getWidth() * sc, dh = src.getHeight() * sc;
        c.drawBitmap(src, null, new android.graphics.RectF((w - dw) / 2, (h - dh) / 2, (w + dw) / 2, (h + dh) / 2), new android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG | android.graphics.Paint.ANTI_ALIAS_FLAG));
        int[] px = new int[w * h]; bm.getPixels(px, 0, w, 0, 0, w, h); bm.recycle();
        final float[] mean = {0.485f, 0.456f, 0.406f}, std = {0.229f, 0.224f, 0.225f};
        ByteBuffer b = ByteBuffer.allocate(w * h * 12).order(ByteOrder.LITTLE_ENDIAN);
        for (int ch = 0; ch < 3; ch++) { int sh = 16 - 8 * ch; for (int i = 0; i < w * h; i++) b.putFloat((((px[i] >> sh) & 255) / 255f - mean[ch]) / std[ch]); }
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
    }
}
