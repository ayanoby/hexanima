package com.hexanima.npu;

import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Qwen-Image-2.1 (Turbo, 4-step Lightning LoRA folded in the DiT pack) on the NPU: files under files/qwen, one picture per qwen.sh call; with a source photo it edits it. */
final class Qwen {
    static final String PREFIX_ID = "qwen";
    static boolean is(String id) { return id != null && id.startsWith(PREFIX_ID); }
    static File dir(File home) { return new File(home, "qwen"); }
    static boolean installed(File home) { return new File(dir(home), ".installed").exists(); }
    static final int MAX_IMAGES = 3;   // condition images ("Picture 1…3"); the first is the photo, the others are extra references
    static File editSrc(File home, int i) { return new File(dir(home), i == 0 ? "edit_src.png" : "edit_src" + i + ".png"); }
    static File editSrc(File home) { return editSrc(home, 0); }
    static File editDir(File home, int i) { return i == 0 ? new File(dir(home), "edit") : new File(new File(dir(home), "edit"), "" + i); }
    static File editDir(File home) { return editDir(home, 0); }

    /** size presets (W×H): the patch count (W/16·H/16) must be a multiple of 32 */
    static final int[][] PRESETS = {{512, 512}, {640, 640}, {768, 768}, {1024, 1024}, {512, 768}, {768, 512}, {640, 960}, {960, 640}, {576, 1024}, {1024, 576}, {768, 1024}, {1024, 768}, {896, 1152}, {1152, 896}};
    static final int MAX_KEYS = 6800;   // the attention of one DiT step holds target + condition + text tokens in the 8 MB VTCM
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        int t = (w / 16) * (h / 16);
        if (t % 32 != 0) return L.f("Число патчей (%d) должно делиться на 32, например 640×640, 1024×1024, 704×832", "Patch count (%d) must be divisible by 32, e.g. 640×640, 1024×1024, 704×832", t);
        if (t > 4608) return L.s("Слишком большой размер: максимум 4608 патчей (например 1152×1024)", "Too large: at most 4608 patches (e.g. 1152×1024)");
        return null;
    }

    private static void writeIds(int[] ids, File out) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
    }

    /** text-to-image tokens -> ids file; returns the number of leading system tokens that qwenrun drops */
    static int writeIds(File home, String prompt, File out) throws Exception {
        QwenTok t = new QwenTok(dir(home).getPath());
        int[] ids = t.encodeQwenImage(prompt), drop = {t.dropQwenImage()};
        if (ids.length > 480) { ids = java.util.Arrays.copyOf(ids, 480); }   // qwenrun takes at most 2048 tokens; 480 keeps the encoder quick
        writeIds(ids, out); return drop[0];
    }

    /** decode the picked photo (EXIF rotation applied) at most 2048 px on the long side, keep it as files/qwen/edit_src.png */
    static Bitmap importPhoto(android.content.ContentResolver cr, Uri uri, File home, int idx) throws Exception {
        Bitmap bm = ImageDecoder.decodeBitmap(ImageDecoder.createSource(cr, uri), (d, info, src) -> {
            int w = info.getSize().getWidth(), h = info.getSize().getHeight(), m = Math.max(w, h);
            if (m > 2048) d.setTargetSize(Math.round(w * 2048f / m), Math.round(h * 2048f / m));
            d.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
        });
        dir(home).mkdirs();
        try (FileOutputStream o = new FileOutputStream(editSrc(home, idx))) { bm.compress(Bitmap.CompressFormat.PNG, 100, o); }
        return bm;
    }

    static final class Edit { int drop, cw, ch, nc, nm, tokens; }

    /** condition size: the photo's aspect at about min(target area, 0.4 Mpx), sides multiples of 32 (what the ViT and the VAE both need); with k images each gets a smaller share so that target + conditions + text stay within MAX_KEYS */
    static int[] condSize(int srcW, int srcH, int tw, int th, int k) {
        double area = Math.min((double) tw * th, 393216.0), ratio = (double) srcW / srcH;
        if (k > 1) area = Math.min(area, Math.min(393216.0 * (k == 2 ? 0.75 : 0.5), Math.max(65536.0, (MAX_KEYS - 400 - tw / 16.0 * th / 16) * 256 / k)));
        int cw = (int) Math.max(64, Math.round(Math.sqrt(area * ratio) / 32) * 32), ch = (int) Math.max(64, Math.round(area / cw / 32) * 32);
        return new int[]{cw, ch};
    }

    /** writes EDIT_DIR[/i]/{meta.txt, rgba.f32, px.f32} for every condition image and the ids file for editing `srcs` with `prompt` into a tw×th picture; Edit sums the images (nc latent tokens, nm image slots) */
    static Edit prepareEdit(File home, java.util.List<Bitmap> srcs, String prompt, int tw, int th, File idsOut) throws Exception {
        final int k = srcs.size(); int[] nms = new int[k]; Edit e = new Edit();
        for (int j = 0; j < MAX_IMAGES; j++) if (j >= k) Models.deleteTree(editDir(home, j));   // images dropped since the last run
        for (int j = 0; j < k; j++) {
            Bitmap src = srcs.get(j);
            int[] cs = condSize(src.getWidth(), src.getHeight(), tw, th, k); final int cw = cs[0], ch = cs[1];
            Bitmap b = src.copy(Bitmap.Config.ARGB_8888, false);
            while (b.getWidth() > 2 * cw || b.getHeight() > 2 * ch) b = Bitmap.createScaledBitmap(b, Math.max(cw, b.getWidth() / 2), Math.max(ch, b.getHeight() / 2), true);   // halve first: one big bilinear step would alias
            if (b.getWidth() != cw || b.getHeight() != ch) b = Bitmap.createScaledBitmap(b, cw, ch, true);
            int[] px = new int[cw * ch]; b.getPixels(px, 0, cw, 0, 0, cw, ch);
            File ed = editDir(home, j); ed.mkdirs();
            final String key = src.getWidth() + "x" + src.getHeight() + ":" + cw + "x" + ch + ":" + editSrc(home, j).length() + ":" + editSrc(home, j).lastModified();
            boolean same = false; try { same = new File(ed, "key.txt").exists() && key.equals(Models.read(new File(ed, "key.txt"))) && new File(ed, "rgba.f32").exists() && new File(ed, "px.f32").exists(); } catch (Exception ignored) { }
            int n = cw * ch, gh = ch / 16, gw = cw / 16;
            if (!same) {   // the photo files (and the encoder's latent) only change with the photo or the condition size
                new File(ed, "cond_raw.f32").delete();
                float[] rgba = new float[4 * n]; float[] rgb = new float[3 * n];
                for (int i = 0; i < n; i++) {
                    int p = px[i], a = p >>> 24, r = (p >> 16) & 255, g = (p >> 8) & 255, bl = p & 255;
                    rgba[i] = r / 127.5f - 1f; rgba[n + i] = g / 127.5f - 1f; rgba[2 * n + i] = bl / 127.5f - 1f; rgba[3 * n + i] = a / 127.5f - 1f;
                    float wa = a / 255f;   // the ViT was trained on the picture flattened over white
                    rgb[i] = (r * wa + 255 * (1 - wa)) / 127.5f - 1f; rgb[n + i] = (g * wa + 255 * (1 - wa)) / 127.5f - 1f; rgb[2 * n + i] = (bl * wa + 255 * (1 - wa)) / 127.5f - 1f;
                }
                writeFloats(new File(ed, "rgba.f32"), rgba);
                // ViT input: 16x16 patches in 2x2 merge order, each (c, t=2 copies, 16, 16)
                float[] pv = new float[gh * gw * 1536]; int bw = gw / 2;
                for (int p = 0; p < gh * gw; p++) {
                    int blk = p / 4, ir = (p / 2) % 2, ic = p % 2, row = (blk / bw) * 2 + ir, col = (blk % bw) * 2 + ic, o = p * 1536;
                    for (int c = 0; c < 3; c++) for (int t = 0; t < 2; t++) for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++)
                        pv[o + ((c * 2 + t) * 16 + y) * 16 + x] = rgb[c * n + (row * 16 + y) * cw + col * 16 + x];
                }
                writeFloats(new File(ed, "px.f32"), pv);
                try (FileWriter kw = new FileWriter(new File(ed, "key.txt"))) { kw.write(key); }
            }
            try (FileWriter w = new FileWriter(new File(ed, "meta.txt"))) { w.write(cw + " " + ch + "\n"); }
            nms[j] = gh * gw / 4; e.nc += gh * gw; e.nm += nms[j]; if (j == 0) { e.cw = cw; e.ch = ch; }
        }
        QwenTok t = new QwenTok(dir(home).getPath());
        int[] ids = t.encodeQwenEdit(prompt, nms); e.drop = t.dropQwenImage(); e.tokens = ids.length;
        writeIds(ids, idsOut);
        return e;
    }
    private static void writeFloats(File f, float[] a) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(a.length * 4).order(ByteOrder.LITTLE_ENDIAN); bb.asFloatBuffer().put(a);
        try (FileOutputStream o = new FileOutputStream(f)) { o.write(bb.array()); }
    }

    /** number of decoder tiles along an axis of n latents for tile size T (the same overlap rule as the vaetile runner) */
    static int tiles(int n, int T) {
        if (n <= T) return 1;
        int om = Math.max(1, T / 8); return Math.max(2, (n - om + (T - om) - 1) / (T - om));
    }
    /** decoder context for the picture: the 512 px tile or, when installed, the 1024 px one, whichever decodes the size with fewer pixels computed (a 1024x1024 picture is 9 small tiles but one big one) */
    static String vaeCtx(File home, int w, int h) {
        long c32 = (long) tiles(w / 16, 32) * tiles(h / 16, 32) * 512 * 512, c64 = (long) tiles(w / 16, 64) * tiles(h / 16, 64) * 1024 * 1024;
        return new File(dir(home), "vae/vae_t64.bin").exists() && c64 < c32 ? "vae_t64.bin" : "vae_t32.bin";
    }

    /** raw fp32 RGB [3][h][w] in [-1,1] -> PNG bytes */
    static byte[] png(File raw, int w, int h) throws IOException { return Moss.png(raw, w, h); }
}
