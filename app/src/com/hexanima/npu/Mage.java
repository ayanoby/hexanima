package com.hexanima.npu;

import android.graphics.Bitmap;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Mage-Flow-Edit-Turbo (Microsoft, MIT; 4 steps) on the NPU: files under files/mage, one picture per mage.sh call; with a source photo it edits it. The photo is picked and stored by the Qwen code (files/qwen/edit_src.png). */
final class Mage {
    static boolean is(String id) { return id != null && id.startsWith("mage"); }
    static File dir(File home) { return new File(home, "mage"); }
    static boolean installed(File home) { return new File(dir(home), ".installed").exists(); }
    static File editDir(File home) { return new File(dir(home), "edit"); }

    static final int[][] PRESETS = Qwen.PRESETS;
    /** the presets for the picker: while a photo is edited only the sizes that fit the attention with the reference (the shared list stays untouched) */
    static java.util.List<int[]> presets(boolean edit) {
        java.util.List<int[]> l = new java.util.ArrayList<>();
        for (int[] z : PRESETS) if (!edit || (z[0] / 16) * (z[1] / 16) <= MAX_EDIT_PATCHES) l.add(z);
        return l;
    }
    static final int MAX_EDIT_PATCHES = 3200;   // target + reference tokens (2x) must fit the 8 MB VTCM attention together with the text
    static final int MAX_PATCHES = 6144, MAX_KEYS = 6500;   // MAX_KEYS: the attention of one DiT step holds all target + reference + text tokens in the 8 MB VTCM (measured: 6244 and 6144+text work)
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        int t = (w / 16) * (h / 16);
        if (t % 32 != 0) return L.f("Число патчей (%d) должно делиться на 32, например 640×640, 1024×1024, 704×832", "Patch count (%d) must be divisible by 32, e.g. 640×640, 1024×1024, 704×832", t);
        if (t > MAX_PATCHES) return L.s("Слишком большой размер: максимум 6144 патчей (1536×1024)", "Too large: at most 6144 patches (1536×1024)");
        return null;
    }

    private static void writeInts(int[] ids, File out) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
    }
    private static void writeFloats(File f, float[] a) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(a.length * 4).order(ByteOrder.LITTLE_ENDIAN); bb.asFloatBuffer().put(a);
        try (FileOutputStream o = new FileOutputStream(f)) { o.write(bb.array()); }
    }

    /** text-to-image: ids file, returns the number of system tokens to drop */
    static int writeIds(File home, String prompt, File out) throws Exception {
        QwenTok t = new QwenTok(dir(home).getPath());
        writeInts(t.encodeMage(prompt), out); return t.dropMage(false);
    }

    private static Bitmap fit(Bitmap src, int w, int h) {
        Bitmap b = src.copy(Bitmap.Config.ARGB_8888, false);
        while (b.getWidth() > 2 * w || b.getHeight() > 2 * h) b = Bitmap.createScaledBitmap(b, Math.max(w, b.getWidth() / 2), Math.max(h, b.getHeight() / 2), true);   // halve first: one big step would alias
        return b.getWidth() != w || b.getHeight() != h ? Bitmap.createScaledBitmap(b, w, h, true) : b;
    }

    static final class Edit { int drop, tokens, nm; }

    /** writes EDIT_DIR/{meta.txt "gh gw h w", px.f32 (ViT patches of the vision-language copy, long edge <= 384, sides multiples of 32), ref.rgb (the photo at the target size, fp32 HWC in [-1,1])} and the ids file */
    static Edit prepareEdit(File home, Bitmap src, String prompt, int tw, int th, File idsOut) throws Exception {
        int w0 = src.getWidth(), h0 = src.getHeight(), m = Math.max(w0, h0);
        double s = m > 384 ? 384.0 / m : 1;
        int vw = Math.max(1, (int) Math.round(w0 * s)), vh = Math.max(1, (int) Math.round(h0 * s));
        final int Wv = Math.max(32, Math.round(vw / 32f) * 32), Hv = Math.max(32, Math.round(vh / 32f) * 32), gh = Hv / 16, gw = Wv / 16;
        int[] px = new int[Wv * Hv]; fit(src, Wv, Hv).getPixels(px, 0, Wv, 0, 0, Wv, Hv);
        int n = Wv * Hv; float[] pv = new float[gh * gw * 1536]; int bw = gw / 2;
        for (int p = 0; p < gh * gw; p++) {   // 16x16 patches in 2x2 merge order, each (c, t = 2 copies, 16, 16)
            int blk = p / 4, ir = (p / 2) % 2, ic = p % 2, row = (blk / bw) * 2 + ir, col = (blk % bw) * 2 + ic, o = p * 1536;
            for (int c = 0; c < 3; c++) for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) {
                int q = px[(row * 16 + y) * Wv + col * 16 + x]; float v = (((q >> (16 - 8 * c)) & 255) / 255f - .5f) / .5f;
                pv[o + (c * 2) * 256 + y * 16 + x] = v; pv[o + (c * 2 + 1) * 256 + y * 16 + x] = v;
            }
        }
        File ed = editDir(home); ed.mkdirs();
        writeFloats(new File(ed, "px.f32"), pv);
        int[] rp = new int[tw * th]; fit(src, tw, th).getPixels(rp, 0, tw, 0, 0, tw, th);
        float[] ref = new float[tw * th * 3];
        for (int i = 0; i < rp.length; i++) { int q = rp[i]; ref[3 * i] = ((q >> 16) & 255) / 127.5f - 1; ref[3 * i + 1] = ((q >> 8) & 255) / 127.5f - 1; ref[3 * i + 2] = (q & 255) / 127.5f - 1; }
        writeFloats(new File(ed, "ref.rgb"), ref);
        try (FileWriter w = new FileWriter(new File(ed, "meta.txt"))) { w.write(gh + " " + gw + " " + th / 16 + " " + tw / 16 + "\n"); }
        Edit e = new Edit(); e.nm = gh * gw / 4;
        QwenTok t = new QwenTok(dir(home).getPath());
        int[] ids = t.encodeMageEdit(prompt, e.nm); e.drop = t.dropMage(true); e.tokens = ids.length;
        writeInts(ids, idsOut);
        return e;
    }

    /** fp32 RGB [h][w][3] in [-1,1] -> PNG bytes */
    static byte[] png(File raw, int w, int h) throws IOException {
        byte[] bytes = Models.readBytes(raw);
        if (bytes.length != 3 * w * h * 4) throw new IOException("decoder output has " + bytes.length + " bytes");
        java.nio.FloatBuffer f = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) px[i] = 0xFF000000 | (c8(f.get(3 * i)) << 16) | (c8(f.get(3 * i + 1)) << 8) | c8(f.get(3 * i + 2));
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888).compress(Bitmap.CompressFormat.PNG, 100, bo);
        return bo.toByteArray();
    }
    private static int c8(float v) { int x = Math.round((v + 1f) * 127.5f); return x < 0 ? 0 : x > 255 ? 255 : x; }
}
