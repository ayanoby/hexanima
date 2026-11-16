package com.hexanima.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.*;
import org.json.JSONObject;

/** Real-ESRGAN x4 (BSD-3-Clause, Xintao Wang et al.) on the NPU: a photo network (x4plus) and an anime network (x4plus anime 6B), each a QNN context of one 128 px input tile
 *  (downloaded to files/upscale on first use); the native helper libkup.so tiles the picture with blended overlaps. up.sh: raw uint8 RGB in, raw uint8 RGB out. */
final class Upscale {
    static File dir(File home) { return new File(home, "upscale"); }
    /** the result is at most 4096 x 4096 (the helper and the PNG encoder need a few hundred MB) */
    static final long MAX_OUT_PIXELS = 4096L * 4096L;

    /** the QNN contexts are not in the APK: they are downloaded once from Hugging Face (public repo ayanoby/hexanima-upscale, BSD-3 weights converted with the QNN tools) */
    static final String BASE = "https://huggingface.co/ayanoby/hexanima-upscale/resolve/main/";
    static final String[][] FILES = {{"kup_photo_t128.bin", "38706040", "55a720f62417a540e90b6de9e20b35b018534ba760142363df179cf0313a34e2"}, {"kup_anime_t128.bin", "10578456", "222f105f8dc1e40b0830b8efe2889226abc406f2e09e5a97e08ae27b0c2bfaeb"}};
    static String[] file(boolean anime) { return FILES[anime ? 1 : 0]; }
    static long size(boolean anime) { return Long.parseLong(file(anime)[1]); }
    static boolean has(File home, boolean anime) { File f = new File(dir(home), file(anime)[0]); return f.exists() && f.length() == size(anime); }
    /** blocking, resumable; call off the UI thread */
    static void download(Context c, boolean anime, Models.Count cb, java.util.concurrent.atomic.AtomicBoolean cancel) throws Exception {
        File d = dir(c.getFilesDir()); d.mkdirs(); String[] f = file(anime);
        Models.fetch(BASE + f[0], new File(d, f[0]), size(anime), f[2], cb, cancel);
    }

    static void delete(File home, boolean anime) { new File(dir(home), file(anime)[0]).delete(); }

    /** error text when the picture is too large for this scale, else null */
    static String sizeError(int w, int h, boolean half) {
        long out = (long) w * h * (half ? 4 : 16);
        if (out > MAX_OUT_PIXELS) return half ? L.s("Картинка слишком большая для увеличения", "The picture is too large to upscale") : L.s("Для ×4 картинка слишком большая: выберите ×2", "Too large for x4: choose x2");
        return null;
    }

    /** upscales `src` (a gallery picture) and saves the result as a new gallery picture; blocks (runs the NPU helper), call off the UI thread */
    static File run(Context c, Main.Sh sh, File src, boolean anime, boolean half) throws Exception {
        File home = c.getFilesDir(), um = dir(home), in = new File(um, "in.rgb"), out = new File(um, "out.rgb");
        Bitmap bm = BitmapFactory.decodeFile(src.getPath());
        if (bm == null) throw new IOException("cannot decode " + src.getName());
        final int w = bm.getWidth(), h = bm.getHeight(), s = half ? 2 : 4, ow = w * s, oh = h * s;
        String bad = sizeError(w, h, half); if (bad != null) throw new IOException(bad);
        try (BufferedOutputStream o = new BufferedOutputStream(new FileOutputStream(in), 1 << 20)) {
            int[] row = new int[w * 16]; byte[] b = new byte[w * 16 * 3];
            for (int y = 0; y < h; y += 16) {
                int n = Math.min(16, h - y); bm.getPixels(row, 0, w, 0, y, w, n);
                for (int i = 0; i < w * n; i++) { int p = row[i]; b[3 * i] = (byte) (p >> 16); b[3 * i + 1] = (byte) (p >> 8); b[3 * i + 2] = (byte) p; }
                o.write(b, 0, w * n * 3);
            }
        }
        bm.recycle();
        sh.run("rm -f " + out.getPath() + "; UM=" + um.getPath() + " sh " + new File(home, "anima/up.sh").getPath() + " " + in.getPath() + " " + w + " " + h + " " + out.getPath() + " " + (anime ? "anime" : "photo") + " " + (half ? 1 : 0) + " < /dev/null > /dev/null 2>&1");
        in.delete();
        if (!out.exists() || out.length() != (long) ow * oh * 3) throw new IOException("the upscaler produced no picture (files/upscale/up.log)");
        File dst = new File(Gallery.dir(c), System.currentTimeMillis() + ".png");
        Bitmap ob = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
        try (InputStream r = new BufferedInputStream(new FileInputStream(out), 1 << 20)) {
            int[] px = new int[ow * 16]; byte[] b = new byte[ow * 16 * 3];
            for (int y = 0; y < oh; y += 16) {
                int n = Math.min(16, oh - y), need = ow * n * 3, got = 0;
                while (got < need) { int k = r.read(b, got, need - got); if (k <= 0) throw new IOException("short read"); got += k; }
                for (int i = 0; i < ow * n; i++) px[i] = 0xFF000000 | ((b[3 * i] & 255) << 16) | ((b[3 * i + 1] & 255) << 8) | (b[3 * i + 2] & 255);
                ob.setPixels(px, 0, ow, 0, y, ow, n);
            }
        }
        out.delete();
        try (FileOutputStream o = new FileOutputStream(dst)) { ob.compress(Bitmap.CompressFormat.PNG, 100, o); }
        ob.recycle();
        JSONObject m = new JSONObject(Gallery.meta(src).toString());
        String tag = "Real-ESRGAN ×" + s + (anime ? " anime" : "");
        m.put("w", ow).put("h", oh).put("model", m.optString("model", "") + " + " + tag).put("upscaled", tag).put("from", src.getName());
        Gallery.writeMeta(dst, m);
        return dst;
    }
}
