package com.hexanima.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.*;
import org.json.JSONObject;

/** SeedVR2-3B (ByteDance Seed, Apache-2.0): one-step diffusion super-resolution on the NPU. The picture is resized bicubically to the target size, encoded by the VAE (QNN tiles of 512 px), restored by the 32-block DiT
 *  (int8 blocks on the HMX, window attention, see seedvr2/NOTES.md) in ONE step and decoded again. Files in files/seedvr2: pack/blkNN.bin (+ glue/), vae/svr_enc_t64.bin and svr_dec_t64.bin; the helper is svr.sh -> libsvsr.so. */
final class SeedVr {
    static File dir(File home) { return new File(home, "seedvr2"); }
    static final String BASE = "https://huggingface.co/ayanoby/hexanima-upscale/resolve/main/seedvr2/";
    /** the files of the model with sizes and hashes (manifest.json of the Hugging Face repository) */
    static org.json.JSONArray files() throws Exception {
        java.net.HttpURLConnection h = Models.open(BASE + "manifest.json", 0);
        try (InputStream in = h.getInputStream(); ByteArrayOutputStream o = new ByteArrayOutputStream()) { byte[] b = new byte[1 << 14]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n); return new JSONObject(o.toString("UTF-8")).getJSONArray("files"); }
    }
    /** downloads what is missing (finished files are kept, partial ones resume); progress = bytes done of the whole model */
    static void download(Context c, Models.Progress pg, java.util.concurrent.atomic.AtomicBoolean cancel) throws Exception {
        org.json.JSONArray a = files(); long total = Models.total(a), done = 0;
        for (int i = 0; i < a.length(); i++) {
            final JSONObject f = a.getJSONObject(i); final long before = done;
            Models.fetch(BASE + f.getString("path"), new File(dir(c.getFilesDir()), f.getString("path")), f.getLong("size"), f.optString("sha256", null), n -> { if (pg != null) pg.on(before + n, total, f.optString("path")); }, cancel);
            done += f.getLong("size");
        }
    }
    /** the DiT works on 16 x 16 pixel tokens in the DSP buffers: 16384 tokens (about 4.2 megapixels) is the limit of the FastRPC mapping */
    static final long MAX_TOKENS = 16384;
    static final String[] GLUE = {"txt0", "vid_in_w", "vid_in_b", "out_w", "out_b", "out_norm", "out_scale", "out_shift", "meta.json"};
    static long blockSize(int i) { return i < 10 ? 158959616L : 79481856L; }
    static boolean has(File home) {
        File d = dir(home);
        for (int i = 0; i < 32; i++) if (new File(d, String.format("pack/blk%02d.bin", i)).length() != blockSize(i)) return false;
        for (String g : GLUE) if (!new File(d, "pack/glue/" + g).exists()) return false;
        return new File(d, "vae/svr_enc_t64.bin").length() == 75609696L && new File(d, "vae/svr_dec_t64.bin").length() == 114006880L;
    }
    /** output size: the sides are multiples of 16 */
    static int[] outSize(int w, int h, double scale) { return new int[]{Math.max(16, (int) (w * scale) / 16 * 16), Math.max(16, (int) (h * scale) / 16 * 16)}; }
    static String sizeError(int ow, int oh) {
        return (long) (ow / 16) * (oh / 16) > MAX_TOKENS ? L.s("Слишком большая картинка для SeedVR2: результат не больше около 4 мегапикселей (2048×2048)", "The picture is too large for SeedVR2: the result is limited to about 4 megapixels (2048×2048)") : null;
    }
    interface Progress { void on(String msg); }

    /** restores `src` (a gallery picture) at `scale` times its size and saves the result as a new gallery picture; blocks, call off the UI thread */
    static File run(Context c, Main.Sh sh, File src, double scale, Progress pr) throws Exception {
        File home = c.getFilesDir(), sd = dir(home), in = new File(sd, "in.rgb"), out = new File(sd, "out.rgb"), log = new File(sd, "svr.log");
        Bitmap bm = BitmapFactory.decodeFile(src.getPath());
        if (bm == null) throw new IOException("cannot decode " + src.getName());
        final int w = bm.getWidth(), h = bm.getHeight(); final int[] o = outSize(w, h, scale), ow = {o[0]}, oh = {o[1]};
        String bad = sizeError(ow[0], oh[0]); if (bad != null) throw new IOException(bad);
        try (BufferedOutputStream os = new BufferedOutputStream(new FileOutputStream(in), 1 << 20)) {
            int[] row = new int[w * 16]; byte[] b = new byte[w * 16 * 3];
            for (int y = 0; y < h; y += 16) {
                int n = Math.min(16, h - y); bm.getPixels(row, 0, w, 0, y, w, n);
                for (int i = 0; i < w * n; i++) { int p = row[i]; b[3 * i] = (byte) (p >> 16); b[3 * i + 1] = (byte) (p >> 8); b[3 * i + 2] = (byte) p; }
                os.write(b, 0, w * n * 3);
            }
        }
        bm.recycle();
        final boolean[] done = {false};
        Thread poll = new Thread(() -> {   // the helper writes "stage <name> <done> <total>" lines to svr.log
            while (!done[0]) {
                try { Thread.sleep(700); String last = ""; try (BufferedReader r = new BufferedReader(new FileReader(log))) { for (String l; (l = r.readLine()) != null; ) if (l.startsWith("stage ")) last = l; }
                    String[] p = last.split(" "); if (p.length == 4 && pr != null) pr.on(p[1].equals("encode") ? L.f("Кодирование %s/%s", "Encoding %s/%s", p[2], p[3]) : p[1].equals("dit") ? (p[2].equals("0") ? L.s("Восстановление (DiT)…", "Restoring (DiT)…") : L.s("Декодирование…", "Decoding…")) : L.f("Декодирование %s/%s", "Decoding %s/%s", p[2], p[3])); } catch (Exception ignored) { }
            }
        });
        poll.start();
        try {
            sh.run("rm -f " + out.getPath() + "; S=" + sd.getPath() + " sh " + new File(home, "anima/svr.sh").getPath() + " " + in.getPath() + " " + w + " " + h + " " + out.getPath() + " " + ow[0] + " " + oh[0] + " " + (int) (System.currentTimeMillis() % 100000) + " < /dev/null > /dev/null 2>&1");
        } finally { done[0] = true; in.delete(); }
        if (!out.exists() || out.length() != (long) ow[0] * oh[0] * 3) throw new IOException("SeedVR2 produced no picture (files/seedvr2/svr.log)");
        File dst = new File(Gallery.dir(c), System.currentTimeMillis() + ".png");
        Bitmap ob = Bitmap.createBitmap(ow[0], oh[0], Bitmap.Config.ARGB_8888);
        try (InputStream r = new BufferedInputStream(new FileInputStream(out), 1 << 20)) {
            int[] px = new int[ow[0] * 16]; byte[] b = new byte[ow[0] * 16 * 3];
            for (int y = 0; y < oh[0]; y += 16) {
                int n = Math.min(16, oh[0] - y), need = ow[0] * n * 3, got = 0;
                while (got < need) { int k = r.read(b, got, need - got); if (k <= 0) throw new IOException("short read"); got += k; }
                for (int i = 0; i < ow[0] * n; i++) px[i] = 0xFF000000 | ((b[3 * i] & 255) << 16) | ((b[3 * i + 1] & 255) << 8) | (b[3 * i + 2] & 255);
                ob.setPixels(px, 0, ow[0], 0, y, ow[0], n);
            }
        }
        out.delete();
        try (FileOutputStream os = new FileOutputStream(dst)) { ob.compress(Bitmap.CompressFormat.PNG, 100, os); }
        ob.recycle();
        JSONObject m = new JSONObject(Gallery.meta(src).toString());
        String tag = "SeedVR2 ×" + (scale == Math.rint(scale) ? String.valueOf((int) scale) : String.valueOf(scale));
        m.put("w", ow[0]).put("h", oh[0]).put("model", m.optString("model", "") + " + " + tag).put("upscaled", tag).put("from", src.getName());
        Gallery.writeMeta(dst, m);
        return dst;
    }
    static void stop(Main.Sh sh, File home) { sh.run("S=" + dir(home).getPath() + " sh " + new File(home, "anima/svr.sh").getPath() + " stop"); }
}
