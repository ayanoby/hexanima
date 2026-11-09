package com.hexanima.npu;

import android.content.Context;
import android.graphics.Bitmap;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Live preview while a picture is made: the runner writes the current x0 estimate of the latent after every step (header int32 {h, w, C, step}, then fp32 [C][h][w]),
 *  a fitted linear map (assets/preview_NAME.bin: int32 C, B, blur passes, fp32 W[C][3*B*B], bias[3*B*B]) turns every latent token into a B×B block of pixels. No VAE runs, so it costs a few milliseconds. */
final class Preview {
    private final int C, B, passes; private final float[] W, bias; private long seen;

    private Preview(int c, int b, int passes, float[] w, float[] bias) { C = c; B = b; this.passes = passes; W = w; this.bias = bias; }

    static Preview load(Context cx, String name) {
        try (InputStream in = new BufferedInputStream(cx.getAssets().open("preview_" + name + ".bin"))) {
            byte[] all = in.readAllBytes(); ByteBuffer bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
            int c = bb.getInt(), b = bb.getInt(), ps = bb.getInt(), k = 3 * b * b; float[] w = new float[c * k], bi = new float[k];
            bb.asFloatBuffer().get(w); bb.position(bb.position() + 4 * w.length); bb.asFloatBuffer().get(bi);
            return new Preview(c, b, ps, w, bi);
        } catch (Exception e) { return null; }
    }

    /** the picture for the file when it has changed since the last call, else null */
    Bitmap poll(File f) {
        long m = f.lastModified(); if (m == 0 || m == seen) return null;
        try (RandomAccessFile r = new RandomAccessFile(f, "r")) {
            byte[] all = new byte[(int) r.length()]; r.readFully(all); ByteBuffer bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
            int h = bb.getInt(), w = bb.getInt(), c = bb.getInt(); bb.getInt();
            if (c != C || all.length < 16 + 4L * c * h * w) return null;
            float[] z = new float[c * h * w]; bb.asFloatBuffer().get(z); seen = m;
            return render(z, h, w);
        } catch (Exception e) { return null; }
    }

    Bitmap render(float[] z, int h, int w) {
        final int k = 3 * B * B, n = h * w, W2 = w * B, H2 = h * B; int[] px = new int[W2 * H2]; float[] acc = new float[k];
        for (int t = 0; t < n; t++) {
            System.arraycopy(bias, 0, acc, 0, k);
            for (int c = 0; c < C; c++) { float v = z[c * n + t]; int o = c * k; for (int j = 0; j < k; j++) acc[j] += v * W[o + j]; }
            int ty = t / w, tx = t % w;
            for (int py = 0; py < B; py++) for (int pxx = 0; pxx < B; pxx++) {
                int j = (py * B + pxx) * 3; int r = cl(acc[j]), g = cl(acc[j + 1]), b = cl(acc[j + 2]);
                px[(ty * B + py) * W2 + tx * B + pxx] = 0xFF000000 | r << 16 | g << 8 | b;
            }
        }
        for (int i = 0; i < passes; i++) blur(px, W2, H2);   // the block structure of the fit shows as a grid: [1 2 1] passes (two = about sigma 1.2) hide it
        return Bitmap.createBitmap(px, W2, H2, Bitmap.Config.ARGB_8888);
    }
    /** separable [1 2 1] / 4 blur in place, edges replicated */
    private static void blur(int[] px, int w, int h) {
        int[] tmp = new int[px.length];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { int a = px[y * w + Math.max(x - 1, 0)], b = px[y * w + x], c = px[y * w + Math.min(x + 1, w - 1)]; tmp[y * w + x] = mix(a, b, c); }
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { int a = tmp[Math.max(y - 1, 0) * w + x], b = tmp[y * w + x], c = tmp[Math.min(y + 1, h - 1) * w + x]; px[y * w + x] = mix(a, b, c); }
    }
    private static int mix(int a, int b, int c) {
        int r = (((a >> 16) & 255) + 2 * ((b >> 16) & 255) + ((c >> 16) & 255) + 2) >> 2, g = (((a >> 8) & 255) + 2 * ((b >> 8) & 255) + ((c >> 8) & 255) + 2) >> 2, bl = ((a & 255) + 2 * (b & 255) + (c & 255) + 2) >> 2;
        return 0xFF000000 | r << 16 | g << 8 | bl;
    }
    private static int cl(float v) { int i = Math.round((v + 1f) * 127.5f); return i < 0 ? 0 : i > 255 ? 255 : i; }
}
