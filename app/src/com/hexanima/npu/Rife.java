package com.hexanima.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.*;
import android.net.Uri;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONObject;

/** Frame interpolation: RIFE v4.6 (Practical-RIFE, hzwer, MIT) on the Hexagon NPU. The picture pair goes in as one [1,6,H,W] tensor, the middle frame comes out ([1,3,H,W], 0..1); one QNN context per frame size (sides multiples of 32,
 *  files rife_tHxW.bin, downloaded on demand), the video is padded (edge replicate) to the smallest context that covers it. x2 = one new frame between every two, x4 = three (recursively). Hard cuts are not interpolated (the frame is repeated).
 *  The pipeline is MediaExtractor + MediaCodec (decode) -> NPU -> MediaCodec + MediaMuxer (H.264, the audio track is copied as it is). Runs libvaerun.so (a runner for one input / one output). */
final class Rife {
    /** the contexts that exist: {H, W} */
    static final int[][] CTX = {{256, 256}, {256, 448}, {384, 512}, {512, 384}, {512, 512}, {512, 768}, {576, 1024}, {768, 512}, {768, 768}, {768, 1024}, {768, 1280}, {1024, 576}, {1024, 768}, {1024, 1024}, {1280, 768}};
    static final String BASE = "https://huggingface.co/ayanoby/hexanima-rife/resolve/main/";
    static File dir(File home) { return new File(home, "rife"); }
    static String name(int[] c) { return "rife_t" + c[0] + "x" + c[1] + ".bin"; }
    static File file(File home, int[] c) { return new File(dir(home), name(c)); }
    /** the smallest context (by area) that covers a w x h video, or null */
    static int[] choose(int w, int h) { int[] best = null; for (int[] c : CTX) if (c[0] >= h && c[1] >= w && (best == null || c[0] * c[1] < best[0] * best[1])) best = c; return best; }
    static boolean has(File home, int[] c) { return file(home, c).length() > (1 << 20); }
    /** milliseconds one pair takes on the NPU (measured: 37 ms at 256 x 448; the rest of a pair is file I/O), for the estimate shown to the user */
    static double pairSec(int[] c) { return 0.04 + c[0] * (double) c[1] * 0.45e-6; }

    static JSONObject manifest() throws Exception {
        java.net.HttpURLConnection h = Models.open(BASE + "manifest.json", 0);
        try (InputStream in = h.getInputStream(); ByteArrayOutputStream o = new ByteArrayOutputStream()) { byte[] b = new byte[1 << 14]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n); return new JSONObject(o.toString("UTF-8")); }
    }
    static JSONObject entry(int[] c) throws Exception { org.json.JSONArray a = manifest().getJSONArray("files"); for (int i = 0; i < a.length(); i++) if (a.getJSONObject(i).getString("path").equals(name(c))) return a.getJSONObject(i); throw new IOException("not in the manifest: " + name(c)); }
    static void download(Context c, int[] ctx, Models.Count cb, AtomicBoolean cancel) throws Exception { JSONObject e = entry(ctx); Models.fetch(BASE + name(ctx), file(c.getFilesDir(), ctx), e.getLong("size"), e.getString("sha256"), cb, cancel); }

    interface Progress { void on(int done, int total, String msg); }

    /** the running NPU context */
    static final class Net implements Closeable {
        final int H, W; final Process p; final Writer w; final BufferedReader r; final File in, out;
        Net(Context c, int[] hw) throws IOException {
            H = hw[0]; W = hw[1]; File d = dir(c.getFilesDir()); d.mkdirs(); in = new File(d, "in.bin"); out = new File(d, "out.bin");
            String bin = c.getApplicationInfo().nativeLibraryDir;
            ProcessBuilder pb = new ProcessBuilder(bin + "/libvaerun.so", bin + "/libQnnHtp.so", bin + "/libQnnSystem.so", file(c.getFilesDir(), hw).getPath());
            pb.environment().put("LD_LIBRARY_PATH", bin + ":/system/lib64:/vendor/lib64"); pb.environment().put("ADSP_LIBRARY_PATH", bin); pb.redirectError(new File("/dev/null"));
            p = pb.start(); w = new OutputStreamWriter(p.getOutputStream()); r = new BufferedReader(new InputStreamReader(p.getInputStream()));
            for (String l; (l = r.readLine()) != null && !l.startsWith("ready"); ) { }
        }
        /** a, b: padded planar RGB [3][H][W] bytes; the middle frame goes to o */
        void mid(byte[] a, byte[] b, byte[] o) throws IOException {
            final int n = 3 * H * W; ByteBuffer bb = ByteBuffer.allocateDirect(1 << 22).order(ByteOrder.LITTLE_ENDIAN);
            try (FileChannel ch = new FileOutputStream(in).getChannel()) {
                for (byte[] f : new byte[][]{a, b}) for (int i = 0; i < n; ) { bb.clear(); int m = Math.min(n - i, bb.capacity() / 4); for (int k = 0; k < m; k++) bb.putFloat((f[i + k] & 255) * (1f / 255f)); bb.flip(); while (bb.hasRemaining()) ch.write(bb); i += m; }
            }
            w.write("run " + in.getPath() + " " + out.getPath() + "\n"); w.flush();
            String l = r.readLine(); if (l == null || !l.startsWith("done")) throw new IOException("RIFE failed");
            try (FileChannel ch = new FileInputStream(out).getChannel()) {
                for (int i = 0; i < n; ) { bb.clear(); int m = Math.min(n - i, bb.capacity() / 4); bb.limit(m * 4); while (bb.hasRemaining()) if (ch.read(bb) < 0) throw new IOException("short output"); bb.flip();
                    for (int k = 0; k < m; k++) { float v = bb.getFloat(); o[i + k] = (byte) (v <= 0 ? 0 : v >= 1 ? 255 : (int) (v * 255f + .5f)); } i += m; }
            }
        }
        @Override public void close() { try { w.write("quit\n"); w.flush(); } catch (Exception ignored) { } p.destroy(); in.delete(); out.delete(); }
    }

    /** the video side of a file: size, rotation, frame rate, duration */
    static final class Info { int w, h, rot, fps; long durUs; String name = ""; int[] ctx; }
    static Info probe(Context c, Uri uri) throws IOException {
        MediaExtractor ex = new MediaExtractor();
        try {
            ex.setDataSource(c, uri, null); Info i = new Info();
            for (int t = 0; t < ex.getTrackCount(); t++) {
                MediaFormat f = ex.getTrackFormat(t); String m = f.getString(MediaFormat.KEY_MIME);
                if (m != null && m.startsWith("video/")) {
                    i.w = f.getInteger(MediaFormat.KEY_WIDTH); i.h = f.getInteger(MediaFormat.KEY_HEIGHT);
                    i.rot = f.containsKey(MediaFormat.KEY_ROTATION) ? f.getInteger(MediaFormat.KEY_ROTATION) : 0;
                    i.fps = f.containsKey(MediaFormat.KEY_FRAME_RATE) ? f.getInteger(MediaFormat.KEY_FRAME_RATE) : 30;
                    i.durUs = f.containsKey(MediaFormat.KEY_DURATION) ? f.getLong(MediaFormat.KEY_DURATION) : 0; i.ctx = choose(i.w, i.h); return i;
                }
            }
            throw new IOException("no video track");
        } finally { ex.release(); }
    }

    // ---- decoding ------------------------------------------------------------------------------
    static final class Dec {
        final MediaExtractor ex = new MediaExtractor(); final MediaCodec c; final int W, H; final boolean bt709; boolean inEos, outEos; final MediaCodec.BufferInfo bi = new MediaCodec.BufferInfo();
        Dec(Context ctx, Uri uri, int[] hw) throws IOException {
            ex.setDataSource(ctx, uri, null); int tr = -1; MediaFormat f = null;
            for (int t = 0; t < ex.getTrackCount(); t++) { MediaFormat g = ex.getTrackFormat(t); String m = g.getString(MediaFormat.KEY_MIME); if (m != null && m.startsWith("video/")) { tr = t; f = g; break; } }
            if (f == null) throw new IOException("no video track");
            ex.selectTrack(tr); H = hw[0]; W = hw[1]; bt709 = f.getInteger(MediaFormat.KEY_HEIGHT) >= 720;
            f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            c = MediaCodec.createDecoderByType(f.getString(MediaFormat.KEY_MIME)); c.configure(f, null, null, 0); c.start();
        }
        /** the next frame as padded planar RGB into dst; its presentation time in microseconds, or -1 at the end. vw x vh = the real picture size */
        long next(byte[] dst, int[] vwh) throws IOException {
            for (;;) {
                if (!inEos) {
                    int ii = c.dequeueInputBuffer(5000);
                    if (ii >= 0) { ByteBuffer b = c.getInputBuffer(ii); int n = ex.readSampleData(b, 0); if (n < 0) { c.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inEos = true; } else { c.queueInputBuffer(ii, 0, n, ex.getSampleTime(), 0); ex.advance(); } }
                }
                if (outEos) return -1;
                int oi = c.dequeueOutputBuffer(bi, 5000);
                if (oi >= 0) {
                    boolean eos = (bi.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0; long pts = bi.presentationTimeUs; boolean got = false;
                    if (bi.size > 0) { Image img = c.getOutputImage(oi); if (img != null) { try { fill(img, dst, vwh); got = true; } finally { img.close(); } } }
                    c.releaseOutputBuffer(oi, false); if (eos) outEos = true;
                    if (got) return pts; if (eos) return -1;
                }
            }
        }
        void fill(Image img, byte[] dst, int[] vwh) {
            android.graphics.Rect cr = img.getCropRect(); final int w = cr.width() & ~1, h = cr.height() & ~1; vwh[0] = w; vwh[1] = h;
            Image.Plane[] p = img.getPlanes(); ByteBuffer by = p[0].getBuffer(), bu = p[1].getBuffer(), bv = p[2].getBuffer(); int ys = p[0].getRowStride(), us = p[1].getRowStride(), vs = p[2].getRowStride(), up = p[1].getPixelStride(), vp = p[2].getPixelStride();
            final int plane = H * W; final float kr = bt709 ? 1.793f : 1.596f, kgu = bt709 ? .213f : .392f, kgv = bt709 ? .533f : .813f, kb = bt709 ? 2.112f : 2.017f;
            for (int y = 0; y < h; y++) {
                int yy = y + cr.top, cy = yy >> 1;
                for (int x = 0; x < w; x++) {
                    int xx = x + cr.left, cx = xx >> 1; float Y = 1.164f * ((by.get(yy * ys + xx) & 255) - 16), U = (bu.get(cy * us + cx * up) & 255) - 128, V = (bv.get(cy * vs + cx * vp) & 255) - 128;
                    int r = (int) (Y + kr * V + .5f), g = (int) (Y - kgu * U - kgv * V + .5f), b = (int) (Y + kb * U + .5f); int o = y * W + x;
                    dst[o] = (byte) (r < 0 ? 0 : r > 255 ? 255 : r); dst[plane + o] = (byte) (g < 0 ? 0 : g > 255 ? 255 : g); dst[2 * plane + o] = (byte) (b < 0 ? 0 : b > 255 ? 255 : b);
                }
                for (int x = w; x < W; x++) { int o = y * W + x, s = y * W + w - 1; dst[o] = dst[s]; dst[plane + o] = dst[plane + s]; dst[2 * plane + o] = dst[2 * plane + s]; }   // replicate the right edge
            }
            for (int y = h; y < H; y++) { int s = (h - 1) * W; for (int ch = 0; ch < 3; ch++) System.arraycopy(dst, ch * plane + s, dst, ch * plane + y * W, W); }   // and the bottom edge
        }
        void close() { try { c.stop(); } catch (Exception ignored) { } c.release(); ex.release(); }
    }

    // ---- encoding ------------------------------------------------------------------------------
    static final class Enc {
        final MediaCodec c; final MediaMuxer m; final int w, h, W, H; final boolean bt709; int track = -1, audio = -1; boolean muxing; final MediaCodec.BufferInfo bi = new MediaCodec.BufferInfo(); final MediaExtractor aex; int aTrack = -1;
        Enc(Context ctx, Uri src, File out, int w, int h, int[] hw, int fps, int rot) throws IOException {
            this.w = w & ~1; this.h = h & ~1; H = hw[0]; W = hw[1]; bt709 = h >= 720;
            MediaFormat f = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, this.w, this.h);
            f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible); f.setInteger(MediaFormat.KEY_BIT_RATE, (int) Math.max(3_000_000, Math.min(40_000_000, this.w * (double) this.h * fps * 0.12)));
            f.setInteger(MediaFormat.KEY_FRAME_RATE, fps); f.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
            c = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC); c.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE); c.start();
            m = new MediaMuxer(out.getPath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4); if (rot != 0) m.setOrientationHint(rot);
            aex = new MediaExtractor(); try { aex.setDataSource(ctx, src, null); for (int t = 0; t < aex.getTrackCount(); t++) { String mm = aex.getTrackFormat(t).getString(MediaFormat.KEY_MIME); if (mm != null && mm.startsWith("audio/")) { aTrack = t; break; } } } catch (Exception e) { aTrack = -1; }
        }
        void put(byte[] planar, long ptsUs, boolean eos) {
            for (;;) {
                int ii = c.dequeueInputBuffer(10000);
                if (ii >= 0) {
                    if (eos) c.queueInputBuffer(ii, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                    else { Image img = c.getInputImage(ii); fill(img, planar); c.queueInputBuffer(ii, 0, w * h * 3 / 2, ptsUs, 0); }
                    drain(false); return;
                }
                drain(false);
            }
        }
        void drain(boolean wait) {
            for (;;) {
                int oi = c.dequeueOutputBuffer(bi, wait ? 10000 : 0);
                if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    track = m.addTrack(c.getOutputFormat());
                    if (aTrack >= 0) { try { aex.selectTrack(aTrack); audio = m.addTrack(aex.getTrackFormat(aTrack)); } catch (Exception e) { audio = -1; } }
                    m.start(); muxing = true;
                } else if (oi >= 0) {
                    ByteBuffer ob = c.getOutputBuffer(oi); if ((bi.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) bi.size = 0;
                    if (bi.size > 0 && muxing) { ob.position(bi.offset); ob.limit(bi.offset + bi.size); m.writeSampleData(track, ob, bi); }
                    c.releaseOutputBuffer(oi, false); if ((bi.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) { eosSeen = true; return; }
                } else if (oi == MediaCodec.INFO_TRY_AGAIN_LATER) return;
            }
        }
        boolean eosSeen;
        void fill(Image img, byte[] pl) {
            Image.Plane[] p = img.getPlanes(); ByteBuffer by = p[0].getBuffer(), bu = p[1].getBuffer(), bv = p[2].getBuffer(); int ys = p[0].getRowStride(), us = p[1].getRowStride(), vs = p[2].getRowStride(), up = p[1].getPixelStride(), vp = p[2].getPixelStride(); final int plane = H * W;
            final float yr = bt709 ? .183f : .257f, yg = bt709 ? .614f : .504f, yb = bt709 ? .062f : .098f, ur = bt709 ? -.101f : -.148f, ug = bt709 ? -.339f : -.291f, vg = bt709 ? -.399f : -.368f, vb = bt709 ? -.040f : -.071f;
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                int o = y * W + x; float r = pl[o] & 255, g = pl[plane + o] & 255, b = pl[2 * plane + o] & 255; by.put(y * ys + x, (byte) clamp(16f + yr * r + yg * g + yb * b));
                if ((y & 1) == 0 && (x & 1) == 0) {
                    float rr = 0, gg = 0, bb = 0; for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) { int k = (y + dy) * W + x + dx; rr += pl[k] & 255; gg += pl[plane + k] & 255; bb += pl[2 * plane + k] & 255; } rr /= 4; gg /= 4; bb /= 4;
                    bu.put((y / 2) * us + (x / 2) * up, (byte) clamp(128f + ur * rr + ug * gg + .439f * bb)); bv.put((y / 2) * vs + (x / 2) * vp, (byte) clamp(128f + .439f * rr + vg * gg + vb * bb));
                }
            }
        }
        static int clamp(float v) { return v < 0 ? 0 : v > 255 ? 255 : (int) (v + .5f); }
        void finish() {
            try {
                put(null, 0, true); while (!eosSeen) drain(true);
                if (audio >= 0 && muxing) { try { ByteBuffer ab = ByteBuffer.allocate(1 << 18); MediaCodec.BufferInfo ai = new MediaCodec.BufferInfo(); aex.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC);
                    for (int n; (n = aex.readSampleData(ab, 0)) >= 0; aex.advance()) { ai.set(0, n, aex.getSampleTime(), (aex.getSampleFlags() & MediaExtractor.SAMPLE_FLAG_SYNC) != 0 ? MediaCodec.BUFFER_FLAG_KEY_FRAME : 0); m.writeSampleData(audio, ab, ai); } } catch (Exception ignored) { } }
            } finally { close(); }
        }
        void close() { try { c.stop(); } catch (Exception ignored) { } c.release(); try { if (muxing) m.stop(); } catch (Exception ignored) { } m.release(); aex.release(); }
    }

    /** x2 / x4 interpolation of a whole video into a new gallery entry (the MP4 + a poster); blocks, call off the UI thread; returns null when cancelled */
    static File run(Context c, Uri src, String srcName, int factor, Progress pr, AtomicBoolean cancel) throws Exception {
        final Info inf = probe(c, src); if (inf.ctx == null) throw new IOException(L.s("Видео больше 1280×768 (720p): не поддерживается", "The video is larger than 1280x768 (720p): not supported"));
        if (!has(c.getFilesDir(), inf.ctx)) throw new IOException("missing " + name(inf.ctx));
        final int H = inf.ctx[0], W = inf.ctx[1], sz = 3 * H * W; final int fpsOut = inf.fps * factor;
        final String stamp = String.valueOf(System.currentTimeMillis()); final File mp4 = new File(Gallery.dir(c), stamp + ".mp4");
        Net net = null; Dec dec = null; Enc enc = null; boolean ok = false;
        try {
            net = new Net(c, inf.ctx); dec = new Dec(c, src, inf.ctx); final int[] vwh = new int[2];
            byte[] prev = new byte[sz], cur = new byte[sz], tmp = new byte[sz], tmp2 = new byte[sz], tmp3 = new byte[sz]; byte[] poster = null;
            long t0 = dec.next(prev, vwh); if (t0 < 0) throw new IOException("no frames"); final long base = t0;
            enc = new Enc(c, src, mp4, vwh[0], vwh[1], inf.ctx, fpsOut, inf.rot); final int vw = vwh[0], vh = vwh[1];
            final long estFrames = inf.durUs > 0 ? Math.max(2, inf.durUs * inf.fps / 1000000L) : 0; long pp = t0; int n = 1, outN = 0; final long mid = inf.durUs / 2;
            enc.put(prev, 0, false); outN++;
            for (;;) {
                if (cancel.get()) return null;
                long pc = dec.next(cur, vwh); if (pc < 0) break;
                boolean cut = cut(prev, cur, H, W);
                if (cut) { for (int k = 1; k < factor; k++) { enc.put(prev, (pp - base) + (pc - pp) * k / factor, false); outN++; } }
                else if (factor == 2) { net.mid(prev, cur, tmp); enc.put(tmp, (pp - base) + (pc - pp) / 2, false); outN++; if (poster == null && (pp - base) >= mid) poster = png(tmp, vw, vh, H, W); }
                else { net.mid(prev, cur, tmp); net.mid(prev, tmp, tmp2); net.mid(tmp, cur, tmp3);
                    enc.put(tmp2, (pp - base) + (pc - pp) / 4, false); enc.put(tmp, (pp - base) + (pc - pp) / 2, false); enc.put(tmp3, (pp - base) + (pc - pp) * 3 / 4, false); outN += 3; if (poster == null && (pp - base) >= mid) poster = png(tmp, vw, vh, H, W); }
                enc.put(cur, pc - base, false); outN++; n++; pp = pc;
                byte[] sw = prev; prev = cur; cur = sw;
                if (pr != null) pr.on(n, (int) estFrames, null);
            }
            if (poster == null) poster = png(prev, vw, vh, H, W);
            enc.finish(); enc = null; ok = true;
            JSONObject meta = new JSONObject().put("prompt", "").put("seed", "").put("w", vw).put("h", vh).put("model", "RIFE ×" + factor + (srcName.isEmpty() ? "" : " · " + srcName)).put("video", mp4.getName()).put("frames", outN).put("fps", fpsOut).put("upscaled", "RIFE ×" + factor);
            return Gallery.save(c, poster, meta);
        } finally {
            if (enc != null) enc.close(); if (dec != null) dec.close(); if (net != null) net.close(); if (!ok) mp4.delete();
        }
    }

    /** a hard cut: a 32 x 32 sample of both frames differs by more than 22 % on average */
    static boolean cut(byte[] a, byte[] b, int H, int W) {
        long s = 0; int cnt = 0; final int plane = H * W;
        for (int y = 0; y < H; y += H / 32) for (int x = 0; x < W; x += W / 32) for (int ch = 0; ch < 3; ch++) { int o = ch * plane + y * W + x; s += Math.abs((a[o] & 255) - (b[o] & 255)); cnt++; }
        return s / (255.0 * cnt) > 0.22;
    }
    static byte[] png(byte[] pl, int w, int h, int H, int W) {
        int[] px = new int[w * h]; final int plane = H * W;
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { int o = y * W + x; px[y * w + x] = 0xFF000000 | ((pl[o] & 255) << 16) | ((pl[plane + o] & 255) << 8) | (pl[2 * plane + o] & 255); }
        Bitmap bm = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888); ByteArrayOutputStream o = new ByteArrayOutputStream(); bm.compress(Bitmap.CompressFormat.PNG, 100, o); bm.recycle(); return o.toByteArray();
    }
}
