package com.hexanima.npu;

import android.graphics.Bitmap;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/** raw fp32 video [frames][3][H][W] in [-1,1] (liblvae) -> H.264 MP4 (MediaCodec + MediaMuxer) and a PNG of the middle frame (the gallery entry) */
final class VideoOut {
    /** @return the PNG bytes of the middle frame; writes the MP4 to out */
    static byte[] encode(File raw, int frames, int w, int h, int fps, File out) throws Exception {
        final long fb = (long) 3 * w * h * 4;
        if (raw.length() != fb * frames) throw new IOException("raw video has " + raw.length() + " bytes, expected " + fb * frames);
        MediaFormat f = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h);
        f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible); f.setInteger(MediaFormat.KEY_BIT_RATE, Math.max(4_000_000, w * h * fps / 2));
        f.setInteger(MediaFormat.KEY_FRAME_RATE, fps); f.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
        MediaCodec codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC); codec.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE); codec.start();
        MediaMuxer mux = new MediaMuxer(out.getPath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4); int track = -1; boolean muxing = false;
        byte[] poster = null; final MediaCodec.BufferInfo bi = new MediaCodec.BufferInfo();
        try (FileChannel ch = new RandomAccessFile(raw, "r").getChannel()) {
            ByteBuffer fbuf = ByteBuffer.allocateDirect((int) fb).order(ByteOrder.LITTLE_ENDIAN); final float[] fr = new float[3 * w * h];
            int sent = 0; boolean eos = false;
            while (!eos) {
                if (sent <= frames) {
                    int ii = codec.dequeueInputBuffer(10000);
                    if (ii >= 0) {
                        if (sent < frames) {
                            fbuf.clear(); ch.position(sent * fb); while (fbuf.hasRemaining() && ch.read(fbuf) > 0) { } fbuf.flip(); fbuf.asFloatBuffer().get(fr);
                            Image img = codec.getInputImage(ii); fill(img, fr, w, h); codec.queueInputBuffer(ii, 0, w * h * 3 / 2, (long) sent * 1_000_000L / fps, 0);
                            if (sent == frames / 2) poster = png(fr, w, h);
                        } else codec.queueInputBuffer(ii, 0, 0, (long) sent * 1_000_000L / fps, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        sent++;
                    }
                }
                int oi = codec.dequeueOutputBuffer(bi, 10000);
                if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) { track = mux.addTrack(codec.getOutputFormat()); mux.start(); muxing = true; }
                else if (oi >= 0) {
                    ByteBuffer ob = codec.getOutputBuffer(oi);
                    if ((bi.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) bi.size = 0;
                    if (bi.size > 0 && muxing) { ob.position(bi.offset); ob.limit(bi.offset + bi.size); mux.writeSampleData(track, ob, bi); }
                    codec.releaseOutputBuffer(oi, false); if ((bi.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) eos = true;
                }
            }
        } finally { try { codec.stop(); } catch (Exception ignored) { } codec.release(); try { if (muxing) mux.stop(); } catch (Exception ignored) { } mux.release(); }
        return poster;
    }

    /** RGB [3][H][W] in [-1,1] -> the encoder's YUV420 image (BT.601, limited range) */
    private static void fill(Image img, float[] fr, int w, int h) {
        final Image.Plane[] p = img.getPlanes(); final ByteBuffer by = p[0].getBuffer(), bu = p[1].getBuffer(), bv = p[2].getBuffer(); final int ys = p[0].getRowStride(), us = p[1].getRowStride(), vs = p[2].getRowStride(), up = p[1].getPixelStride(), vp = p[2].getPixelStride();
        final int n = w * h;
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            final float r = (fr[y * w + x] + 1f) * 127.5f, g = (fr[n + y * w + x] + 1f) * 127.5f, b = (fr[2 * n + y * w + x] + 1f) * 127.5f;
            by.put(y * ys + x, (byte) clamp(16f + 0.257f * r + 0.504f * g + 0.098f * b));
            if ((y & 1) == 0 && (x & 1) == 0) {   // chroma from the 2x2 average
                float rr = 0, gg = 0, bb = 0; for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) { int k = (y + dy) * w + x + dx; rr += (fr[k] + 1f) * 127.5f; gg += (fr[n + k] + 1f) * 127.5f; bb += (fr[2 * n + k] + 1f) * 127.5f; }
                rr /= 4; gg /= 4; bb /= 4;
                bu.put((y / 2) * us + (x / 2) * up, (byte) clamp(128f - 0.148f * rr - 0.291f * gg + 0.439f * bb)); bv.put((y / 2) * vs + (x / 2) * vp, (byte) clamp(128f + 0.439f * rr - 0.368f * gg - 0.071f * bb));
            }
        }
    }
    private static int clamp(float v) { return v < 0 ? 0 : v > 255 ? 255 : (int) (v + 0.5f); }
    private static byte[] png(float[] fr, int w, int h) {
        int[] px = new int[w * h]; final int n = w * h;
        for (int i = 0; i < n; i++) px[i] = 0xFF000000 | (clamp((fr[i] + 1f) * 127.5f) << 16) | (clamp((fr[n + i] + 1f) * 127.5f) << 8) | clamp((fr[2 * n + i] + 1f) * 127.5f);
        Bitmap bm = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888); ByteArrayOutputStream o = new ByteArrayOutputStream(); bm.compress(Bitmap.CompressFormat.PNG, 100, o); bm.recycle(); return o.toByteArray();
    }
}
