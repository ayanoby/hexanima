package com.hexanima.npu;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/** "Tags from a picture": WD EVA02-Large Tagger v3 (SmilingWolf, Apache-2.0) converted for the Hexagon NPU (QNN HTP context, fp16, input 448 x 448, 10861 tags). The picture is padded to a square with white, scaled to 448 px and
 *  sent as BGR floats 0..255 (NHWC) to libvaerun.so; the output is the probability of every tag (assets/wd_tags.csv lists them: general tags >= 0.35, characters >= 0.85). About 3 s on the phone; the context (674 MB) is downloaded once. */
final class Tagger {
    static final String URL = "https://huggingface.co/ayanoby/hexanima-tagger/resolve/main/wd_eva02_v3.bin";
    static final long SIZE = 673689584L; static final String SHA = "d96140714c969e2b20df792f0f92864bbbb63c755cb43468f6a15dfc3937d709";
    static final int N = 10861, SZ = 448;
    static File dir(File home) { return new File(home, "wd"); }
    static File model(File home) { return new File(dir(home), "wd_eva02_v3.bin"); }
    static boolean has(File home) { return model(home).length() == SIZE; }
    static void download(Context c, Models.Count cb, AtomicBoolean cancel) throws Exception { Models.fetch(URL, model(c.getFilesDir()), SIZE, SHA, cb, cancel); }
    static void delete(File home) { model(home).delete(); }

    /** blocks (about 5 s: the context is loaded, then one run), call off the UI thread; returns "tag, tag, ..." with spaces instead of underscores, the most certain first */
    static String run(Context c, Bitmap src) throws Exception {
        File d = dir(c.getFilesDir()); d.mkdirs(); File in = new File(d, "in.bin"), out = new File(d, "out.bin"); out.delete();
        int s = Math.max(src.getWidth(), src.getHeight());
        Bitmap sq = Bitmap.createBitmap(SZ, SZ, Bitmap.Config.ARGB_8888); Canvas cv = new Canvas(sq); cv.drawColor(Color.WHITE);   // padded to a square with white, then scaled to 448
        float k = (float) SZ / s; android.graphics.Matrix m = new android.graphics.Matrix(); m.postScale(k, k); m.postTranslate((SZ - src.getWidth() * k) / 2, (SZ - src.getHeight() * k) / 2);
        cv.drawBitmap(src, m, new android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG | android.graphics.Paint.ANTI_ALIAS_FLAG));
        int[] px = new int[SZ * SZ]; sq.getPixels(px, 0, SZ, 0, 0, SZ, SZ); sq.recycle();
        ByteBuffer b = ByteBuffer.allocate(SZ * SZ * 12).order(ByteOrder.LITTLE_ENDIAN);
        for (int p : px) { b.putFloat(p & 255); b.putFloat((p >> 8) & 255); b.putFloat((p >> 16) & 255); }   // BGR, 0..255
        try (OutputStream o = new FileOutputStream(in)) { o.write(b.array()); }
        String bin = c.getApplicationInfo().nativeLibraryDir;
        ProcessBuilder pb = new ProcessBuilder(bin + "/libvaerun.so", bin + "/libQnnHtp.so", bin + "/libQnnSystem.so", model(c.getFilesDir()).getPath());
        pb.environment().put("LD_LIBRARY_PATH", bin + ":/system/lib64:/vendor/lib64"); pb.environment().put("ADSP_LIBRARY_PATH", bin);
        pb.redirectError(new File("/dev/null"));
        Process p = pb.start();
        try (Writer w = new OutputStreamWriter(p.getOutputStream()); BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            for (String l; (l = r.readLine()) != null && !l.startsWith("ready"); ) { }
            w.write("run " + in.getPath() + " " + out.getPath() + "\nquit\n"); w.flush();
            for (String l; (l = r.readLine()) != null && !l.startsWith("done") && !l.startsWith("error"); ) { }
        } finally { p.destroy(); in.delete(); }
        if (out.length() != N * 4L) throw new IOException("the tagger produced no result");
        float[] pr = new float[N]; ByteBuffer ob = ByteBuffer.wrap(readAll(out)).order(ByteOrder.LITTLE_ENDIAN); for (int i = 0; i < N; i++) pr[i] = ob.getFloat(); out.delete();
        List<String[]> rows = rows(c); List<float[]> hit = new ArrayList<>(); List<String> names = new ArrayList<>();
        for (int i = 0; i < Math.min(N, rows.size()); i++) {
            String cat = rows.get(i)[2]; float t = cat.equals("0") ? .35f : cat.equals("4") ? .85f : 2f;   // general tags and characters only (not the ratings)
            if (pr[i] >= t) { hit.add(new float[]{pr[i], names.size()}); names.add(rows.get(i)[1].replace('_', ' ')); }
        }
        hit.sort((x, y) -> Float.compare(y[0], x[0]));
        StringBuilder sb = new StringBuilder(); for (float[] h : hit) { if (sb.length() > 0) sb.append(", "); sb.append(names.get((int) h[1])); }
        return sb.toString();
    }
    static byte[] readAll(File f) throws IOException { try (InputStream i = new FileInputStream(f); ByteArrayOutputStream o = new ByteArrayOutputStream()) { byte[] b = new byte[1 << 16]; for (int n; (n = i.read(b)) > 0; ) o.write(b, 0, n); return o.toByteArray(); } }
    /** assets/wd_tags.csv: tag_id,name,category,count (the order of the model's outputs); tag names have no commas that matter here (a few do: "tag" is quoted) */
    static List<String[]> rows(Context c) throws IOException {
        List<String[]> l = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(c.getAssets().open("wd_tags.csv"), StandardCharsets.UTF_8))) {
            r.readLine();
            for (String line; (line = r.readLine()) != null; ) {
                String[] p; if (line.contains("\"")) { int a = line.indexOf('"'), e = line.lastIndexOf('"'); String[] h = line.substring(0, a).split(","), t = line.substring(e + 1).split(","); p = new String[]{h[0], line.substring(a + 1, e), t.length > 1 ? t[1] : "0", ""}; }
                else p = line.split(",");
                l.add(p);
            }
        }
        return l;
    }
}
