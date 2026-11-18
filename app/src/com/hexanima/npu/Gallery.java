package com.hexanima.npu;

import android.content.ClipData;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.provider.MediaStore;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.json.JSONObject;

/** The app's own picture history: files/gallery/<ts>.png + <ts>.json (prompt, seed, size, model, time, fav). Deleted pictures move to gallery/trash for 30 days.
 *  Nothing goes to the phone gallery until the user taps "Download". */
final class Gallery {
    static final String AUTH = "com.hexanima.npu.gallery";
    static final int ALL = 0, FAV = 1, TRASH = 2;
    static final long TRASH_DAYS = 30;

    static File dir(Context c) { File d = new File(c.getFilesDir(), "gallery"); d.mkdirs(); return d; }
    static File trashDir(Context c) { File d = new File(dir(c), "trash"); d.mkdirs(); return d; }

    static File save(Context c, byte[] png, JSONObject meta) throws IOException {
        String base = String.valueOf(System.currentTimeMillis());
        File f = new File(dir(c), base + ".png");
        try (FileOutputStream o = new FileOutputStream(f)) { o.write(png); }
        writeMeta(f, meta);
        if (Settings.autoSave(c)) try { export(c, f); final String m = savedMsg(c); new android.os.Handler(android.os.Looper.getMainLooper()).post(() -> android.widget.Toast.makeText(c.getApplicationContext(), m, android.widget.Toast.LENGTH_SHORT).show()); } catch (Exception ignored) { }   // a failed copy must not lose the picture: it is in the app gallery
        return f;
    }

    /** newest first; FAV = favourites among the live pictures */
    static File[] list(Context c, int mode) {
        File[] fs = (mode == TRASH ? trashDir(c) : dir(c)).listFiles((d, n) -> n.endsWith(".png"));
        if (fs == null) return new File[0];
        Arrays.sort(fs, Comparator.comparing(File::getName).reversed());
        if (mode != FAV) return fs;
        List<File> out = new ArrayList<>();
        for (File f : fs) if (fav(f)) out.add(f);
        return out.toArray(new File[0]);
    }

    static File json(File png) { return new File(png.getPath().replaceAll("\\.png$", ".json")); }

    static JSONObject meta(File png) {
        try { return new JSONObject(Models.read(json(png))); } catch (Exception e) { return new JSONObject(); }
    }

    static void writeMeta(File png, JSONObject m) throws IOException { try (FileWriter w = new FileWriter(json(png))) { w.write(m.toString()); } }

    static boolean fav(File png) { return meta(png).optBoolean("fav"); }

    static void setFav(File png, boolean v) {
        try { writeMeta(png, meta(png).put("fav", v)); } catch (Exception ignored) { }
    }

    /** soft delete / restore: move the pair between gallery/ and gallery/trash/ */
    /** a video entry (LTX): the MP4 named in the metadata lies beside the poster PNG */
    static File videoOf(File png) { String v = meta(png).optString("video", ""); return v.isEmpty() ? null : new File(png.getParentFile(), v); }
    static void move(File png, File to) {
        File dst = new File(to, png.getName()); File vid = videoOf(png);
        png.renameTo(dst); json(png).renameTo(json(dst)); if (vid != null) vid.renameTo(new File(to, vid.getName()));
        dst.setLastModified(System.currentTimeMillis());
    }
    static void trash(Context c, File png) { move(png, trashDir(c)); }
    static void restore(Context c, File png) { move(png, dir(c)); }

    /** permanent delete */
    static void delete(File png) { File vid = videoOf(png); png.delete(); json(png).delete(); if (vid != null) vid.delete(); }

    static void purgeOld(Context c) {
        long limit = System.currentTimeMillis() - TRASH_DAYS * 86400000L;
        for (File f : list(c, TRASH)) if (f.lastModified() < limit) delete(f);
    }

    /** copy into the phone gallery (Pictures/Anima) */
    static Uri export(Context c, File png) throws IOException {
        android.net.Uri tree = Settings.pictureDir(c);
        if (tree != null) { Uri u = exportToTree(c, png, tree); if (u != null) return u; }   // the folder is gone (permission revoked): the phone gallery gets it instead
        ContentValues cv = new ContentValues();
        JSONObject m = meta(png); File vid = videoOf(png);
        if (vid != null) {   // the video goes to Movies/Anima
            cv.put(MediaStore.Video.Media.DISPLAY_NAME, "anima_" + m.optInt("w") + "x" + m.optInt("h") + "_" + m.optString("seed", "0") + "_" + vid.getName()); cv.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4"); cv.put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Anima");
            Uri vu = c.getContentResolver().insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, cv);
            try (OutputStream o = c.getContentResolver().openOutputStream(vu); InputStream in = new FileInputStream(vid)) { byte[] b = new byte[1 << 16]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n); }
            return vu;
        }
        cv.put(MediaStore.Images.Media.DISPLAY_NAME, "anima_" + m.optInt("w") + "x" + m.optInt("h") + "_" + m.optString("seed", "0") + "_" + png.getName());
        cv.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        cv.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Anima");
        Uri u = c.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
        try (OutputStream o = c.getContentResolver().openOutputStream(u); InputStream in = new FileInputStream(png)) {
            byte[] b = new byte[1 << 16]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n);
        }
        return u;
    }

    /** " (text 3 · DiT 62 · decode 8 s)" from the stage times kept in the metadata ("st"), or "" */
    static String stageText(JSONObject m) {
        JSONObject st = m.optJSONObject("st"); if (st == null) return "";
        return L.f("  (текст %.1f · DiT %.1f · декод. %.1f с)", "  (text %.1f · DiT %.1f · decode %.1f s)", st.optDouble("t"), st.optDouble("d"), st.optDouble("v"));
    }

    /** where the Download button puts a picture now, for the "saved" message */
    static String where(Context c) { return Settings.pictureDir(c) == null ? "Pictures/Anima" : Settings.pictureDirLabel(c); }
    static String savedMsg(Context c) { return L.f("Сохранено: %s", "Saved: %s", where(c)); }

    private static Uri exportToTree(Context c, File png, android.net.Uri tree) {
        try {
            JSONObject m = meta(png); File vid = videoOf(png); File src = vid != null ? vid : png;
            String name = "anima_" + m.optInt("w") + "x" + m.optInt("h") + "_" + m.optString("seed", "0") + "_" + src.getName();
            Uri top = android.provider.DocumentsContract.buildDocumentUriUsingTree(tree, android.provider.DocumentsContract.getTreeDocumentId(tree));
            Uri u = android.provider.DocumentsContract.createDocument(c.getContentResolver(), top, vid != null ? "video/mp4" : "image/png", name);
            if (u == null) return null;
            try (OutputStream o = c.getContentResolver().openOutputStream(u); InputStream in = new FileInputStream(src)) { byte[] b = new byte[1 << 16]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n); }
            return u;
        } catch (Exception e) { return null; }
    }

    interface Pack { boolean on(int done, int total); }   // false = stop

    /** every picture into one ZIP in Downloads (PNG is already compressed, so ZIP costs almost nothing and opens everywhere); null when stopped. The metadata stays in each PNG. */
    static String zipName() { return "HexAnima_" + new java.text.SimpleDateFormat("yyyyMMdd_HHmm", java.util.Locale.US).format(new java.util.Date()); }
    /** the archive goes to the folder picked in Settings (tree = its URI) or, by default, to Downloads; name = file name with or without ".zip" */
    static Uri packZip(Context c, File[] pngs, String name, Uri tree, Pack cb) throws IOException {
        name = name.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_").trim(); if (name.isEmpty()) name = zipName(); if (!name.toLowerCase(java.util.Locale.ROOT).endsWith(".zip")) name += ".zip";
        Uri u;
        if (tree != null) {
            try { u = android.provider.DocumentsContract.createDocument(c.getContentResolver(), android.provider.DocumentsContract.buildDocumentUriUsingTree(tree, android.provider.DocumentsContract.getTreeDocumentId(tree)), "application/zip", name); }
            catch (Exception e) { throw new IOException(e.getMessage()); }
            if (u == null) throw new IOException("cannot create the file");
        } else {
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.Downloads.DISPLAY_NAME, name);
            cv.put(MediaStore.Downloads.MIME_TYPE, "application/zip");
            cv.put(MediaStore.Downloads.RELATIVE_PATH, "Download");
            cv.put(MediaStore.Downloads.IS_PENDING, 1);
            u = c.getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv);
        }
        final boolean viaTree = tree != null;
        boolean ok = false;
        try {
            try (java.util.zip.ZipOutputStream z = new java.util.zip.ZipOutputStream(new BufferedOutputStream(c.getContentResolver().openOutputStream(u), 1 << 16))) {
                java.util.HashSet<String> used = new java.util.HashSet<>(); byte[] b = new byte[1 << 16];
                for (int i = 0; i < pngs.length; i++) {
                    if (cb != null && !cb.on(i, pngs.length)) return null;
                    JSONObject m = meta(pngs[i]); String nm = "anima_" + m.optInt("w") + "x" + m.optInt("h") + "_" + m.optString("seed", "0") + "_" + pngs[i].getName();
                    if (!used.add(nm)) continue;
                    java.util.zip.ZipEntry e = new java.util.zip.ZipEntry(nm); e.setTime(pngs[i].lastModified()); z.putNextEntry(e);
                    try (InputStream in = new FileInputStream(pngs[i])) { for (int n; (n = in.read(b)) > 0; ) z.write(b, 0, n); }
                    z.closeEntry();
                }
            }
            if (!viaTree) { ContentValues done = new ContentValues(); done.put(MediaStore.Downloads.IS_PENDING, 0); c.getContentResolver().update(u, done, null, null); }
            ok = true; return u;
        } finally { if (!ok) try { if (viaTree) android.provider.DocumentsContract.deleteDocument(c.getContentResolver(), u); else c.getContentResolver().delete(u, null, null); } catch (Exception ignored) { } }   // stopped or failed: no half-written archive stays
    }

    static Uri uri(File png) { return Uri.parse("content://" + AUTH + "/" + png.getName()); }

    static void share(Context c, List<File> pngs) {
        ArrayList<Uri> us = new ArrayList<>();
        boolean allVideo = pngs.size() == 1 && videoOf(pngs.get(0)) != null;
        for (File f : pngs) us.add(allVideo ? uri(videoOf(f)) : uri(f));
        Intent s = new Intent(us.size() == 1 ? Intent.ACTION_SEND : Intent.ACTION_SEND_MULTIPLE).setType(allVideo ? "video/mp4" : "image/png").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        if (us.size() == 1) s.putExtra(Intent.EXTRA_STREAM, us.get(0)); else s.putParcelableArrayListExtra(Intent.EXTRA_STREAM, us);
        ClipData cd = ClipData.newRawUri("", us.get(0));
        for (int i = 1; i < us.size(); i++) cd.addItem(new ClipData.Item(us.get(i)));
        s.setClipData(cd);
        c.startActivity(Intent.createChooser(s, L.s("Поделиться", "Share")).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION));
    }
    static void share(Context c, File png) { share(c, java.util.Collections.singletonList(png)); }

    static void view(Context c, File png) {
        File vid = videoOf(png); Uri u = vid != null ? uri(vid) : uri(png);
        Intent v = new Intent(Intent.ACTION_VIEW).setDataAndType(u, vid != null ? "video/mp4" : "image/png").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        v.setClipData(ClipData.newRawUri("", u));
        c.startActivity(v);
    }
}
