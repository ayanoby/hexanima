package com.hexanima.npu;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import java.io.*;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONArray;
import org.json.JSONObject;

/** Export / import of an installed model to a folder the user picks (system folder picker: internal storage, SD card, USB drive).
 *  The folder gets the model's files exactly as they are in files/ plus hexanima-model.json (relative paths, sizes, sha256; written last, so an unfinished export is refused).
 *  The weights are not run from there: the NPU streams them from the phone's own flash. This is a backup that comes back without a download. */
final class ModelIO {
    static final String MANIFEST = "hexanima-model.json";
    private static final String[] ROOTS = {"anima/", "sd/", "h3/", "r2v/", "ltx/", "klein/", "moss/", "qwen/", "mage/"};

    /** the 4 step Qwen / MOSS add-ons share the folder of the base pack: they are exported with it, not alone */
    static boolean supported(String id) { return !(Qwen.is8(id) || Moss.is8(id) || Moss.isRs(id)); }

    private static boolean skip(File f) {
        String n = f.getName();
        if (f.isDirectory()) return n.equals("dsp") || n.startsWith("cache") || n.equals("lcache") || n.equals("tmp");
        return n.endsWith(".part") || n.endsWith(".log") || n.equals(".installed") || n.equals("feat.f32") || n.equals("z.f32") || n.equals("ids.i32") || n.equals("ids_t.i32");
    }
    private static void walk(File home, File f, List<String> out) {
        if (skip(f)) return;
        if (f.isDirectory()) { File[] c = f.listFiles(); if (c != null) { java.util.Arrays.sort(c); for (File x : c) walk(home, x, out); } }
        else out.add(f.getPath().substring(home.getPath().length() + 1));
    }
    /** paths (relative to files/) that make up the model; Ref2VA brings the MiniMax-H3 text encoder it needs */
    static List<String> files(Models st, String id) {
        List<String> out = new ArrayList<>();
        walk(st.home, st.modelDir(id), out);
        if (R2v.is(id)) {
            File hd = H3.dir(st.home); File[] fs = hd.listFiles();
            if (fs != null) for (File f : fs) if (f.getName().equals("te") || f.getName().startsWith("tok_") || f.getName().equals("LICENSE") || f.getName().equals("NOTICE")) walk(st.home, f, out);
        }
        return out;
    }
    static long bytes(Models st, List<String> files) { long t = 0; for (String p : files) t += new File(st.home, p).length(); return t; }

    // ---- export ---------------------------------------------------------------------------------
    static void export(Context c, Models st, String id, String name, Uri tree, Models.Progress pg, AtomicBoolean cancel) throws Exception {
        ContentResolver cr = c.getContentResolver();
        Uri top = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree));
        List<String> files = files(st, id);
        if (files.isEmpty()) throw new IOException(L.s("Нет файлов модели", "The model has no files"));
        long total = bytes(st, files), done = 0;
        Uri root = DocumentsContract.createDocument(cr, top, DocumentsContract.Document.MIME_TYPE_DIR, "HexAnima-" + id.replaceAll("[^A-Za-z0-9._-]", "_"));
        if (root == null) throw new IOException(L.s("Не удалось создать папку", "Could not create the folder"));
        try {
            Map<String, Uri> dirs = new HashMap<>(); dirs.put("", root);
            JSONArray fa = new JSONArray(); byte[] buf = new byte[1 << 20];
            for (String p : files) {
                File src = new File(st.home, p);
                Uri parent = dir(cr, dirs, root, p.contains("/") ? p.substring(0, p.lastIndexOf('/')) : "");
                Uri dst = DocumentsContract.createDocument(cr, parent, "application/octet-stream", p.substring(p.lastIndexOf('/') + 1));
                if (dst == null) throw new IOException(L.s("Не удалось создать файл ", "Could not create the file ") + p);
                MessageDigest md = MessageDigest.getInstance("SHA-256"); long n = 0;
                try (InputStream in = new FileInputStream(src); OutputStream o = cr.openOutputStream(dst)) {
                    for (int r; (r = in.read(buf)) > 0; ) {
                        if (cancel.get()) throw new InterruptedIOException();
                        o.write(buf, 0, r); md.update(buf, 0, r); n += r; done += r; pg.on(done, total, p);
                    }
                }
                fa.put(new JSONObject().put("p", p).put("size", n).put("sha", hex(md.digest())));
            }
            JSONObject m = new JSONObject().put("format", 1).put("id", id).put("name", name).put("files", fa);
            if (st.modelDir(id).getPath().startsWith(st.anima.getPath() + "/m_")) {   // an Anima model: the sparse safetensors stub that sd.cpp opens is rebuilt from hdr.bin
                File stub = new File(st.sd, id + ".safetensors");
                if (stub.exists()) m.put("stub", new JSONObject().put("hdr", "anima/m_" + id + "/hdr.bin").put("out", "sd/" + id + ".safetensors").put("size", stub.length()));
            }
            Uri mf = DocumentsContract.createDocument(cr, root, "application/json", MANIFEST);
            if (mf == null) throw new IOException(L.s("Не удалось записать описание модели", "Could not write the model description"));
            try (OutputStream o = cr.openOutputStream(mf)) { o.write(m.toString(1).getBytes("UTF-8")); }
        } catch (Exception e) {
            try { DocumentsContract.deleteDocument(cr, root); } catch (Exception ignored) { }   // no half-finished copy is left behind
            throw e;
        }
    }

    private static Uri dir(ContentResolver cr, Map<String, Uri> dirs, Uri root, String rel) throws IOException {
        Uri u = dirs.get(rel); if (u != null) return u;
        Uri parent = dir(cr, dirs, root, rel.contains("/") ? rel.substring(0, rel.lastIndexOf('/')) : "");
        u = DocumentsContract.createDocument(cr, parent, DocumentsContract.Document.MIME_TYPE_DIR, rel.substring(rel.lastIndexOf('/') + 1));
        if (u == null) throw new IOException(L.s("Не удалось создать папку ", "Could not create the folder ") + rel);
        dirs.put(rel, u); return u;
    }

    // ---- import ---------------------------------------------------------------------------------
    /** @param tree the folder that holds hexanima-model.json (the one the export made); returns the model's name */
    static String importFrom(Context c, Models st, Uri tree, Models.Progress pg, AtomicBoolean cancel) throws Exception {
        ContentResolver cr = c.getContentResolver();
        Map<String, Uri> docs = new HashMap<>();
        list(cr, tree, DocumentsContract.getTreeDocumentId(tree), "", docs);
        Uri mf = docs.get(MANIFEST);
        if (mf == null) throw new IOException(L.s("В этой папке нет hexanima-model.json: выберите папку, которую создал экспорт", "This folder has no hexanima-model.json: pick the folder an export made"));
        JSONObject m; try (InputStream in = cr.openInputStream(mf)) { m = new JSONObject(new String(readAll(in), "UTF-8")); }
        if (m.optInt("format") != 1) throw new IOException(L.s("Неизвестный формат экспорта", "Unknown export format"));
        String id = m.getString("id"), name = m.optString("name", id); JSONArray fa = m.getJSONArray("files");
        long total = 0; for (int i = 0; i < fa.length(); i++) total += fa.getJSONObject(i).getLong("size");
        if (st.home.getUsableSpace() < total + 300_000_000L) throw new IOException(L.f("Не хватает места: нужно %s ГБ", "Not enough space: need %s GB", String.format("%.1f", total / 1e9)));
        String home = st.home.getCanonicalPath() + "/"; byte[] buf = new byte[1 << 20]; long done = 0;
        for (int i = 0; i < fa.length(); i++) {
            JSONObject f = fa.getJSONObject(i); String p = f.getString("p"); long size = f.getLong("size"); String sha = f.getString("sha");
            boolean ok = false; for (String r : ROOTS) if (p.startsWith(r)) ok = true;
            File dst = new File(st.home, p);
            if (!ok || p.contains("..") || !dst.getCanonicalPath().startsWith(home)) throw new IOException("bad path " + p);   // a hand-edited description must not write outside files/
            Uri src = docs.get(p); if (src == null) throw new IOException(L.s("В папке нет файла ", "The folder has no file ") + p);
            if (dst.length() == size && sha.equals(Models.sha256(dst))) { done += size; pg.on(done, total, p); continue; }   // already there (a retried import)
            dst.getParentFile().mkdirs(); File part = new File(dst.getPath() + ".part");
            MessageDigest md = MessageDigest.getInstance("SHA-256"); long n = 0;
            try (InputStream in = cr.openInputStream(src); OutputStream o = new FileOutputStream(part)) {
                for (int r; (r = in.read(buf)) > 0; ) {
                    if (cancel.get()) { o.close(); part.delete(); throw new InterruptedIOException(); }
                    o.write(buf, 0, r); md.update(buf, 0, r); n += r; done += r; pg.on(done, total, p);
                }
            }
            if (n != size || !hex(md.digest()).equalsIgnoreCase(sha)) { part.delete(); throw new IOException(L.s("Файл повреждён: ", "The file is damaged: ") + p); }
            dst.delete(); if (!part.renameTo(dst)) throw new IOException("rename " + p);
        }
        JSONObject sb = m.optJSONObject("stub");
        if (sb != null) Models.stub(new File(st.home, sb.getString("hdr")), new File(st.home, sb.getString("out")), sb.getLong("size"));
        File dir = st.modelDir(id); dir.mkdirs(); st.doneMark(dir).createNewFile();
        return name;
    }

    private static void list(ContentResolver cr, Uri tree, String docId, String rel, Map<String, Uri> out) {
        Uri kids = DocumentsContract.buildChildDocumentsUriUsingTree(tree, docId);
        try (Cursor q = cr.query(kids, new String[]{DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME, DocumentsContract.Document.COLUMN_MIME_TYPE}, null, null, null)) {
            if (q == null) return;
            while (q.moveToNext()) {
                String did = q.getString(0), n = q.getString(1), path = rel.isEmpty() ? n : rel + "/" + n;
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(q.getString(2))) list(cr, tree, did, path, out);
                else out.put(path, DocumentsContract.buildDocumentUriUsingTree(tree, did));
            }
        }
    }

    private static byte[] readAll(InputStream in) throws IOException { ByteArrayOutputStream o = new ByteArrayOutputStream(); byte[] b = new byte[8192]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n); return o.toByteArray(); }
    private static String hex(byte[] d) { StringBuilder s = new StringBuilder(); for (byte b : d) s.append(String.format("%02x", b)); return s.toString(); }
}
