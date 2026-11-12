package com.hexanima.npu;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Model packs: a catalog (catalog.json) lists models; every model and the shared "common" pack have a manifest with raw files (size + sha256) that are
 * downloaded one by one (resumable, verified). Nothing is zipped: the weights do not compress and per-file resume/verification is simpler.
 * The multi-GB safetensors files that sd.cpp opens are NOT downloaded: only their JSON header (hdr.bin) is, and the app rebuilds a sparse stub of the real size.
 * Layout: files/anima/{te*.bin, te_embed.f16, vae.bin} (common), files/anima/m_ID/{small.bin, blkNN.bin}, files/sd/{ID.safetensors stub, llm stub, taew2_1.safetensors}.
 */
final class Models {
    static class Item { String id, name, desc, descEn, params = "", quant = "", author = "", manifest; long size; int speed; boolean moss, qwen, mage; }   // speed: seconds per picture at 512x768
    interface Progress { void on(long done, long total, String what); }

    final File anima, sd, home;
    Models(File home) { this.home = home; anima = new File(home, "anima"); sd = new File(home, "sd"); anima.mkdirs(); sd.mkdirs(); }

    static String get(String url) throws IOException {
        HttpURLConnection c = open(url, 0);
        try (InputStream in = c.getInputStream(); ByteArrayOutputStream bo = new ByteArrayOutputStream()) {
            byte[] b = new byte[8192]; for (int n; (n = in.read(b)) > 0; ) bo.write(b, 0, n);
            return bo.toString("UTF-8");
        } finally { c.disconnect(); }
    }

    static HttpURLConnection open(String url, long from) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(20000); c.setReadTimeout(30000); c.setInstanceFollowRedirects(true);
        if (from > 0) c.setRequestProperty("Range", "bytes=" + from + "-");
        return c;
    }

    long commonSize;
    List<Item> catalog(String base) throws Exception {
        String txt;   // catalog3.json adds Mage-Flow (1.4.1+); catalog2.json lists the Anima models, MOSS-Image and Qwen-Image (1.3+); catalog.json stays as is for older apps, which would misread such entries
        try { return catalog5(get(base + "catalog7.json"), base); } catch (Exception e7) { }   // catalog7.json = catalog6 + the common files with the 512 token text encoder (1.5.4+; older apps expect 128 tokens)
        try { return catalog5(get(base + "catalog6.json"), base); } catch (Exception e6) { }   // catalog6.json = catalog5 + the MiaoMiao MOSS fine-tunes (older apps do not read it)
        try { return catalog5(get(base + "catalog5.json"), base); } catch (Exception e5) { }   // catalog5.json: packs + models with switchable modes (flattened here to one Item per mode)
        try { txt = get(base + "catalog4.json"); } catch (Exception e0) { try { txt = get(base + "catalog3.json"); } catch (Exception e) { try { txt = get(base + "catalog2.json"); } catch (Exception e2) { txt = get(base + "catalog.json"); } } }
        JSONObject j = new JSONObject(txt);
        commonSize = j.optLong("common_size", 0);
        List<Item> out = new ArrayList<>();
        JSONArray a = j.getJSONArray("models");
        for (int i = 0; i < a.length(); i++) {
            JSONObject m = a.getJSONObject(i); Item it = new Item();
            it.id = m.getString("id"); it.name = niceName(it.id, m.getString("name")); it.desc = m.optString("desc", ""); it.size = m.getLong("size"); it.manifest = m.getString("manifest");
            it.descEn = m.optString("desc_en", it.desc); it.params = m.optString("params", ""); it.quant = m.optString("quant", ""); it.author = m.optString("author", ""); it.speed = m.optInt("speed", 0); it.moss = m.optString("kind", "").equals("moss"); it.qwen = m.optString("kind", "").equals("qwen"); it.mage = m.optString("kind", "").equals("mage");
            out.add(it);
        }
        return out;
    }

    /** catalog5: {"packs": {id: {url | path, manifest, size, needs, arch}}, "models": [{family, line, name, desc, modes: [{id, label, name, pack, steps, speed}]}]} */
    String commonUrl;   // directory URL of the shared Anima pack for this phone's DSP arch (null: base + commonDir())
    List<Item> catalog5(String txt, String base) throws Exception {
        JSONObject j = new JSONObject(txt), packs = j.getJSONObject("packs");
        String arch = Compat.dspArch().isEmpty() ? "v69" : Compat.dspArch();
        JSONObject cp = packs.optJSONObject("anima-common@" + arch); if (cp != null) commonUrl = packUrl(cp, base);
        List<Item> out = new ArrayList<>();
        JSONArray ms = j.getJSONArray("models");
        for (int i = 0; i < ms.length(); i++) {
            JSONObject m = ms.getJSONObject(i); JSONArray modes = m.getJSONArray("modes");
            for (int k = 0; k < modes.length(); k++) {
                JSONObject md = modes.getJSONObject(k), p = packs.getJSONObject(md.getString("pack")); Item it = new Item();
                it.id = md.getString("id"); it.name = niceName(it.id, md.optString("name", m.getString("name"))); it.size = p.optLong("size", 0);
                it.manifest = packUrl(p, base) + p.optString("manifest", "manifest.json");
                it.desc = m.optString("desc", ""); it.descEn = m.optString("desc_en", it.desc); it.params = m.optString("params", ""); it.author = m.optString("author", "");
                it.speed = md.optInt("speed", 0); String f = m.optString("family", "Anima");
                it.moss = f.equals("MOSS-Image"); it.qwen = f.equals("Qwen-Image"); it.mage = f.equals("Mage-Flow");
                out.add(it);
            }
        }
        return out;
    }
    static String packUrl(JSONObject p, String base) { String u = p.has("url") ? p.optString("url") : base + p.optString("path"); return u.endsWith("/") ? u : u + "/"; }

    // ---- state ----------------------------------------------------------------------------------
    File modelDir(String id) { return H3.is(id) ? H3.dir(home) : Moss.is(id) ? Moss.dir(home) : Mage.is(id) ? Mage.dir(home) : Qwen.is(id) ? Qwen.dir(home) : new File(anima, "m_" + id); }
    File doneMark(File dir) { return new File(dir, ".installed"); }
    boolean modelInstalled(String id) { if (H3.is(id)) return H3.installed(home); if (Moss.is(id)) return Moss.installed(home) && (Moss.is8(id) ? Moss.installed8(home) : Moss.isRs(id) ? Moss.installedRs(home, id) : id.equals(mossId())); if (Qwen.is(id)) return Qwen.is8(id) ? Qwen.installed8(home) : Qwen.installed(home) && id.equals(qwenId()); if (Mage.is(id)) return Mage.installed(home) && id.equals(mageId()); return commonInstalled() && doneMark(modelDir(id)).exists() && new File(sd, id + ".safetensors").exists(); }
    /** v69 keeps the original "common" folder (older apps use it); other Hexagon generations have their own QNN contexts in common_<arch> */
    static String commonDir() { String a = Compat.dspArch(); return a.isEmpty() || a.equals("v69") ? "common" : "common_" + a; }
    boolean commonInstalled() { return doneMark(new File(anima, commonDir())).exists(); }
    String qwenId() { try { return new JSONObject(read(new File(Qwen.dir(home), "meta.json"))).getString("id"); } catch (Exception e) { return ""; } }
    String mageId() { try { return new JSONObject(read(new File(Mage.dir(home), "meta.json"))).getString("id"); } catch (Exception e) { return ""; } }
    String mossId() { try { return new JSONObject(read(new File(Moss.dir(home), "meta.json"))).getString("id"); } catch (Exception e) { return ""; } }
    /** ids of installed models, in catalog order of installation directory names */
    List<String> installed() {
        List<String> out = new ArrayList<>();
        if (Moss.installed(home)) { out.add(mossId()); if (Moss.installed8(home)) out.add(Moss.ID8); if (Moss.installedRs(home, Moss.IDRS)) out.add(Moss.IDRS); if (Moss.installedRs(home, Moss.IDHM)) out.add(Moss.IDHM); }
        if (Qwen.installed(home)) out.add(qwenId()); if (Qwen.installed8(home)) out.add(Qwen.ID8);
        if (Mage.installed(home)) out.add(mageId());
        if (H3.installed(home)) out.add(H3.ID);
        File[] ds = anima.listFiles();
        if (ds != null) for (File d : ds) if (d.getName().startsWith("m_") && modelInstalled(d.getName().substring(2))) out.add(d.getName().substring(2));
        java.util.Collections.sort(out);
        return out;
    }
    /** the name shown in the app: the kind of the model is part of it (the catalog and meta.json keep the plain model names) */
    static String niceName(String id, String name) {
        switch (id) {
            case "harem2b": case "real2b": case "harem29": case "real29": return name.endsWith(" Anima Turbo 4step") ? name.substring(0, name.length() - " Anima Turbo 4step".length()) : name;   // turbo and base are told apart by the group in the picker
            case "qwen-image-21-turbo": return "Qwen-Image 2.1 Turbo 4 step";
            case Qwen.ID8: return "Qwen-Image-2.1-Turbo";
            case Moss.ID8: return "MOSS-Image 8 step";
            case Moss.IDRS: return "MiaoMiao RealSkin · MOSS-Image 1.0";
            case Moss.IDHM: return "MiaoMiao Harem · MOSS-Image 1.1";
            case H3.ID: return "MiniMax-H3-Image";
            default: return name;
        }
    }
    // ---- names given by the user (a pencil in the model picker) -------------------------------------------------
    /** the key an alias is stored under: both variants of an Anima model share one (the turbo id and id + "base") */
    static String aliasKey(String id) { return Qwen.is(id) ? "qwen" : Moss.is(id) ? (Moss.isRs(id) ? id : "moss") : H3.is(id) || Mage.is(id) ? id : (id.endsWith("base") ? id.substring(0, id.length() - 4) : id); }
    File aliasFile() { return new File(home, "model_names.json"); }
    String alias(String key) { try { String v = new JSONObject(read(aliasFile())).optString(key, ""); return v.isEmpty() ? null : v; } catch (Exception e) { return null; } }
    void setAlias(String key, String name) {
        JSONObject j; try { j = new JSONObject(read(aliasFile())); } catch (Exception e) { j = new JSONObject(); }
        try { if (name == null || name.trim().isEmpty()) j.remove(key); else j.put(key, name.trim()); try (FileWriter w = new FileWriter(aliasFile())) { w.write(j.toString()); } } catch (Exception ignored) { }
    }
    String name(String id) { String al = alias(aliasKey(id)); if (al != null && !Qwen.is8(id) && !Moss.is8(id)) return al + (!Qwen.is(id) && !Moss.is(id) && !H3.is(id) && !Mage.is(id) && id.endsWith("base") ? " (base)" : ""); if (H3.is(id)) return niceName(id, ""); if (Qwen.is8(id) || Moss.is8(id) || Moss.isRs(id)) return niceName(id, ""); try { return niceName(id, new JSONObject(read(new File(modelDir(id), "meta.json"))).getString("name")); } catch (Exception e) { return id; } }

    /** seconds per picture at 512x768 (from the catalog at install time; models installed before that get the measured value of their size class) */
    int speed(String id) { if (H3.is(id)) return 110; if (Moss.isRs(id)) return speed(mossId()); if (Qwen.is8(id)) return 80; if (Moss.is8(id)) return 3 * speed(mossId()) / 2; try { int v = new JSONObject(read(new File(modelDir(id), "meta.json"))).optInt("speed", 0); if (v > 0) return v; } catch (Exception e) { } return id.contains("29") ? 12 : 7; }

    static byte[] readBytes(File f) throws IOException {
        try (FileInputStream in = new FileInputStream(f)) { byte[] b = new byte[(int) f.length()]; int o = 0; for (int n; o < b.length && (n = in.read(b, o, b.length - o)) > 0; ) o += n; return b; }
    }
    static String read(File f) throws IOException {
        try (FileInputStream in = new FileInputStream(f)) { byte[] b = new byte[(int) f.length()]; int o = 0; for (int n; o < b.length && (n = in.read(b, o, b.length - o)) > 0; ) o += n; return new String(b, "UTF-8"); }
    }

    // ---- install --------------------------------------------------------------------------------
    void installCommon(String base, Progress pg, AtomicBoolean cancel) throws Exception {
        if (commonInstalled()) return;
        final String cu = commonUrl != null ? commonUrl : base + commonDir() + "/";
        JSONObject m = new JSONObject(get(cu + "manifest.json"));
        File marker = new File(anima, commonDir()); marker.mkdirs();
        long total = total(m.getJSONArray("files")) + total(m.getJSONArray("sd"));
        long[] done = {0};
        fetchAll(cu, m.getJSONArray("files"), anima, total, done, pg, cancel);
        fetchAll(cu, m.getJSONArray("sd"), sd, total, done, pg, cancel);
        JSONObject llm = m.getJSONObject("llm_stub");
        File hdr = new File(marker, "llm_hdr.bin");
        fetch(cu + llm.getString("hdr"), hdr, llm.optLong("hdr_size", -1), llm.optString("hdr_sha256", null), null, cancel);
        stub(hdr, new File(sd, llm.getString("name")), llm.getLong("size"));
        doneMark(marker).createNewFile();
    }

    void installModel(String base, Item it, Progress pg, AtomicBoolean cancel) throws Exception {
        final String murl = it.manifest.startsWith("http") ? it.manifest : base + it.manifest;   // a pack may live in another repository (MOSS-Image: Apache-2.0, not in the Anima repo)
        JSONObject m = new JSONObject(get(murl));
        File dir = modelDir(it.id); dir.mkdirs();
        JSONArray files = m.getJSONArray("files");
        long[] done = {0};
        if (Qwen.is8(it.id)) {   // the plain Turbo: shared files + dit8 (files already there, e.g. from an old 4 step pack, are kept); no meta.json / marker, those belong to the 4 step pack
            fetchAll(murl.substring(0, murl.lastIndexOf('/') + 1), files, dir, total(files), done, pg, cancel); return;
        }
        if (Moss.isRs(it.id)) {   // RealSkin: dit/ref/glue into files/moss/rs, the shared files come from the installed MOSS-Image
            if (!Moss.installed(home)) throw new IOException(L.s("Сначала установите MOSS-Image", "Install MOSS-Image first"));
            File rs = new File(dir, Moss.sub(it.id)); rs.mkdirs(); fetchAll(murl.substring(0, murl.lastIndexOf('/') + 1), files, rs, total(files), done, pg, cancel); return;
        }
        if (Moss.is8(it.id)) {   // an add-on to the installed pack: its files only, the default pack's meta and marker stay
            if (!Moss.installed(home)) throw new IOException(L.s("Сначала установите MOSS-Image", "Install MOSS-Image first"));
            fetchAll(murl.substring(0, murl.lastIndexOf('/') + 1), files, dir, total(files), done, pg, cancel); return;
        }
        doneMark(dir).delete();
        fetchAll(murl.substring(0, murl.lastIndexOf('/') + 1), files, dir, total(files), done, pg, cancel);
        if (it.moss || it.qwen || it.mage) {   // MOSS-Image / Qwen-Image: raw files only, no sd.cpp stub
            try (FileWriter w = new FileWriter(new File(dir, "meta.json"))) { w.write(new JSONObject().put("id", it.id).put("name", it.name).put("speed", it.speed).toString()); }
            doneMark(dir).createNewFile(); return;
        }
        JSONObject st = m.getJSONObject("stub");
        File hdr = new File(dir, st.getString("hdr"));
        stub(hdr, new File(sd, it.id + ".safetensors"), st.getLong("size"));
        try (FileWriter w = new FileWriter(new File(dir, "meta.json"))) { w.write(new JSONObject().put("id", it.id).put("name", it.name).put("speed", it.speed).toString()); }
        doneMark(dir).createNewFile();
    }

    /** installed models take the name / speed of the current catalog (names are shown from meta.json) */
    void refreshMeta(Item it) {
        if (Qwen.is8(it.id) || Moss.is8(it.id) || Moss.isRs(it.id) || !modelInstalled(it.id)) return;   // the 8-step add-on shares the folder (and meta.json) of the default Qwen pack
        try (FileWriter w = new FileWriter(new File(modelDir(it.id), "meta.json"))) { w.write(new JSONObject().put("id", it.id).put("name", it.name).put("speed", it.speed).toString()); } catch (Exception ignored) { }
    }

    void delete(String id) {
        File dir = modelDir(id);
        if (Qwen.is8(id) && !new File(dir, "dit/blk31.bin").exists()) { deleteTree(dir); return; }   // no old 4 step pack: the plain Turbo owns the whole folder
        if (Qwen.is8(id)) { deleteTree(new File(dir, "dit8")); new File(dir, "glue8.bin").delete(); deleteTree(new File(dir, "v8")); deleteTree(new File(dir, "cache8")); return; }   // only the extra pack; the default Qwen stays
        if (Moss.isRs(id)) { deleteTree(new File(dir, Moss.sub(id))); deleteTree(new File(dir, "v" + Moss.sub(id))); return; }
        if (Moss.is8(id)) { deleteTree(new File(dir, "lora8")); new File(dir, "glue8.bin").delete(); return; }
        if (Moss.is(id) || Qwen.is(id) || Mage.is(id) || H3.is(id)) { deleteTree(dir); return; }
        File[] fs = dir.listFiles(); if (fs != null) for (File f : fs) f.delete();
        dir.delete(); new File(sd, id + ".safetensors").delete();
    }

    static final String CUSTOM = ".custom";   // marker in m_ID of a checkpoint converted on the phone
    java.util.List<String> customIds() {
        java.util.List<String> out = new java.util.ArrayList<>(); File[] ds = anima.listFiles();
        if (ds != null) for (File d : ds) if (d.getName().startsWith("m_") && (new File(d, CUSTOM).exists() || d.getName().startsWith("m_my")) && doneMark(d).exists()) out.add(d.getName().substring(2));   // models converted by an earlier build have no marker: their ids start with "my"
        java.util.Collections.sort(out); return out;
    }
    /** the shared Anima files (text encoder, VAE, embeddings, TAESD) that every Anima model needs; 0 while they are not installed */
    long commonBytes() {
        if (!commonInstalled()) return 0; long n = 0; File[] fs = anima.listFiles();
        if (fs != null) for (File f : fs) if (f.isFile() && (f.getName().matches("te\\d+\\.bin") || f.getName().equals("te_embed.f16") || f.getName().equals("vae.bin"))) n += f.length();
        return n + new File(sd, "taew2_1.safetensors").length();
    }
    long dirSize(File d) { long n = 0; File[] fs = d.listFiles(); if (fs != null) for (File f : fs) n += f.isDirectory() ? dirSize(f) : f.length(); return n; }

    static void deleteTree(File f) { File[] fs = f.listFiles(); if (fs != null) for (File c : fs) deleteTree(c); f.delete(); }

    /** sparse stub of a safetensors file: the real header, then a hole up to the real size (sd.cpp maps the file but never touches tensor data when the DiT/LLM run on the NPU) */
    static void stub(File hdr, File out, long size) throws IOException {
        try (FileInputStream in = new FileInputStream(hdr); FileOutputStream o = new FileOutputStream(out)) {
            byte[] b = new byte[65536]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n);
        }
        try (RandomAccessFile r = new RandomAccessFile(out, "rw")) { r.setLength(size); }
    }

    static long total(JSONArray a) throws Exception { long t = 0; for (int i = 0; i < a.length(); i++) t += a.getJSONObject(i).getLong("size"); return t; }

    void fetchAll(String urlBase, JSONArray files, File dir, long total, long[] done, Progress pg, AtomicBoolean cancel) throws Exception {
        for (int i = 0; i < files.length(); i++) {
            JSONObject f = files.getJSONObject(i);
            File out = new File(dir, f.getString("path"));
            final long before = done[0];
            fetch(urlBase + f.getString("path"), out, f.getLong("size"), f.optString("sha256", null), n -> { if (pg != null) pg.on(before + n, total, f.optString("path")); }, cancel);
            done[0] += f.getLong("size");
        }
    }

    interface Count { void on(long n); }

    /** download url to out (resumes a .part file), verify size and sha256, rename; a finished file with the right size is kept as is */
    static void fetch(String url, File out, long size, String sha, Count cb, AtomicBoolean cancel) throws Exception {
        if (out.exists() && (size < 0 || out.length() == size)) { if (cb != null) cb.on(out.length()); return; }
        File part = new File(out.getPath() + ".part");
        out.getParentFile().mkdirs();
        for (int attempt = 0; ; attempt++) {
            try {
                long have = part.exists() ? part.length() : 0;
                if (size >= 0 && have > size) { part.delete(); have = 0; }
                if (size < 0 || have < size) {
                    HttpURLConnection c = open(url, have);
                    int code = c.getResponseCode();
                    if (code == 200 && have > 0) { part.delete(); have = 0; }   // server ignored Range
                    else if (code != 200 && code != 206) throw new IOException("HTTP " + code + " " + url);
                    try (InputStream in = c.getInputStream(); FileOutputStream o = new FileOutputStream(part, have > 0)) {
                        byte[] b = new byte[1 << 16]; long n = have;
                        for (int r; (r = in.read(b)) > 0; ) {
                            if (cancel != null && cancel.get()) throw new InterruptedIOException("cancelled");
                            o.write(b, 0, r); n += r; if (cb != null) cb.on(n);
                        }
                    } finally { c.disconnect(); }
                }
                if (size >= 0 && part.length() != size) throw new IOException("size " + part.length() + " != " + size + " " + url);
                if (sha != null && !sha.isEmpty() && !sha256(part).equalsIgnoreCase(sha)) { part.delete(); throw new IOException("sha256 mismatch " + url); }
                if (!part.renameTo(out)) throw new IOException("rename " + out);
                return;
            } catch (InterruptedIOException e) { throw e;
            } catch (IOException e) { if (attempt >= 5) throw e; Thread.sleep(2000L * (attempt + 1)); }   // flaky network: resume from the .part file
        }
    }

    static String sha256(File f) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (FileInputStream in = new FileInputStream(f)) { byte[] b = new byte[1 << 20]; for (int n; (n = in.read(b)) > 0; ) md.update(b, 0, n); }
        StringBuilder sb = new StringBuilder(); for (byte x : md.digest()) sb.append(String.format("%02x", x));
        return sb.toString();
    }
}
