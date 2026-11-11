package com.hexanima.npu;

import java.io.*;

/** Lab: a turbo LoRA picked by the user for Qwen-Image. The file is copied to files/qwen/lora/turbo.safetensors and checked by libqwenfold.so ("check", header only); at generation
 *  "replace" turns it into slot overlays for the plain pack (dit8) + a glue override (host layers), ~2 s, cached by strength in files/qwen/lora/turbo_ovl. */
final class QwenLora {
    static File dir(File home) { return new File(Qwen.dir(home), "lora"); }
    static File turboSrc(File home) { return new File(dir(home), "turbo.safetensors"); }
    static File turboOvl(File home) { return new File(dir(home), "turbo_ovl"); }

    static final class Check {
        boolean ok; String code = "", detail = ""; int layers, mods, rank, host; String warn = "";
        String message() {
            if (ok) return L.f("Подходит: ранг %d, блоков %d из 32, слоёв вне блоков %d", "Compatible: rank %d, %d of 32 blocks, %d host layers", rank, layers, host);
            switch (code) {
                case "OTHERMODEL": return L.s("Это LoRA для другой модели. Для Qwen-Image она не подходит.", "This LoRA is for another model. It does not fit Qwen-Image.");
                case "NOTLORA": return L.s("Файл не похож на LoRA для Qwen-Image: не найдены нужные слои (transformer_blocks).", "This file does not look like a Qwen-Image LoRA: the expected layers (transformer_blocks) are missing.");
                case "SHAPE": return L.f("Размеры слоёв не совпадают с Qwen-Image (%s).", "Layer sizes do not match Qwen-Image (%s).", detail);
                case "RANK": return L.f("Слишком большой ранг (%s). Поддерживается до 128.", "The rank is too large (%s). Up to 128 is supported.", detail);
                case "RANK64": return L.f("Для слоёв gate/proj поддерживается ранг до 64 (%s).", "The gate/proj layers take rank up to 64 (%s).", detail);
                case "LAYER": return L.f("В LoRA есть блоки, которых нет в Qwen-Image (%s).", "The LoRA has blocks that Qwen-Image does not have (%s).", detail);
                case "FILE": return L.f("Не удалось прочитать файл (%s). Нужен .safetensors.", "Cannot read the file (%s). A .safetensors file is needed.", detail);
                default: return L.f("Ошибка проверки: %s %s", "Check failed: %s %s", code, detail);
            }
        }
    }
    static Check parse(String line) {
        Check c = new Check(); String[] a = line.trim().split("\t");
        if (a.length > 0 && a[0].equals("OK")) {
            c.ok = true;
            for (String f : a) {
                if (f.startsWith("layers=")) c.layers = Integer.parseInt(f.substring(7)); else if (f.startsWith("mods=")) c.mods = Integer.parseInt(f.substring(5)); else if (f.startsWith("rank=")) c.rank = Integer.parseInt(f.substring(5));
                else if (f.startsWith("host=")) c.host = Integer.parseInt(f.substring(5)); else if (f.startsWith("warn=")) c.warn = f.substring(5);
            }
        } else { c.code = a.length > 1 ? a[1] : "?"; c.detail = a.length > 2 ? a[2] : ""; }
        return c;
    }
    static String run(String... cmd) throws IOException {
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start(); StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) { for (String l; (l = r.readLine()) != null; ) sb.append(l).append('\n'); }
        try { p.waitFor(); } catch (InterruptedException ignored) { }
        return sb.toString();
    }
    static Check check(String bin, File f) {
        try {
            String last = ""; for (String l : run(bin + "/libqwenfold.so", "check", f.getPath()).split("\n")) if (l.startsWith("OK") || l.startsWith("ERR")) last = l;
            return last.isEmpty() ? parse("ERR\tFILE\t") : parse(last);
        } catch (IOException e) { return parse("ERR\tFILE\t" + e.getMessage()); }
    }
    /** overlays for the picked LoRA at this strength (cached); null on failure (msg[0] = reason) */
    static File ensureTurbo(File home, String bin, float strength, String[] msg) {
        File out = turboOvl(home), ok = new File(out, ".ok"); final String tag = String.format(java.util.Locale.US, "%.2f", strength);
        try { if (ok.exists() && tag.equals(Models.read(ok).trim())) return out; } catch (IOException ignored) { }
        MossLora.deleteTree(out); out.mkdirs();
        try {
            String last = ""; for (String l : run(bin + "/libqwenfold.so", "replace", new File(Qwen.dir(home), "dit8").getPath(), turboSrc(home).getPath(), tag, out.getPath(), new File(Qwen.dir(home), "glue8.bin").getPath()).split("\n")) if (!l.startsWith("progress ") && !l.isEmpty()) last = l;
            if (!last.startsWith("OK")) { msg[0] = last; MossLora.deleteTree(out); return null; }
            try (FileWriter w = new FileWriter(ok)) { w.write(tag); } return out;
        } catch (Exception e) { msg[0] = String.valueOf(e); MossLora.deleteTree(out); return null; }
    }
}
