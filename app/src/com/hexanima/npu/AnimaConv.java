package com.hexanima.npu;

import android.app.AlertDialog;
import android.content.Intent;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.*;
import java.io.*;
import java.io.InterruptedIOException;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicReference;

/** Own Anima checkpoint (.safetensors, bf16 / fp16 / fp32) -> the files of the NPU runner, converted on the phone by libanimaconv.so (same output as export/export_all.py --q8). */
final class AnimaConv {
    static final int REQ = 21;
    final Base a; final Models store; final String baseUrl; final Runnable onDone;
    AnimaConv(Base a, Models store, String baseUrl, Runnable onDone) { this.a = a; this.store = store; this.baseUrl = baseUrl; this.onDone = onDone; }

    static void pick(android.app.Activity a) { a.startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("*/*").addCategory(Intent.CATEGORY_OPENABLE), REQ); }

    File exe() { return new File(a.getApplicationInfo().nativeLibraryDir, "libanimaconv.so"); }

    /** runs the converter, returns all its output lines (stops at the first "error"/"ok"/"done" for `check`) */
    String run(String... args) throws Exception {
        String[] cmd = new String[args.length + 1]; cmd[0] = exe().getPath(); System.arraycopy(args, 0, cmd, 1, args.length);
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) { for (String l; (l = r.readLine()) != null; ) sb.append(l).append('\n'); }
        p.waitFor(); return sb.toString().trim();
    }

    /** a file picked in the document picker: checked at once (header only), then name and kind are asked */
    void start(final Uri uri) {
        new Thread(() -> {
            ParcelFileDescriptor pfd = null; File tmp = null;
            try {
                pfd = a.getContentResolver().openFileDescriptor(uri, "r");
                String src = "/proc/" + android.os.Process.myPid() + "/fd/" + pfd.getFd(), res = run("check", src);
                if (res.startsWith("error cannot open")) {   // the provider does not give a file the converter may open: copy it (needs the space twice)
                    tmp = new File(a.getFilesDir(), "anima/conv_src.tmp"); tmp.getParentFile().mkdirs();
                    final long need = pfd.getStatSize();
                    if (a.getFilesDir().getUsableSpace() < need + 4_000_000_000L) { fail(L.s("Мало места для копии файла", "Not enough space for a copy of the file")); return; }
                    ui(() -> Toast.makeText(a, L.s("Копирую файл…", "Copying the file…"), Toast.LENGTH_LONG).show());
                    try (InputStream in = a.getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) { byte[] b = new byte[1 << 20]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n); }
                    src = tmp.getPath(); res = run("check", src);
                }
                if (!res.startsWith("ok")) { if (tmp != null) tmp.delete(); fail(L.f("Файл не подходит: %s", "The file does not fit: %s", res.replaceFirst("^error ", ""))); return; }
                final int blocks = Integer.parseInt(res.replaceAll(".*blocks=(\\d+).*", "$1"));
                final String srcF = src; final File tmpF = tmp; final ParcelFileDescriptor pf = pfd;
                final String dn = name(uri);
                ui(() -> common(() -> ask(srcF, tmpF, pf, blocks, dn), () -> cleanup(tmpF, pf)));
                pfd = null;   // kept open until the conversion ends
            } catch (Exception e) { if (tmp != null) tmp.delete(); fail(String.valueOf(e.getMessage())); }
            finally { if (pfd != null) try { pfd.close(); } catch (IOException ignored) { } }
        }).start();
    }

    String name(Uri uri) {
        String n = uri.getLastPathSegment(); if (n == null) return "custom";
        n = n.substring(n.lastIndexOf('/') + 1).replaceAll("(?i)\\.safetensors$", ""); return n.isEmpty() ? "custom" : n;
    }
    void ui(Runnable r) { a.runOnUiThread(r); }
    void fail(String m) { ui(() -> new AlertDialog.Builder(a).setMessage(m).setPositiveButton("OK", null).show()); }

    void ask(final String src, final File tmp, final ParcelFileDescriptor pfd, final int blocks, String defName) {
        final long need = blocks * 83_365_888L + 310_000_000L;
        final EditText e = new EditText(a); e.setText(defName); e.setSingleLine(); e.setSelectAllOnFocus(true); e.setInputType(InputType.TYPE_CLASS_TEXT);
        final String[] kinds = {L.s("Обычная (20 шагов, CFG и негативный промпт; Turbo-LoRA можно добавить отдельным файлом)", "Regular (20 steps, CFG and negative prompt; a Turbo LoRA can be added as a file)"), L.s("Уже ускоренная Turbo (4 шага, без CFG)", "Already distilled Turbo (4 steps, no CFG)")};
        final int[] kind = {0};
        LinearLayout col = new LinearLayout(a); col.setOrientation(LinearLayout.VERTICAL); col.setPadding(a.dp(20), a.dp(8), a.dp(20), 0);
        TextView info = new TextView(a); info.setText(L.f("Файл BF16 / FP16 / FP32, блоков: %d (%s). Понадобится %.1f ГБ свободного места, около минуты.", "A BF16 / FP16 / FP32 file, blocks: %d (%s). Needs %.1f GB of free space, about a minute.", blocks, blocks == 40 ? "2.9B" : "2B", need / 1e9));
        col.addView(info); col.addView(e);
        RadioGroup rg = new RadioGroup(a);
        for (int i = 0; i < kinds.length; i++) { RadioButton rb = new RadioButton(a); rb.setText(kinds[i]); rb.setId(100 + i); rg.addView(rb); }
        rg.check(100); rg.setOnCheckedChangeListener((g, id) -> kind[0] = id - 100); col.addView(rg);
        TextView hint = new TextView(a); hint.setTextSize(12); hint.setAlpha(0.7f); hint.setPadding(0, a.dp(10), 0, 0);
        hint.setText(L.s("Что выбрать.\n• Обычная (base): чекпоинт без ускорения, как его выложил автор (на Civitai это base, v1.x и т.п.). Работает на 20–30 шагах с CFG 3–5, качество лучше; чтобы ускорить, добавьте файл Turbo-LoRA в настройках картинки.\n• Turbo: в названии или описании автора есть turbo, 4 step или distilled, то есть ускорение уже вшито в веса. Работает на 4 шагах без CFG, это самый быстрый вариант.\nНе уверены, выбирайте «Обычная»: Turbo-чекпоинт на ней тоже работает, просто покажутся лишние поля (CFG, негативный промпт).",
            "What to pick.\n• Regular (base): a checkpoint without acceleration, as the author published it (on Civitai: base, v1.x and so on). It runs at 20-30 steps with CFG 3-5 for the best quality; add a Turbo LoRA file in the picture settings to speed it up.\n• Turbo: the author's name or description says turbo, 4 step or distilled, so the acceleration is already merged into the weights. It runs at 4 steps without CFG and is the fastest.\nNot sure? Pick Regular: a Turbo checkpoint works there too, you will just see extra fields (CFG, negative prompt)."));
        col.addView(hint);
        new AlertDialog.Builder(a).setTitle(L.s("Свой чекпоинт Anima", "Own Anima checkpoint")).setView(sv(col))
            .setNegativeButton(L.s("Отмена", "Cancel"), (d, w) -> cleanup(tmp, pfd))
            .setPositiveButton(L.s("Конвертировать", "Convert"), (d, w) -> {
                String nm = e.getText().toString().trim(); if (nm.isEmpty()) nm = "custom";
                String id = "my" + nm.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", ""); if (id.length() < 3) id = "mymodel"; if (id.length() > 24) id = id.substring(0, 24);
                if (kind[0] == 0) id += "base";
                convert(src, tmp, pfd, id, nm, blocks, need);
            }).setOnCancelListener(d -> cleanup(tmp, pfd)).show();
    }

    /** the common Anima files (text encoder, VAE, tokenizer stub) are needed to run any converted model: offer to download them first */
    void common(final Runnable next, final Runnable abort) {
        if (store.commonInstalled()) { next.run(); return; }
        new AlertDialog.Builder(a).setTitle(L.s("Нужны базовые файлы", "Base files are needed"))
            .setMessage(L.f("Для работы любой модели Anima нужны общие файлы: текстовый энкодер и VAE (около %.1f ГБ). Скачать их сейчас и продолжить?", "Every Anima model needs the common files: the text encoder and the VAE (about %.1f GB). Download them now and go on?", Math.max(store.commonSize, 1_100_000_000L) / 1e9))
            .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> downloadCommon(next, abort)).setNegativeButton(L.s("Отмена", "Cancel"), (d, w) -> abort.run()).setOnCancelListener(d -> abort.run()).show();
    }

    void downloadCommon(final Runnable next, final Runnable abort) {
        final ProgressBar bar = new ProgressBar(a, null, android.R.attr.progressBarStyleHorizontal); bar.setMax(1000);
        final TextView msg = new TextView(a); msg.setText(L.s("Подключаюсь…", "Connecting…"));
        LinearLayout col = new LinearLayout(a); col.setOrientation(LinearLayout.VERTICAL); col.setPadding(a.dp(20), a.dp(12), a.dp(20), 0); col.addView(msg); col.addView(bar);
        final java.util.concurrent.atomic.AtomicBoolean cancel = new java.util.concurrent.atomic.AtomicBoolean(false);
        final AlertDialog dlg = new AlertDialog.Builder(a).setTitle(L.s("Базовые файлы Anima", "Anima base files")).setView(col).setCancelable(false)
            .setNegativeButton(L.s("Отмена", "Cancel"), (d, w) -> cancel.set(true)).show();
        a.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        a.startForegroundService(new Intent(a, DownloadService.class));
        new Thread(() -> {
            String err = null; final long[] last = {0};
            try {
                store.installCommon(baseUrl, (d, tot, what) -> {
                    long now = System.currentTimeMillis(); if (now - last[0] < 300) return; last[0] = now;
                    final String t = String.format(Locale.US, "%.1f / %.1f GB  ·  %s", d / 1e9, tot / 1e9, what); final int pm = tot > 0 ? (int) (1000L * d / tot) : 0;
                    ui(() -> { bar.setProgress(pm); msg.setText(t); });
                }, cancel);
            } catch (InterruptedIOException e) { err = "cancelled"; } catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", String.valueOf(e.getMessage())); }
            final String er = err;
            ui(() -> {
                dlg.dismiss(); a.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); DownloadService.stop(a.getApplicationContext());
                if (er == null) next.run(); else { abort.run(); if (!er.equals("cancelled")) new AlertDialog.Builder(a).setMessage(er).setPositiveButton("OK", null).show(); }
            });
        }).start();
    }

    ScrollView sv(View v) { ScrollView s = new ScrollView(a); s.addView(v); return s; }

    void cleanup(File tmp, ParcelFileDescriptor pfd) { if (tmp != null) tmp.delete(); try { if (pfd != null) pfd.close(); } catch (IOException ignored) { } }

    void convert(final String src, final File tmp, final ParcelFileDescriptor pfd, final String id, final String name, final int blocks, final long need) {
        final File dir = new File(a.getFilesDir(), "anima/m_" + id), stub = new File(a.getFilesDir(), "sd/" + id + ".safetensors");
        if (dir.exists() && new File(dir, ".installed").exists() || a.getFilesDir().getUsableSpace() < need + 1_000_000_000L) {
            cleanup(tmp, pfd); fail(dir.exists() ? L.s("Модель с таким названием уже есть: удалите её или назовите по-другому", "A model with this name exists: delete it or use another name") : L.s("Мало свободного места", "Not enough free space")); return;
        }
        final ProgressBar bar = new ProgressBar(a, null, android.R.attr.progressBarStyleHorizontal); bar.setMax(1000);
        final TextView msg = new TextView(a); msg.setText(L.s("Конвертирую…", "Converting…"));
        LinearLayout col = new LinearLayout(a); col.setOrientation(LinearLayout.VERTICAL); col.setPadding(a.dp(20), a.dp(12), a.dp(20), 0); col.addView(msg); col.addView(bar);
        final AtomicReference<Process> proc = new AtomicReference<>(); final boolean[] cancelled = {false};
        final AlertDialog dlg = new AlertDialog.Builder(a).setTitle(name).setView(col).setCancelable(false)
            .setNegativeButton(L.s("Отмена", "Cancel"), (d, w) -> { cancelled[0] = true; Process p = proc.get(); if (p != null) p.destroy(); }).show();
        a.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        a.startForegroundService(new Intent(a, DownloadService.class));
        new Thread(() -> {
            String result = "";
            try {
                dir.mkdirs(); stub.getParentFile().mkdirs();
                Process p = new ProcessBuilder(exe().getPath(), "convert", src, dir.getPath(), stub.getPath()).redirectErrorStream(true).start(); proc.set(p);
                try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    for (String l; (l = r.readLine()) != null; ) {
                        result = l;
                        if (l.startsWith("progress ")) { String[] s = l.split(" "); final int pm = (int) (1000L * Integer.parseInt(s[1]) / Math.max(1, Integer.parseInt(s[2]))); final String t = L.f("Блок %s из %s", "Block %s of %s", s[1], s[2]); ui(() -> { bar.setProgress(pm); msg.setText(t); }); }
                    }
                }
                p.waitFor();
                if (!cancelled[0] && result.equals("done")) {
                    try (FileWriter w = new FileWriter(new File(dir, "meta.json"))) { w.write(new org.json.JSONObject().put("id", id).put("name", name).put("speed", 100).toString()); }
                    new File(dir, Models.CUSTOM).createNewFile();
                    new File(dir, ".installed").createNewFile();
                }
            } catch (Exception e) { result = "error " + e.getMessage(); }
            final boolean ok = !cancelled[0] && result.equals("done"); final String res = result;
            if (!ok) { deleteAll(dir); stub.delete(); }
            cleanup(tmp, pfd);
            ui(() -> {
                dlg.dismiss(); a.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); DownloadService.stop(a.getApplicationContext());
                if (ok) { Toast.makeText(a, L.s("Готово: модель добавлена", "Done: the model is added"), Toast.LENGTH_LONG).show(); if (onDone != null) onDone.run(); }
                else if (!cancelled[0]) new AlertDialog.Builder(a).setMessage(L.f("Не удалось: %s", "Failed: %s", res.replaceFirst("^error ", ""))).setPositiveButton("OK", null).show();
            });
        }).start();
    }

    static void deleteAll(File f) { File[] c = f.listFiles(); if (c != null) for (File x : c) deleteAll(x); f.delete(); }
}
