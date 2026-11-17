package com.hexanima.npu;

import android.widget.Toast;
import java.io.File;
import org.json.JSONObject;

/** the upscale menu (Real-ESRGAN x2 / x4, SeedVR2 x2-x4) shared by the viewer and the main screen: runs on the NPU, the result is a new gallery picture handed to `done` */
final class UpscaleUi {
    final Main.Sh sh = new Main.Sh(); boolean busy;
    /** Real-ESRGAN on the NPU: x4 or x2, photo or anime network; the result becomes a new picture next to this one */
    void dialog(final Base act, final File f, final JSONObject m, final java.util.function.Consumer<File> done) {
        if (busy) { Toast.makeText(act, L.s("Увеличение уже идёт", "An upscale is already running"), Toast.LENGTH_SHORT).show(); return; }
        final String mn = m.optString("model", ""); final boolean photoFirst = mn.contains("Klein") || mn.contains("H3") || mn.contains("Qwen") || mn.contains("Mage") || mn.contains("MOSS");
        final String[] names = photoFirst
            ? new String[]{L.s("×4 · фото", "×4 · photo"), L.s("×4 · аниме", "×4 · anime"), L.s("×2 · фото", "×2 · photo"), L.s("×2 · аниме", "×2 · anime")}
            : new String[]{L.s("×4 · аниме", "×4 · anime"), L.s("×4 · фото", "×4 · photo"), L.s("×2 · аниме", "×2 · anime"), L.s("×2 · фото", "×2 · photo")};
        final boolean[] anime = photoFirst ? new boolean[]{false, true, false, true} : new boolean[]{true, false, true, false}, half = {false, false, true, true};
        final String[] all = new String[names.length + 3]; System.arraycopy(names, 0, all, 0, names.length);
        all[names.length] = L.s("×2 · SeedVR2 (диффузия, лучшее качество)", "×2 · SeedVR2 (diffusion, best quality)"); all[names.length + 1] = "×3 · SeedVR2"; all[names.length + 2] = "×4 · SeedVR2";
        act.builder().setTitle(L.s("Увеличить (NPU)", "Upscale (NPU)")).setItems(all, (d, i) -> {
            if (i >= names.length) { seedVr(act, f, m, 2 + (i - names.length), done); return; }
            final boolean an = anime[i], hf = half[i];
            int w = m.optInt("w", 512), h = m.optInt("h", 512); String bad = Upscale.sizeError(w, h, hf);
            if (bad != null) { Toast.makeText(act, bad, Toast.LENGTH_LONG).show(); return; }
            if (!Upscale.has(act.getFilesDir(), an)) { askDownload(act, f, an, hf, done); return; }
            startUpscale(act, f, an, hf, done);
        }).show();
    }
    /** SeedVR2-3B (one-step diffusion) on the NPU: the result becomes a new picture next to this one */
    void seedVr(final Base act, final File f, final JSONObject m, final int scale, final java.util.function.Consumer<File> done) {
        int[] o = SeedVr.outSize(m.optInt("w", 512), m.optInt("h", 512), scale); String bad = SeedVr.sizeError(o[0], o[1]);
        if (bad != null) { Toast.makeText(act, bad, Toast.LENGTH_LONG).show(); return; }
        if (!SeedVr.has(act.getFilesDir())) { act.builder().setTitle("SeedVR2").setMessage(L.s("Модель SeedVR2 (около 3,6 ГБ) ещё не установлена: файлы должны лежать в files/seedvr2 (pack/, vae/). Загрузка из каталога появится позже.", "The SeedVR2 model (about 3.6 GB) is not installed yet: the files must be in files/seedvr2 (pack/, vae/). Download from the catalog will come later.")).setPositiveButton("OK", null).show(); return; }
        busy = true; final android.app.AlertDialog dlg = act.builder().setTitle("SeedVR2 ×" + scale).setMessage(L.s("Подготовка…", "Preparing…")).setCancelable(false).setNegativeButton(L.s("Остановить", "Stop"), (d, i) -> new Thread(() -> SeedVr.stop(sh, act.getFilesDir())).start()).create(); dlg.show();
        new Thread(() -> {
            File nf = null; String err = null;
            try {
                sh.env = new String[]{new File(act.getFilesDir(), "anima").getPath(), new File(act.getFilesDir(), "sd").getPath(), act.getApplicationInfo().nativeLibraryDir};
                nf = SeedVr.run(act, sh, f, scale, msg -> act.runOnUiThread(() -> dlg.setMessage(msg)));
            } catch (Throwable e) { err = String.valueOf(e.getMessage() != null ? e.getMessage() : e); }
            final File res = nf; final String er = err;
            act.runOnUiThread(() -> {
                dlg.dismiss(); busy = false;
                if (res == null) { Toast.makeText(act, L.f("Не удалось: %s", "Failed: %s", er), Toast.LENGTH_LONG).show(); return; }
                done.accept(res);
                Toast.makeText(act, L.s("Готово: новая картинка в галерее", "Done: a new picture in the gallery"), Toast.LENGTH_SHORT).show();
            });
        }).start();
    }
    void askDownload(final Base act, final File f, final boolean an, final boolean hf, final java.util.function.Consumer<File> done) {
        act.builder().setTitle(L.s("Нужна загрузка", "Download needed")).setMessage(L.f("Сеть для увеличения (%s, %d МБ) скачивается один раз с Hugging Face. Лицензия Real-ESRGAN — BSD-3.", "The upscaling network (%s, %d MB) is downloaded once from Hugging Face. Real-ESRGAN licence: BSD-3.", an ? L.s("аниме", "anime") : L.s("фото", "photo"), Upscale.size(an) / 1000000))
            .setPositiveButton(L.s("Скачать", "Download"), (d, i) -> downloadThenUpscale(act, f, an, hf, done)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
    }
    void downloadThenUpscale(final Base act, final File f, final boolean an, final boolean hf, final java.util.function.Consumer<File> done) {
        busy = true; final android.app.AlertDialog dlg = act.builder().setTitle(L.s("Загрузка…", "Downloading…")).setMessage("0 %").setCancelable(false).create(); dlg.show();
        new Thread(() -> {
            String err = null;
            try { final long tot = Upscale.size(an); Upscale.download(act, an, n -> act.runOnUiThread(() -> dlg.setMessage((100 * n / tot) + " %")), null); } catch (Throwable e) { err = String.valueOf(e.getMessage() != null ? e.getMessage() : e); }
            final String er = err;
            act.runOnUiThread(() -> {
                dlg.dismiss(); busy = false;
                if (er != null) { Toast.makeText(act, L.f("Не удалось скачать: %s", "Download failed: %s", er), Toast.LENGTH_LONG).show(); return; }
                startUpscale(act, f, an, hf, done);
            });
        }).start();
    }
    void startUpscale(final Base act, final File f, final boolean an, final boolean hf, final java.util.function.Consumer<File> done) {
        {
            busy = true; Toast.makeText(act, L.s("Увеличение…", "Upscaling…"), Toast.LENGTH_SHORT).show();
            new Thread(() -> {
                File nf = null; String err = null;
                try {
                    sh.env = new String[]{new File(act.getFilesDir(), "anima").getPath(), new File(act.getFilesDir(), "sd").getPath(), act.getApplicationInfo().nativeLibraryDir};
                    nf = Upscale.run(act, sh, f, an, hf);
                } catch (Throwable e) { err = String.valueOf(e.getMessage() != null ? e.getMessage() : e); }
                final File res = nf; final String er = err;
                act.runOnUiThread(() -> {
                    busy = false;
                    if (res == null) { Toast.makeText(act, L.f("Не удалось увеличить: %s", "Upscale failed: %s", er), Toast.LENGTH_LONG).show(); return; }
                    done.accept(res);
                    Toast.makeText(act, L.s("Готово: новая картинка в галерее", "Done: a new picture in the gallery"), Toast.LENGTH_SHORT).show();
                });
            }).start();
        }
    }

}
