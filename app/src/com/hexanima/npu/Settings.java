package com.hexanima.npu;

import android.content.res.ColorStateList;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.*;

/** Settings: language, theme, auto-load / idle unload of the model, screen on, quality-tag prefix. Everything is stored in the "anima" prefs and applied at once. */
public class Settings extends Base {
    static final String DEFAULT_PREFIX = "best quality, score_7, score_9, very aesthetic, ultra detailed, fair skin, high contrast, anime coloring, ";
    static final int[] IDLE = {0, 3, 5, 10, 30};

    static SharedPreferences prefs(Context c) { return c.getSharedPreferences("anima", 0); }
    static boolean autoload(Context c) { return prefs(c).getBoolean("autoload", false); }
    static int idleMin(Context c) { return prefs(c).getInt("idle", 10); }
    /** low-memory mode for the Anima models (blocks are streamed from storage instead of being pinned in RAM): on by default below ~9.5 GB of RAM */
    static final int[] BASE_STEPS = {20, 25, 30};
    static final String[] BASE_CFG = {"3", "4", "5", "6"};
    static final String DEFAULT_NEG = "worst quality, low quality, blurry, bad anatomy";
    /** the Anima base models (no turbo LoRA): sampling steps, CFG scale and negative prompt */
    static int baseSteps(Context c) { return prefs(c).getInt("base_steps", 20); }
    static String baseCfg(Context c) { return prefs(c).getString("base_cfg", "4"); }
    static String baseNeg(Context c) { String n = prefs(c).getString("base_neg", DEFAULT_NEG).trim(); return n.isEmpty() ? DEFAULT_NEG : n; }   // an empty negative prompt would make sd.cpp skip the second pass
    /** Anima sampler and scheduler (sd.cpp names); the scheduler "" = the model default */
    static final String[] SAMPLERS = {"euler", "euler_a", "er_sde", "dpm++2m", "dpm++2mv2", "ipndm", "lcm"};
    static final String[] SCHEDULERS = {"", "simple", "discrete", "karras", "exponential", "sgm_uniform", "bong_tangent"};
    static String sampler(Context c) { String v = prefs(c).getString("sampler", "euler"); return java.util.Arrays.asList(SAMPLERS).contains(v) ? v : "euler"; }
    static String scheduler(Context c) { String v = prefs(c).getString("scheduler", ""); return java.util.Arrays.asList(SCHEDULERS).contains(v) ? v : ""; }
    /** write the prompt, seed, model... into the PNG itself (the app's gallery keeps its own copy either way) */
    static boolean metaInPng(Context c) { return prefs(c).getBoolean("png_meta", true); }
    /** the folder picked for archives (a SAF tree URI), or null = Downloads; a folder whose permission is gone counts as not set */
    static android.net.Uri archiveDir(Context c) { return dirPref(c, "zip_dir"); }
    /** the folder picked for saved pictures and videos (the Download button), or null = the phone gallery (Pictures/Anima, Movies/Anima) */
    static android.net.Uri pictureDir(Context c) { return dirPref(c, "pic_dir"); }
    static String pictureDirLabel(Context c) { return pictureDir(c) == null ? L.s("Галерея телефона (Pictures/Anima)", "Phone gallery (Pictures/Anima)") : folderLabel(pictureDir(c)); }
    static android.net.Uri dirPref(Context c, String key) {
        String v = prefs(c).getString(key, ""); if (v.isEmpty()) return null;
        android.net.Uri u = android.net.Uri.parse(v);
        for (android.content.UriPermission up : c.getContentResolver().getPersistedUriPermissions()) if (up.getUri().equals(u) && up.isWritePermission()) return u;
        return null;
    }
    static String archiveDirLabel(Context c) { android.net.Uri u = archiveDir(c); return u == null ? L.s("Загрузки", "Downloads") : folderLabel(u); }
    static String folderLabel(android.net.Uri u) {
        try { String id = android.provider.DocumentsContract.getTreeDocumentId(u); int i = id.indexOf(':'); String r = i >= 0 ? id.substring(i + 1) : id; return r.isEmpty() ? L.s("Внутренняя память", "Internal storage") : r; } catch (Exception e) { return u.getLastPathSegment(); }
    }
    static boolean keepOn(Context c) { return prefs(c).getBoolean("keepon", true); }
    static String rawPrefix(Context c) { return prefs(c).getString("prefix", DEFAULT_PREFIX); }
    /** what goes before the prompt: trimmed, ending with a comma and a space; a single space when the user cleared it (anima.sh treats an empty PREFIX as "use the default") */
    static String prefix(Context c) { String p = rawPrefix(c).trim(); return p.isEmpty() ? " " : (p.endsWith(",") ? p : p + ",") + " "; }

    EditText prefixEdit;

    @Override protected void onActivityResult(int req, int res, android.content.Intent d) {
        super.onActivityResult(req, res, d);
        if (req == 32 && res == RESULT_OK && d != null && d.getData() != null) {
            try {   // scaled to at most 1600 px on the long side, kept as files/wallpaper.jpg (the picture itself is not touched)
                android.graphics.Bitmap bm = android.graphics.ImageDecoder.decodeBitmap(android.graphics.ImageDecoder.createSource(getContentResolver(), d.getData()), (dec, info, src) -> {
                    int mx = Math.max(info.getSize().getWidth(), info.getSize().getHeight()); if (mx > 1600) dec.setTargetSize(info.getSize().getWidth() * 1600 / mx, info.getSize().getHeight() * 1600 / mx);
                    dec.setAllocator(android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE);
                });
                try (java.io.FileOutputStream o = new java.io.FileOutputStream(wallFile(this))) { bm.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, o); }
                wallBm = null; SharedPreferences sp = prefs(this); sp.edit().putFloat("wall_zoom", 1f).putFloat("wall_x", .5f).putFloat("wall_y", .5f).putLong("wall_ts", System.currentTimeMillis()).putInt("card_trans", sp.getInt("card_trans", 35)).apply();
                startActivity(new android.content.Intent(this, WallEdit.class));   // straight to placing it; this screen is rebuilt on return (the theme key changed)
                return;
            } catch (Exception e) { Toast.makeText(this, L.f("Не удалось открыть картинку: %s", "Could not open the picture: %s", String.valueOf(e.getMessage())), Toast.LENGTH_LONG).show(); }
            recreate();
        }
        if (req == 33 && res == RESULT_OK && d != null && d.getData() != null) {
            try { getContentResolver().takePersistableUriPermission(d.getData(), android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION | android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION); prefs(this).edit().putString("pic_dir", d.getData().toString()).apply(); }
            catch (Exception e) { Toast.makeText(this, String.valueOf(e.getMessage()), Toast.LENGTH_LONG).show(); }
            recreate();
        }
        if (req == 31 && res == RESULT_OK && d != null && d.getData() != null) {
            try { getContentResolver().takePersistableUriPermission(d.getData(), android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION | android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION); prefs(this).edit().putString("zip_dir", d.getData().toString()).apply(); }
            catch (Exception e) { Toast.makeText(this, String.valueOf(e.getMessage()), Toast.LENGTH_LONG).show(); }
            recreate();
        }
    }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout col = screen(L.s("Настройки", "Settings"), null);
        SharedPreferences p = prefs(this);

        LinearLayout c1 = card(col);
        c1.addView(choice(L.s("Язык", "Language"), L.NAMES[L.index()], v -> {
            builder().setTitle(L.s("Язык", "Language")).setSingleChoiceItems(L.NAMES, L.index(), (d, i) -> { d.dismiss(); L.set(this, L.CODES[i]); recreate(); }).show();
        }));
        String[] tn = {L.s("Как в системе", "Follow system"), L.s("Светлая", "Light"), L.s("Тёмная", "Dark"), L.s("Чёрная (AMOLED)", "Black (AMOLED)")}, tv = {"system", "light", "dark", "amoled"};
        int ti = java.util.Arrays.asList(tv).indexOf(p.getString("theme", "system"));
        c1.addView(choice(L.s("Тема", "Theme"), tn[Math.max(ti, 0)], v -> {
            builder().setTitle(L.s("Тема", "Theme")).setSingleChoiceItems(tn, Math.max(ti, 0), (d, i) -> { d.dismiss(); p.edit().putString("theme", tv[i]).apply(); recreate(); }).show();
        }));
        if (android.os.Build.VERSION.SDK_INT >= 31) c1.addView(toggle(L.s("Цвет акцента из системы", "Accent colour from the system"), L.s("Берётся из обоев (Android 12 и новее)", "Taken from the wallpaper (Android 12 and newer)"), p.getBoolean("dyn_accent", false), on -> { p.edit().putBoolean("dyn_accent", on).apply(); recreate(); }));

        // own wallpaper behind every screen; the blocks turn see-through so it shows
        c1.addView(choice(L.s("Обои", "Wallpaper"), hasWall(this) ? L.s("Свои", "Own") : L.s("Нет", "None"), v -> {
            final boolean has = hasWall(this);
            builder().setTitle(L.s("Обои", "Wallpaper")).setItems(has ? new String[]{L.s("Выбрать картинку…", "Choose a picture…"), L.s("Положение и масштаб", "Position and scale"), L.s("Убрать обои", "Remove the wallpaper")} : new String[]{L.s("Выбрать картинку…", "Choose a picture…")}, (d, i) -> {
                if (i == 0) startActivityForResult(new android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT).setType("image/*").addCategory(android.content.Intent.CATEGORY_OPENABLE), 32);
                else if (i == 1) startActivity(new android.content.Intent(this, WallEdit.class));
                else { wallFile(this).delete(); wallBm = null; p.edit().putLong("wall_ts", 0).apply(); recreate(); }
            }).show();
        }));
        if (hasWall(this)) {
            c1.addView(slider(L.s("Прозрачность блоков", "Block transparency"), cardTrans(this), 90, v -> p.edit().putInt("card_trans", v).apply(), v -> liveTrans(v)));
            c1.addView(slider(L.s("Затемнение обоев", "Wallpaper dimming"), wallDim(this), 80, v -> p.edit().putInt("wall_dim", v).apply(), v -> { dimNow = v; ((android.view.ViewGroup) findViewById(android.R.id.content)).getChildAt(0).invalidate(); }));
        }

        LinearLayout c2 = card(col);
        c2.addView(toggle(L.s("Загружать модель при запуске", "Load the model at start"), L.s("Первая картинка появится быстрее, но модель занимает ОЗУ", "The first picture is faster, but the model holds RAM"), autoload(this), on -> p.edit().putBoolean("autoload", on).apply()));
        String[] in = new String[IDLE.length];
        for (int i = 0; i < IDLE.length; i++) in[i] = IDLE[i] == 0 ? L.s("Никогда", "Never") : L.f("%d мин", "%d min", IDLE[i]);
        int ii = Math.max(0, java.util.Arrays.binarySearch(IDLE, idleMin(this)));
        c2.addView(choice(L.s("Выгружать при простое", "Unload when idle"), in[ii], v -> {
            builder().setTitle(L.s("Выгружать при простое", "Unload when idle")).setSingleChoiceItems(in, ii, (d, i) -> { d.dismiss(); p.edit().putInt("idle", IDLE[i]).apply(); recreate(); }).show();
        }));
        TextView idleNote = text(L.s("Освобождает ОЗУ, если вы долго ничего не генерируете", "Frees RAM when you have not generated anything for a while"), 12, DIM, false); idleNote.setPadding(dp(2), 0, 0, dp(4));
        c2.addView(idleNote);

        LinearLayout c3 = card(col);
        c3.addView(toggle(L.s("Не гасить экран при генерации", "Keep the screen on while generating"), "", keepOn(this), on -> p.edit().putBoolean("keepon", on).apply()));
        c3.addView(toggle(L.s("Хранить метаданные в картинке", "Keep metadata in the picture"), L.s("Промпт, seed и модель записываются в PNG: по ним можно восстановить настройки. Если выключить, файл будет чистым (удобно перед публикацией); в галерее приложения подробности остаются.", "The prompt, seed and model are written into the PNG so the settings can be restored from it. Off: the file stays clean (handy before sharing); the details are still kept in the app's gallery."), metaInPng(this), on -> p.edit().putBoolean("png_meta", on).apply()));
        c3.addView(choice(L.s("Куда сохранять картинки", "Where to save pictures"), pictureDirLabel(this), v -> builder().setTitle(L.s("Куда сохранять картинки", "Where to save pictures"))
            .setSingleChoiceItems(new String[]{L.s("Галерея телефона (Pictures/Anima)", "Phone gallery (Pictures/Anima)"), L.s("Выбрать папку…", "Choose a folder…")}, pictureDir(this) == null ? 0 : -1, (d, i) -> {
                d.dismiss();
                if (i == 0) { p.edit().remove("pic_dir").apply(); recreate(); }
                else startActivityForResult(new android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT_TREE).addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION | android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION | android.content.Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION | android.content.Intent.FLAG_GRANT_PREFIX_URI_PERMISSION), 33);
            }).show()));
        c3.addView(choice(L.s("Куда сохранять архивы", "Where to save archives"), archiveDirLabel(this), v -> builder().setTitle(L.s("Куда сохранять архивы", "Where to save archives"))
            .setSingleChoiceItems(new String[]{L.s("Загрузки", "Downloads"), L.s("Выбрать папку…", "Choose a folder…")}, archiveDir(this) == null ? 0 : -1, (d, i) -> {
                d.dismiss();
                if (i == 0) { p.edit().remove("zip_dir").apply(); recreate(); }
                else startActivityForResult(new android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT_TREE).addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION | android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION | android.content.Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION | android.content.Intent.FLAG_GRANT_PREFIX_URI_PERMISSION), 31);
            }).show()));
        LinearLayout c5 = card(col); c5.addView(githubRow());   // about: the project page
        if (Moss.is(p.getString("model", "")) || Qwen.is(p.getString("model", "")) || Mage.is(p.getString("model", "")) || H3.is(p.getString("model", "")) || Klein.is(p.getString("model", "")) || Ltx.is(p.getString("model", ""))) return;
        TextView pl = text(L.s("Теги качества перед описанием (модели Anima)", "Quality tags before the prompt (Anima models)"), 15, TXT, true); pl.setPadding(dp(2), dp(14), 0, dp(6)); c3.addView(pl);
        prefixEdit = field("");
        prefixEdit.setText(rawPrefix(this)); prefixEdit.setTextSize(13); prefixEdit.setMinLines(2); prefixEdit.setGravity(Gravity.TOP);
        prefixEdit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        c3.addView(prefixEdit);
        Button reset = button(L.s("Сбросить по умолчанию", "Reset to default"), FIELD, ACC); reset.setTextSize(13);
        reset.setOnClickListener(v -> prefixEdit.setText(DEFAULT_PREFIX));
        LinearLayout.LayoutParams rl = new LinearLayout.LayoutParams(-2, dp(38)); rl.topMargin = dp(8);
        c3.addView(reset, rl);
    }

    @Override protected void onPause() { super.onPause(); if (prefixEdit != null) prefs(this).edit().putString("prefix", prefixEdit.getText().toString()).apply(); }

    View choice(String title, String value, View.OnClickListener l) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setPadding(dp(2), dp(12), dp(2), dp(12));
        r.addView(text(title, 15, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        r.addView(text(value + "  ▾", 14, ACC, false));
        r.setOnClickListener(l);
        return r;
    }

    interface OnToggle { void on(boolean v); }

    /** title and the value in %, a bar under them; the change is applied (the screen is rebuilt) when the finger lifts */
    View slider(String title, int val, int max, java.util.function.IntConsumer cb, java.util.function.IntConsumer live) {
        LinearLayout r = new LinearLayout(this); r.setOrientation(LinearLayout.VERTICAL); r.setPadding(dp(2), dp(10), dp(2), dp(6));
        LinearLayout t = new LinearLayout(this); t.setGravity(Gravity.CENTER_VERTICAL); t.addView(text(title, 15, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        final TextView vt = text(val + " %", 14, DIM, false); t.addView(vt); r.addView(t);
        SeekBar sb = new SeekBar(this); sb.setMax(max); sb.setProgress(val); sb.setProgressTintList(ColorStateList.valueOf(ACC)); sb.setThumbTintList(ColorStateList.valueOf(ACC));
        sb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b, int v, boolean user) { vt.setText(v + " %"); if (user) live.accept(v); }
            public void onStartTrackingTouch(SeekBar b) { }
            public void onStopTrackingTouch(SeekBar b) { cb.accept(b.getProgress()); recreate(); }
        });
        r.addView(sb, new LinearLayout.LayoutParams(-1, dp(36)));
        return r;
    }

    View toggle(String title, String sub, boolean on, OnToggle cb) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setPadding(dp(2), dp(10), dp(2), dp(10));
        LinearLayout t = new LinearLayout(this); t.setOrientation(LinearLayout.VERTICAL);
        t.addView(text(title, 15, TXT, true));
        if (!sub.isEmpty()) { TextView s = text(sub, 12, DIM, false); s.setPadding(0, dp(2), 0, 0); t.addView(s); }
        r.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
        Switch sw = new Switch(this); sw.setChecked(on);
        sw.setThumbTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{ACC, DIM}));
        sw.setTrackTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{(ACC & 0x00FFFFFF) | 0x66000000, (DIM & 0x00FFFFFF) | 0x44000000}));
        sw.setOnCheckedChangeListener((v, c) -> cb.on(c));
        r.addView(sw);
        r.setOnClickListener(v -> sw.toggle());
        return r;
    }
}
