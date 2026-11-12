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
    static boolean keepOn(Context c) { return prefs(c).getBoolean("keepon", true); }
    static String rawPrefix(Context c) { return prefs(c).getString("prefix", DEFAULT_PREFIX); }
    /** what goes before the prompt: trimmed, ending with a comma and a space; a single space when the user cleared it (anima.sh treats an empty PREFIX as "use the default") */
    static String prefix(Context c) { String p = rawPrefix(c).trim(); return p.isEmpty() ? " " : (p.endsWith(",") ? p : p + ",") + " "; }

    EditText prefixEdit;

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
        LinearLayout c5 = card(col); c5.addView(githubRow());   // about: the project page
        if (Moss.is(p.getString("model", "")) || Qwen.is(p.getString("model", "")) || Mage.is(p.getString("model", "")) || H3.is(p.getString("model", ""))) return;
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
