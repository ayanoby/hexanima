package com.hexanima.npu;

import android.app.AlertDialog;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Outline;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewOutlineProvider;
import android.view.WindowManager;
import android.widget.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONArray;
import org.json.JSONObject;

/** Anima on the NPU: load/unload the model (animad server via anima.sh) and generate. No root: the bundled native binaries run from the app's own uid. */
public class Main extends Base {
    static final String DEFAULT_BASE = "https://huggingface.co/ayanoby/anima-npu-packs/resolve/main/";
    static final int STEPS = 4;   // the models are distilled for 4 steps (see Info)
    /** an Anima base model (no turbo LoRA, id ends with "base"): many steps, CFG with a negative prompt (two DiT passes per step) */
    static boolean isBase(String id) { return id.endsWith("base"); }
    boolean base() { return isBase(model) && !customTurboAnima(); }
    /** Lab: a turbo LoRA file on an Anima base pack (megarun folds it with the other LoRAs when the model loads: LORA="file*strength|file*strength") */
    boolean customTurboAnima() { return lab && isAnima() && isBase(model) && prefs().getBoolean("aturbo_on", false) && AnimaLora.turboSrc(getFilesDir()).exists(); }
    /** Lab: a turbo LoRA file on the plain Qwen pack (dit8): slot overlays + glue override made by libqwenfold; sampler = the sigma table of the LoRA (4 or 8) */
    boolean customTurboQwen() { return lab && isQwen() && Qwen.is8(model) && prefs().getBoolean("qturbo_on", false) && QwenLora.turboSrc(getFilesDir()).exists(); }
    String turboStrKey() { return isAnima() ? "aturbo_str" : isQwen() ? "qturbo_str" : "turbo_str"; }
    /** sampling steps, typed or stepped by hand for every model; the range is what the engine of the model accepts (MOSS: the step codes of its LoRA, Qwen / Mage: their sigma tables), the default is the recipe of the model */
    /** Lab: the turbo LoRA of MOSS is a file picked by the user (sampler: 4 = rapid-moss like, 8 = DMD like) */
    boolean customTurbo() { return lab && isMoss() && prefs().getBoolean("turbo_on", false) && MossLora.turboSrc(getFilesDir()).exists(); }
    /** RealSkin MOSS has one model id; its 4 / 8 step mode (rapid-moss / the DMD add-on of the installed MOSS-Image) is the "rs_mode" preference */
    boolean rs8() { return Moss.isRs(model) && prefs().getString("rs_mode", "4").equals("8"); }
    boolean moss8() { return Moss.is8(model) || rs8(); }
    int mossSteps() { return customTurbo() ? prefs().getInt("turbo_sampler", 4) : rs8() ? 8 : Moss.steps(model); }
    boolean h3Lora() { return isH3() && prefs().getBoolean("h3_lora", false) && H3.loraInstalled(getFilesDir()); }
    int h3Steps() { return steps(); }   // 6 fixed; with the pDMD LoRA 2..4 (default 2)
    int stepsDef() { return isH3() ? (h3Lora() ? H3.LORA_STEPS : H3.STEPS) : base() ? Settings.baseSteps(this) : isMoss() ? mossSteps() : isQwen() ? (customTurboQwen() ? prefs().getInt("qturbo_sampler", 4) : Qwen.steps(model)) : STEPS; }
    int stepsMin() { return isH3() ? (h3Lora() ? H3.LORA_STEPS : 1) : 1; }
    int stepsMax() { return isH3() ? (h3Lora() ? H3.LORA_MAX_STEPS : H3.STEPS) : base() ? 50 : isMoss() ? mossSteps() : isQwen() ? 50 : isMage() ? 4 : 50; }
    String stepsKey() { return "steps_" + model + (customTurbo() || customTurboAnima() || customTurboQwen() ? "_t" : "") + (rs8() && !customTurbo() ? "_8" : "") + (isH3() && h3Lora() ? "_hl" : ""); }   // the steps of a model with a turbo LoRA picked by the user are kept apart
    int steps() { return Math.max(stepsMin(), Math.min(stepsMax(), prefs().getInt(stepsKey(), stepsDef()))); }
    void setSteps(int v) { prefs().edit().putInt(stepsKey(), Math.max(stepsMin(), Math.min(stepsMax(), v))).apply(); if (base()) prefs().edit().putInt("base_steps", steps()).apply(); refreshBaseUi(); }
    int cm() { return base() ? 2 : 1; }   // DiT passes per step
    /** popular sizes, all valid for the NPU pipeline (multiples of 16, patch count divisible by 32, at most 6912); grouped by aspect ratio, a portrait and its landscape twin next to each other */
    static final int[][] SIZES = {{512, 512}, {640, 640}, {768, 768}, {1024, 1024}, {512, 768}, {768, 512}, {768, 1152}, {1152, 768}, {1024, 1536}, {1536, 1024}, {768, 1024}, {1024, 768}, {960, 1280}, {1280, 960},
        {576, 1024}, {1024, 576}, {640, 1152}, {1152, 640}, {960, 1536}, {1536, 960}, {768, 1216}, {1216, 768}, {1280, 1280}, {1152, 1536}, {1536, 1152}};
    static volatile boolean visible;   // the main screen is in front (otherwise a finished picture is announced by a notification)

    String D, BIN;   // work dir (files/anima) and the native lib dir with the bundled executables
    Models store;
    java.util.List<String> ids = new java.util.ArrayList<>();   // installed model ids
    volatile int resW = 512, resH = 768;
    final Handler ui = new Handler(Looper.getMainLooper());
    final ExecutorService ex = Executors.newSingleThreadExecutor();
    final ExecutorService ex2 = Executors.newSingleThreadExecutor();   // the Stop button
    volatile boolean loaded, busy, cancelled, oomAbort;
    String model = "";   // id of the selected installed model; anima.sh gets MID/MDIR
    int count = 1;       // pictures per press of Generate
    volatile int curIdx, curN = 1;
    boolean statusSeen;
    int tick;

    TextView chipName, chipDot, state, status, saved, phSize, ramTxt, tempTxt, warnTxt, sizeText;
    View placeholder; LinearLayout sizeChips, advBox; ImageView advArrow; TextView advSum;
    LinearLayout customRow, imgBox;
    LinearLayout baseRow, stepsRow, variantRow, turboStrRow; SeekBar turboBar; TextView countVal;
    Button loadBtn, genBtn; LinearLayout baseBody;
    ImageButton dlBtn, shareBtn, favBtn, againBtn;
    MultiAutoCompleteTextView prompt;
    EditText seed, wEdit, hEdit;
    TextView tokView, negTokView; volatile boolean tokOver; QwenTok tokQ, tokM;   // prompt token counter (Anima / MOSS / Qwen text-to-image); generation is refused over the limit
    final Runnable tokTick = this::updateTokens;
    LinearLayout baseBox; EditText negEdit; TextView stepsVal, cfgVal, samplerVal, schedVal;   // Anima base models: negative prompt, steps and CFG under the prompt
    LinearLayout aloraBox, aloraList; TextView aloraInfo;   // Anima: the LoRA library (AnimaLoras)
    LinearLayout loraBox; TextView loraName, loraInfo, strText; SeekBar strBar; Button loraPick, loraDrop;
    LinearLayout editBox; TextView editNote; ImageView editThumb; TextView editInfo; Button editPick, editDrop, editAdd; LinearLayout extraBox; volatile Bitmap editBm; final java.util.List<Bitmap> extraBm = new java.util.concurrent.CopyOnWriteArrayList<>();   // Qwen-Image: the photo to edit
    ProgressBar bar, ramBar;
    ImageView img;
    File lastFile;
    boolean customOpen;

    String mp() { return "MID=" + model + " MDIR=m_" + model + " IDLE=" + Settings.idleMin(this) * 60 + " " + "MEGA_STREAM=1 " + animaLoraEnv(); }
    String animaLoraEnv() {
        StringBuilder l = new StringBuilder();
        if (customTurboAnima()) l.append(AnimaLora.turboSrc(getFilesDir()).getPath()).append('*').append(String.format(java.util.Locale.US, "%.2f", prefs().getFloat("aturbo_str", 1f)));
        String lu = AnimaLoras.env(prefs(), getFilesDir()); if (!lu.isEmpty()) { if (l.length() > 0) l.append('|'); l.append(lu); }
        return l.length() == 0 ? "" : "LORA=" + q(l.toString()) + " ";
    }
    /** the LoRA the Anima server has loaded (changing it restarts the server, anima.sh keeps the LoRA in the server key) */
    String servedSig = "";
    String loraSig() { return isAnima() ? animaLoraEnv() : ""; }   // the whole LORA= value: another set, strength or turbo LoRA restarts the server

    GradientDrawable gradient(boolean enabled) { return bg(enabled ? ACC : CARD, 12); }
    /** "HexAnima Lab" (second launcher icon, activity-alias .Lab -> this activity): the layout of the new loader design: only the base of a model is listed, the turbo LoRA is a row under it, the manual LoRA stays; own preferences, the same models and pictures */
    boolean lab;
    SharedPreferences prefs() { return getSharedPreferences(lab ? "lab" : "anima", 0); }
    void setFavIcon(boolean on) { favBtn.setImageResource(getResources().getIdentifier(on ? "ic_star" : "ic_star_off", "drawable", getPackageName())); favBtn.setImageTintList(on ? null : ColorStateList.valueOf(TXT)); }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        lab = true;   // the former Lab layout is the main app now (base + separate turbo-LoRA file); the flag only keeps the old branches reachable
        D = new File(getFilesDir(), "anima").getPath(); BIN = getApplicationInfo().nativeLibraryDir;
        store = new Models(getFilesDir());
        main.env = pollSh.env = ctlSh.env = new String[]{D, new File(getFilesDir(), "sd").getPath(), BIN};
        installScript();
        getWindow().setNavigationBarColor(0);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        LinearLayout frame = new LinearLayout(this);
        frame.setOrientation(LinearLayout.VERTICAL);
        frame.setBackgroundColor(BG);
        frame.setFitsSystemWindows(true);
        ScrollView sv = new ScrollView(this);
        sv.setFillViewport(true);
        sv.setVerticalScrollBarEnabled(false);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(4), dp(20), dp(20));
        sv.addView(root, new ViewGroup.LayoutParams(-1, -2));
        frame.addView(sv, new LinearLayout.LayoutParams(-1, 0, 1));

        // header: name, history, menu
        LinearLayout tr = new LinearLayout(this); tr.setGravity(Gravity.CENTER_VERTICAL);
        tr.addView(text("HexAnima", 20, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        ramTxt = statusItem("ic_memory"); tempTxt = statusItem("ic_thermo");
        tr.addView(ramTxt); tr.addView(tempTxt);
        ImageButton hist = icon("ic_photo", 0); hist.setContentDescription(L.s("История", "History"));
        hist.setOnClickListener(v -> startActivityForResult(new Intent(this, History.class), 7));
        tr.addView(hist, new LinearLayout.LayoutParams(dp(44), dp(44)));
        ImageButton more = icon("ic_more", 0); more.setContentDescription(L.s("Меню", "Menu"));
        more.setOnClickListener(v -> menu(v));
        tr.addView(more, new LinearLayout.LayoutParams(dp(44), dp(44)));
        root.addView(tr, new LinearLayout.LayoutParams(-1, dp(56)));
        ramTxt.setOnClickListener(v -> Toast.makeText(this, ramDetail, Toast.LENGTH_LONG).show());
        tempTxt.setOnClickListener(v -> Toast.makeText(this, tempDetail, Toast.LENGTH_LONG).show());
        ramBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); ramBar.setMax(1000);   // kept for the tick code
        warnTxt = text("", 13, WARN, false); warnTxt.setPadding(0, 0, 0, dp(8)); warnTxt.setVisibility(View.GONE);
        root.addView(warnTxt);

        // model: name and state; tap = choose another; Load / Unload on the right
        LinearLayout mc = new LinearLayout(this); mc.setGravity(Gravity.CENTER_VERTICAL); mc.setPadding(dp(16), dp(12), dp(12), dp(12)); mc.setBackground(bg(CARD, 14));
        chipDot = text("●", 9, DIM, false); chipDot.setPadding(0, 0, dp(10), 0); mc.addView(chipDot);
        LinearLayout names = new LinearLayout(this); names.setOrientation(LinearLayout.VERTICAL);
        chipName = text("…", 16, TXT, true); chipName.setMaxLines(2); chipName.setEllipsize(android.text.TextUtils.TruncateAt.END);
        state = text(L.s("Проверяю…", "Checking…"), 13, DIM, false); state.setMaxLines(1); state.setEllipsize(android.text.TextUtils.TruncateAt.END); state.setPadding(0, dp(2), 0, 0);
        names.addView(chipName); names.addView(state);
        mc.addView(names, new LinearLayout.LayoutParams(0, -2, 1));
        loadBtn = button(L.s("Загрузить", "Load"), BG, TXT); loadBtn.setTextSize(13); loadBtn.setPadding(dp(14), 0, dp(14), 0);
        loadBtn.setOnClickListener(v -> toggleModel());
        LinearLayout.LayoutParams lbl = new LinearLayout.LayoutParams(-2, dp(34)); lbl.leftMargin = dp(10); mc.addView(loadBtn, lbl);
        ImageView chev = new ImageView(this); chev.setImageDrawable(vec("ic_chevron", DIM, 20)); LinearLayout.LayoutParams cvl = new LinearLayout.LayoutParams(dp(20), dp(20)); cvl.leftMargin = dp(8); mc.addView(chev, cvl);
        mc.setOnClickListener(v -> pickModel());
        root.addView(mc, new LinearLayout.LayoutParams(-1, -2));
        variantRow = new LinearLayout(this); variantRow.setBackground(bg(CARD, 12)); variantRow.setPadding(dp(3), dp(3), dp(3), dp(3)); variantRow.setVisibility(View.GONE);
        LinearLayout.LayoutParams vrl = new LinearLayout.LayoutParams(-1, dp(42)); vrl.topMargin = dp(8); root.addView(variantRow, vrl);

        // the picture
        imgBox = new LinearLayout(this);
        imgBox.setOrientation(LinearLayout.VERTICAL);
        imgBox.setGravity(Gravity.CENTER);
        imgBox.setBackground(bg(CARD, 14));
        imgBox.setClipToOutline(true);
        imgBox.setOutlineProvider(new ViewOutlineProvider() {
            @Override public void getOutline(View v, Outline o) { o.setRoundRect(0, 0, v.getWidth(), v.getHeight(), dp(16)); }
        });
        LinearLayout ph = new LinearLayout(this); ph.setOrientation(LinearLayout.VERTICAL); ph.setGravity(Gravity.CENTER);
        phSize = text("", 14, DIM, false); phSize.setGravity(Gravity.CENTER); ph.addView(phSize, new LinearLayout.LayoutParams(-1, -2));
        placeholder = ph;
        imgBox.addView(placeholder, new LinearLayout.LayoutParams(-1, -1));   // the picture area has a fixed size: the placeholder and the picture fill it (the picture is fitted inside)
        img = new ImageView(this);
        img.setAdjustViewBounds(false);
        img.setScaleType(ImageView.ScaleType.FIT_CENTER);
        img.setVisibility(View.GONE);
        img.setOnClickListener(v -> { if (lastFile != null) { File[] all = Gallery.list(this, Gallery.ALL); int k = java.util.Arrays.asList(all).indexOf(lastFile); Viewer.open(this, all, Math.max(0, k), Gallery.ALL, true); } });
        imgBox.addView(img, new LinearLayout.LayoutParams(-1, -1));
        int boxH = getResources().getDisplayMetrics().widthPixels - dp(40);   // a square picture area
        LinearLayout.LayoutParams ibl = new LinearLayout.LayoutParams(-1, boxH); ibl.topMargin = dp(8); root.addView(imgBox, ibl);
        LinearLayout resRow = new LinearLayout(this);
        resRow.setGravity(Gravity.CENTER_VERTICAL);
        saved = text("", 12, DIM, false);
        resRow.addView(saved, new LinearLayout.LayoutParams(0, -2, 1));
        againBtn = icon("ic_refresh", 0);
        againBtn.setOnClickListener(v -> generate(true));   // the same prompt with a new random seed
        favBtn = icon("ic_star_off", 0);
        favBtn.setOnClickListener(v -> { if (lastFile == null) return; boolean nv = !Gallery.fav(lastFile); Gallery.setFav(lastFile, nv); setFavIcon(nv); });
        dlBtn = icon("ic_download", 0);
        dlBtn.setOnClickListener(v -> { if (lastFile == null) return; try { Gallery.export(this, lastFile); Toast.makeText(this, L.s("Сохранено в галерею: Pictures/Anima", "Saved to gallery: Pictures/Anima"), Toast.LENGTH_SHORT).show(); } catch (Exception e) { Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show(); } });
        shareBtn = icon("ic_share", 0);
        shareBtn.setOnClickListener(v -> { if (lastFile != null) Gallery.share(this, lastFile); });
        for (ImageButton x : new ImageButton[]{againBtn, favBtn, dlBtn, shareBtn}) { x.setVisibility(View.GONE); resRow.addView(x, new LinearLayout.LayoutParams(dp(44), dp(44))); }
        root.addView(resRow);

        // prompt
        LinearLayout pl = new LinearLayout(this); pl.setGravity(Gravity.BOTTOM);
        pl.addView(label(L.s("Описание", "Prompt")), new LinearLayout.LayoutParams(0, -2, 1));
        TextView tpl = text(L.s("Шаблоны", "Templates"), 13, TXT, false); tpl.setPadding(dp(8), dp(14), 0, dp(6)); tpl.setOnClickListener(v -> templates()); pl.addView(tpl);
        root.addView(pl);
        prompt = new MultiAutoCompleteTextView(this);   // Danbooru tag suggestions while typing, comma separated
        prompt.setHint("1girl, smile, outdoors, cherry blossoms…");
        prompt.setHintTextColor(DIM); prompt.setTextColor(TXT); prompt.setTextSize(15);
        prompt.setMinLines(3); prompt.setMaxLines(Integer.MAX_VALUE);   // grows down with the text
        prompt.setGravity(Gravity.TOP);
        prompt.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        prompt.setBackground(bg(FIELD, 12));
        prompt.setPadding(dp(14), dp(12), dp(14), dp(12));
        prompt.setThreshold(2);
        prompt.setTokenizer(new MultiAutoCompleteTextView.CommaTokenizer());
        prompt.setDropDownBackgroundDrawable(bg(CARD, 12));
        prompt.setDropDownHeight(dp(240));
        root.addView(prompt, new LinearLayout.LayoutParams(-1, -2));
        tokView = text("", 12, DIM, false); tokView.setGravity(Gravity.END); tokView.setPadding(0, dp(6), 0, 0); tokView.setVisibility(View.GONE); root.addView(tokView, new LinearLayout.LayoutParams(-1, -2));
        prompt.addTextChangedListener(new android.text.TextWatcher() {
            public void beforeTextChanged(CharSequence a, int b, int c, int d) { }
            public void onTextChanged(CharSequence a, int b, int c, int d) { }
            public void afterTextChanged(android.text.Editable e) { ui.removeCallbacks(tokTick); ui.postDelayed(tokTick, 350); }
        });
        loadTags();

        // Anima base models: negative prompt
        baseBox = new LinearLayout(this); baseBox.setOrientation(LinearLayout.VERTICAL); baseBox.setVisibility(View.GONE); baseBody = baseBox;
        baseBox.addView(label(L.s("Негативный промпт", "Negative prompt")));
        negEdit = field(Settings.DEFAULT_NEG); negEdit.setTextSize(14);
        negEdit.setMinLines(2); negEdit.setGravity(Gravity.TOP); negEdit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        negEdit.setPadding(dp(14), dp(12), dp(14), dp(12));
        negEdit.setText(prefs().getString("base_neg", Settings.DEFAULT_NEG));
        negEdit.addTextChangedListener(new android.text.TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int b, int c) { }
            public void onTextChanged(CharSequence s, int a, int b, int c) { }
            public void afterTextChanged(android.text.Editable e) { prefs().edit().putString("base_neg", e.toString()).apply(); ui.removeCallbacks(tokTick); ui.postDelayed(tokTick, 350); }
        });
        baseBox.addView(negEdit, new LinearLayout.LayoutParams(-1, -2));
        negTokView = text("", 12, DIM, false); negTokView.setGravity(Gravity.END); negTokView.setPadding(0, dp(6), 0, 0); baseBox.addView(negTokView, new LinearLayout.LayoutParams(-1, -2));   // the negative prompt has its own counter and limit
        root.addView(baseBox);

        // Qwen-Image / Mage-Flow: the photo to edit
        editBox = new LinearLayout(this); editBox.setOrientation(LinearLayout.VERTICAL);
        editBox.addView(label(L.s("Фото для правки", "Pictures to edit")));
        LinearLayout er = new LinearLayout(this); er.setGravity(Gravity.CENTER_VERTICAL);
        editThumb = new ImageView(this); editThumb.setScaleType(ImageView.ScaleType.CENTER_CROP); editThumb.setBackground(bg(CARD, 10)); editThumb.setClipToOutline(true);
        er.addView(editThumb, new LinearLayout.LayoutParams(dp(56), dp(56)));
        editInfo = text("", 13, DIM, false); editInfo.setPadding(dp(12), 0, dp(8), 0); er.addView(editInfo, new LinearLayout.LayoutParams(0, -2, 1));
        editPick = button(L.s("Выбрать", "Choose"), CARD, TXT); editPick.setTextSize(13); editPick.setPadding(dp(14), 0, dp(14), 0); editPick.setOnClickListener(v -> pickEdit());
        er.addView(editPick, new LinearLayout.LayoutParams(-2, dp(36)));
        editDrop = button("✕", BG, DIM); editDrop.setOnClickListener(v -> dropEdit());
        er.addView(editDrop, new LinearLayout.LayoutParams(dp(40), dp(36)));
        editBox.addView(er);
        extraBox = new LinearLayout(this); extraBox.setOrientation(LinearLayout.VERTICAL); editBox.addView(extraBox);   // Qwen: up to two more reference pictures
        editAdd = button("", BG, TXT); editAdd.setTextSize(13); editAdd.setGravity(Gravity.START | Gravity.CENTER_VERTICAL); editAdd.setPadding(0, 0, 0, 0); editAdd.setOnClickListener(v -> pickExtra(extraBm.size()));
        editBox.addView(editAdd, new LinearLayout.LayoutParams(0, 0));
        editNote = text("", 12, DIM, false); editNote.setPadding(0, dp(10), 0, 0); editBox.addView(editNote);
        root.addView(editBox);
        refreshEdit();

        // size: ratio segments, the exact size (tap = full list), swap
        sizeText = text(L.s("Готовые ▾", "Presets ▾"), 13, TXT, true); sizeText.setGravity(Gravity.CENTER); sizeText.setPadding(dp(12), 0, dp(12), 0); sizeText.setBackground(bg(CARD, 12)); sizeText.setOnClickListener(v -> pickSize());   // the list of ready sizes: right end of the size row
        sizeChips = null;   // the ratio segments are gone: the exact size below and the list of ready sizes are enough
        customRow = new LinearLayout(this);
        customRow.setGravity(Gravity.CENTER_VERTICAL);
        customRow.setPadding(0, dp(6), 0, 0);
        wEdit = field(L.s("ширина", "width")); hEdit = field(L.s("высота", "height"));
        wEdit.setInputType(InputType.TYPE_CLASS_NUMBER); hEdit.setInputType(InputType.TYPE_CLASS_NUMBER);
        customRow.addView(wEdit, new LinearLayout.LayoutParams(0, -2, 1));
        TextView times = text("⇄", 18, TXT, false); times.setGravity(Gravity.CENTER); times.setContentDescription(L.s("Повернуть", "Swap"));   // swaps width and height
        times.setOnClickListener(v -> {
            int w = resW, h = resH;
            try { w = Integer.parseInt(wEdit.getText().toString().trim()); h = Integer.parseInt(hEdit.getText().toString().trim()); } catch (NumberFormatException e) { }
            setRes(h, w);
        });
        customRow.addView(times, new LinearLayout.LayoutParams(dp(48), dp(44)));
        customRow.addView(hEdit, new LinearLayout.LayoutParams(0, -2, 1));
        LinearLayout.LayoutParams stl = new LinearLayout.LayoutParams(dp(124), dp(44)); stl.leftMargin = dp(8); customRow.addView(sizeText, stl);   // the two fields share the rest; Presets gets more room
        // the exact size is always editable; a size typed here is taken over (after a short pause) by the chips, the size label and the placeholder
        final Runnable applyTyped = () -> {
            try { int w = Integer.parseInt(wEdit.getText().toString().trim()), h = Integer.parseInt(hEdit.getText().toString().trim()); if (w >= 16 && h >= 16 && w <= 16384 && h <= 16384 && (w != resW || h != resH)) setRes(w, h); } catch (NumberFormatException ignored) { }
        };
        android.text.TextWatcher tw = new android.text.TextWatcher() {
            public void beforeTextChanged(CharSequence x, int a, int b, int c) { }
            public void onTextChanged(CharSequence x, int a, int b, int c) { }
            public void afterTextChanged(android.text.Editable e) { ui.removeCallbacks(applyTyped); ui.postDelayed(applyTyped, 700); }
        };
        wEdit.addTextChangedListener(tw); hEdit.addTextChangedListener(tw);
        root.addView(customRow);
        { int at = root.indexOfChild(resRow) + 1; root.removeView(customRow); root.addView(customRow, at); }   // the size row sits right under the picture area

        // parameters: one row each, label on the left, the control on the right
        root.addView(label(L.s("Параметры", "Settings")));
        LinearLayout pcard = new LinearLayout(this); pcard.setOrientation(LinearLayout.VERTICAL); pcard.setBackground(bg(CARD, 14)); pcard.setPadding(dp(16), 0, dp(12), 0);
        root.addView(pcard, new LinearLayout.LayoutParams(-1, -2));
        baseRow = new LinearLayout(this); baseRow.setOrientation(LinearLayout.VERTICAL); baseRow.setVisibility(View.GONE);
        stepsVal = text("", 15, TXT, false); cfgVal = text("", 15, TXT, false);
        stepsRow = new LinearLayout(this); stepsRow.setOrientation(LinearLayout.VERTICAL);
        stepsRow.addView(paramRow(L.s("Шаги", "Steps"), stepper(stepsVal, 0))); stepsRow.addView(divider());
        pcard.addView(stepsRow);
        baseRow.addView(paramRow("CFG", stepper(cfgVal, 1))); baseRow.addView(divider());
        samplerVal = text("", 15, TXT, false); samplerVal.setOnClickListener(v -> builder().setTitle(L.s("Сэмплер", "Sampler")).setSingleChoiceItems(Settings.SAMPLERS, java.util.Arrays.asList(Settings.SAMPLERS).indexOf(smp()), (d, i) -> { d.dismiss(); prefs().edit().putString("sampler", Settings.SAMPLERS[i]).apply(); refreshBaseUi(); }).show());
        schedVal = text("", 15, TXT, false); schedVal.setOnClickListener(v -> { String[] sn = Settings.SCHEDULERS.clone(); sn[0] = L.s("по умолчанию", "default");
            builder().setTitle(L.s("Шедулер", "Scheduler")).setSingleChoiceItems(sn, java.util.Arrays.asList(Settings.SCHEDULERS).indexOf(sch()), (d, i) -> { d.dismiss(); prefs().edit().putString("scheduler", Settings.SCHEDULERS[i]).apply(); refreshBaseUi(); }).show(); });
        for (TextView t : new TextView[]{samplerVal, schedVal}) { t.setGravity(Gravity.END); t.setPadding(dp(12), dp(10), dp(4), dp(10)); }
        baseRow.addView(paramRow(L.s("Сэмплер", "Sampler"), samplerVal)); baseRow.addView(divider());
        baseRow.addView(paramRow(L.s("Шедулер", "Scheduler"), schedVal)); baseRow.addView(divider());
        pcard.addView(baseRow);
        LinearLayout sr = new LinearLayout(this); sr.setGravity(Gravity.CENTER_VERTICAL);
        seed = field("-1"); seed.setText("-1"); seed.setGravity(Gravity.END); seed.setBackground(null); seed.setPadding(0, 0, dp(4), 0);
        seed.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED);
        sr.addView(seed, new LinearLayout.LayoutParams(dp(130), -2));
        TextView dice = text("↻", 18, DIM, false); dice.setGravity(Gravity.CENTER); dice.setContentDescription(L.s("Случайный", "Random"));
        dice.setOnClickListener(v -> seed.setText("-1"));
        sr.addView(dice, new LinearLayout.LayoutParams(dp(36), dp(36)));
        pcard.addView(paramRow(L.s("Seed  (−1 = случайный)", "Seed  (−1 = random)"), sr)); pcard.addView(divider());
        count = Math.max(1, Math.min(4, prefs().getInt("count", 1)));
        countVal = text(String.valueOf(count), 15, TXT, false);
        pcard.addView(paramRow(L.s("Картинок за раз", "Pictures per run"), stepper(countVal, 2)));

        // LoRA (Anima, MOSS-Image)
        loraBox = new LinearLayout(this); loraBox.setOrientation(LinearLayout.VERTICAL);
        LinearLayout lr = new LinearLayout(this); lr.setGravity(Gravity.CENTER_VERTICAL);
        loraName = text(L.s("не выбрана", "none"), 14, DIM, false); loraName.setSingleLine(true); loraName.setEllipsize(android.text.TextUtils.TruncateAt.MIDDLE); loraName.setGravity(Gravity.END);
        lr.addView(loraName, new LinearLayout.LayoutParams(dp(150), -2));
        loraPick = button(L.s("Выбрать", "Choose"), BG, TXT); loraPick.setTextSize(14); loraPick.setPadding(dp(12), 0, 0, 0); loraPick.setBackground(null); loraPick.setOnClickListener(v -> pickLora());
        lr.addView(loraPick, new LinearLayout.LayoutParams(-2, dp(36)));
        loraDrop = button("✕", BG, DIM); loraDrop.setBackground(null); loraDrop.setOnClickListener(v -> dropLora());
        lr.addView(loraDrop, new LinearLayout.LayoutParams(dp(36), dp(36)));
        loraBox.addView(divider()); loraBox.addView(paramRow("LoRA", lr));
        loraInfo = text("", 12, DIM, false); loraInfo.setPadding(0, 0, 0, dp(8)); loraBox.addView(loraInfo);
        LinearLayout sr2 = new LinearLayout(this); sr2.setGravity(Gravity.CENTER_VERTICAL); sr2.setPadding(0, 0, 0, dp(8));
        sr2.addView(text(L.s("Сила", "Strength"), 15, TXT, false));
        strBar = new SeekBar(this); strBar.setMax(40); strBar.setProgress(Math.round(prefs().getFloat("moss_lora_str", 1f) * 20));
        strBar.setProgressTintList(ColorStateList.valueOf(TXT)); strBar.setThumbTintList(ColorStateList.valueOf(TXT)); strBar.setProgressBackgroundTintList(ColorStateList.valueOf(LINE));
        sr2.addView(strBar, new LinearLayout.LayoutParams(0, -2, 1));
        strText = text("", 15, TXT, false); strText.setMinWidth(dp(40)); strText.setGravity(Gravity.END); sr2.addView(strText);
        strBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b, int p, boolean u) { strText.setText(String.format(java.util.Locale.US, "%.2f", p * 0.05f)); if (u) prefs().edit().putFloat(lp() + "str", p * 0.05f).apply(); }
            public void onStartTrackingTouch(SeekBar b) { }
            public void onStopTrackingTouch(SeekBar b) { }
        });
        strText.setText(String.format(java.util.Locale.US, "%.2f", strBar.getProgress() * 0.05f));
        loraBox.addView(sr2);
        pcard.addView(loraBox);
        // Anima: any number of LoRA files, each with a switch and a strength
        aloraBox = new LinearLayout(this); aloraBox.setOrientation(LinearLayout.VERTICAL);
        LinearLayout ah = new LinearLayout(this); ah.setGravity(Gravity.CENTER_VERTICAL);
        ah.addView(text("LoRA", 15, TXT, false), new LinearLayout.LayoutParams(0, -2, 1));
        Button aadd = button(L.s("＋ Добавить", "＋ Add"), BG, TXT); aadd.setTextSize(14); aadd.setBackground(null); aadd.setOnClickListener(v -> pickLora());
        ah.addView(aadd, new LinearLayout.LayoutParams(-2, dp(40)));
        aloraBox.addView(divider()); LinearLayout.LayoutParams ahl = new LinearLayout.LayoutParams(-1, dp(48)); aloraBox.addView(ah, ahl);
        aloraList = new LinearLayout(this); aloraList.setOrientation(LinearLayout.VERTICAL); aloraBox.addView(aloraList);
        aloraInfo = text("", 12, DIM, false); aloraInfo.setPadding(0, 0, 0, dp(8)); aloraBox.addView(aloraInfo);
        aloraBox.setVisibility(View.GONE);
        pcard.addView(aloraBox);
        turboStrRow = new LinearLayout(this); turboStrRow.setGravity(Gravity.CENTER_VERTICAL); turboStrRow.setPadding(0, dp(8), 0, dp(8)); turboStrRow.setVisibility(View.GONE);
        turboStrRow.addView(text(L.s("Сила turbo LoRA", "Turbo LoRA strength"), 15, TXT, false));
        turboBar = new SeekBar(this); final SeekBar tb = turboBar; tb.setMax(40); tb.setProgress(Math.round(prefs().getFloat(turboStrKey(), 1f) * 20));
        tb.setProgressTintList(ColorStateList.valueOf(TXT)); tb.setThumbTintList(ColorStateList.valueOf(TXT)); tb.setProgressBackgroundTintList(ColorStateList.valueOf(LINE));
        turboStrRow.addView(tb, new LinearLayout.LayoutParams(0, -2, 1));
        final TextView tt = text(String.format(java.util.Locale.US, "%.2f", tb.getProgress() * 0.05f), 15, TXT, false); tt.setMinWidth(dp(40)); tt.setGravity(Gravity.END); turboStrRow.addView(tt);
        tb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b, int p, boolean u) { tt.setText(String.format(java.util.Locale.US, "%.2f", p * 0.05f)); if (u) prefs().edit().putFloat(turboStrKey(), Math.max(0.05f, p * 0.05f)).apply(); }
            public void onStartTrackingTouch(SeekBar b) { }
            public void onStopTrackingTouch(SeekBar b) { }
        });
        pcard.addView(divider()); pcard.addView(turboStrRow);
        refreshLora();

        // bottom: progress, status, Generate
        LinearLayout bottom = new LinearLayout(this);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setBackgroundColor(BG);
        bottom.setPadding(dp(20), dp(8), dp(20), dp(14));
        bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        bar.setProgressTintList(ColorStateList.valueOf(TXT));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(LINE));
        bar.setVisibility(View.GONE);
        LinearLayout.LayoutParams bl = new LinearLayout.LayoutParams(-1, dp(2)); bl.bottomMargin = dp(8);
        bottom.addView(bar, bl);
        status = text("", 13, DIM, false);
        status.setVisibility(View.GONE);
        status.setPadding(0, 0, 0, dp(8));
        bottom.addView(status);
        genBtn = button("", ACC, ON_ACC);
        genBtn.setTextSize(16);
        genBtn.setOnClickListener(v -> { if (busy) cancel(); else generate(false); });
        bottom.addView(genBtn, new LinearLayout.LayoutParams(-1, dp(52)));
        frame.addView(divider());
        frame.addView(bottom, new LinearLayout.LayoutParams(-1, -2));

        setContentView(frame);
        SharedPreferences p = prefs();   // restore the last session's inputs
        prompt.setText(p.getString("prompt", "")); seed.setText(p.getString("seed", "-1")); model = p.getString("model", "");
        refreshModels();
        setRes(p.getInt("w", 512), p.getInt("h", 768));
        refreshState();
        if (!getSharedPreferences("anima", 0).getBoolean("welcomed", false)) startActivity(new Intent(this, Welcome.class));
    }

    /** the tag list for the suggestions is read in the background (8000 lines in assets/tags.txt) */
    void loadTags() {
        new Thread(() -> {
            final ArrayList<String> tags = new ArrayList<>();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(getAssets().open("tags.txt"), "UTF-8"))) { for (String l; (l = r.readLine()) != null; ) if (!l.isEmpty()) tags.add(l); } catch (IOException ignored) { }
            ui.post(() -> prompt.setAdapter(new ArrayAdapter<String>(this, android.R.layout.simple_dropdown_item_1line, tags) {
                @Override public View getView(int i, View v, ViewGroup p) { View r = super.getView(i, v, p); ((TextView) r).setTextColor(TXT); ((TextView) r).setTextSize(15); r.setBackgroundColor(CARD); return r; }
            }));
        }).start();
    }

    @Override protected void onPause() {
        super.onPause();
        visible = false;
        ui.removeCallbacks(ramTick);
        prefs().edit().putString("prompt", prompt.getText().toString()).putString("seed", seed.getText().toString()).putString("model", model).putInt("w", resW).putInt("h", resH).apply();
    }

    @Override protected void onResume() {
        super.onResume();
        visible = true;
        ((NotificationManager) getSystemService(NOTIFICATION_SERVICE)).cancel(2);
        String del = ModelsActivity.deleted; ModelsActivity.deleted = "";
        if (!del.isEmpty() && del.equals(model) && loaded && !busy) { loaded = false; ex.submit(() -> su("sh " + D + "/anima.sh stop < /dev/null >/dev/null 2>&1", false)); }
        if (genBtn != null) { refreshModels(); ui.post(ramTick); }
        if (android.os.Build.VERSION.SDK_INT >= 33 && prefs().getBoolean("welcomed", false) && !prefs().getBoolean("asked_notif", false)) {   // once, after the welcome screen
            prefs().edit().putBoolean("asked_notif", true).apply();
            if (checkSelfPermission("android.permission.POST_NOTIFICATIONS") != android.content.pm.PackageManager.PERMISSION_GRANTED) requestPermissions(new String[]{"android.permission.POST_NOTIFICATIONS"}, 5);
        }
    }

    /** every 2 s: RAM strip, thermal / battery / chip warnings; every 10 s the model state (the idle watchdog may have unloaded it) */
    volatile double appGb; volatile boolean appBusy; volatile String ramDetail = "", tempDetail = "";
    final Runnable ramTick = new Runnable() {
        @Override public void run() {
            android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo();
            ((android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(mi);
            double g = 1024.0 * 1024 * 1024, tot = mi.totalMem / g, free = mi.availMem / g, used = tot - free;
            if (!appBusy) { appBusy = true; new Thread(() -> { appGb = Compat.appRamGb(); appBusy = false; }).start(); }   // /proc scan off the UI thread
            ramDetail = L.f("ОЗУ  всего %.1f  ·  занято %.1f  ·  свободно %.1f ГБ", "RAM  total %.1f  ·  used %.1f  ·  free %.1f GB", tot, used, free) + (appGb > 0.05 ? L.f("  ·  приложение %.1f ГБ", "  ·  app %.1f GB", appGb) : "");
            setPill(ramTxt, String.format(java.util.Locale.US, "%.1f/%.1f/%.1f", tot, free, appGb), free < 0.8 ? BAD : free < 1.5 ? WARN : TXT);
            ramBar.setProgress((int) (1000 * used / tot));
            List<String> w = new ArrayList<>();
            if (!Compat.chipAllowed(Main.this)) w.add(L.f("Чип %s не поддерживается", "Chip %s is not supported", Compat.soc()));
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            if (pm.getCurrentThermalStatus() >= PowerManager.THERMAL_STATUS_MODERATE) w.add(L.s("Телефон горячий: скорость снижена", "The phone is hot: speed is reduced"));
            if (pm.isPowerSaveMode()) w.add(L.s("Экономия заряда замедляет NPU", "Battery saver slows the NPU"));
            Intent bat = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
            if (bat != null) { int lvl = bat.getIntExtra("level", -1) * 100 / Math.max(1, bat.getIntExtra("scale", 100)), st = bat.getIntExtra("status", 0); if (lvl >= 0 && lvl < 15 && st != 2 && st != 5) w.add(L.f("Низкий заряд батареи (%d%%)", "Low battery (%d%%)", lvl)); }
            if (bat != null && bat.hasExtra("temperature")) {   // battery temperature (tenths of a degree) is the one reading an app may take; the headroom (Android 11+) says how far the SoC is from throttling
                double tc = bat.getIntExtra("temperature", 0) / 10.0; float hr = android.os.Build.VERSION.SDK_INT >= 30 ? pm.getThermalHeadroom(0) : Float.NaN;
                String t = L.f("Температура  батарея %.1f °C", "Temperature  battery %.1f °C", tc);
                int margin = Float.isNaN(hr) ? -1 : Math.max(0, Math.round(100 * (1 - Math.min(1f, hr))));
                if (margin >= 0) t += L.f("  ·  запас до замедления %d%%", "  ·  margin before throttling %d%%", margin);
                tempDetail = t;
                setPill(tempTxt, String.format(java.util.Locale.US, "%.0f°", tc), tc >= 45 || (margin >= 0 && margin < 10) ? BAD : tc >= 40 || (margin >= 0 && margin < 30) ? WARN : TXT);
            }
            warnTxt.setVisibility(w.isEmpty() ? View.GONE : View.VISIBLE);
            warnTxt.setText("⚠  " + android.text.TextUtils.join("  ·  ", w));
            if (++tick % 5 == 0 && !busy) refreshState();
            ui.postDelayed(this, 2000);
        }
    };

    @Override protected void onActivityResult(int req, int res, Intent d) {
        if (req == AnimaConv.REQ && res == RESULT_OK && d != null && d.getData() != null) { new AnimaConv(this, store, prefs().getString("base", DEFAULT_BASE), () -> refreshModels()).start(d.getData()); return; }
        if (req == 9 && res == RESULT_OK && d != null && d.getData() != null) { importLora(d.getData()); return; }
        if (req == 10 && res == RESULT_OK && d != null && d.getData() != null) { importEdit(d.getData()); return; }
        if (req == 14 && res == RESULT_OK && d != null && d.getData() != null) { if (isAnima()) importTurboAnima(d.getData()); else if (isQwen()) importTurboQwen(d.getData()); else importTurbo(d.getData()); return; }
        if (req == 12 && res == RESULT_OK && d != null && d.getData() != null) { importExtra(d.getData()); return; }
        if ((req == 7 || req == Viewer.REQ) && res == RESULT_OK && d != null && d.hasExtra("prompt")) {
            prompt.setText(d.getStringExtra("prompt")); seed.setText(d.getStringExtra("seed"));
            customOpen = false;
            setRes(d.getIntExtra("w", 512), d.getIntExtra("h", 768));
        } else if (req == 8 && res == RESULT_OK && d != null && d.getData() != null) {
            try (InputStream in = getContentResolver().openInputStream(d.getData()); ByteArrayOutputStream bo = new ByteArrayOutputStream()) {
                byte[] b = new byte[1 << 16]; int total = 0;
                for (int n; (n = in.read(b)) > 0 && total < (40 << 20); ) { bo.write(b, 0, n); total += n; }
                JSONObject m = PngMeta.settings(bo.toByteArray());
                if (m == null || m.optString("prompt").isEmpty()) { Toast.makeText(this, L.s("В этом файле нет настроек генерации", "This file has no generation settings"), Toast.LENGTH_LONG).show(); return; }
                prompt.setText(m.optString("prompt")); seed.setText(m.optString("seed", "-1"));
                customOpen = false; if (m.optInt("w") > 0) setRes(m.optInt("w"), m.optInt("h"));
                Toast.makeText(this, L.s("Настройки загружены из файла", "Settings loaded from the file"), Toast.LENGTH_SHORT).show();
            } catch (Exception e) { Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show(); }
        }
    }


    // ---- Qwen-Image: photo to edit ------------------------------------------------------------------
    boolean isQwen() { return Qwen.is(model); }
    /** Qwen-Image: three fixed rows for the pictures (the empty ones wait in order), Mage-Flow: one photo */
    void refreshEdit() {
        if (editBm == null && prefs().getBoolean("qwen_edit_on", false) && Qwen.editSrc(getFilesDir()).exists()) editBm = BitmapFactory.decodeFile(Qwen.editSrc(getFilesDir()).getPath());
        if (extraBm.isEmpty() && editBm != null) for (int i = 1; i < Qwen.MAX_IMAGES; i++) { File f = Qwen.editSrc(getFilesDir(), i); Bitmap x = f.exists() ? BitmapFactory.decodeFile(f.getPath()) : null; if (x == null) break; extraBm.add(x); }
        final Bitmap b = editBm; final boolean multi = isQwen();
        editThumb.setImageBitmap(b);
        editInfo.setText(b != null ? (multi ? L.f("Картинка 1  ·  %d×%d", "Picture 1  ·  %d×%d", b.getWidth(), b.getHeight()) : L.f("Фото  ·  %d×%d", "Photo  ·  %d×%d", b.getWidth(), b.getHeight())) : (multi ? L.s("Картинка 1", "Picture 1") : L.s("Фото", "Photo")));
        editInfo.setTextColor(b != null ? TXT : DIM);
        editPick.setText(b != null ? L.s("Заменить", "Replace") : L.s("Выбрать", "Choose"));
        editDrop.setVisibility(b != null ? View.VISIBLE : View.INVISIBLE);
        extraBox.removeAllViews();
        if (multi) for (int i = 0; i < Qwen.MAX_IMAGES - 1; i++) {
            final int idx = i; final boolean has = i < extraBm.size(), next = i == extraBm.size() && b != null;
            LinearLayout row = new LinearLayout(this); row.setGravity(Gravity.CENTER_VERTICAL);
            ImageView th = new ImageView(this); th.setScaleType(ImageView.ScaleType.CENTER_CROP); th.setBackground(bg(CARD, 10)); th.setClipToOutline(true); if (has) th.setImageBitmap(extraBm.get(i));
            row.addView(th, new LinearLayout.LayoutParams(dp(56), dp(56)));
            TextView tt = text(has ? L.f("Картинка %d  ·  %d×%d", "Picture %d  ·  %d×%d", i + 2, extraBm.get(i).getWidth(), extraBm.get(i).getHeight()) : L.f("Картинка %d", "Picture %d", i + 2), 13, has ? TXT : DIM, false); tt.setPadding(dp(12), 0, dp(8), 0); row.addView(tt, new LinearLayout.LayoutParams(0, -2, 1));
            Button pk = button(has ? L.s("Заменить", "Replace") : L.s("Выбрать", "Choose"), CARD, TXT); pk.setTextSize(13); pk.setPadding(dp(14), 0, dp(14), 0); pk.setEnabled(has || next); pk.setAlpha(has || next ? 1f : .35f); pk.setOnClickListener(v -> pickExtra(idx));
            row.addView(pk, new LinearLayout.LayoutParams(-2, dp(36)));
            Button x = button("✕", BG, DIM); x.setVisibility(has ? View.VISIBLE : View.INVISIBLE); x.setOnClickListener(v -> dropExtra(idx));
            row.addView(x, new LinearLayout.LayoutParams(dp(40), dp(36)));
            LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(-1, -2); rp.topMargin = dp(8); extraBox.addView(row, rp);
        }
        editAdd.setVisibility(View.GONE);
        if (editNote != null) editNote.setText(b != null && multi ? L.s("В описании ссылайтесь на картинки: «картинка 1», «картинка 2», например «персонаж с картинки 1 стоит на фоне картинки 2».", "Refer to the pictures in the prompt: \"picture 1\", \"picture 2\", e.g. \"the character from picture 1 stands in the scene of picture 2\".")
            : b != null ? L.s("Описание — что в фото изменить, например «сделай небо закатным».", "The prompt says what to change, e.g. \"make the sky a sunset\".") : L.s("Выберите фото, чтобы изменить его по описанию. Без фото модель просто рисует картинку.", "Pick a photo to change it by the prompt. Without one the model just draws a picture."));
    }
    void pickEdit() {
        if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; }
        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("image/*").addCategory(Intent.CATEGORY_OPENABLE), 10);
    }
    void dropEdit() {
        if (busy) return;
        editBm = null; extraBm.clear(); prefs().edit().putBoolean("qwen_edit_on", false).apply();
        ex2.submit(() -> { for (int i = 0; i < Qwen.MAX_IMAGES; i++) Qwen.editSrc(getFilesDir(), i).delete(); Models.deleteTree(Qwen.editDir(getFilesDir())); ui.post(this::refreshEdit); });
        setRes(resW, resH);
    }
    int extraSlot = -1;   // which extra row the picker was opened for
    void pickExtra(int slot) {
        extraSlot = slot;
        if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; }
        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("image/*").addCategory(Intent.CATEGORY_OPENABLE), 12);
    }
    void importExtra(Uri uri) {
        ex2.submit(() -> {
            try {
                final int slot = Math.min(extraSlot < 0 ? extraBm.size() : extraSlot, extraBm.size()); final int idx = slot + 1; if (idx >= Qwen.MAX_IMAGES) return;
                final Bitmap bm = Qwen.importPhoto(getContentResolver(), uri, getFilesDir(), idx);
                Models.deleteTree(Qwen.editDir(getFilesDir(), idx));   // the cached condition of a replaced picture is stale
                ui.post(() -> { if (slot < extraBm.size()) extraBm.set(slot, bm); else extraBm.add(bm); refreshEdit(); });
            } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Не удалось открыть фото: %s", "Cannot open the photo: %s", e.getMessage()), Toast.LENGTH_LONG).show()); }
        });
    }
    /** removes extra picture idx (0 = "picture 2"); the later ones move up, their cached condition files are rebuilt */
    void dropExtra(int idx) {
        if (busy || idx >= extraBm.size()) return;
        extraBm.remove(idx);
        ex2.submit(() -> {
            final File home = getFilesDir(); int n = extraBm.size();
            for (int i = idx + 1; i <= n + 1; i++) { File from = Qwen.editSrc(home, i + 1), to = Qwen.editSrc(home, i); to.delete(); if (from.exists()) from.renameTo(to); }
            Qwen.editSrc(home, n + 1).delete();
            for (int i = 1; i < Qwen.MAX_IMAGES; i++) Models.deleteTree(Qwen.editDir(home, i));
            ui.post(this::refreshEdit);
        });
    }
    void importEdit(Uri uri) {
        ex2.submit(() -> {
            try {
                final Bitmap bm = Qwen.importPhoto(getContentResolver(), uri, getFilesDir(), 0);
                prefs().edit().putBoolean("qwen_edit_on", true).apply();
                ui.post(() -> {
                    editBm = bm; refreshEdit();
                    // the output size follows the photo's shape: the preset with the closest aspect ratio, at most 0.8 Mpx
                    double r = (double) bm.getWidth() / bm.getHeight(), best = 1e9; int[] pick = null;
                    for (int[] z : sizeList()) { if (z[0] * z[1] > 800000) continue; double d = Math.abs(Math.log((double) z[0] / z[1] / r)) + Math.abs(z[0] * z[1] - 393216) / 5e7; if (d < best) { best = d; pick = z; } }
                    if (pick != null) { customOpen = false; setRes(pick[0], pick[1]); }
                });
            } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Не удалось открыть фото: %s", "Cannot open the photo: %s", e.getMessage()), Toast.LENGTH_LONG).show()); }
        });
    }

    // ---- MOSS-Image user LoRA ---------------------------------------------------------------------
    /** the LoRA box serves MOSS-Image and the Anima models; each keeps its own file, name, info and strength under its own prefix */
    String lp() { return isMoss() ? "moss_lora_" : "anima_lora_"; }
    File loraSrc() { return isMoss() ? MossLora.src(getFilesDir()) : AnimaLora.src(getFilesDir()); }
    String loraId() { return prefs().getString(lp() + "id", ""); }
    boolean loraOn() { if (isAnima()) return !AnimaLoras.env(prefs(), getFilesDir()).isEmpty(); return isMoss() && !moss8() && !customTurbo() && !loraId().isEmpty() && loraSrc().exists() && strBar.getProgress() > 0; }
    void refreshLora() {
        if (loraName == null) return;
        boolean has = !loraId().isEmpty() && loraSrc().exists();
        strBar.setProgress(Math.round(prefs().getFloat(lp() + "str", 1f) * 20));
        loraName.setText(has ? prefs().getString(lp() + "name", "LoRA") : L.s("не выбрана", "none"));
        loraInfo.setText(has ? prefs().getString(lp() + "info", "") : isMoss() ? L.s("Файл .safetensors для MOSS-Image: проверяется сразу после выбора", "A .safetensors file for MOSS-Image: it is checked as soon as you pick it") : L.s("Файл .safetensors для Anima: проверяется сразу после выбора", "A .safetensors file for Anima: it is checked as soon as you pick it"));
        loraInfo.setTextColor(DIM);
        loraDrop.setVisibility(has ? View.VISIBLE : View.GONE); ((View) strBar.getParent()).setVisibility(has ? View.VISIBLE : View.GONE); loraName.setTextColor(has ? TXT : DIM); loraInfo.setVisibility(has || loraInfo.getCurrentTextColor() == WARN ? View.VISIBLE : View.GONE);
    }
    /** the Anima LoRA library as rows: switch, name, remove, and the strength under it */
    void refreshAnimaLoras() {
        if (aloraList == null) return;
        aloraList.removeAllViews();
        final java.util.List<AnimaLoras.Item> ls = AnimaLoras.load(prefs(), getFilesDir()); int act = 0;
        for (final AnimaLoras.Item it : ls) {
            if (AnimaLoras.active(it)) act++;
            LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL); row.setPadding(0, dp(4), 0, dp(6));
            LinearLayout top = new LinearLayout(this); top.setGravity(Gravity.CENTER_VERTICAL);
            Switch sw = new Switch(this); sw.setChecked(it.on);
            sw.setThumbTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{ACC, DIM}));
            sw.setTrackTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{(ACC & 0x00FFFFFF) | 0x66000000, (DIM & 0x00FFFFFF) | 0x44000000}));
            top.addView(sw);
            TextView nm = text(it.name.replaceAll("(?i)\\.safetensors$", ""), 14, it.on ? TXT : DIM, false); nm.setSingleLine(true); nm.setEllipsize(android.text.TextUtils.TruncateAt.MIDDLE); nm.setPadding(dp(10), 0, 0, 0);
            top.addView(nm, new LinearLayout.LayoutParams(0, -2, 1));
            TextView pen = text("✎", 16, DIM, false); pen.setGravity(Gravity.CENTER); pen.setContentDescription(L.s("Переименовать", "Rename"));
            pen.setOnClickListener(v -> {
                if (busy) return;
                final EditText e = new EditText(this); e.setText(it.name.replaceAll("(?i)\\.safetensors$", "")); e.setSingleLine(true); e.setSelectAllOnFocus(true); e.setTextColor(TXT);
                builder().setTitle(L.s("Название LoRA", "LoRA name")).setView(e).setNegativeButton(L.s("Отмена", "Cancel"), null)
                    .setPositiveButton("OK", (d, w) -> { String n = e.getText().toString().trim(); if (!n.isEmpty()) { it.name = n; AnimaLoras.save(prefs(), ls); refreshAnimaLoras(); } }).show();
            });
            top.addView(pen, new LinearLayout.LayoutParams(dp(40), dp(36)));
            TextView del = text("✕", 16, DIM, false); del.setGravity(Gravity.CENTER); del.setContentDescription(L.s("Убрать", "Remove"));
            del.setOnClickListener(v -> { if (busy) return; builder().setMessage(L.f("Убрать «%s» из списка?", "Remove \"%s\" from the list?", it.name)).setNegativeButton(L.s("Отмена", "Cancel"), null)
                .setPositiveButton(L.s("Убрать", "Remove"), (d, w) -> { ls.remove(it); AnimaLoras.file(getFilesDir(), it.id).delete(); AnimaLoras.save(prefs(), ls); refreshAnimaLoras(); }).show(); });
            top.addView(del, new LinearLayout.LayoutParams(dp(40), dp(36)));
            row.addView(top);
            LinearLayout sr = new LinearLayout(this); sr.setGravity(Gravity.CENTER_VERTICAL);
            final SeekBar bar = new SeekBar(this); bar.setMax(40); bar.setProgress(Math.round(it.str * 20));
            bar.setProgressTintList(ColorStateList.valueOf(TXT)); bar.setThumbTintList(ColorStateList.valueOf(TXT)); bar.setProgressBackgroundTintList(ColorStateList.valueOf(LINE));
            sr.addView(bar, new LinearLayout.LayoutParams(0, -2, 1));
            final TextView st = text(String.format(java.util.Locale.US, "%.2f", it.str), 14, TXT, false); st.setMinWidth(dp(40)); st.setGravity(Gravity.END); sr.addView(st);
            bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                public void onProgressChanged(SeekBar b, int p, boolean u) { st.setText(String.format(java.util.Locale.US, "%.2f", p * 0.05f)); if (u) { it.str = p * 0.05f; AnimaLoras.save(prefs(), ls); } }
                public void onStartTrackingTouch(SeekBar b) { }
                public void onStopTrackingTouch(SeekBar b) { refreshAnimaInfo(ls); }
            });
            sw.setOnCheckedChangeListener((b, on) -> { if (busy) { b.setChecked(it.on); return; } it.on = on; AnimaLoras.save(prefs(), ls); refreshAnimaLoras(); });
            row.addView(sr); aloraList.addView(row);
        }
        refreshAnimaInfo(ls);
    }
    void refreshAnimaInfo(java.util.List<AnimaLoras.Item> ls) {
        int act = 0; for (AnimaLoras.Item it : ls) if (AnimaLoras.active(it)) act++;
        aloraInfo.setTextColor(DIM);
        aloraInfo.setText(ls.isEmpty() ? L.s("Файлы .safetensors для Anima (kohya, diffusers/PEFT, LoKr, LoHa): добавляйте сколько нужно. У каждого свой выключатель и сила. LoRA вливаются в модель при загрузке, поэтому первая картинка с новым набором на 10-15 секунд дольше.", "Own .safetensors files for Anima (kohya, diffusers/PEFT, LoKr, LoHa): add as many as you need, each with its own switch and strength. They are folded into the model when it loads, so the first picture with a new set takes 10-15 seconds longer.")
            : L.f("Включено: %d из %d. Новый набор или сила: первая картинка на 10-15 секунд дольше.", "Active: %d of %d. A new set or strength makes the first picture 10-15 seconds slower.", act, ls.size()));
    }
    void importAnimaLora(Uri uri) {
        String nm = "LoRA";
        try (android.database.Cursor c = getContentResolver().query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) nm = c.getString(i); } } catch (Exception ignored) { }
        final String name = nm;
        aloraInfo.setTextColor(WARN); aloraInfo.setText(L.s("Проверяю файл…", "Checking the file…"));
        ex2.submit(() -> {
            final File dir = AnimaLoras.dir(getFilesDir()); dir.mkdirs(); final File tmp = new File(dir, "new.tmp");
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 16]; long total = 0;
                for (int n; (n = in.read(b)) > 0; ) { o.write(b, 0, n); total += n; if (total > (1L << 30)) throw new IOException(L.s("файл больше 1 ГБ", "the file is larger than 1 GB")); }
            } catch (Exception e) { tmp.delete(); ui.post(() -> { refreshAnimaLoras(); loraFail(name, L.f("Не удалось прочитать файл: %s", "Cannot read the file: %s", e.getMessage())); }); return; }
            AnimaLora.Check c = AnimaLora.check(tmp);
            if (!c.ok) { tmp.delete(); final String msg = c.message(); ui.post(() -> { refreshAnimaLoras(); loraFail(name, msg); }); return; }
            AnimaLoras.Item it = new AnimaLoras.Item(); it.id = Long.toHexString(System.currentTimeMillis()); it.name = name; it.info = "✓ " + c.message();
            tmp.renameTo(AnimaLoras.file(getFilesDir(), it.id));
            java.util.List<AnimaLoras.Item> ls = AnimaLoras.load(prefs(), getFilesDir()); ls.add(it); AnimaLoras.save(prefs(), ls);
            ui.post(this::refreshAnimaLoras);
        });
    }
    void pickLora() {
        if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; }
        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("*/*").addCategory(Intent.CATEGORY_OPENABLE), 9);
    }
    void dropLora() {
        if (busy) return;
        final String pf = lp(); final boolean moss = isMoss();
        prefs().edit().remove(pf + "id").remove(pf + "name").remove(pf + "info").apply();
        ex2.submit(() -> { File d = moss ? MossLora.dir(getFilesDir()) : AnimaLora.dir(getFilesDir()); File[] fs = d.listFiles(); if (fs != null) for (File f : fs) if (!f.getName().startsWith("turbo")) MossLora.deleteTree(f); ui.post(this::refreshLora); });
    }
    void importLora(Uri uri) {
        if (isAnima()) { importAnimaLora(uri); return; }
        String nm = "LoRA";
        try (android.database.Cursor c = getContentResolver().query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) nm = c.getString(i); } } catch (Exception ignored) { }
        final String name = nm;
        loraInfo.setTextColor(WARN); loraInfo.setVisibility(View.VISIBLE); loraInfo.setText(L.s("Проверяю файл…", "Checking the file…"));
        ex2.submit(() -> {
            final boolean moss = isMoss(); final String pf = lp(); final File src = loraSrc();
            final File dir = moss ? MossLora.dir(getFilesDir()) : AnimaLora.dir(getFilesDir()); dir.mkdirs(); final File tmp = new File(dir, "new.safetensors");
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 16]; long total = 0;
                for (int n; (n = in.read(b)) > 0; ) { o.write(b, 0, n); total += n; if (total > (1L << 30)) throw new IOException(L.s("файл больше 1 ГБ", "the file is larger than 1 GB")); }
            } catch (Exception e) { tmp.delete(); ui.post(() -> { refreshLora(); loraFail(name, L.f("Не удалось прочитать файл: %s", "Cannot read the file: %s", e.getMessage())); }); return; }
            final boolean ok; final String msg;
            if (moss) { MossLora.Check c = MossLora.check(BIN, tmp); ok = c.ok; msg = c.message(); } else { AnimaLora.Check c = AnimaLora.check(tmp); ok = c.ok; msg = c.message(); }
            if (!ok) { tmp.delete(); ui.post(() -> { refreshLora(); loraFail(name, msg); }); return; }
            File[] old = dir.listFiles(); if (old != null) for (File f : old) if (!f.equals(tmp) && !f.getName().startsWith("turbo")) MossLora.deleteTree(f);   // the old LoRA and its folded patches go (the Lab turbo LoRA stays)
            tmp.renameTo(src);
            prefs().edit().putString(pf + "id", Long.toHexString(System.currentTimeMillis())).putString(pf + "name", name).putString(pf + "info", "✓ " + msg).apply();
            ui.post(this::refreshLora);
        });
    }
    /** Lab: import a turbo LoRA file for MOSS (checked like a user LoRA), then ask which sampler it was trained with */
    void importTurbo(Uri uri) {
        String nm = "LoRA";
        try (android.database.Cursor c = getContentResolver().query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) nm = c.getString(i); } } catch (Exception ignored) { }
        final String name = nm;
        ex2.submit(() -> {
            final File dir = MossLora.dir(getFilesDir()); dir.mkdirs(); final File tmp = new File(dir, "turbo_new.safetensors");
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 16]; long total = 0;
                for (int n; (n = in.read(b)) > 0; ) { o.write(b, 0, n); total += n; if (total > (2L << 30)) throw new IOException(L.s("файл больше 2 ГБ", "the file is larger than 2 GB")); }
            } catch (Exception e) { tmp.delete(); ui.post(() -> loraFail(name, L.f("Не удалось прочитать файл: %s", "Cannot read the file: %s", e.getMessage()))); return; }
            final MossLora.Check c = MossLora.check(BIN, tmp);
            if (!c.ok) { tmp.delete(); ui.post(() -> loraFail(name, c.message())); return; }
            ui.post(() -> builder().setTitle(L.s("Для какого расписания эта LoRA?", "Which sampler is this LoRA made for?")).setItems(new String[]{L.s("4 шага (как rapid-moss)", "4 steps (like rapid-moss)"), L.s("8 шагов (как DMD, нужен 8 step)", "8 steps (like DMD, needs 8 step)")}, (dd, k) -> {
                final int sampler = k == 0 ? 4 : 8;
                if (sampler == 8 && !ids.contains(Moss.ID8)) { tmp.delete(); Toast.makeText(this, L.s("Для 8 шагов нужна добавка MOSS 8 step", "8 steps need the MOSS 8 step add-on"), Toast.LENGTH_LONG).show(); return; }
                ex2.submit(() -> {
                    MossLora.deleteTree(MossLora.turboOvl(getFilesDir())); MossLora.turboSrc(getFilesDir()).delete(); tmp.renameTo(MossLora.turboSrc(getFilesDir()));
                    prefs().edit().putString("turbo_name", name).putInt("turbo_sampler", sampler).putBoolean("turbo_on", true).apply();
                    ui.post(() -> { setModelUi(false, false, ""); Toast.makeText(this, "✓ " + c.message(), Toast.LENGTH_LONG).show(); });
                });
            }).setNegativeButton(L.s("Отмена", "Cancel"), (dd, k) -> tmp.delete()).setOnCancelListener(dd -> tmp.delete()).show());
        });
    }
    /** Lab: import a turbo LoRA file for Qwen-Image, then ask which sigma table it was trained with */
    void importTurboQwen(Uri uri) {
        String nm = "LoRA";
        try (android.database.Cursor c = getContentResolver().query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) nm = c.getString(i); } } catch (Exception ignored) { }
        final String name = nm;
        ex2.submit(() -> {
            final File dir = QwenLora.dir(getFilesDir()); dir.mkdirs(); final File tmp = new File(dir, "turbo_new.safetensors");
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 16]; long total = 0;
                for (int n; (n = in.read(b)) > 0; ) { o.write(b, 0, n); total += n; if (total > (2L << 30)) throw new IOException(L.s("файл больше 2 ГБ", "the file is larger than 2 GB")); }
            } catch (Exception e) { tmp.delete(); ui.post(() -> loraFail(name, L.f("Не удалось прочитать файл: %s", "Cannot read the file: %s", e.getMessage()))); return; }
            final QwenLora.Check c = QwenLora.check(BIN, tmp);
            if (!c.ok) { tmp.delete(); ui.post(() -> loraFail(name, c.message())); return; }
            ui.post(() -> builder().setTitle(L.s("Для какого расписания эта LoRA?", "Which sampler is this LoRA made for?")).setItems(new String[]{L.s("4 шага (как Lightning)", "4 steps (like Lightning)"), L.s("8 шагов (как обычный Turbo)", "8 steps (like plain Turbo)")}, (dd, k) -> {
                final int sampler = k == 0 ? 4 : 8;
                ex2.submit(() -> {
                    MossLora.deleteTree(QwenLora.turboOvl(getFilesDir())); QwenLora.turboSrc(getFilesDir()).delete(); tmp.renameTo(QwenLora.turboSrc(getFilesDir()));
                    prefs().edit().putString("qturbo_name", name).putInt("qturbo_sampler", sampler).putBoolean("qturbo_on", true).apply();
                    ui.post(() -> { Toast.makeText(this, "✓ " + c.message(), Toast.LENGTH_LONG).show(); if (!Qwen.is8(model) && ids.contains(Qwen.ID8)) setModel(Qwen.ID8); else setModelUi(false, false, ""); });
                });
            }).setNegativeButton(L.s("Отмена", "Cancel"), (dd, k) -> tmp.delete()).setOnCancelListener(dd -> tmp.delete()).show());
        });
    }
    /** Lab: a turbo LoRA file for an Anima base pack, checked like a user LoRA (header only) */
    void importTurboAnima(Uri uri) {
        String nm = "LoRA";
        try (android.database.Cursor c = getContentResolver().query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) nm = c.getString(i); } } catch (Exception ignored) { }
        final String name = nm;
        ex2.submit(() -> {
            final File dir = AnimaLora.dir(getFilesDir()); dir.mkdirs(); final File tmp = new File(dir, "turbo_new.safetensors");
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 16]; long total = 0;
                for (int n; (n = in.read(b)) > 0; ) { o.write(b, 0, n); total += n; if (total > (1L << 30)) throw new IOException(L.s("файл больше 1 ГБ", "the file is larger than 1 GB")); }
            } catch (Exception e) { tmp.delete(); ui.post(() -> loraFail(name, L.f("Не удалось прочитать файл: %s", "Cannot read the file: %s", e.getMessage()))); return; }
            final AnimaLora.Check c = AnimaLora.check(tmp);
            if (!c.ok) { tmp.delete(); ui.post(() -> loraFail(name, c.message())); return; }
            AnimaLora.turboSrc(getFilesDir()).delete(); tmp.renameTo(AnimaLora.turboSrc(getFilesDir()));
            prefs().edit().putString("aturbo_name", name).putBoolean("aturbo_on", true).apply();
            ui.post(() -> { Toast.makeText(this, "✓ " + c.message(), Toast.LENGTH_LONG).show(); final String fam = famOf(model); if (!isBase(model) && ids.contains(fam + "base")) setModel(fam + "base"); else setModelUi(false, false, ""); });
        });
    }
    void loraFail(String name, String why) {
        builder().setTitle(L.f("LoRA «%s» не подходит", "LoRA \"%s\" does not fit", name)).setMessage(why).setPositiveButton("OK", null).show();
    }

    // ---- menu, dialogs ------------------------------------------------------------------------
    void menu(View anchor) {
        final LinearLayout col = new LinearLayout(this); col.setOrientation(LinearLayout.VERTICAL); col.setPadding(dp(10), dp(14), dp(10), dp(14));
        final AlertDialog[] dlg = new AlertDialog[1];
        final String[] icons = {"ic_layers", "ic_add_photo", "ic_tune", "ic_info"};
        final String[] names = {L.s("Модели", "Models"), L.s("Открыть настройки из PNG", "Load settings from a PNG"), L.s("Настройки", "Settings"), L.s("Инфо", "Info")};
        for (int i = 0; i < names.length; i++) {
            final int k = i;
            LinearLayout row = new LinearLayout(this); row.setGravity(Gravity.CENTER_VERTICAL); row.setPadding(dp(16), dp(16), dp(16), dp(16)); row.setBackground(bg(CARD, 12));
            TextView t = text(names[i], 16, TXT, false); row.addView(t);
            row.setOnClickListener(v -> {
                dlg[0].dismiss();
                if (k == 0) openModels();
                else if (k == 1) startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("image/png").addCategory(Intent.CATEGORY_OPENABLE), 8);
                else if (k == 2) startActivity(new Intent(this, Settings.class));
                else startActivity(new Intent(this, Info.class));
            });
            LinearLayout.LayoutParams rlp = new LinearLayout.LayoutParams(-1, -2); rlp.bottomMargin = dp(8); col.addView(row, rlp);
        }
        dlg[0] = sheet(col);
    }

    void openModels() { if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; } startActivity(new Intent(this, ModelsActivity.class)); }

    /** saved and recent prompts: tap to use; saved ones can be removed with ✕ */
    void templates() {
        final LinearLayout box = new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setPadding(dp(14), dp(14), dp(14), dp(10));
        final AlertDialog[] dlg = new AlertDialog[1];
        final Runnable[] fill = new Runnable[1];
        fill[0] = () -> {
            box.removeAllViews();
            Button save = button(L.s("Сохранить текущее описание", "Save the current prompt"), CARD, TXT); save.setTextSize(14);
            save.setOnClickListener(v -> {
                String t = prompt.getText().toString().trim();
                if (t.isEmpty()) { Toast.makeText(this, L.s("Введите описание", "Enter a prompt"), Toast.LENGTH_SHORT).show(); return; }
                List<String> l = savedPrompts(); l.remove(t); l.add(0, t); while (l.size() > 30) l.remove(l.size() - 1);
                prefs().edit().putString("saved", new JSONArray(l).toString()).apply(); fill[0].run();
            });
            box.addView(save, new LinearLayout.LayoutParams(-1, dp(44)));
            List<String> sv = savedPrompts();
            if (!sv.isEmpty()) { box.addView(sect(L.s("Сохранённые", "Saved"))); for (String t : sv) box.addView(promptRow(t, true, dlg, fill)); }
            List<String> rc = recentPrompts();
            if (!rc.isEmpty()) { box.addView(sect(L.s("Недавние", "Recent"))); for (String t : rc) box.addView(promptRow(t, false, dlg, fill)); }
        };
        fill[0].run();
        ScrollView s = new ScrollView(this); s.addView(box);
        dlg[0] = dialog(s);
    }

    TextView sect(String s) { TextView t = text(s, 12, DIM, true); t.setPadding(dp(4), dp(14), 0, dp(6)); return t; }

    View promptRow(String t, boolean removable, AlertDialog[] dlg, Runnable[] fill) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setPadding(dp(4), dp(12), dp(4), dp(12));
        TextView tv = text(t, 13, TXT, false); tv.setMaxLines(2); tv.setEllipsize(android.text.TextUtils.TruncateAt.END);
        r.addView(tv, new LinearLayout.LayoutParams(0, -2, 1));
        if (removable) {
            TextView x = text("✕", 16, DIM, false); x.setPadding(dp(12), dp(4), dp(8), dp(4));
            x.setOnClickListener(v -> { List<String> l = savedPrompts(); l.remove(t); prefs().edit().putString("saved", new JSONArray(l).toString()).apply(); fill[0].run(); });
            r.addView(x);
        }
        r.setOnClickListener(v -> { prompt.setText(t); prompt.setSelection(t.length()); dlg[0].dismiss(); });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.topMargin = dp(4);
        r.setLayoutParams(lp);
        return r;
    }

    List<String> savedPrompts() {
        List<String> l = new ArrayList<>();
        try { JSONArray a = new JSONArray(prefs().getString("saved", "[]")); for (int i = 0; i < a.length(); i++) l.add(a.getString(i)); } catch (Exception ignored) { }
        return l;
    }

    List<String> recentPrompts() {
        List<String> l = new ArrayList<>();
        File[] fs = Gallery.list(this, Gallery.ALL);
        for (int i = 0; i < fs.length && l.size() < 10 && i < 60; i++) { String p = Gallery.meta(fs[i]).optString("prompt").trim(); if (!p.isEmpty() && !l.contains(p)) l.add(p); }
        return l;
    }

    static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
    static String sizeLabel(int w, int h) { int g = gcd(w, h); return w + "×" + h + "  ·  " + w / g + ":" + h / g; }

    /** Size picker: the portrait sizes in the left column, their landscape twins in the right one, squares below; every group runs from the smallest to the largest picture. */
    void pickSize() {
        final AlertDialog[] dlg = new AlertDialog[1];
        LinearLayout box = new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setPadding(dp(14), dp(14), dp(14), dp(10));
        java.util.List<int[]> por = new java.util.ArrayList<>(), lan = new java.util.ArrayList<>(), sq = new java.util.ArrayList<>();
        for (int[] z : sizeList()) (z[1] > z[0] ? por : z[0] > z[1] ? lan : sq).add(z);
        java.util.Comparator<int[]> byArea = (x, y) -> { long ax = (long) x[0] * x[1], ay = (long) y[0] * y[1]; return ax != ay ? Long.compare(ax, ay) : Integer.compare(Math.abs(x[1] - x[0]), Math.abs(y[1] - y[0])); };
        java.util.Collections.sort(por, byArea); java.util.Collections.sort(lan, byArea); java.util.Collections.sort(sq, byArea);
        box.addView(cols(text(L.s("Портрет", "Portrait"), 12, DIM, true), text(L.s("Альбом", "Landscape"), 12, DIM, true), true));
        for (int i = 0; i < Math.max(por.size(), lan.size()); i++) box.addView(cols(i < por.size() ? cell(por.get(i), dlg) : new View(this), i < lan.size() ? cell(lan.get(i), dlg) : new View(this), false));
        TextView sqh = text(L.s("Квадрат", "Square"), 12, DIM, true); sqh.setPadding(dp(4), dp(14), 0, dp(6)); box.addView(sqh);
        for (int i = 0; i < sq.size(); i += 2) box.addView(cols(cell(sq.get(i), dlg), i + 1 < sq.size() ? cell(sq.get(i + 1), dlg) : new View(this), false));
        ScrollView sv = new ScrollView(this); sv.addView(box);
        dlg[0] = dialog(sv);
    }

    LinearLayout cols(View l, View r, boolean header) {
        LinearLayout row = new LinearLayout(this); row.setPadding(0, dp(3), 0, dp(3));
        LinearLayout.LayoutParams a = new LinearLayout.LayoutParams(0, -2, 1); a.rightMargin = dp(4);
        LinearLayout.LayoutParams b = new LinearLayout.LayoutParams(0, -2, 1); b.leftMargin = dp(4);
        if (header) { l.setPadding(dp(4), 0, 0, dp(2)); r.setPadding(dp(4), 0, 0, dp(2)); }
        row.addView(l, a); row.addView(r, b);
        return row;
    }

    TextView cell(int[] z, AlertDialog[] dlg) {
        boolean on = z[0] == resW && z[1] == resH;
        TextView t = text(sizeLabel(z[0], z[1]), 14, on ? ON_ACC : TXT, on);
        t.setGravity(Gravity.CENTER); t.setPadding(dp(6), dp(12), dp(6), dp(12)); t.setBackground(bg(on ? ACC : FIELD, 14));
        t.setOnClickListener(v -> { customOpen = false; setRes(z[0], z[1]); dlg[0].dismiss(); });
        return t;
    }

    boolean isMoss() { return Moss.is(model); }
    boolean isMage() { return Mage.is(model); }
    boolean isH3() { return H3.is(model); }
    /** CFG of the picture: the base models keep theirs (base_cfg, 4), the turbo ones have their own (turbo_cfg, 1 = no second pass) */
    // the picture settings live in prefs() (the "lab" file in lab mode), not in Settings.prefs(): read them from the same place they are written
    String smp() { String v = prefs().getString("sampler", "euler"); return java.util.Arrays.asList(Settings.SAMPLERS).contains(v) ? v : "euler"; }
    String sch() { String v = prefs().getString("scheduler", ""); return java.util.Arrays.asList(Settings.SCHEDULERS).contains(v) ? v : ""; }
    String negStr() { String n = prefs().getString("base_neg", Settings.DEFAULT_NEG).trim(); return n.isEmpty() ? Settings.DEFAULT_NEG : n; }
    String cfgKey() { return base() ? "base_cfg" : "turbo_cfg"; }
    String cfgStr() { return prefs().getString(cfgKey(), base() ? "4" : "1"); }
    boolean isAnima() { return !model.isEmpty() && !isMoss() && !isQwen() && !isMage() && !isH3(); }
    /** the size presets of the selected model: Anima's list, or MOSS-Image's own presets (any size with sides multiple of 16 works) */
    java.util.List<int[]> sizeList() { return isH3() ? java.util.Arrays.asList(H3.PRESETS) : isMage() ? Mage.presets(editBm != null) : java.util.Arrays.asList(isMoss() ? Moss.PRESETS : isQwen() ? Qwen.PRESETS : SIZES); }
    String mossSizeError(int w, int h) { return Moss.sizeError(w, h); }

    // ---- model state --------------------------------------------------------------------------
    LinearLayout dropBtn(String title, TextView value, View.OnClickListener l) {
        LinearLayout b = new LinearLayout(this); b.setGravity(Gravity.CENTER_VERTICAL); b.setBackground(bg(CARD, 16)); b.setPadding(dp(16), 0, dp(16), 0);
        b.addView(text(title, 12, DIM, true)); value.setPadding(dp(12), 0, 0, 0); b.addView(value, new LinearLayout.LayoutParams(0, -2, 1)); b.addView(text("▾", 14, DIM, false));
        b.setOnClickListener(l); return b;
    }
    void refreshBaseUi() {
        if (stepsVal != null) { int st = steps(); stepsVal.setText(String.valueOf(st)); stepsVal.setTextColor(st == stepsDef() ? TXT : WARN); }   // an orange number: not the recipe of the model
        if (cfgVal != null) cfgVal.setText(cfgStr());
        if (samplerVal != null) samplerVal.setText(smp() + "  ▾");
        if (schedVal != null) { String sc = sch(); schedVal.setText((sc.isEmpty() ? L.s("по умолчанию", "default") : sc) + "  ▾"); }
        updateChips();
        if (genBtn != null && !busy) genBtn.setText(genLabel());
    }
    void pickBaseSteps() {
        final EditText e = new EditText(this); e.setText(String.valueOf(steps())); e.setSelectAllOnFocus(true); e.setTextColor(TXT); e.setInputType(InputType.TYPE_CLASS_NUMBER);
        builder().setTitle(L.f("Шагов (%d–%d, по умолчанию %d)", "Steps (%d–%d, default %d)", stepsMin(), stepsMax(), stepsDef())).setView(e).setNegativeButton(L.s("Отмена", "Cancel"), null).setPositiveButton("OK", (dd, w) -> {
            try { setSteps(Integer.parseInt(e.getText().toString().trim())); } catch (NumberFormatException x) { Toast.makeText(this, L.f("Введите число от %d до %d", "Enter a number from %d to %d", stepsMin(), stepsMax()), Toast.LENGTH_SHORT).show(); }
        }).show();
    }
    void pickBaseCfg() {
        final String cur = cfgStr(); final String[] cfgs = base() ? Settings.BASE_CFG : new String[]{"1", "2", "3", "4"}; final String ck = cfgKey();
        final String[] opts = new String[cfgs.length + 1]; System.arraycopy(cfgs, 0, opts, 0, cfgs.length); opts[opts.length - 1] = L.s("Своё значение…", "Custom value…");
        int ci = java.util.Arrays.asList(cfgs).indexOf(cur); if (ci < 0) ci = opts.length - 1;
        builder().setTitle("CFG").setSingleChoiceItems(opts, ci, (d, i) -> {
            d.dismiss();
            if (i < cfgs.length) { prefs().edit().putString(ck, cfgs[i]).apply(); refreshBaseUi(); return; }
            final EditText e = new EditText(this); e.setText(cur); e.setSelectAllOnFocus(true); e.setTextColor(TXT); e.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
            builder().setTitle("CFG (1–20)").setView(e).setNegativeButton(L.s("Отмена", "Cancel"), null).setPositiveButton("OK", (dd, w) -> {
                try { float v = Float.parseFloat(e.getText().toString().trim().replace(',', '.')); if (v < 1 || v > 20) throw new NumberFormatException();
                    prefs().edit().putString(ck, String.valueOf(Math.round(v * 100) / 100f).replaceAll("\\.0$", "")).apply(); refreshBaseUi(); }
                catch (NumberFormatException x) { Toast.makeText(this, L.s("Введите число от 1 до 20", "Enter a number from 1 to 20"), Toast.LENGTH_SHORT).show(); }
            }).show();
        }).show();
    }
    String modelName() {   // the family name as in the model list: the 4/8 step variant is picked by the LoRA and the steps, it does not rename the model
        if (model.isEmpty()) return L.s("нет модели", "no model");
        String al = store.alias(Models.aliasKey(model)); if (al != null) return al;   // a name the user gave
        if (isQwen()) return "Qwen-Image-2.1-Turbo";
        if (Moss.isRs(model)) return store.name(model);
        if (isMoss() && ids.contains(mossBase())) return store.name(mossBase()).replace(" (base)", "");
        return store.name(model);
    }

    /** a name of the user's own for a model (stored in model_names.json; an empty name or Reset brings the original back) */
    void renameModel(final android.app.AlertDialog picker, final String key, String current, boolean hasAlias) {
        final EditText e = new EditText(this); e.setText(current); e.setSingleLine(true); e.setSelectAllOnFocus(true); e.setTextColor(TXT);
        android.app.AlertDialog.Builder b = builder().setTitle(L.s("Название модели", "Model name")).setView(e).setNegativeButton(L.s("Отмена", "Cancel"), null)
            .setPositiveButton("OK", (d, w) -> { store.setAlias(key, e.getText().toString()); picker.dismiss(); if (chipName != null) chipName.setText(modelName()); pickModel(); });
        if (hasAlias) b.setNeutralButton(L.s("Сбросить", "Reset"), (d, w) -> { store.setAlias(key, null); picker.dismiss(); if (chipName != null) chipName.setText(modelName()); pickModel(); });
        b.show();
    }

    void pickModel() {
        if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; }
        if (ids.isEmpty()) { openModels(); return; }
        final LinearLayout col = new LinearLayout(this); col.setOrientation(LinearLayout.VERTICAL); col.setPadding(dp(16), dp(14), dp(16), dp(10));
        final android.app.AlertDialog[] dlg = new android.app.AlertDialog[1];
        TextView title = text(L.s("Модель", "Model"), 20, TXT, true); title.setPadding(dp(2), 0, 0, dp(6)); col.addView(title);
        final String[][] groups = {{"anima", "ANIMA"}, {"moss", "MOSS-IMAGE"}, {"qwen", "QWEN-IMAGE"}, {"mage", "MAGE-FLOW"}, {"h3", "MINIMAX-H3"}};
        for (String[] g : groups) {
            boolean head = false; final java.util.Set<String> seen = new java.util.HashSet<>();
            for (final String id0 : ids) {
                String mg = modelGroup(id0); if (!(g[0].equals("anima") ? mg.equals("turbo") || mg.equals("base") : mg.equals(g[0]))) continue;
                final boolean anima = g[0].equals("anima"), qw = g[0].equals("qwen"), ms = g[0].equals("moss"); final String fam = anima ? famOf(id0) : qw ? "qwen" : ms ? (Moss.isRs(id0) ? "moss-" + Moss.sub(id0) : "moss") : id0;
                if (!seen.add(fam)) continue;
                final String id = anima && ids.contains(fam) ? fam : id0; final boolean fm = anima || qw;   // the name of a family is the turbo one when it is installed
                if (!head) { head = true; TextView h = text(g[1], 12, DIM, false); h.setLetterSpacing(0.04f); h.setPadding(dp(4), dp(18), 0, dp(6)); col.addView(h); }
                final boolean sel = anima ? famOf(model).equals(fam) && isAnima() : qw ? isQwen() : ms ? isMoss() && (Moss.isRs(id0) ? model.equals(id0) : !Moss.isRs(model)) : id.equals(model);
                LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL); row.setPadding(dp(14), dp(10), dp(14), dp(10));
                { GradientDrawable rb = bg(CARD, 12); if (sel) rb.setStroke(dp(2), TXT); row.setBackground(rb); }
                final String akey = Models.aliasKey(id0), shown = qw ? "Qwen-Image-2.1-Turbo" : ms && Moss.isRs(id0) ? store.name(id0) : store.name(id).replace(" (base)", "");
                LinearLayout trow = new LinearLayout(this); trow.setGravity(Gravity.CENTER_VERTICAL);
                final String alias0 = store.alias(akey);
                trow.addView(text(alias0 != null ? alias0 : shown, 15, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
                TextView pen = text("✎", 16, DIM, false); pen.setGravity(Gravity.CENTER); pen.setContentDescription(L.s("Переименовать", "Rename"));
                pen.setOnClickListener(v -> renameModel(dlg[0], akey, alias0 != null ? alias0 : shown, alias0 != null));
                trow.addView(pen, new LinearLayout.LayoutParams(dp(40), dp(32)));
                row.addView(trow);
                TextView cap = text(anima ? familyCaption(fam) : qw ? qwenCaption() : ms ? mossCaption() : modelCaption(id), 12, DIM, false); cap.setPadding(0, dp(3), 0, 0); row.addView(cap);
                row.setOnClickListener(v -> { dlg[0].dismiss(); if (anima) { if (!sel) setModel(variantOf(fam)); } else if (qw) { if (!sel) setModel(qwenVariant()); } else if (ms) { if (!sel) setModel(Moss.isRs(id0) ? id0 : mossVariant()); } else if (!sel) setModel(id); });
                LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.bottomMargin = dp(8); col.addView(row, lp);
            }
        }
        TextView own = text(L.s("＋ Свой чекпоинт Anima (BF16/FP16/FP32)", "＋ Own Anima checkpoint (BF16/FP16/FP32)"), 15, TXT, false); own.setPadding(dp(14), dp(14), dp(14), dp(4));
        own.setOnClickListener(v -> { dlg[0].dismiss(); AnimaConv.pick(this); }); col.addView(own);
        TextView cat = text(L.s("Каталог моделей", "Model catalog"), 15, TXT, false); cat.setPadding(dp(14), dp(14), dp(14), dp(14)); 
        cat.setOnClickListener(v -> { dlg[0].dismiss(); openModels(); }); col.addView(cat);
        android.widget.ScrollView sv = new android.widget.ScrollView(this); sv.addView(col);
        dlg[0] = builder().setView(sv).create(); dlg[0].show();
        if (dlg[0].getWindow() != null) dlg[0].getWindow().setBackgroundDrawable(framed(20));
    }
    static String modelGroup(String id) { return H3.is(id) ? "h3" : Moss.is(id) ? "moss" : Qwen.is(id) ? "qwen" : Mage.is(id) ? "mage" : id.endsWith("base") ? "base" : "turbo"; }
    /** an Anima family = the turbo model and its base twin (id + "base") */
    static String famOf(String id) { return id.endsWith("base") ? id.substring(0, id.length() - 4) : id; }
    /** the variant to open for a family: the last one used, else turbo, else base */
    String variantOf(String fam) {
        boolean t = ids.contains(fam), b = ids.contains(fam + "base");
        if (t && b) return lab || prefs().getString("var_" + fam, "turbo").equals("base") ? fam + "base" : fam;
        return t ? fam : fam + "base";
    }
    /** Qwen-Image: the default pack (4 step, Lightning LoRA) and the 8 step add-on are one entry with a switch */
    String qwenBase() { return store.qwenId(); }
    boolean qwenBoth() { return ids.contains(qwenBase()) && ids.contains(Qwen.ID8); }
    String qwenVariant() {
        if (qwenBoth()) return prefs().getString("var_qwen", "4").equals("8") ? Qwen.ID8 : qwenBase();
        return ids.contains(qwenBase()) ? qwenBase() : Qwen.ID8;
    }
    String qwenCaption() { return modelCaption(qwenBase()); }
    /** MOSS-Image: the default pack (4 step, rapid LoRA) and the 8 step DMD add-on are one entry with a switch */
    String mossBase() { return store.mossId(); }
    boolean mossBoth() { return ids.contains(mossBase()) && ids.contains(Moss.ID8); }
    String mossVariant() {
        if (mossBoth()) return prefs().getString("var_moss", "4").equals("8") ? Moss.ID8 : mossBase();
        return mossBase();
    }
    String mossCaption() { return modelCaption(mossBase()); }
    String familyCaption(String fam) {
        boolean t = ids.contains(fam), b = ids.contains(fam + "base");
        String sz = fam.contains("29") ? "2.9B" : "2B";
        if (lab) return sz + "  ·  base" + (b ? "" : " (turbo pack only)");
        return sz + "  ·  " + (t && b ? "Turbo  /  Base" : t ? "Turbo" : "Base");
    }
    /** Turbo | Base switch under the model: shown when both variants of the family are installed */
    void updateVariant() {
        if (variantRow == null) return;
        if (lab) { updateLabLora(); return; }
        final boolean qw = isQwen(), ms = isMoss(); final String fam = famOf(model);
        boolean both = ms ? mossBoth() && !Moss.isRs(model) : qw ? qwenBoth() : isAnima() && ids.contains(fam) && ids.contains(fam + "base");
        variantRow.setVisibility(both ? View.VISIBLE : View.GONE);
        if (!both) return;
        variantRow.removeAllViews();
        for (int k = 0; k < 2; k++) {
            final boolean second = k == 1; final boolean on = ms ? second == Moss.is8(model) : qw ? second == Qwen.is8(model) : second == model.endsWith("base");
            TextView c = text(qw || ms ? (second ? "8 step" : "4 step") : (second ? "Base" : "Turbo"), 14, on ? BG : DIM, on); c.setGravity(Gravity.CENTER);
            c.setBackground(on ? bg(TXT, 9) : null);
            c.setOnClickListener(v -> {
                if (busy) return;
                if (ms) { prefs().edit().putString("var_moss", second ? "8" : "4").apply(); setModel(second ? Moss.ID8 : mossBase()); }
                else if (qw) { prefs().edit().putString("var_qwen", second ? "8" : "4").apply(); setModel(second ? Qwen.ID8 : qwenBase()); }
                else { prefs().edit().putString("var_" + fam, second ? "base" : "turbo").apply(); setModel(second ? fam + "base" : fam); }
            });
            variantRow.addView(c, new LinearLayout.LayoutParams(0, -1, 1));
        }
    }
    /** Lab: one row "Turbo LoRA" under the model; its choices are the packs of the family (Anima: the base and the turbo pack, MOSS / Qwen: 4 step and 8 step). Until the kernels apply a LoRA on top of a base, a choice switches the pack. */
    void updateLabLora() {
        final java.util.List<String[]> o = new java.util.ArrayList<>();   // {label, model id}
        final String fam = famOf(model);
        if (isAnima()) {
            if (ids.contains(fam + "base")) o.add(new String[]{L.s("Нет", "None"), fam + "base"}); if (ids.contains(fam)) o.add(new String[]{"Turbo · 4 step", fam});
            if (ids.contains(fam + "base")) {
                if (AnimaLora.turboSrc(getFilesDir()).exists()) o.add(new String[]{prefs().getString("aturbo_name", "LoRA"), "custom"});
                o.add(new String[]{L.s("Выбрать файл…", "Pick a file…"), "pick"});
            }
        }
        else if (isMoss()) {
            if (Moss.isRs(model)) { o.add(new String[]{"4 step · rapid-moss", "rs4"}); if (Moss.installed8(getFilesDir())) o.add(new String[]{"8 step · DMD", "rs8"}); }
            else { if (ids.contains(mossBase())) o.add(new String[]{"4 step · rapid-moss", mossBase()}); } if (!Moss.isRs(model) && ids.contains(Moss.ID8)) o.add(new String[]{"8 step · DMD", Moss.ID8});
            if (MossLora.turboSrc(getFilesDir()).exists()) o.add(new String[]{prefs().getString("turbo_name", "LoRA") + " (" + prefs().getInt("turbo_sampler", 4) + " step)", "custom"});
            o.add(new String[]{L.s("Выбрать файл…", "Pick a file…"), "pick"});
        }
        else if (isH3()) { o.add(new String[]{L.s("Нет · 6 шагов", "None · 6 steps"), "h3none"}); if (H3.loraInstalled(getFilesDir())) o.add(new String[]{"pDMD · 2 step", "h3pdmd2"}); }
        else if (isQwen()) {
            if (ids.contains(Qwen.ID8)) o.add(new String[]{L.s("Нет", "None"), Qwen.ID8}); if (ids.contains(qwenBase())) o.add(new String[]{"4 step · Lightning", qwenBase()});
            if (ids.contains(Qwen.ID8)) { if (QwenLora.turboSrc(getFilesDir()).exists()) o.add(new String[]{prefs().getString("qturbo_name", "LoRA") + " (" + prefs().getInt("qturbo_sampler", 4) + " step)", "custom"}); o.add(new String[]{L.s("Выбрать файл…", "Pick a file…"), "pick"}); }
        }
        variantRow.setVisibility(o.isEmpty() ? View.GONE : View.VISIBLE);
        variantRow.removeAllViews(); variantRow.setPadding(dp(16), 0, dp(16), 0); variantRow.setGravity(Gravity.CENTER_VERTICAL);
        if (o.isEmpty()) return;
        int cur = 0; for (int i = 0; i < o.size(); i++) if (customTurbo() || customTurboAnima() || customTurboQwen() ? o.get(i)[1].equals("custom") : o.get(i)[1].equals(model)) cur = i;
        if (Moss.isRs(model) && !customTurbo()) for (int i = 0; i < o.size(); i++) if (o.get(i)[1].equals(rs8() ? "rs8" : "rs4")) cur = i;
        if (isH3()) for (int i = 0; i < o.size(); i++) if (o.get(i)[1].equals(h3Lora() ? "h3pdmd2" : "h3none")) cur = i;
        variantRow.addView(text(L.s("Turbo LoRA", "Turbo LoRA"), 14, DIM, false), new LinearLayout.LayoutParams(0, -2, 1));
        variantRow.addView(text((cur < 0 ? L.s("не выбрана", "none") : o.get(cur)[0]) + "  ▾", 15, cur < 0 ? DIM : TXT, false));
        final int sel = cur;
        variantRow.setOnClickListener(v -> {
            if (busy) return;
            String[] names = new String[o.size()]; for (int i = 0; i < names.length; i++) names[i] = o.get(i)[0];
            builder().setTitle("Turbo LoRA").setSingleChoiceItems(names, sel, (d, i) -> {
                d.dismiss(); final String id = o.get(i)[1];
                if (id.equals("pick")) { startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("*/*").addCategory(Intent.CATEGORY_OPENABLE), 14); return; }
                if (id.equals("custom") && isQwen() && i == sel) { prefs().edit().putBoolean("qturbo_on", false).apply(); setModelUi(false, false, ""); return; }   // tapping the chosen file again turns the LoRA off
                if (id.equals("custom")) { if (isQwen()) { prefs().edit().putBoolean("qturbo_on", true).apply(); if (!Qwen.is8(model)) { setModel(Qwen.ID8); return; } } else if (isAnima()) { prefs().edit().putBoolean("aturbo_on", true).apply(); if (!isBase(model)) { setModel(fam + "base"); return; } } else prefs().edit().putBoolean("turbo_on", true).apply(); setModelUi(false, false, ""); return; }
                if (id.equals("h3none") || id.equals("h3pdmd2")) { prefs().edit().putBoolean("h3_lora", id.equals("h3pdmd2")).apply(); setModelUi(false, false, ""); refreshBaseUi(); return; }
                if (id.equals("rs4") || id.equals("rs8")) { prefs().edit().putBoolean("turbo_on", false).putString("rs_mode", id.substring(2)).apply(); setModelUi(false, false, ""); refreshBaseUi(); return; }
                prefs().edit().putBoolean("turbo_on", false).putBoolean("aturbo_on", false).putBoolean("qturbo_on", false).apply();
                if (!id.equals(model)) setModel(id); else setModelUi(false, false, "");
            }).show();
        });
    }
    /** the line under a model in the picker: what differs between the models of one group (the group header says the rest) */
    String pickCaption(String id) {
        String g = modelGroup(id);
        if (g.equals("turbo") || g.equals("base")) return (id.contains("29") ? "2.9B" : "2B") + "  ·  " + L.f("~%d с", "~%d s", store.speed(id));
        return modelCaption(id);
    }
    String modelCaption(String id) {
        switch (modelGroup(id)) {
            case "turbo": return L.s("4 шага · без CFG · одна картинка за секунды", "4 steps · no CFG · a picture in seconds");
            case "base": return L.s("20–30 шагов · CFG и негативный промпт", "20–30 steps · CFG and negative prompt");
            case "moss": return L.s("описания обычными предложениями, надписи", "plain-sentence prompts, text in pictures");
            case "qwen": return L.s("правка фото, до 3 картинок", "photo editing, up to 3 pictures");
            case "h3": return L.s("33B · 6 шагов · картинки от 1 МП, минуты на картинку", "33B · 6 steps · pictures from 1 MP, minutes per picture");
            default: return L.s("4 шага · правка фото", "4 steps · photo editing");
        }
    }

    /** The NPU pipeline needs width/height multiples of 16, patch count divisible by 32 and at most 6912 patches (megarun TMAX; the attention fills the 8 MB VTCM at 7392 tokens). */
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        int t = (w / 16) * (h / 16);
        if (t % 32 != 0) return L.f("Число патчей (%d) должно делиться на 32, например 640×640, 1024×1024, 704×832", "Patch count (%d) must be divisible by 32, e.g. 640×640, 1024×1024, 704×832", t);
        if (t > 6912) return L.s("Слишком большой размер: максимум 6912 патчей (например 1152×1536)", "Too large: at most 6912 patches (e.g. 1152×1536)");
        return null;
    }

    void setRes(int w, int h) {
        resW = w; resH = h;
        boolean preset = false;
        for (int[] z : sizeList()) if (z[0] == w && z[1] == h) preset = true;
                if (customRow != null) { customRow.setVisibility(View.VISIBLE); if (!wEdit.getText().toString().equals(String.valueOf(w)) || !hEdit.getText().toString().equals(String.valueOf(h))) { wEdit.setText(String.valueOf(w)); hEdit.setText(String.valueOf(h)); } }
        if (chipName != null) chipName.setText(modelName());
        rebuildSizeChips(); updatePlaceholder(); updateChips();
        if (genBtn != null && !busy) genBtn.setText(genLabel());
    }

    void updatePlaceholder() {
        if (placeholder == null || phSize == null) return;
        phSize.setText(resW + " × " + resH);
    }

    /** ratio shortcuts next to the current size: for each common ratio the preset of this model whose area is closest to the current one; the last chip opens the full picker */
    void rebuildSizeChips() {
        if (sizeChips == null) return;
        sizeChips.removeAllViews();
        final boolean land = resW > resH;
        final String[] ratios = land ? new String[]{"1:1", "3:2", "4:3", "16:9"} : new String[]{"1:1", "2:3", "3:4", "9:16"};
        final double area = (double) resW * resH;
        for (String rt : ratios) {
            int[] best = null;
            for (int[] z : sizeList()) { int g = gcd(z[0], z[1]); if (!(z[0] / g + ":" + z[1] / g).equals(rt)) continue; if (best == null || Math.abs(z[0] * (double) z[1] - area) < Math.abs(best[0] * (double) best[1] - area)) best = z; }
            if (best == null) continue;
            final int[] z = best; final boolean on = z[0] == resW && z[1] == resH;
            TextView c = text(rt, 14, on ? BG : DIM, on); c.setGravity(Gravity.CENTER);
            c.setBackground(on ? bg(TXT, 9) : null);
            c.setOnClickListener(v -> { customOpen = false; setRes(z[0], z[1]); });
            sizeChips.addView(c, new LinearLayout.LayoutParams(0, -1, 1));
        }
    }

    /** rough time to a picture: the catalog speed at 512x768 scaled by the patch count, times the number of pictures */
    /** Qwen time from phone measurements: a DiT step takes about 5 + 3.8e-4*N + 7.7e-7*N^2 s for N target tokens (7.4 s at 1536, 36 s at 6144), the tiled decoder 4.9 s per 512 px tile, text ~10 s, ~16 s per photo (ViT + encoder) */
    long qwenEst() {
        final int n = (resW / 16) * (resH / 16), st = steps(), imgs = editBm != null ? 1 + extraBm.size() : 0;
        final double step = 5 + 3.8e-4 * n + 7.7e-7 * (double) n * n;
        final int tiles = Math.max(1, (int) Math.ceil((resW - 64) / 448.0)) * Math.max(1, (int) Math.ceil((resH - 64) / 448.0));
        return Math.round(10 + st * step + 4.9 * tiles + 16 * imgs);
    }
    String genLabel() {
        if (model.isEmpty()) return L.s("Скачать модель", "Get a model");
        if (isH3()) return count > 1 ? L.f("Генерировать ×%d  ·  ~%d с", "Generate ×%d  ·  ~%d s", count, H3.estimate(resW, resH, h3Steps()) * count) : L.f("Генерировать  ·  ~%d с", "Generate  ·  ~%d s", H3.estimate(resW, resH, h3Steps()));
        if (isQwen()) return count > 1 ? L.f("Генерировать ×%d  ·  ~%d с", "Generate ×%d  ·  ~%d s", count, qwenEst() * count) : L.f("Генерировать  ·  ~%d с", "Generate  ·  ~%d s", qwenEst());
        long est = Math.max(3, Math.round(store.speed(model) * (steps() / (base() ? 20.0 : customTurboAnima() ? 40.0 : stepsDef())) * ((resW / 16) * (resH / 16)) / 1536.0 * (isQwen() && editBm != null ? 1.3 + 0.25 * extraBm.size() : 1))) * count;
        return count > 1 ? L.f("Генерировать ×%d  ·  ~%d с", "Generate ×%d  ·  ~%d s", count, est) : L.f("Генерировать  ·  ~%d с", "Generate  ·  ~%d s", est);
    }

    /** Share of the DiT time spent after `st` steps: the first two of the 4 steps use a half-size latent and cost ~0.3 (512x768) / ~0.22 (960x1536) of a full step. */
    double stepShare(int st) {
        double lo = resW * resH < 500000 ? 0.3 : 0.22;
        double done = 0, total = 0;
        for (int i = 0; i < steps(); i++) { double c = i < 2 && !base() ? lo : 1.0; total += c; if (i < st) done += c; }
        return done / total;
    }

    /** token limit of the prompt of the current model (0 = none): Anima keeps 512 text tokens, MOSS-Image and Qwen-Image text-to-image 480 (with their chat template) */
    int tokLimit() { return isH3() ? 480 : isAnima() ? 512 : isMoss() ? 480 : isQwen() && editBm == null ? 480 : 0; }
    /** tokens of `text` for the current model (the quality prefix counts for Anima); runs off the UI thread (the vocabulary takes a second to load) */
    int tokCount(String text) throws Exception {
        if (isMoss()) { if (tokM == null) tokM = new QwenTok(Moss.dir(getFilesDir()).getPath()); return tokM.encodePrompt(text).length; }
        if (tokQ == null) {   // Anima and Qwen use the Qwen vocabulary (bundled; the installed Qwen pack is not required)
            File d = new File(getFilesDir(), "tok"); d.mkdirs();
            for (String f : new String[]{"tok_vocab.txt", "tok_merges.txt", "tok_special.txt"}) {
                File o = new File(d, f); if (o.exists()) continue;
                try (java.io.InputStream in = getAssets().open("tok/" + f); java.io.FileOutputStream w = new java.io.FileOutputStream(o)) { byte[] b = new byte[1 << 16]; for (int n; (n = in.read(b)) > 0; ) w.write(b, 0, n); }
            }
            tokQ = new QwenTok(d.getPath());
        }
        return isH3() ? tokQ.encode(text.isEmpty() ? " " : text).length : isQwen() ? tokQ.encodeQwenImage(text).length : tokQ.encode(text).length + 1;   // Anima: + end token; H3: the prompt verbatim
    }
    void updateTokens() {
        final int lim = tokLimit();
        if (tokView == null) return;
        if (lim == 0) { tokOver = false; tokView.setVisibility(View.GONE); if (negTokView != null) negTokView.setVisibility(View.GONE); return; }
        final String pr = prompt.getText().toString().trim(), neg = base() ? negStr() : "";
        final String full = isAnima() ? Settings.prefix(this) + pr : pr; final boolean an = isAnima();
        new Thread(() -> {
            int n = 0, nn = 0;
            try { n = pr.isEmpty() ? 0 : tokCount(full); if (!neg.isEmpty()) nn = tokCount(neg); } catch (Exception e) { return; }
            final int a = n, b = nn;
            ui.post(() -> {
                if (tokLimit() != lim) return;
                tokOver = a > lim || b > lim;
                String t = L.f("Токенов: %d / %d", "Tokens: %d / %d", a, lim);
                tokView.setText(a > lim ? t + L.s("  ·  слишком длинно", "  ·  too long") : t); tokView.setTextColor(a > lim ? BAD : DIM); tokView.setVisibility(View.VISIBLE);
                if (negTokView != null) {
                    String tn = L.f("Токенов: %d / %d", "Tokens: %d / %d", b, lim);
                    negTokView.setText(b > lim ? tn + L.s("  ·  слишком длинно", "  ·  too long") : tn); negTokView.setTextColor(b > lim ? BAD : DIM);
                }
            });
        }).start();
    }
    void applyFamily() { rebuildSizeChips(); updateChips(); updateVariant(); }
    void setModelUi(boolean isLoaded, boolean working, String msg) {
        ui.post(() -> {
            int c = working ? WARN : isLoaded ? OK : DIM;
            chipDot.setTextColor(c);
            if (turboStrRow != null) { boolean tv = customTurbo() || customTurboAnima() || customTurboQwen(); turboStrRow.setVisibility(tv ? View.VISIBLE : View.GONE); if (tv) turboBar.setProgress(Math.round(prefs().getFloat(turboStrKey(), 1f) * 20)); }
            if (loraBox != null) { loraBox.setVisibility(isMoss() && !moss8() && !customTurbo() ? View.VISIBLE : View.GONE); refreshLora(); }
            if (aloraBox != null) { aloraBox.setVisibility(isAnima() ? View.VISIBLE : View.GONE); refreshAnimaLoras(); }
            if (baseBox != null) { baseBox.setVisibility(base() ? View.VISIBLE : View.GONE); refreshBaseUi(); }
            if (editBox != null) { editBox.setVisibility(isQwen() || isMage() ? View.VISIBLE : View.GONE); refreshEdit(); }
            applyFamily(); updateTokens();
            state.setText(working ? msg : !isAnima() ? modelCaption(model) : isLoaded ? L.s("Загружена в NPU, генерация стартует сразу", "Loaded on the NPU, generation starts at once") : L.s("Выгружена · загрузится сама при генерации (~5 с)", "Unloaded · loads itself when you generate (~5 s)"));
            loadBtn.setText(isLoaded ? L.s("Выгрузить", "Unload") : L.s("Загрузить", "Load"));
            loadBtn.setEnabled(!working && !busy && !model.isEmpty() && !isMoss() && !isQwen() && !isMage() && !isH3());
            loadBtn.setVisibility(loadBtn.isEnabled() || (isLoaded && isAnima()) ? View.VISIBLE : View.GONE);
            boolean can = busy ? !cancelled : !working;   // while a picture is made the main button is "Stop"
            genBtn.setEnabled(can);
            genBtn.setBackground(busy ? bg(CARD, 12) : gradient(can));
            genBtn.setTextColor(busy ? BAD : can ? ON_ACC : DIM);
            genBtn.setText(busy ? (cancelled ? L.s("Останавливаю…", "Stopping…") : L.s("Остановить", "Stop")) : genLabel());
        });
    }

    void refreshState() {
        if (isMoss() || isQwen() || isMage() || isH3()) { setModelUi(false, false, ""); return; }
        ex.submit(() -> {
            String s = su("sh " + D + "/anima.sh status", true);
            if (s.startsWith("ERR") || s.isEmpty()) {
                ui.post(() -> { state.setText(L.f("Не запускается оболочка: %s", "Shell does not start: %s", s)); chipDot.setTextColor(BAD); });
                return;
            }
            if (busy) return;
            loaded = s.contains("unloaded") ? false : s.contains("loaded");
            setModelUi(loaded, false, "");
            if (!statusSeen) { statusSeen = true; if (!loaded && Settings.autoload(this) && !model.isEmpty() && Compat.chipAllowed(this)) ui.post(this::toggleModel); }   // "load at start"
        });
    }

    void setModel(String m) {
        if (m.equals(model) || busy) return;   // no model switch mid-generation
        model = m;
        setRes(resW, resH);
        if (loaded) {   // the loaded server holds the other model's weights
            setModelUi(true, true, L.s("Меняю модель…", "Switching model…"));
            ex.submit(() -> { su("sh " + D + "/anima.sh stop < /dev/null >/dev/null 2>&1", false); loaded = false; setModelUi(false, false, ""); });
        } else setModelUi(false, false, "");
    }

    void toggleModel() {
        if (isMoss() || isQwen() || isMage() || isH3()) return;   // nothing to preload
        final boolean want = !loaded;
        setModelUi(loaded, true, want ? L.s("Загружаю модель…", "Loading the model…") : L.s("Выгружаю модель…", "Unloading the model…"));
        ex.submit(() -> {
            su("rm -f " + D + "/cancel; " + mp() + "sh " + D + "/anima.sh " + (want ? "load" : "stop") + " < /dev/null >/dev/null 2>&1", false);
            loaded = want;
            setModelUi(loaded, false, "");
        });
    }

    /** anima.sh ships in assets and is rewritten at every start (it is a script, so it may live in the app's data dir) */
    void installScript() {
        for (String nm : new String[]{"anima.sh", "moss.sh", "qwen.sh", "mage.sh", "h3.sh"})
            try (InputStream in = getAssets().open(nm); FileOutputStream o = new FileOutputStream(new File(D, nm))) {
                byte[] b = new byte[8192]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n);
            } catch (IOException e) { Toast.makeText(this, nm + ": " + e, Toast.LENGTH_LONG).show(); }
    }

    void refreshModels() {
        ids = store.installed();
        model = ids.isEmpty() ? "" : ids.contains(model) ? model : ids.get(0);
        setRes(resW, resH);
        setModelUi(loaded, false, "");
    }

    // ---- shell --------------------------------------------------------------------------------
    /** One long-lived /system/bin/sh with A, S, BIN in its environment: the helper binaries are started from it with the app's own uid. */
    static class Sh {
        String[] env; Process p; BufferedWriter w; BufferedReader r;
        synchronized String run(String cmd) {
            try {
                if (p == null) {
                    ProcessBuilder pb = new ProcessBuilder("/system/bin/sh").redirectErrorStream(true);
                    pb.environment().put("A", env[0]); pb.environment().put("S", env[1]); pb.environment().put("BIN", env[2]); pb.directory(new File(env[0]));
                    p = pb.start();
                    w = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
                    r = new BufferedReader(new InputStreamReader(p.getInputStream()));
                }
                w.write(cmd + "\necho __END__\n");
                w.flush();
                StringBuilder sb = new StringBuilder();
                for (String l; (l = r.readLine()) != null && !l.equals("__END__"); ) sb.append(l).append('\n');
                return sb.toString().trim();
            } catch (Exception e) {
                p = null;
                return "ERR " + e;
            }
        }
    }

    /** main: model load / generation; pollSh: progress and status polling; ctlSh: the Stop button (it must not wait for a running generation) */
    final Sh main = new Sh(), pollSh = new Sh(), ctlSh = new Sh();

    /** Runs a command in the main shell (blocking) and returns its stdout; `poll` uses the second shell so progress polling never waits for a running generation. */
    String su(String cmd, boolean poll) { return (poll ? pollSh : main).run(cmd); }

    static String q(String s) { return "'" + s.replace("'", "'\\''") + "'"; }

    // ---- generation ---------------------------------------------------------------------------
    /** Stop: the server is stopped (the model unloads; the next picture reloads it in ~5 s), anima.sh sees the cancel marker and leaves its waiting loops */
    void cancel() {
        if (!busy || cancelled) return;
        cancelled = true;
        setModelUi(loaded, false, "");
        if (isH3()) ex2.submit(() -> ctlSh.run("sh " + D + "/h3.sh stop < /dev/null >/dev/null 2>&1"));
        else if (isMage()) ex2.submit(() -> ctlSh.run("sh " + D + "/mage.sh stop < /dev/null >/dev/null 2>&1"));
        else if (isQwen()) ex2.submit(() -> ctlSh.run("sh " + D + "/qwen.sh stop < /dev/null >/dev/null 2>&1"));
        else if (isMoss()) ex2.submit(() -> ctlSh.run("sh " + D + "/moss.sh stop < /dev/null >/dev/null 2>&1"));
        else ex2.submit(() -> ctlSh.run("touch " + D + "/cancel; sh " + D + "/anima.sh stop < /dev/null >/dev/null 2>&1"));
    }

    /** {MemAvailable in MB, memory pressure "some avg10" in %}; pressure is -1 when the kernel does not let the app read it */
    static double[] memState() {
        double avail = -1, psi = -1;
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader("/proc/meminfo"))) { for (String l; (l = r.readLine()) != null; ) if (l.startsWith("MemAvailable:")) { avail = Long.parseLong(l.replaceAll("\\D+", "")) / 1024.0; break; } } catch (Exception ignored) { }
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader("/proc/pressure/memory"))) { String l = r.readLine(); if (l != null && l.startsWith("some")) psi = Double.parseDouble(l.replaceAll(".*avg10=([0-9.]+).*", "$1")); } catch (Exception ignored) { }
        return new double[]{avail, psi};
    }
    /** while a picture is made: when the phone starts to thrash (the low-memory killer would close the app next) stop the generation and warn instead */
    void startMemWatch() {
        Thread t = new Thread(() -> {
            int low = 0;
            while (busy && !cancelled) {
                try { Thread.sleep(1000); } catch (InterruptedException e) { return; }
                double[] m = memState(); if (m[0] < 0) continue;
                boolean thrash = m[1] >= 0 ? m[1] >= 35 && m[0] < 900 : m[0] < 350;   // pressure readable: heavy stalls with little free memory; otherwise just very little free memory
                low = thrash ? low + 1 : 0;
                if (low >= (m[1] >= 0 ? 2 : 4)) { oomAbort = true; cancel(); return; }
            }
        });
        t.setDaemon(true); t.start();
    }
    void memWarning() {
        builder().setTitle(L.s("Не хватило оперативной памяти", "Out of memory")).setMessage(L.s("Телефону не хватило оперативной памяти, поэтому генерация остановлена, чтобы приложение не закрылось. Закройте другие приложения или выберите размер (и фото для правки) поменьше и повторите.", "The phone ran out of memory, so the generation was stopped to keep the app from being closed. Close other apps, or pick a smaller size (and a smaller photo for editing), and try again.")).setPositiveButton("OK", null).show();
    }

    void generate(boolean newSeed) {
        if (model.isEmpty()) { openModels(); return; }
        int cw = 0, ch = 0;
        try { cw = Integer.parseInt(wEdit.getText().toString().trim()); ch = Integer.parseInt(hEdit.getText().toString().trim()); } catch (NumberFormatException e) { }
        String bad = isH3() ? H3.sizeError(cw, ch) : isMoss() ? mossSizeError(cw, ch) : isMage() ? Mage.sizeError(cw, ch) : isQwen() ? Qwen.sizeError(cw, ch) : sizeError(cw, ch);
        if (bad != null) { Toast.makeText(this, bad, Toast.LENGTH_LONG).show(); return; }
        if (cw != resW || ch != resH) setRes(cw, ch);
        final String pr = prompt.getText().toString().trim();
        if (pr.isEmpty()) { Toast.makeText(this, L.s("Введите описание", "Enter a prompt"), Toast.LENGTH_SHORT).show(); return; }
        if (tokLimit() > 0) {   // counted here again (the counter on screen may be a keystroke behind)
            int n; try { n = tokCount(isAnima() ? Settings.prefix(this) + pr : pr); if (base()) n = Math.max(n, tokCount(negStr())); } catch (Exception e) { n = 0; }
            if (n > tokLimit()) { updateTokens(); Toast.makeText(this, L.f("Слишком длинный промпт: %d токенов, максимум %d. Сократите описание.", "Prompt too long: %d tokens, the limit is %d. Shorten it.", n, tokLimit()), Toast.LENGTH_LONG).show(); return; }
        }
        String sd = seed.getText().toString().trim();
        final boolean randomSeed = newSeed || sd.isEmpty() || sd.equals("-1");   // -1 = random; the seed used is shown under the picture
        long base0 = 0; try { base0 = Long.parseLong(sd); } catch (NumberFormatException ignored) { }
        final long baseSeed = base0;
        final int n = count, w = resW, h = resH;
        busy = true; cancelled = false; oomAbort = false; curIdx = 0; curN = n;
        { double[] m0 = memState(); if (m0[0] >= 0 && m0[0] < 1200) Toast.makeText(this, L.f("Мало свободной памяти (%d МБ): закройте другие приложения, иначе генерация может остановиться", "Little free memory (%d MB): close other apps, or the generation may stop", (int) m0[0]), Toast.LENGTH_LONG).show(); }
        startMemWatch();
        setModelUi(loaded, false, "");
        View f0 = getCurrentFocus(); if (f0 != null) ((android.view.inputmethod.InputMethodManager) getSystemService(INPUT_METHOD_SERVICE)).hideSoftInputFromWindow(f0.getWindowToken(), 0);
        bar.setProgress(0);
        bar.setVisibility(View.VISIBLE);
        status.setVisibility(View.VISIBLE);
        status.setText(L.s("Запуск…", "Starting…"));
        if (Settings.keepOn(this)) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (((PowerManager) getSystemService(POWER_SERVICE)).getCurrentThermalStatus() >= PowerManager.THERMAL_STATUS_SEVERE) Toast.makeText(this, L.s("Телефон сильно нагрелся: генерация будет медленной", "The phone is very hot: generation will be slow"), Toast.LENGTH_LONG).show();
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class).putExtra("mode", "gen"));   // keeps the process alive when the app goes to the background
        final long t0 = System.currentTimeMillis();
        ex.submit(() -> {
            int made = 0; Bitmap lastBm = null;
            for (int i = 0; i < curN && !cancelled; i++) {
                curIdx = i;
                final String seedS = String.valueOf(randomSeed ? new Random().nextInt(1 << 30) : baseSeed + i);
                File fr = null;
                try { fr = runOne(pr, seedS, w, h, t0); } catch (OutOfMemoryError e) { oomAbort = true; }
                final File f = fr;
                if (f == null) break;
                made++;
                final Bitmap bm = BitmapFactory.decodeFile(f.getPath());
                lastBm = bm;
                ui.post(() -> {
                    lastFile = f; img.setImageBitmap(bm); img.setVisibility(View.VISIBLE); placeholder.setVisibility(View.GONE); img.setAlpha(0f); img.animate().alpha(1f).setDuration(250).start();
                    setFavIcon(false);
                    for (ImageButton x : new ImageButton[]{againBtn, favBtn, dlBtn, shareBtn}) x.setVisibility(View.VISIBLE);
                    final ScrollView svv = (ScrollView) imgBox.getParent().getParent(); svv.post(() -> svv.smoothScrollTo(0, 0));   // after the image has been laid out
                    saved.setText("seed " + seedS);
                });
            }
            final double sec = (System.currentTimeMillis() - t0) / 1000.0;
            final boolean wasCancelled = cancelled, okAny = made > 0, oom = oomAbort; oomAbort = false;
            final int total = made;
            if (wasCancelled) loaded = false;   // Stop shut the server down
            busy = false; cancelled = false;
            DownloadService.stop(app);
            final Bitmap nb = lastBm;
            ui.post(() -> {
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                bar.setVisibility(View.GONE);
                if (!okAny) {   // a preview may still be on screen: back to the last real picture (or the placeholder)
                    Bitmap prev = lastFile != null ? BitmapFactory.decodeFile(lastFile.getPath()) : null;
                    if (prev != null) img.setImageBitmap(prev); else { img.setVisibility(View.GONE); placeholder.setVisibility(View.VISIBLE); }
                }
                if (oom) { status.setText(L.s("Остановлено: не хватило памяти", "Stopped: out of memory")); memWarning(); }
                else if (wasCancelled) status.setText(L.s("Остановлено", "Stopped"));
                else if (!okAny) status.setText(L.s("Ошибка генерации (лог: files/anima/animad.log)", "Generation failed (log: files/anima/animad.log)"));
                else status.setText(L.f("Готово за %.1f с", "Done in %.1f s", sec));
                setModelUi(loaded, false, "");
                if (okAny && !visible) notifyDone(app, nb, pr, total);
            });
        });
    }

    /** one picture: loads the model if needed, runs anima.sh, stores the PNG (with the settings inside) in the history; null on failure or Stop */
    File runOne(String pr, String seedS, int w, int h, long t0) {
        if (isH3()) return runOneH3(pr, seedS, w, h, t0);
        if (isMage()) return runOneMage(pr, seedS, w, h, t0);
        if (isQwen()) return runOneQwen(pr, seedS, w, h, t0);
        if (isMoss()) return runOneMoss(pr, seedS, w, h, t0);
        final String sig = loraSig();
        if (!loaded || !sig.equals(servedSig)) {   // a new LoRA (or none) restarts the server: megarun folds it into the blocks while loading
            setProgressUi(0, !animaLoraEnv().isEmpty() ? L.s("Подготавливаю LoRA… (один раз для этого набора)", "Preparing the LoRA… (once for this set)") : loaded ? L.s("Загружаю LoRA…", "Loading the LoRA…") : L.s("Загружаю модель…", "Loading the model…"), t0);
            su("rm -f " + D + "/cancel; " + mp() + "sh " + D + "/anima.sh load < /dev/null >/dev/null 2>&1", false);
            if (cancelled) return null;
            loaded = true; servedSig = sig;
        }
        final String f = D + "/last.png";
        final boolean[] done = {false};
        final int[] base = count();
        final Preview pv = Preview.load(this, "anima"); final File pvf = new File(D, "prev.bin"); pvf.delete();
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                if (pv != null) showPreview(pv.poll(pvf));
                int[] c = count();
                int st = (c[0] - base[0]) / cm();
                if (c[1] > base[1]) setProgressUi(97, L.s("Сохраняю…", "Saving…"), t0);
                else if (st >= steps()) setProgressUi(92, L.s("Декодирование…", "Decoding…"), t0);
                else if (c[2] > base[2] || st > 0) setProgressUi(15 + (int) (76 * stepShare(st)), L.f("Шаг %d/%d", "Step %d/%d", st + 1, steps()), t0);  // adapter line = step 1 started
                else setProgressUi(8, L.s("Кодирование текста…", "Encoding the text…"), t0);
            }
        });
        poll.start();
        su("cd " + D + " && rm -f last.png && " + mp() + "PREFIX=" + q(Settings.prefix(this)) + " STEPS=" + steps() + " SAMPLER=" + smp() + (sch().isEmpty() ? "" : " SCHED=" + sch()) + (base() || !cfgStr().equals("1") ? " CFG=" + cfgStr() + " NEG=" + q(negStr()) : "") + " W=" + w + " H=" + h + " LOGRE='learned_condition|completed in' sh ./anima.sh " + q(pr) + " " + seedS + " " + f + " < /dev/null >> gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        byte[] png = new byte[0];
        try { png = Models.readBytes(new File(f)); } catch (Exception ignored) { }
        if (png.length == 0 || BitmapFactory.decodeByteArray(png, 0, png.length) == null) return null;
        try {
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            if (!AnimaLoras.names(prefs(), getFilesDir()).isEmpty()) meta.put("lora", AnimaLoras.names(prefs(), getFilesDir()));
            return Gallery.save(this, Settings.metaInPng(this) ? PngMeta.embed(png, meta) : png, meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show()); return null; }
    }

    /** one MOSS-Image picture: tokenise here, text encoder + DiT + VAE on the NPU through moss.sh, PNG with the settings inside */
    File runOneMoss(String pr, String seedS, int w, int h, long t0) {
        final File home = getFilesDir(), M = Moss.dir(home), ids = new File(M, "ids.i32"), raw = new File(M, "img.f32"); final int STEPS = steps();
        final File MR;   // the dir mossrun runs in: the pack itself, or the RealSkin link dir
        try { MR = Moss.isRs(model) ? Moss.variantRs(home, model) : M; } catch (Exception e) { ui.post(() -> Toast.makeText(this, "RealSkin: " + e, Toast.LENGTH_LONG).show()); return null; }
        setProgressUi(3, L.s("Токенизация…", "Tokenising…"), t0);
        try { Moss.writeIds(home, pr, ids); } catch (Exception e) { ui.post(() -> Toast.makeText(this, "tokenizer: " + e, Toast.LENGTH_LONG).show()); return null; }
        String loraEnv = moss8() ? "MODE=8 " : ""; final boolean useLora = loraOn();
        if (customTurbo()) {
            final String[] cm = {""}; setProgressUi(2, L.s("Подготовка turbo LoRA…", "Preparing the turbo LoRA…"), t0);
            File ovl = MossLora.ensureTurbo(home, BIN, prefs().getFloat("turbo_str", 1f), cm);
            if (ovl == null) { ui.post(() -> Toast.makeText(this, L.f("Не удалось подготовить LoRA: %s", "Could not prepare the LoRA: %s", cm[0]), Toast.LENGTH_LONG).show()); return null; }
            loraEnv = "MOSS_LORA=" + q(ovl.getPath()) + " MODE=" + prefs().getInt("turbo_sampler", 4) + " ";
        } final float lstr = strBar.getProgress() * 0.05f;
        if (useLora) {
            final String[] msg = {""};
            File ovl = MossLora.ensure(home, BIN, loraId(), lstr, (d, t) -> setProgressUi(1 + 5 * d / t, L.f("Подготовка LoRA… %d/%d", "Preparing the LoRA… %d/%d", d, t), t0), () -> cancelled, msg);
            if (ovl == null) { if (!cancelled) ui.post(() -> Toast.makeText(this, L.f("Не удалось подготовить LoRA: %s", "Could not prepare the LoRA: %s", msg[0]), Toast.LENGTH_LONG).show()); return null; }
            loraEnv = "MOSS_LORA=" + q(ovl.getPath()) + " ";
        }
        final boolean[] done = {false};
        final Preview pv = Preview.load(this, "moss"); final File pvf = new File(MR, "prev.bin"); pvf.delete();
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                if (pv != null) showPreview(pv.poll(pvf));
                String s = su("echo $(grep -c '^te:' " + MR + "/moss.log) $(grep -c '^refiner:' " + MR + "/moss.log) $(grep -c '^step' " + MR + "/moss.log) $(grep -c '^done' " + MR + "/moss.log)", true);
                try {
                    String[] a = s.split("\\s+"); int te = Integer.parseInt(a[0]), rf = Integer.parseInt(a[1]), st = Integer.parseInt(a[2]), dn = Integer.parseInt(a[3]);
                    if (dn > 0) setProgressUi(94, L.s("Декодирование…", "Decoding…"), t0);
                    else if (rf > 0 || st > 0) setProgressUi(22 + 70 * st / STEPS, L.f("Шаг %d/%d", "Step %d/%d", Math.min(st + 1, STEPS), STEPS), t0);
                    else if (te > 0) setProgressUi(20, L.s("Подготовка текста…", "Preparing the text…"), t0);
                    else setProgressUi(8, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                } catch (Exception ignored) { }
            }
        });
        poll.start();
        su("rm -f " + q(raw.getPath()) + "; " + loraEnv + "M=" + q(MR.getPath()) + " sh " + D + "/moss.sh " + q(ids.getPath()) + " " + w + " " + h + " " + seedS + " " + STEPS + " " + q(raw.getPath()) + " < /dev/null >> " + D + "/gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        try {
            byte[] png = Moss.png(raw, w, h); raw.delete();
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            if (useLora) meta.put("lora", prefs().getString("moss_lora_name", "LoRA")).put("lora_strength", lstr);
            return Gallery.save(this, Settings.metaInPng(this) ? PngMeta.embed(png, meta) : png, meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Ошибка генерации: %s (лог files/moss/moss.log)", "Generation failed: %s (log files/moss/moss.log)", e.getMessage()), Toast.LENGTH_LONG).show()); return null; }
    }

    /** one Mage-Flow picture (or an edit of the picked photo): tokenise / prepare the photo here, everything else on the NPU through mage.sh */
    File runOneMage(String pr, String seedS, int w, int h, long t0) {
        final File home = getFilesDir(), M = Mage.dir(home), ids = new File(M, "ids.i32"), raw = new File(M, "img.rgb"); final int STEPS = steps();
        final Bitmap src = editBm; final boolean edit = src != null;
        setProgressUi(2, edit ? L.s("Подготовка фото…", "Preparing the photo…") : L.s("Токенизация…", "Tokenising…"), t0);
        int drop, keys; String edArg = "";
        try {
            if (edit) {
                if ((w / 16) * (h / 16) > Mage.MAX_EDIT_PATCHES) { ui.post(() -> Toast.makeText(this, L.s("Для редактирования размер не больше 1024×768", "For editing the size is at most 1024×768"), Toast.LENGTH_LONG).show()); return null; }
                Mage.Edit e = Mage.prepareEdit(home, src, pr, w, h, new File(Mage.editDir(home), "ids.i32")); drop = e.drop; edArg = " " + q(Mage.editDir(home).getPath());
                keys = 2 * (w / 16) * (h / 16) + e.tokens - e.drop;
            } else { drop = Mage.writeIds(home, pr, ids); keys = (w / 16) * (h / 16) + (int) (ids.length() / 4) - drop; }
            if (keys > Mage.MAX_KEYS) { ui.post(() -> Toast.makeText(this, L.s("Описание слишком длинное для этого размера: сократите его или выберите размер поменьше", "The prompt is too long for this size: shorten it or pick a smaller size"), Toast.LENGTH_LONG).show()); return null; }
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, "mage: " + e, Toast.LENGTH_LONG).show()); return null; }
        final boolean[] done = {false};
        final Preview pv = Preview.load(this, "mage"); final File pvf = new File(M, "prev.bin"); pvf.delete();
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                if (pv != null) showPreview(pv.poll(pvf));
                String s = su("echo $(grep -c '^vit:' " + M + "/mage.log) $(grep -c '^te:' " + M + "/mage.log) $(grep -c '^txt_in:' " + M + "/mage.log) $(grep -c '^step' " + M + "/mage.log) $(grep -c '^vae dec total' " + M + "/mage.log)", true);
                try {
                    String[] a = s.split("\\s+"); int vit = Integer.parseInt(a[0]), te = Integer.parseInt(a[1]), ti = Integer.parseInt(a[2]), st = Integer.parseInt(a[3]), dn = Integer.parseInt(a[4]);
                    if (dn > 0) setProgressUi(97, L.s("Декодирование…", "Decoding…"), t0);
                    else if (st >= STEPS) setProgressUi(90, L.s("Декодирование…", "Decoding…"), t0);
                    else if (ti > 0 || st > 0) setProgressUi(30 + 60 * st / STEPS, L.f("Шаг %d/%d", "Step %d/%d", Math.min(st + 1, STEPS), STEPS), t0);
                    else if (te > 0) setProgressUi(28, L.s("Подготовка текста…", "Preparing the text…"), t0);
                    else if (edit && vit > 0) setProgressUi(15, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                    else if (edit) setProgressUi(5, L.s("Анализ фото на NPU…", "Analysing the photo on the NPU…"), t0);
                    else setProgressUi(8, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                } catch (Exception ignored) { }
            }
        });
        poll.start();
        su("rm -f " + q(raw.getPath()) + "; M=" + q(M.getPath()) + " sh " + D + "/mage.sh " + q(ids.getPath()) + " " + drop + " " + w + " " + h + " " + seedS + " " + STEPS + " " + q(raw.getPath()) + edArg + " < /dev/null >> " + D + "/gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        try {
            byte[] png = Mage.png(raw, w, h); raw.delete();
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            if (edit) meta.put("edit", true);
            return Gallery.save(this, Settings.metaInPng(this) ? PngMeta.embed(png, meta) : png, meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Ошибка генерации: %s (лог files/mage/mage.log)", "Generation failed: %s (log files/mage/mage.log)", e.getMessage()), Toast.LENGTH_LONG).show()); return null; }
    }

    /** one MiniMax-H3-Image picture: tokenise here, text encoder + DiT + VAE on the NPU through h3.sh (log files/h3/h3.log) */
    File runOneH3(String pr, String seedS, int w, int h, long t0) {
        final File home = getFilesDir(), M = H3.dir(home), ids = new File(M, "ids.i32"), raw = new File(M, "img.f32"); final boolean lora = h3Lora(); final int STEPS = h3Steps();
        final String loraEnv = lora ? "H3_LORA=" + q(H3.lora(home).getPath()) + " H3_SIG=" + H3.loraSigmas(STEPS) + " H3_HEADS=" + H3.loraHeads(STEPS) + " " : STEPS != H3.STEPS ? "H3_SIG=" + H3.pddSigmas(STEPS) + " H3_HEADS=" + H3.pddHeads(STEPS) + " " : "";
        setProgressUi(3, L.s("Токенизация…", "Tokenising…"), t0);
        try { H3.writeIds(home, pr, ids); } catch (Exception e) { ui.post(() -> Toast.makeText(this, "tokenizer: " + e, Toast.LENGTH_LONG).show()); return null; }
        final boolean[] done = {false};
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                String s = su("echo $(grep -c '^text encoder' " + M + "/h3.log) $(grep -c '^step' " + M + "/h3.log) $(grep -c '^done' " + M + "/h3.log)", true);
                try {
                    String[] a = s.split("\\s+"); int te = Integer.parseInt(a[0]), st = Integer.parseInt(a[1]), dn = Integer.parseInt(a[2]);
                    if (dn > 0) setProgressUi(97, L.s("Декодирование…", "Decoding…"), t0);
                    else if (st >= STEPS) setProgressUi(93, L.s("Декодирование…", "Decoding…"), t0);
                    else if (te > 0) setProgressUi(14 + 78 * st / STEPS, L.f("Шаг %d/%d", "Step %d/%d", Math.min(st + 1, STEPS), STEPS), t0);
                    else setProgressUi(6, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                } catch (Exception ignored) { }
            }
        });
        poll.start();
        su("rm -f " + q(raw.getPath()) + "; " + loraEnv + "HM=" + q(M.getPath()) + " sh " + D + "/h3.sh " + q(ids.getPath()) + " " + w + " " + h + " " + seedS + " " + q(raw.getPath()) + " < /dev/null >> " + D + "/gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        try {
            byte[] png = Moss.png(raw, w, h); raw.delete();
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            if (lora) meta.put("lora", "pDMD " + STEPS + "-step");
            return Gallery.save(this, Settings.metaInPng(this) ? PngMeta.embed(png, meta) : png, meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Ошибка генерации: %s (лог files/h3/h3.log)", "Generation failed: %s (log files/h3/h3.log)", e.getMessage()), Toast.LENGTH_LONG).show()); return null; }
    }

    /** shows the preview of the picture being made in place of the last result (the final picture replaces it) */
    void showPreview(Bitmap pb) {
        if (pb == null) return;
        ui.post(() -> { if (!busy) return; img.setImageBitmap(pb); img.setVisibility(View.VISIBLE); placeholder.setVisibility(View.GONE); });
    }

    /** Lab: a turbo LoRA file on the plain pack: overlays (made now, cached by strength) + its sigma table */
    String qwenLoraEnv() {
        if (!customTurboQwen()) return "";
        final String[] cm = {""}; File ovl = QwenLora.ensureTurbo(getFilesDir(), BIN, prefs().getFloat("qturbo_str", 1f), cm);
        if (ovl == null) { ui.post(() -> Toast.makeText(this, L.f("Не удалось подготовить LoRA: %s", "Could not prepare the LoRA: %s", cm[0]), Toast.LENGTH_LONG).show()); return " QLORA_FAILED=1"; }
        return " QLORA=" + q(ovl.getPath()) + " QSCHED=" + prefs().getInt("qturbo_sampler", 4);
    }
    /** one Qwen-Image picture (or an edit of the picked photo): tokenise / prepare the photo here, everything else on the NPU through qwen.sh */
    File runOneQwen(String pr, String seedS, int w, int h, long t0) {
        final File home = getFilesDir(), Q = Qwen.dir(home), ids = new File(Q, "ids.i32"), raw = new File(Q, "img.f32");
        final int QS = steps(); final Bitmap src = editBm; final boolean edit = src != null;
        setProgressUi(2, edit ? L.s("Подготовка фото…", "Preparing the photo…") : L.s("Токенизация…", "Tokenising…"), t0);
        int drop; String edArg = "";
        try {
            if (edit) {
                java.util.List<Bitmap> srcs = new java.util.ArrayList<>(); srcs.add(src); srcs.addAll(extraBm);
                Qwen.Edit e = Qwen.prepareEdit(home, srcs, pr, w, h, ids); drop = e.drop; edArg = " " + q(Qwen.editDir(home).getPath());
                int keys = (w / 16) * (h / 16) + e.nc + (e.tokens - e.drop - e.nm);
                if (keys > Qwen.MAX_KEYS) { ui.post(() -> Toast.makeText(this, L.s("Фото и результат вместе слишком большие: выберите размер поменьше", "The photo and the result together are too large: pick a smaller size"), Toast.LENGTH_LONG).show()); return null; }
            } else drop = Qwen.writeIds(home, pr, ids);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, "qwen: " + e, Toast.LENGTH_LONG).show()); return null; }
        final boolean[] done = {false};
        final Preview pv = Preview.load(this, "qwen"); final File pvf = new File(Q, "prev.bin"); pvf.delete();
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                if (pv != null) showPreview(pv.poll(pvf));
                String s = su("echo $(grep -c '^vit:' " + Q + "/qwen.log) $(grep -c '^te:' " + Q + "/qwen.log) $(grep -c '^txt_in:' " + Q + "/qwen.log) $(grep -c '^step' " + Q + "/qwen.log) $(grep -c '^done' " + Q + "/qwen.log)", true);
                try {
                    String[] a = s.split("\\s+"); int vit = Integer.parseInt(a[0]), te = Integer.parseInt(a[1]), ti = Integer.parseInt(a[2]), st = Integer.parseInt(a[3]), dn = Integer.parseInt(a[4]);
                    if (dn >= (edit ? 2 : 1)) setProgressUi(94, L.s("Декодирование…", "Decoding…"), t0);
                    else if (ti > 0 || st > 0) setProgressUi(30 + 62 * st / QS, st == 0 && edit ? L.s("Шаг 1/" + QS + " (разбор фото)", "Step 1/" + QS + " (reading the photo)") : L.f("Шаг %d/%d", "Step %d/%d", Math.min(st + 1, QS), QS), t0);
                    else if (te > 0) setProgressUi(28, L.s("Подготовка текста…", "Preparing the text…"), t0);
                    else if (edit && vit > 0) setProgressUi(15, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                    else if (edit) setProgressUi(5, L.s("Анализ фото на NPU…", "Analysing the photo on the NPU…"), t0);
                    else setProgressUi(8, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                } catch (Exception ignored) { }
            }
        });
        poll.start();
        su("rm -f " + q(raw.getPath()) + "; Q=" + q(Q.getPath()) + (Qwen.is8(model) ? " QV=8" : "") + qwenLoraEnv() + " VAE=" + Qwen.vaeCtx(home, w, h) + " sh " + D + "/qwen.sh " + q(ids.getPath()) + " " + drop + " " + w + " " + h + " " + seedS + " " + QS + " " + q(raw.getPath()) + edArg + " < /dev/null >> " + D + "/gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        try {
            byte[] png = Qwen.png(raw, w, h); raw.delete();
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            if (edit) meta.put("edit", true).put("pictures", 1 + extraBm.size());
            return Gallery.save(this, Settings.metaInPng(this) ? PngMeta.embed(png, meta) : png, meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Ошибка генерации: %s (лог files/qwen/qwen.log)", "Generation failed: %s (log files/qwen/qwen.log)", e.getMessage()), Toast.LENGTH_LONG).show()); return null; }
    }

    /** "Done" notification when the app is not on screen; tapping it opens the app */
    void notifyDone(android.content.Context app, Bitmap bm, String pr, int made) {
        try {
            NotificationManager nm = (NotificationManager) app.getSystemService(NOTIFICATION_SERVICE);
            nm.createNotificationChannel(new NotificationChannel("done", L.s("Готовые картинки", "Finished pictures"), NotificationManager.IMPORTANCE_DEFAULT));
            Notification.Builder nb = new Notification.Builder(app, "done").setSmallIcon(android.R.drawable.ic_menu_gallery).setAutoCancel(true)
                .setContentTitle(made > 1 ? L.f("Готово: %d картинки", "Done: %d pictures", made) : L.s("Картинка готова", "Picture is ready")).setContentText(pr)
                .setContentIntent(PendingIntent.getActivity(app, 0, new Intent(app, Main.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP), PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
            if (bm != null) nb.setLargeIcon(bm).setStyle(new Notification.BigPictureStyle().bigPicture(bm));
            nm.notify(2, nb.build());
        } catch (Exception ignored) { }
    }

    /** [steps done, vae runs, adapter runs (text encoder finished)] from the animad log. */
    int[] count() {
        String s = su("echo $(grep -c 'animad: step' " + D + "/animad.log) $(grep -c 'animad: vae' " + D + "/animad.log) $(grep -c 'animad: adapter' " + D + "/animad.log)", true);
        try {
            String[] a = s.split("\\s+");
            return new int[]{Integer.parseInt(a[0]), Integer.parseInt(a[1]), Integer.parseInt(a[2])};
        } catch (Exception e) {
            return new int[]{0, 0, 0};
        }
    }

    /** pct is the progress of the current picture; with several pictures the bar and the text cover the whole batch */
    /** the ring replaces the "your picture will appear here" hint while a picture is made and no preview shows yet */
    /** small header status: a white icon and a short value; a tap shows the details */
    TextView statusItem(String icon) {
        TextView t = text("", 13, TXT, false); t.setGravity(Gravity.CENTER_VERTICAL); t.setPadding(dp(8), 0, dp(8), 0);
        t.setCompoundDrawablePadding(dp(4)); t.setCompoundDrawablesRelative(vec(icon, TXT, 16), null, null, null); return t;
    }
    /** label on the left, the control on the right, one line */
    LinearLayout paramRow(String label, View control) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setMinimumHeight(dp(52));
        r.addView(text(label, 15, TXT, false), new LinearLayout.LayoutParams(0, -2, 1));
        r.addView(control);
        return r;
    }
    /** −  value  + : one tap changes the value; a tap on the value opens the full list (steps) or the free input (CFG) */
    LinearLayout stepper(final TextView val, final int kind) {
        LinearLayout b = new LinearLayout(this); b.setGravity(Gravity.CENTER_VERTICAL);
        TextView mi = text("−", 20, DIM, false), pl = text("+", 20, DIM, false);
        for (TextView t : new TextView[]{mi, pl}) t.setGravity(Gravity.CENTER);
        mi.setOnClickListener(v -> bump(kind, -1)); pl.setOnClickListener(v -> bump(kind, 1));
        val.setGravity(Gravity.CENTER); val.setMinWidth(dp(44));
        val.setOnClickListener(v -> { if (kind == 0) pickBaseSteps(); else if (kind == 1) pickBaseCfg(); else pickCount(); });
        b.addView(mi, new LinearLayout.LayoutParams(dp(40), dp(40))); b.addView(val); b.addView(pl, new LinearLayout.LayoutParams(dp(40), dp(40)));
        return b;
    }
    /** pictures per run: any number; while a batch runs the value is the length of the queue (never below the picture being made) and the loop follows it */
    void setCount(int v) {
        v = Math.max(busy ? curIdx + 1 : 1, Math.min(999, v)); count = v; prefs().edit().putInt("count", count).apply();
        if (busy) { curN = count; Toast.makeText(this, L.f("В очереди: %d", "In the queue: %d", count), Toast.LENGTH_SHORT).show(); }
        refreshBaseUi(); updateChips();
    }
    void pickCount() {
        final EditText e = new EditText(this); e.setText(String.valueOf(count)); e.setSelectAllOnFocus(true); e.setTextColor(TXT); e.setInputType(InputType.TYPE_CLASS_NUMBER);
        builder().setTitle(L.s("Картинок за раз", "Pictures per run")).setView(e).setNegativeButton(L.s("Отмена", "Cancel"), null).setPositiveButton("OK", (dd, w) -> {
            try { setCount(Integer.parseInt(e.getText().toString().trim())); } catch (NumberFormatException x) { Toast.makeText(this, L.s("Введите число", "Enter a number"), Toast.LENGTH_SHORT).show(); }
        }).show();
    }
    void bump(int kind, int dir) {
        if (kind == 0) {
            setSteps(steps() + dir); return;
        } else if (kind == 1) {
            float v; try { v = Float.parseFloat(cfgStr()); } catch (NumberFormatException e) { v = base() ? 4 : 1; }
            v = Math.max(1, Math.min(20, Math.round((v + 0.5f * dir) * 2) / 2f));
            prefs().edit().putString(cfgKey(), v == (int) v ? String.valueOf((int) v) : String.valueOf(v)).apply();
        } else {
            setCount(count + dir); return;
        }
        refreshBaseUi();
    }
    void updateChips() {
        if (baseRow != null) baseRow.setVisibility(isAnima() ? View.VISIBLE : View.GONE);
        if (stepsRow != null) stepsRow.setVisibility(model.isEmpty() ? View.GONE : View.VISIBLE);
        if (countVal != null) countVal.setText(String.valueOf(count));
    }
    void setProgressUi(int pct, String msg, long t0) {
        final int all = (curIdx * 100 + pct) / curN;
        final String pre = curN > 1 ? "(" + (curIdx + 1) + "/" + curN + ")  " : "";
        final String t = L.f("%s  ·  %.0f с", "%s  ·  %.0f s", pre + msg, (System.currentTimeMillis() - t0) / 1000.0);
        ui.post(() -> { bar.setProgress(all); status.setText(t); });
    }
}
