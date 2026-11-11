package com.hexanima.npu;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.WindowManager;
import android.widget.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Model catalog: a card per model (size, speed at the base resolution, installed state) with download / delete. The download itself runs on a thread with a foreground service,
 *  its progress lives in statics so that leaving this screen does not lose it. The shared files (text encoder, VAE) come with the first model. */
public class ModelsActivity extends Base {
    static volatile boolean downloading;
    static volatile String dlId = "", dlMsg = "";
    static volatile long dlDone, dlTotal;
    static final AtomicBoolean cancel = new AtomicBoolean();
    static volatile String deleted = "";   // last deleted model id: Main unloads the server if it holds it

    Models store;
    final Handler ui = new Handler(Looper.getMainLooper());
    final ExecutorService io = Executors.newSingleThreadExecutor();
    LinearLayout list;
    TextView info;
    EditText url;
    List<Models.Item> items = new ArrayList<>();
    final java.util.Map<String, ProgressBar> bars = new java.util.HashMap<>();
    final java.util.Map<String, TextView> msgs = new java.util.HashMap<>();
    boolean wasDownloading;
    EditText search; boolean fInst, fFit; final java.util.Set<String> openE = new java.util.HashSet<>();
    final Runnable tick = new Runnable() {
        @Override public void run() {
            if (downloading) {
                ProgressBar pb = bars.get(dlId); TextView tv = msgs.get(dlId);
                if (pb != null && dlTotal > 0) pb.setProgress((int) (1000 * dlDone / dlTotal));
                if (tv != null) tv.setText(dlMsg);
            } else if (wasDownloading) { wasDownloading = false; getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); render(); }
            if (downloading) wasDownloading = true;
            ui.postDelayed(this, 400);
        }
    };

    String prefBase() { return getSharedPreferences("anima", 0).getString("base", Main.DEFAULT_BASE); }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        store = new Models(getFilesDir());
        Button src = button(L.s("Источник", "Source"), CARD, DIM); src.setTextSize(12);
        LinearLayout col = screen(L.s("Модели", "Models"), src);
        info = text("", 12, DIM, false); info.setPadding(dp(2), 0, 0, dp(10));
        col.addView(info);
        search = field(L.s("Поиск по названию, автору…", "Search by name or author…")); search.setSingleLine(); search.setTextSize(14);
        search.addTextChangedListener(new android.text.TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int b, int c) { }
            public void onTextChanged(CharSequence s, int a, int b, int c) { }
            public void afterTextChanged(android.text.Editable e) { if (!items.isEmpty()) render(); }
        });
        col.addView(search, new LinearLayout.LayoutParams(-1, dp(46)));
        LinearLayout fr = new LinearLayout(this); fr.setPadding(0, dp(10), 0, dp(4));
        fr.addView(filterChip(L.s("Установленные", "Installed"), true)); fr.addView(filterChip(L.s("Подходят телефону", "Fits this phone"), false));
        col.addView(fr);
        url = field(L.s("Адрес каталога (…/resolve/main/)", "Catalog URL (…/resolve/main/)")); url.setText(prefBase()); url.setTextSize(12); url.setBackground(bg(CARD, 14)); url.setVisibility(View.GONE);
        col.addView(url);
        src.setOnClickListener(v -> { boolean show = url.getVisibility() != View.VISIBLE; url.setVisibility(show ? View.VISIBLE : View.GONE); if (!show) load(); });
        list = new LinearLayout(this); list.setOrientation(LinearLayout.VERTICAL);
        col.addView(list);
        String soc = android.os.Build.VERSION.SDK_INT >= 31 ? android.os.Build.SOC_MODEL : "?";
        boolean ok = soc.equals("SM8475") || soc.equals("SM8450");
        info.setText((ok ? L.f("Чип %s · подходит (%s)", "Chip %s · supported (%s)", soc, Compat.archLabel()) : L.f("Чип %s · проверено только на Snapdragon 8 Gen 1 / 8+ Gen 1", "Chip %s · tested only on Snapdragon 8 Gen 1 / 8+ Gen 1", soc))
            + "\n" + L.f("Свободно %.1f ГБ", "Free %.1f GB", getFilesDir().getUsableSpace() / 1e9));
        if (downloading) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        load();
    }

    @Override protected void onResume() { super.onResume(); ui.post(tick); }
    @Override protected void onPause() { super.onPause(); ui.removeCallbacks(tick); }
    @Override protected void onDestroy() { super.onDestroy(); io.shutdownNow(); }

    void load() {
        list.removeAllViews();
        list.addView(text(L.s("Загружаю каталог…", "Loading catalog…"), 13, DIM, false));
        final String base = url.getText().toString().trim().replaceAll("/*$", "") + "/";
        getSharedPreferences("anima", 0).edit().putString("base", base).apply();
        io.submit(() -> {
            try { final List<Models.Item> cat = store.catalog(base); for (Models.Item it : cat) store.refreshMeta(it); ui.post(() -> { items = cat; render(); }); }
            catch (Exception e) { ui.post(() -> { list.removeAllViews(); list.addView(text(L.f("Каталог недоступен: %s", "Catalog unavailable: %s", e.getMessage()), 13, BAD, false)); }); }
        });
    }

    static String gb(long b) { return String.format("%.1f", b / 1e9); }

    TextView filterChip(String label, final boolean inst) {
        final TextView t = text(label, 13, TXT, false); t.setGravity(android.view.Gravity.CENTER); t.setPadding(dp(14), 0, dp(14), 0);
        Runnable paint = () -> { boolean on = inst ? fInst : fFit; t.setTextColor(on ? BG : TXT); t.setBackground(on ? bg(TXT, 17) : bg(CARD, 17)); };
        paint.run();
        t.setOnClickListener(v -> { if (inst) fInst = !fInst; else fFit = !fFit; paint.run(); if (!items.isEmpty()) render(); });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, dp(34)); lp.rightMargin = dp(8); t.setLayoutParams(lp);
        return t;
    }

    /** one line of the catalog: a model with its variants (Anima turbo + base, Qwen 4 step + 8 step) */
    static final class Entry { String key, name, family, line = ""; final List<Models.Item> v = new ArrayList<>(); final List<String> lbl = new ArrayList<>(); }

    static String familyOf(Models.Item it) { return it.moss ? "MOSS-Image" : it.qwen ? "Qwen-Image" : it.mage ? "Mage-Flow" : "Anima"; }
    static String lineOf(Models.Item it) { String n = it.name; return n.contains("RealSkin") ? "RealSkin" : n.contains("3D Harem") ? "3D Harem" : "Harem"; }

    List<Entry> entries() {
        java.util.LinkedHashMap<String, Entry> map = new java.util.LinkedHashMap<>();
        for (Models.Item it : items) {
            String fam = familyOf(it), key = it.qwen ? "qwen" : it.moss ? (Moss.isRs(it.id) ? "moss-rs" : "moss") : fam.equals("Anima") ? Main.famOf(it.id) : it.id;
            final boolean legacy = (it.qwen && !Qwen.is8(it.id)) || (fam.equals("Anima") && !it.id.endsWith("base"));   // old scheme: merged turbo checkpoints / the 4 step Qwen pack; new users see base + LoRA only
            if (legacy) { key = "legacy-" + it.id; fam = "Legacy"; }
            Entry e = map.get(key); if (e == null) { e = new Entry(); e.key = key; e.family = fam; e.line = fam.equals("Anima") ? lineOf(it) : ""; map.put(key, e); }
            e.v.add(it);
            e.lbl.add(legacy ? "" : it.moss ? (Moss.isRs(it.id) ? "" : Moss.is8(it.id) ? "8 step" : "4 step") : it.qwen ? "" : fam.equals("Anima") ? "" : "");
            if (e.name == null || (it.moss && !Moss.is8(it.id) && !Moss.isRs(it.id))) e.name = legacy ? (it.qwen ? "Qwen-Image 2.1 Turbo (4 step, Lightning)" : it.name.replace(" (base)", "") + " (turbo)") : it.qwen ? "Qwen-Image-2.1-Turbo" : it.name.replace(" (base)", "");
        }
        return new ArrayList<>(map.values());
    }

    void render() {
        list.removeAllViews(); bars.clear(); msgs.clear();
        final String base = prefBase(), q = search.getText().toString().trim().toLowerCase();
        final boolean needCommonAnima = !store.commonInstalled();
        long used = 0; for (Models.Item it : items) if (store.modelInstalled(it.id)) used += it.size;
        info.setText(info.getText().toString().split("\n")[0] + "\n" + L.f("Моделей на телефоне: %s ГБ  ·  свободно %.1f ГБ", "Models on the phone: %s GB  ·  free %.1f GB", gb(used), getFilesDir().getUsableSpace() / 1e9));
        List<Entry> all = entries(); java.util.LinkedHashMap<String, List<Entry>> fams = new java.util.LinkedHashMap<>();
        for (String f : new String[]{"Anima", "Qwen-Image", "MOSS-Image", "Mage-Flow", "Legacy"}) fams.put(f, new ArrayList<>());
        for (Entry e : all) {
            boolean inst = false, ramOk = true, hit = q.isEmpty();
            for (Models.Item it : e.v) { inst |= store.modelInstalled(it.id); ramOk &= !((it.moss || it.qwen) && Compat.ramGb(this) < 10.5); hit |= (it.name + " " + it.author + " " + e.family).toLowerCase().contains(q); }
            if (!hit || (fInst && !inst) || (fFit && !ramOk)) continue;
            List<Entry> l = fams.get(e.family); if (l == null) { l = new ArrayList<>(); fams.put(e.family, l); } l.add(e);
        }
        boolean any = false;
        for (java.util.Map.Entry<String, List<Entry>> fe : fams.entrySet()) {
            List<Entry> es = fe.getValue(); if (es.isEmpty()) continue; any = true;
            final String fam = fe.getKey(); int have = 0; boolean busyIn = false;
            if (fam.equals("Anima")) { final List<String> order = java.util.Arrays.asList("Harem", "RealSkin", "3D Harem"); java.util.Collections.sort(es, (a, b) -> order.indexOf(a.line) - order.indexOf(b.line)); }   // stable: by line, then catalog order
            for (Entry e : es) { boolean h = false; for (Models.Item it : e.v) { h |= store.modelInstalled(it.id); busyIn |= downloading && it.id.equals(dlId); } if (h) have++; }
            final boolean open = !q.isEmpty() || busyIn ? true : getSharedPreferences("anima", 0).getBoolean("cat_open_" + fam, have > 0);
            LinearLayout hd = new LinearLayout(this); hd.setGravity(android.view.Gravity.CENTER_VERTICAL); hd.setPadding(0, dp(18), 0, dp(8));
            hd.addView(text(fam.equals("Legacy") ? L.s("Старая версия", "Old version") : fam, 18, fam.equals("Legacy") ? DIM : TXT, true));
            TextView cnt = text("   " + L.f("%d из %d установлено", "%d of %d installed", have, es.size()), 13, DIM, false); hd.addView(cnt, new LinearLayout.LayoutParams(0, -2, 1));
            ImageView ch = new ImageView(this); ch.setImageDrawable(vec("ic_chevron", DIM, 20)); ch.setRotation(open ? 180 : 0); hd.addView(ch, new LinearLayout.LayoutParams(dp(22), dp(22)));
            hd.setOnClickListener(v -> { getSharedPreferences("anima", 0).edit().putBoolean("cat_open_" + fam, !open).apply(); render(); });
            list.addView(hd);
            if (!open) continue;
            String lastLine = null;
            for (Entry e : es) {
                if (fam.equals("Anima") && !e.line.equals(lastLine)) { lastLine = e.line; TextView lh = text(e.line, 12, DIM, false); lh.setPadding(dp(2), dp(6), 0, dp(6)); lh.setLetterSpacing(0.04f); list.addView(lh); }
                addEntry(e, base, needCommonAnima);
            }
        }
        if (items.isEmpty()) list.addView(text(L.s("Каталог пуст", "The catalog is empty"), 13, DIM, false));
        else if (!any) list.addView(text(L.s("Ничего не найдено", "Nothing found"), 14, DIM, false));
        else if (!store.commonInstalled() && hasAnima()) { TextView n = text(L.f("К первой модели Anima добавятся общие файлы (%s ГБ): текстовый энкодер и VAE. Они качаются один раз.", "The first Anima model also fetches the shared files (%s GB): text encoder and VAE, once.", gb(store.commonSize)), 12, DIM, false); n.setPadding(dp(2), dp(12), 0, 0); list.addView(n); }
    }

    void addEntry(final Entry e, final String base, final boolean needCommonAnima) {
        boolean anyHave = false, allHave = true, busyThis = false; long minSize = Long.MAX_VALUE;
        for (Models.Item it : e.v) { boolean h = store.modelInstalled(it.id); anyHave |= h; allHave &= h; busyThis |= downloading && it.id.equals(dlId); minSize = Math.min(minSize, it.size); }
        final boolean open = openE.contains(e.key) || busyThis;
        LinearLayout c = new LinearLayout(this); c.setOrientation(LinearLayout.VERTICAL); c.setBackground(bg(CARD, 14)); c.setPadding(dp(16), dp(12), dp(16), dp(12));
        Models.Item f = e.v.get(0);
        LinearLayout top = new LinearLayout(this); top.setGravity(android.view.Gravity.CENTER_VERTICAL);
        top.addView(text(e.name, 16, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        TextView st = text(allHave ? "✓" : anyHave ? "✓ " + L.s("частично", "partly") : "↓ " + gb(minSize) + " " + L.s("ГБ", "GB"), 13, anyHave ? TXT : DIM, false); top.addView(st);
        c.addView(top);
        StringBuilder sb = new StringBuilder(); if (!f.params.isEmpty()) sb.append(f.params); if (f.speed > 0) sb.append(sb.length() > 0 ? "  ·  " : "").append(L.f("~%d с", "~%d s", f.speed)); if (!f.author.isEmpty()) sb.append(sb.length() > 0 ? "  ·  " : "").append(f.author);
        if (e.v.size() > 1) sb.append(sb.length() > 0 ? "  ·  " : "").append(android.text.TextUtils.join(" / ", e.lbl));
        TextView mt = text(sb.toString(), 13, DIM, false); mt.setPadding(0, dp(3), 0, 0); c.addView(mt);
        if (open) {
            String d = L.s(f.desc, f.descEn); if (!d.isEmpty()) { TextView dv = text(d, 13, DIM, false); dv.setPadding(0, dp(8), 0, dp(2)); c.addView(dv); }
            for (int k = 0; k < e.v.size(); k++) {
                final Models.Item it = e.v.get(k); final boolean have = store.modelInstalled(it.id), busy = downloading && it.id.equals(dlId);
                final boolean needCommon = needCommonAnima && !it.moss && !it.qwen && !it.mage;
                LinearLayout vr = new LinearLayout(this); vr.setGravity(android.view.Gravity.CENTER_VERTICAL); vr.setPadding(0, dp(10), 0, 0);
                vr.addView(text((e.lbl.get(k).isEmpty() ? L.s("Модель", "Model") : e.lbl.get(k)) + "   " + L.f("%s ГБ", "%s GB", gb(it.size)), 14, TXT, false), new LinearLayout.LayoutParams(0, -2, 1));
                if (have && !busy) {
                    TextView del = text(L.s("Удалить", "Delete"), 14, BAD, false); del.setPadding(dp(12), dp(4), 0, dp(4));
                    del.setOnClickListener(v -> { if (downloading) return; store.delete(it.id); deleted = it.id; render(); }); vr.addView(del);
                } else if (!busy) {
                    Button bt = button(L.f("Скачать  %s ГБ", "Download  %s GB", gb(it.size + (needCommon ? store.commonSize : 0))), ACC, ON_ACC); bt.setTextSize(13); bt.setPadding(dp(14), 0, dp(14), 0); bt.setEnabled(!downloading); bt.setAlpha(downloading ? .5f : 1f);
                    bt.setOnClickListener(v -> start(base, it, needCommon)); vr.addView(bt, new LinearLayout.LayoutParams(-2, dp(36)));
                }
                c.addView(vr);
                if (busy) {
                    ProgressBar pb = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); pb.setMax(1000);
                    pb.setProgressTintList(android.content.res.ColorStateList.valueOf(ACC)); pb.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(LINE));
                    LinearLayout.LayoutParams pl = new LinearLayout.LayoutParams(-1, dp(3)); pl.topMargin = dp(10); c.addView(pb, pl); bars.put(it.id, pb);
                    TextView tv = text(dlMsg, 12, DIM, false); tv.setPadding(0, dp(6), 0, 0); c.addView(tv); msgs.put(it.id, tv);
                    TextView x = text(L.s("Отменить", "Cancel"), 14, BAD, false); x.setPadding(0, dp(8), 0, dp(2)); x.setOnClickListener(v -> cancel.set(true)); c.addView(x);
                }
            }
        }
        c.setOnClickListener(v -> { if (!openE.remove(e.key)) openE.add(e.key); render(); });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.bottomMargin = dp(8); list.addView(c, lp);
    }

    boolean hasAnima() { for (Models.Item it : items) if (!it.moss && !it.qwen && !it.mage) return true; return false; }

    void start(String base, Models.Item it, boolean needCommon) {
        if (!Compat.chipAllowed(this)) {   // an unsupported chip: the models will not run, so ask before 4 GB are downloaded
            builder().setMessage(L.f("Чип %s не поддерживается: модели рассчитаны на Snapdragon 8 Gen 1 / 8+ Gen 1. Всё равно скачать?", "Chip %s is not supported: the models are built for Snapdragon 8 Gen 1 / 8+ Gen 1. Download anyway?", Compat.soc()))
                .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> { Compat.allowChip(this); begin(base, it, needCommon); }).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        if ((it.moss || it.qwen) && Compat.ramGb(this) < 10.5) {   // text encoder 6 GB and the DiT stream through RAM: 12 GB phones only
            builder().setMessage(L.f("В телефоне %.1f ГБ ОЗУ: для этой модели рекомендуется 12 ГБ. Всё равно скачать?", "This phone has %.1f GB of RAM: 12 GB is recommended for this model. Download anyway?", Compat.ramGb(this)))
                .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> begin(base, it, needCommon)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        begin(base, it, needCommon);
    }

    void begin(String base, Models.Item it, boolean needCommon) {
        long need = it.size + (needCommon ? store.commonSize : 0);
        if (getFilesDir().getUsableSpace() < need + 300_000_000L) { Toast.makeText(this, L.f("Не хватает места: нужно %s ГБ", "Not enough space: need %s GB", gb(need)), Toast.LENGTH_LONG).show(); return; }
        if (downloading) return;
        downloading = true; dlId = it.id; dlDone = 0; dlTotal = need; dlMsg = L.s("Подключаюсь…", "Connecting…"); cancel.set(false);
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        render();
        final long[] last = {0};
        new Thread(() -> {
            Models.Progress pg = (d, tot, what) -> {
                long now = System.currentTimeMillis(); if (now - last[0] < 300) return; last[0] = now;
                dlDone = d; dlTotal = tot; dlMsg = L.f("%s / %s ГБ  ·  %s", "%s / %s GB  ·  %s", gb(d), gb(tot), what);
            };
            String err = null;
            try { if (!it.moss && !it.qwen && !it.mage) store.installCommon(base, pg, cancel); store.installModel(base, it, pg, cancel); }
            catch (java.io.InterruptedIOException e) { err = L.s("Загрузка отменена: файлы сохранены, можно продолжить", "Cancelled: the files are kept, you can resume"); }
            catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", e.getMessage()); }
            final String msg = err;
            downloading = false;
            DownloadService.stop(app);
            if (msg != null) ui.post(() -> Toast.makeText(app, msg, Toast.LENGTH_LONG).show());
        }).start();
    }
}
