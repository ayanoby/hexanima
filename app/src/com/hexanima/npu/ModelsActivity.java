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
import org.json.JSONObject;
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
    org.json.JSONObject staleM; long staleN;   // the shared Anima files that differ from the catalog (null: up to date or not checked)
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
        url = field(L.s("Адрес каталога (…/resolve/main/)", "Catalog URL (…/resolve/main/)")); url.setText(prefBase()); url.setTextSize(13); url.setBackground(bg(CARD, 14)); url.setVisibility(View.GONE);   // opened by "Source": right under the title, above the search
        LinearLayout.LayoutParams ul = new LinearLayout.LayoutParams(-1, -2); ul.bottomMargin = dp(10); col.addView(url, ul);
        search = field(L.s("Поиск по названию, автору…", "Search by name or author…")); search.setSingleLine(); search.setTextSize(14);
        search.addTextChangedListener(new android.text.TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int b, int c) { }
            public void onTextChanged(CharSequence s, int a, int b, int c) { }
            public void afterTextChanged(android.text.Editable e) { if (!items.isEmpty()) render(); }
        });
        col.addView(search, new LinearLayout.LayoutParams(-1, dp(46)));
        LinearLayout fr = new LinearLayout(this); fr.setPadding(0, dp(10), 0, dp(4));
        fr.addView(filterChip(L.s("Установленные", "Installed"), true));
        col.addView(fr);
        Button own = button(L.s("＋ Свой чекпоинт Anima (.safetensors, BF16/FP16/FP32)", "＋ Own Anima checkpoint (.safetensors, BF16/FP16/FP32)"), CARD, TXT); own.setTextSize(13);
        own.setOnClickListener(v -> AnimaConv.pick(this));
        own.setMinHeight(dp(42)); own.setMinimumHeight(dp(42)); own.setPadding(dp(14), dp(8), dp(14), dp(8));
        LinearLayout.LayoutParams ol = new LinearLayout.LayoutParams(-1, -2); ol.topMargin = dp(6); ol.bottomMargin = dp(6); col.addView(own, ol);   // the button grows to two lines when the text does not fit
        src.setOnClickListener(v -> { boolean show = url.getVisibility() != View.VISIBLE; url.setVisibility(show ? View.VISIBLE : View.GONE); if (!show) load(); });
        list = new LinearLayout(this); list.setOrientation(LinearLayout.VERTICAL);
        col.addView(list);
        String soc = Compat.soc(); boolean ok = Compat.chipOk();
        info.setText((ok ? L.f("Чип %s · подходит (%s)", "Chip %s · supported (%s)", soc, Compat.archLabel()) : L.f("Чип %s · проверено только на Snapdragon 8 Gen 1 / 8+ Gen 1", "Chip %s · tested only on Snapdragon 8 Gen 1 / 8+ Gen 1", soc))
            + "\n" + L.f("Свободно %.1f ГБ", "Free %.1f GB", getFilesDir().getUsableSpace() / 1e9));
        if (downloading) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        load();
    }

    @Override protected void onActivityResult(int req, int res, Intent d) {
        if (req == AnimaConv.REQ && res == RESULT_OK && d != null && d.getData() != null) new AnimaConv(this, store, prefBase(), () -> render()).start(d.getData());
        else super.onActivityResult(req, res, d);
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
            try { final List<Models.Item> cat = store.catalog(base); for (Models.Item it : cat) store.refreshMeta(it); ui.post(() -> { items = cat; render(); });
                if (store.commonInstalled() && !downloading) try { JSONObject m = store.commonManifest(base); List<Object[]> st = store.commonStale(m); ui.post(() -> { staleM = st.isEmpty() ? null : m; staleN = Models.staleBytes(st); render(); }); } catch (Exception ignored) { } }
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

    static String familyOf(Models.Item it) { return it.moss ? "MOSS-Image" : it.qwen ? "Qwen-Image" : it.mage ? "Mage-Flow" : it.h3 || it.r2v ? "MiniMax-H3" : it.klein ? "FLUX.3-Klein" : it.ltx ? "LTX-2.5" : "Anima"; }
    static String lineOf(Models.Item it) { String n = it.name; return n.contains("RealSkin") ? "RealSkin" : n.contains("3D Harem") ? "3D Harem" : "Harem"; }

    List<Entry> entries() {
        java.util.LinkedHashMap<String, Entry> map = new java.util.LinkedHashMap<>();
        for (Models.Item it : items) {
            String fam = familyOf(it), key = it.qwen ? "qwen" : it.moss ? (Moss.isRs(it.id) ? "moss-" + Moss.sub(it.id) : "moss") : fam.equals("Anima") ? Main.famOf(it.id) : it.id;
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
        long used = store.commonBytes(); for (Models.Item it : items) if (store.modelInstalled(it.id)) used += it.size; for (String id : store.customIds()) used += store.dirSize(store.modelDir(id));   // the catalog packs, the shared Anima files and the models converted on the phone
        info.setText(info.getText().toString().split("\n")[0] + "\n" + L.f("Моделей на телефоне: %s ГБ  ·  свободно %.1f ГБ", "Models on the phone: %s GB  ·  free %.1f GB", gb(used), getFilesDir().getUsableSpace() / 1e9));
        addUpdate(base);
        addCustom();
        List<Entry> all = entries(); java.util.LinkedHashMap<String, List<Entry>> fams = new java.util.LinkedHashMap<>();
        for (String f : new String[]{"Anima", "Qwen-Image", "MOSS-Image", "Mage-Flow", "MiniMax-H3", "FLUX.3-Klein", "LTX-2.5", "Legacy"}) fams.put(f, new ArrayList<>());
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
        if (q.isEmpty() || "upscaler real-esrgan апскейлер".contains(q)) { addUpscale(); any = true; }
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
                final boolean needCommon = needCommonAnima && it.anima();
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

    boolean customOpen = true; final java.util.Set<String> customE = new java.util.HashSet<>();
    /** "Own models": the checkpoints converted on the phone, each expands to its details and a delete button */
    /** a card at the top when the shared files (text encoder, VAE) changed in the catalog: one button downloads just the changed ones */
    void addUpdate(final String base) {
        final boolean busy = downloading && dlId.equals("common");
        if (staleM == null && !busy) return;
        LinearLayout c = new LinearLayout(this); c.setOrientation(LinearLayout.VERTICAL); c.setPadding(dp(16), dp(14), dp(16), dp(14)); c.setBackground(bg(CARD, 14));
        c.addView(text(L.s("Доступно обновление общих файлов", "Update for the shared files"), 15, TXT, true));
        TextView d = text(L.f("Текстовый энкодер и VAE в каталоге изменились (%s ГБ). Обновление исправляет, например, шум на второй картинке с длинным промптом. Модели скачивать заново не нужно.", "The text encoder / VAE changed in the catalog (%s GB). The update fixes, for example, noise on the second picture with a long prompt. The models stay as they are.", gb(staleN)), 12, DIM, false); d.setPadding(0, dp(6), 0, dp(10)); c.addView(d);
        if (busy) {
            ProgressBar pb = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); pb.setMax(1000); c.addView(pb, new LinearLayout.LayoutParams(-1, dp(3))); bars.put("common", pb);
            TextView tv = text(dlMsg, 12, DIM, false); tv.setPadding(0, dp(6), 0, 0); c.addView(tv); msgs.put("common", tv);
        } else {
            Button bt = button(L.f("Обновить  %s ГБ", "Update  %s GB", gb(staleN)), ACC, ON_ACC); bt.setTextSize(13); bt.setEnabled(!downloading); bt.setAlpha(downloading ? .5f : 1f);
            bt.setOnClickListener(v -> updateCommon(base)); c.addView(bt, new LinearLayout.LayoutParams(-2, dp(36)));
        }
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.bottomMargin = dp(10); list.addView(c, lp);
    }

    void updateCommon(final String base) {
        if (downloading || staleM == null) return;
        if (getFilesDir().getUsableSpace() < staleN + 300_000_000L) { Toast.makeText(this, L.f("Не хватает места: нужно %s ГБ", "Not enough space: need %s GB", gb(staleN)), Toast.LENGTH_LONG).show(); return; }
        final org.json.JSONObject m = staleM;
        downloading = true; dlId = "common"; dlDone = 0; dlTotal = staleN; dlMsg = L.s("Подключаюсь…", "Connecting…"); cancel.set(false);
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        render();
        final long[] last = {0};
        new Thread(() -> {
            Models.Progress pg = (dn, tot, what) -> { long now = System.currentTimeMillis(); if (now - last[0] < 300) return; last[0] = now; dlDone = dn; dlTotal = tot; dlMsg = L.f("%s / %s ГБ  ·  %s", "%s / %s GB  ·  %s", gb(dn), gb(tot), what); };
            String err = null;
            try { store.updateCommon(base, m, pg, cancel); staleM = null; }
            catch (java.io.InterruptedIOException e) { err = L.s("Загрузка отменена: можно продолжить", "Cancelled: you can resume"); }
            catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", e.getMessage()); }
            final String msg = err; downloading = false; DownloadService.stop(app);
            ui.post(() -> { if (msg != null) Toast.makeText(app, msg, Toast.LENGTH_LONG).show(); else Toast.makeText(app, L.s("Общие файлы обновлены", "Shared files updated"), Toast.LENGTH_LONG).show(); render(); });
        }).start();
    }

    void addCustom() {
        final List<String> cs = store.customIds();
        if (cs.isEmpty()) { TextView e = text(L.s("Свои модели: пока нет. Кнопка «＋ Свой чекпоинт Anima» выше сконвертирует файл .safetensors прямо на телефоне.", "Own models: none yet. The “＋ Own Anima checkpoint” button above converts a .safetensors file on the phone."), 12, DIM, false); e.setPadding(dp(4), dp(10), 0, dp(10)); list.addView(e); return; }
        TextView h = text(L.f("Свои модели (%d)  %s", "Own models (%d)  %s", cs.size(), customOpen ? "▴" : "▾"), 15, TXT, true); h.setPadding(dp(4), dp(10), 0, dp(8));
        h.setOnClickListener(v -> { customOpen = !customOpen; render(); }); list.addView(h);
        if (!customOpen) return;
        for (final String id : cs) {
            LinearLayout c = new LinearLayout(this); c.setOrientation(LinearLayout.VERTICAL); c.setPadding(dp(16), dp(12), dp(16), dp(12)); c.setBackground(bg(CARD, 14));
            final long sz = store.dirSize(store.modelDir(id)); final boolean open = customE.contains(id);
            c.addView(text(store.name(id).replace(" (base)", "") + (open ? "" : "  ▾"), 15, TXT, true));
            if (open) {
                int blocks = 0; File[] fs = store.modelDir(id).listFiles(); if (fs != null) for (File f : fs) if (f.getName().matches("blk\\d+\\.bin")) blocks++;
                TextView d = text(L.f("%s · блоков %d · %s ГБ на телефоне\nid: %s", "%s · %d blocks · %s GB on the phone\nid: %s", id.endsWith("base") ? L.s("обычная", "regular") : "Turbo", blocks, gb(sz), id), 12, DIM, false); d.setPadding(0, dp(6), 0, dp(8)); c.addView(d);
                TextView del = text(L.s("Удалить", "Delete"), 14, BAD, true); del.setPadding(0, dp(4), 0, dp(2));
                del.setOnClickListener(v -> builder().setMessage(L.f("Удалить «%s»? Файлы модели будут стёрты с телефона (оригинальный чекпоинт не затрагивается).", "Delete \"%s\"? The model files are removed from the phone (the original checkpoint is not touched).", store.name(id).replace(" (base)", "")))
                    .setNegativeButton(L.s("Отмена", "Cancel"), null).setPositiveButton(L.s("Удалить", "Delete"), (dd, w) -> { store.delete(id); customE.remove(id); render(); }).show());
                c.addView(del);
            }
            c.setOnClickListener(v -> { if (!customE.remove(id)) customE.add(id); render(); });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.bottomMargin = dp(8); list.addView(c, lp);
        }
    }

    /** the Real-ESRGAN upscaler weights (not a model of the generator): two small downloads, used by the button in the picture viewer */
    void addUpscale() {
        LinearLayout hd = new LinearLayout(this); hd.setPadding(0, dp(18), 0, dp(8)); hd.addView(text(L.s("Апскейлер", "Upscaler"), 18, TXT, true)); list.addView(hd);
        LinearLayout c = new LinearLayout(this); c.setOrientation(LinearLayout.VERTICAL); c.setBackground(bg(CARD, 14)); c.setPadding(dp(16), dp(12), dp(16), dp(12));
        c.addView(text("Real-ESRGAN ×2 / ×4", 16, TXT, true));
        TextView mt = text(L.s("Увеличивает готовую картинку на NPU. Кнопка в просмотре картинки. BSD-3  ·  Xintao Wang et al.", "Enlarges a finished picture on the NPU. The button is in the picture viewer. BSD-3  ·  Xintao Wang et al."), 13, DIM, false); mt.setPadding(0, dp(3), 0, dp(2)); c.addView(mt);
        for (final boolean anime : new boolean[]{false, true}) {
            final String id = anime ? "up-anime" : "up-photo"; final boolean have = Upscale.has(getFilesDir(), anime), busy = downloading && id.equals(dlId);
            LinearLayout vr = new LinearLayout(this); vr.setGravity(android.view.Gravity.CENTER_VERTICAL); vr.setPadding(0, dp(10), 0, 0);
            vr.addView(text((anime ? L.s("Аниме", "Anime") : L.s("Фото", "Photo")) + "   " + L.f("%s ГБ", "%s GB", String.format("%.2f", Upscale.size(anime) / 1e9)), 14, TXT, false), new LinearLayout.LayoutParams(0, -2, 1));
            if (have && !busy) { TextView del = text(L.s("Удалить", "Delete"), 14, BAD, false); del.setPadding(dp(12), dp(4), 0, dp(4)); del.setOnClickListener(v -> { if (downloading) return; Upscale.delete(getFilesDir(), anime); render(); }); vr.addView(del); }
            else if (!busy) {
                Button bt = button(L.f("Скачать  %s ГБ", "Download  %s GB", String.format("%.2f", Upscale.size(anime) / 1e9)), ACC, ON_ACC); bt.setTextSize(13); bt.setPadding(dp(14), 0, dp(14), 0); bt.setEnabled(!downloading); bt.setAlpha(downloading ? .5f : 1f);
                bt.setOnClickListener(v -> startUp(anime)); vr.addView(bt, new LinearLayout.LayoutParams(-2, dp(36)));
            }
            c.addView(vr);
            if (busy) {
                ProgressBar pb = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); pb.setMax(1000);
                pb.setProgressTintList(android.content.res.ColorStateList.valueOf(ACC)); pb.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(LINE));
                LinearLayout.LayoutParams pl = new LinearLayout.LayoutParams(-1, dp(3)); pl.topMargin = dp(10); c.addView(pb, pl); bars.put(id, pb);
                TextView tv = text(dlMsg, 12, DIM, false); tv.setPadding(0, dp(6), 0, 0); c.addView(tv); msgs.put(id, tv);
                TextView x = text(L.s("Отменить", "Cancel"), 14, BAD, false); x.setPadding(0, dp(8), 0, dp(2)); x.setOnClickListener(v -> cancel.set(true)); c.addView(x);
            }
        }
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.bottomMargin = dp(8); list.addView(c, lp);
        // SeedVR2 (diffusion upscaler, Apache 2.0): one download of the whole model; also offered by the Upscale screen the first time it is used
        LinearLayout sc = new LinearLayout(this); sc.setOrientation(LinearLayout.VERTICAL); sc.setBackground(bg(CARD, 14)); sc.setPadding(dp(16), dp(12), dp(16), dp(12));
        sc.addView(text("SeedVR2 ×2 / ×3 / ×4", 16, TXT, true));
        TextView sm = text(L.s("Диффузионный апскейлер: лучшее качество, медленнее. Выбирается на экране «Увеличение». Apache 2.0  ·  ByteDance Seed", "A diffusion upscaler: the best quality, slower. Pick it on the Upscale screen. Apache 2.0  ·  ByteDance Seed"), 13, DIM, false); sm.setPadding(0, dp(3), 0, dp(2)); sc.addView(sm);
        final boolean shave = SeedVr.has(getFilesDir()), sbusy = downloading && "up-seedvr".equals(dlId);
        LinearLayout sr = new LinearLayout(this); sr.setGravity(android.view.Gravity.CENTER_VERTICAL); sr.setPadding(0, dp(10), 0, 0);
        sr.addView(text(L.s("Модель", "Model") + "   " + L.f("%s ГБ", "%s GB", String.format("%.2f", SeedVr.SIZE / 1e9)), 14, TXT, false), new LinearLayout.LayoutParams(0, -2, 1));
        if (shave && !sbusy) { TextView del = text(L.s("Удалить", "Delete"), 14, BAD, false); del.setPadding(dp(12), dp(4), 0, dp(4)); del.setOnClickListener(v -> { if (downloading) return; Models.deleteTree(SeedVr.dir(getFilesDir())); render(); }); sr.addView(del); }
        else if (!sbusy) {
            Button bt = button(L.f("Скачать  %s ГБ", "Download  %s GB", String.format("%.2f", SeedVr.SIZE / 1e9)), ACC, ON_ACC); bt.setTextSize(13); bt.setPadding(dp(14), 0, dp(14), 0); bt.setEnabled(!downloading); bt.setAlpha(downloading ? .5f : 1f);
            bt.setOnClickListener(v -> startSeed()); sr.addView(bt, new LinearLayout.LayoutParams(-2, dp(36)));
        }
        sc.addView(sr);
        if (sbusy) {
            ProgressBar pb = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); pb.setMax(1000);
            pb.setProgressTintList(android.content.res.ColorStateList.valueOf(ACC)); pb.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(LINE));
            LinearLayout.LayoutParams pl = new LinearLayout.LayoutParams(-1, dp(3)); pl.topMargin = dp(10); sc.addView(pb, pl); bars.put("up-seedvr", pb);
            TextView tv = text(dlMsg, 12, DIM, false); tv.setPadding(0, dp(6), 0, 0); sc.addView(tv); msgs.put("up-seedvr", tv);
            TextView x = text(L.s("Отменить", "Cancel"), 14, BAD, false); x.setPadding(0, dp(8), 0, dp(2)); x.setOnClickListener(v -> cancel.set(true)); sc.addView(x);
        }
        LinearLayout.LayoutParams sl = new LinearLayout.LayoutParams(-1, -2); sl.bottomMargin = dp(8); list.addView(sc, sl);
    }

    void startSeed() {
        if (downloading) return;
        long need = SeedVr.SIZE; if (getFilesDir().getUsableSpace() < need + 100_000_000L) { Toast.makeText(this, L.f("Не хватает места: нужно %s ГБ", "Not enough space: need %s GB", gb(need)), Toast.LENGTH_LONG).show(); return; }
        downloading = true; dlId = "up-seedvr"; dlDone = 0; dlTotal = need; dlMsg = L.s("Подключаюсь…", "Connecting…"); cancel.set(false);
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        render();
        new Thread(() -> {
            String err = null;
            try { SeedVr.download(app, (n, tot, what) -> { dlDone = n; dlTotal = tot; dlMsg = L.f("%s / %s ГБ", "%s / %s GB", String.format("%.2f", n / 1e9), String.format("%.2f", tot / 1e9)); }, cancel); }
            catch (java.io.InterruptedIOException e) { err = L.s("Загрузка отменена: можно продолжить", "Cancelled: you can resume"); }
            catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", e.getMessage()); }
            final String msg = err; downloading = false; DownloadService.stop(app);
            ui.post(() -> { if (msg != null) Toast.makeText(app, msg, Toast.LENGTH_LONG).show(); render(); });
        }).start();
    }

    void startUp(final boolean anime) {
        if (downloading) return;
        long need = Upscale.size(anime); if (getFilesDir().getUsableSpace() < need + 100_000_000L) { Toast.makeText(this, L.f("Не хватает места: нужно %s ГБ", "Not enough space: need %s GB", gb(need)), Toast.LENGTH_LONG).show(); return; }
        downloading = true; dlId = anime ? "up-anime" : "up-photo"; dlDone = 0; dlTotal = need; dlMsg = L.s("Подключаюсь…", "Connecting…"); cancel.set(false);
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        render();
        new Thread(() -> {
            String err = null;
            try { Upscale.download(app, anime, n -> { dlDone = n; dlMsg = L.f("%s / %s ГБ", "%s / %s GB", String.format("%.2f", n / 1e9), String.format("%.2f", need / 1e9)); }, cancel); }
            catch (java.io.InterruptedIOException e) { err = L.s("Загрузка отменена: можно продолжить", "Cancelled: you can resume"); }
            catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", e.getMessage()); }
            final String msg = err; downloading = false; DownloadService.stop(app);
            ui.post(() -> { if (msg != null) Toast.makeText(app, msg, Toast.LENGTH_LONG).show(); render(); });
        }).start();
    }

    boolean hasAnima() { for (Models.Item it : items) if (it.anima()) return true; return false; }

    void start(String base, Models.Item it, boolean needCommon) {
        if (!Compat.chipAllowed(this)) {   // an unsupported chip: the models will not run, so ask before 4 GB are downloaded
            builder().setMessage(L.f("Чип %s не поддерживается: модели рассчитаны на Snapdragon 8 Gen 1 / 8+ Gen 1. Всё равно скачать?", "Chip %s is not supported: the models are built for Snapdragon 8 Gen 1 / 8+ Gen 1. Download anyway?", Compat.soc()))
                .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> { Compat.allowChip(this); begin(base, it, needCommon); }).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        if (it.h3) {   // the MiniMax licence has to be seen before the download: territories, outputs, notices
            builder().setTitle("MiniMax-H3-Image").setMessage(L.f("Модель распространяется по MiniMax-H3-Image Community License Agreement (текст лицензии скачается вместе с файлами). Нельзя использовать в США, Европейском союзе, Великобритании и Южной Корее; результаты нельзя использовать для улучшения других ИИ-моделей. Размер: %s ГБ. Скачать?", "The model is distributed under the MiniMax-H3-Image Community License Agreement (the licence text downloads with the files). It must not be used in the United States, the European Union, the United Kingdom or South Korea, and its outputs must not be used to improve other AI models. Size: %s GB. Download?", gb(it.size)))
                .setPositiveButton(L.s("Принимаю, скачать", "Accept and download"), (d, w) -> begin(base, it, needCommon)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        if (it.r2v) {   // the video model needs the text encoder of MiniMax-H3-Image; the same licence has to be seen
            if (!R2v.teInstalled(getFilesDir())) { builder().setTitle("MiniMax-H3 Ref2VA").setMessage(L.s("Сначала установите MiniMax-H3-Image: видео-модель использует её текстовый энкодер.", "Install MiniMax-H3-Image first: the video model uses its text encoder.")).setPositiveButton("OK", null).show(); return; }
            builder().setTitle("MiniMax-H3 Ref2VA").setMessage(L.f("Модель распространяется по MiniMax-H3 Community License Agreement (текст лицензии скачается вместе с файлами). Нельзя использовать в США, Европейском союзе, Великобритании и Южной Корее; результаты нельзя использовать для улучшения других ИИ-моделей. Размер: %s ГБ. Скачать?", "The model is distributed under the MiniMax-H3 Community License Agreement (the licence text downloads with the files). It must not be used in the United States, the European Union, the United Kingdom or South Korea, and its outputs must not be used to improve other AI models. Size: %s GB. Download?", gb(it.size)))
                .setPositiveButton(L.s("Принимаю, скачать", "Accept and download"), (d, w) -> begin(base, it, needCommon)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        if (it.klein) {   // the FLUX licence has to be seen before the download
            builder().setTitle("FLUX.3 Image Klein").setMessage(L.f("Модель распространяется по FLUX Kommunity License (Black Forest Labs; текст лицензии скачается вместе с файлами). Использование некоммерческое (коммерческое только на условиях лицензии), результаты нельзя использовать для обучения или улучшения других ИИ-моделей. Размер: %s ГБ. Скачать?", "The model is distributed under the FLUX Kommunity License (Black Forest Labs; the licence text downloads with the files). Use is non-commercial (commercial use only on the licence's terms) and its outputs must not be used to train or improve other AI models. Size: %s GB. Download?", gb(it.size)))
                .setPositiveButton(L.s("Принимаю, скачать", "Accept and download"), (d, w) -> begin(base, it, needCommon)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        if (it.ltx) {   // the LTX licence has to be seen before the download
            builder().setTitle("LTX-2.5 Video").setMessage(L.f("Модель распространяется по LTX-2 Community License Agreement (Lightricks; текст лицензии скачается вместе с файлами): бесплатно для организаций с годовым доходом до 10 млн долларов, действуют ограничения на использование из лицензии. Текстовый энкодер Gemma (Google) используется на условиях Gemma. Размер: %s ГБ. Скачать?", "The model is distributed under the LTX-2 Community License Agreement (Lightricks; the licence text downloads with the files): free for organisations with annual revenue under 10 million USD, and the use restrictions of the licence apply. The Gemma text encoder (Google) is used under the Gemma terms. Size: %s GB. Download?", gb(it.size)))
                .setPositiveButton(L.s("Принимаю, скачать", "Accept and download"), (d, w) -> begin(base, it, needCommon)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
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
            try { if (it.anima()) store.installCommon(base, pg, cancel); store.installModel(base, it, pg, cancel); }
            catch (java.io.InterruptedIOException e) { err = L.s("Загрузка отменена: файлы сохранены, можно продолжить", "Cancelled: the files are kept, you can resume"); }
            catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", e.getMessage()); }
            final String msg = err;
            downloading = false;
            DownloadService.stop(app);
            if (msg != null) ui.post(() -> Toast.makeText(app, msg, Toast.LENGTH_LONG).show());
        }).start();
    }
}
