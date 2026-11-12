package com.hexanima.npu;

import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.LruCache;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** History of generated pictures: tabs All / Favourites / Trash, a thumbnail grid, long-press (or "Select") for multi-select with bulk actions.
 *  Tap = viewer with download / share / favourite / "use these settings". Result: RESULT_OK with prompt, seed, w, h extras. */
public class History extends Base {
    int mode = Gallery.ALL;
    boolean selecting;
    final Set<File> sel = new LinkedHashSet<>();
    File[] cur = new File[0];
    LinearLayout grid, tabs, bar, trashNote, chips;
    HorizontalScrollView chipsScroll;
    String modelF;   // model filter (the name stored with the picture), null = all
    TextView count;
    Button selBtn;
    LinearLayout barBtns;
    final Handler ui = new Handler(Looper.getMainLooper());
    final ExecutorService io = Executors.newSingleThreadExecutor();
    final LruCache<String, Bitmap> cache = new LruCache<>(60);
    int cell;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Gallery.purgeOld(this);
        cell = (getResources().getDisplayMetrics().widthPixels - dp(40)) / 3;   // the screen padding is 20 dp on each side
        selBtn = button(L.s("Выбрать", "Select"), CARD, TXT); selBtn.setTextSize(13);
        selBtn.setOnClickListener(v -> setSelecting(!selecting));
        LinearLayout col = screen(L.s("История", "History"), selBtn);
        tabs = new LinearLayout(this); tabs.setPadding(0, 0, 0, dp(6));
        col.addView(tabs);
        chips = new LinearLayout(this); chips.setPadding(0, 0, 0, dp(8));
        chipsScroll = new HorizontalScrollView(this); chipsScroll.setHorizontalScrollBarEnabled(false); chipsScroll.addView(chips); col.addView(chipsScroll);
        trashNote = new LinearLayout(this); trashNote.setGravity(Gravity.CENTER_VERTICAL); trashNote.setPadding(dp(2), 0, 0, dp(8));
        col.addView(trashNote);
        grid = new LinearLayout(this); grid.setOrientation(LinearLayout.VERTICAL);
        col.addView(grid);

        // bulk-action bar, shown while selecting; it is added under the scroll view
        bar = new LinearLayout(this); bar.setOrientation(LinearLayout.VERTICAL); bar.setBackgroundColor(CARD); bar.setPadding(dp(16), dp(10), dp(16), dp(10)); bar.setVisibility(View.GONE);
        LinearLayout top = new LinearLayout(this); top.setGravity(Gravity.CENTER_VERTICAL);
        count = text("", 14, TXT, true);
        top.addView(count, new LinearLayout.LayoutParams(0, -2, 1));
        Button all = button(L.s("Выбрать все", "Select all"), FIELD, TXT); all.setTextSize(12);
        all.setOnClickListener(v -> { if (sel.size() == cur.length) sel.clear(); else sel.addAll(Arrays.asList(cur)); refreshSel(); });
        top.addView(all, new LinearLayout.LayoutParams(-2, dp(36)));
        bar.addView(top);
        barBtns = new LinearLayout(this); barBtns.setPadding(0, dp(8), 0, 0);
        bar.addView(barBtns);
        ((LinearLayout) ((android.view.ViewGroup) col.getParent().getParent())).addView(bar);
    }

    @Override protected void onResume() { super.onResume(); fill(); }
    @Override protected void onDestroy() { super.onDestroy(); io.shutdownNow(); }
    @Override public void onBackPressed() { if (selecting) setSelecting(false); else super.onBackPressed(); }

    void setSelecting(boolean on) { selecting = on; sel.clear(); selBtn.setText(on ? L.s("Готово", "Done") : L.s("Выбрать", "Select")); bar.setVisibility(on ? View.VISIBLE : View.GONE); fill(); }

    void refreshSel() { updateBar(); paintAll(grid); }
    /** the grid is date headers, and under each one three columns of tiles: walk down to the tiles (the ones that carry their file as a tag) */
    void paintAll(View v) {
        if (v instanceof FrameLayout && v.getTag() instanceof File) { paintCell((FrameLayout) v); return; }
        if (v instanceof android.view.ViewGroup) { android.view.ViewGroup g = (android.view.ViewGroup) v; for (int i = 0; i < g.getChildCount(); i++) paintAll(g.getChildAt(i)); }
    }

    void updateBar() {
        count.setText(L.f("Выбрано: %d", "Selected: %d", sel.size()));
        barBtns.removeAllViews();
        boolean en = !sel.isEmpty();
        if (mode == Gallery.TRASH) {
            addBar("↩ " + L.s("Восстановить", "Restore"), 2, () -> { for (File f : sel) Gallery.restore(this, f); setSelecting(false); });
            addBar(L.s("Удалить навсегда", "Delete forever"), 2, () -> confirm(L.f("Удалить навсегда (%d)?", "Delete %d forever?", sel.size()), () -> { for (File f : sel) Gallery.delete(f); setSelecting(false); }));
        } else {
            addIcon("ic_star_off", () -> { boolean allFav = true; for (File f : sel) allFav &= Gallery.fav(f); for (File f : sel) Gallery.setFav(f, !allFav); setSelecting(false); });
            addIcon("ic_download", () -> { int n = 0; for (File f : sel) try { Gallery.export(this, f); n++; } catch (Exception ignored) { } Toast.makeText(this, L.f("Сохранено в галерею: %d", "Saved to gallery: %d", n), Toast.LENGTH_SHORT).show(); });
            addIcon("ic_share", () -> Gallery.share(this, new ArrayList<>(sel)));
            addIcon("ic_delete", () -> confirm(L.f("В корзину (%d)?", "Move %d to trash?", sel.size()), () -> { for (File f : sel) Gallery.trash(this, f); setSelecting(false); }));
        }
        for (int i = 0; i < barBtns.getChildCount(); i++) { barBtns.getChildAt(i).setEnabled(en); barBtns.getChildAt(i).setAlpha(en ? 1f : .4f); }
    }

    void addIcon(String name, Runnable r) {
        ImageButton b = icon(name, FIELD); b.setOnClickListener(v -> r.run());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(46), 1); lp.rightMargin = dp(6);
        barBtns.addView(b, lp);
    }

    void addBar(String s, float w, Runnable r) {
        Button b = button(s, FIELD, TXT); b.setTextSize(13); b.setOnClickListener(v -> r.run());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(44), w); lp.rightMargin = dp(6);
        barBtns.addView(b, lp);
    }

    void confirm(String msg, Runnable ok) { builder().setMessage(msg).setPositiveButton(L.s("Да", "Yes"), (x, y) -> ok.run()).setNegativeButton(L.s("Отмена", "Cancel"), null).show(); }

    void fill() {
        // tabs with counts
        int[] n = new int[3]; for (int m = 0; m < 3; m++) n[m] = Gallery.list(this, m).length;
        String[] nm = {L.s("Все", "All"), L.s("★ Избранное", "★ Favourites"), L.s("Корзина", "Trash")};
        tabs.removeAllViews();
        for (int m = 0; m < 3; m++) {
            final int mm = m;
            TextView t = text(nm[m] + (n[m] > 0 ? "  " + n[m] : ""), 12, mm == mode ? ON_ACC : TXT, true);
            t.setPadding(dp(12), dp(9), dp(12), dp(9)); t.setBackground(bg(mm == mode ? ACC : CARD, 18));
            t.setOnClickListener(v -> { mode = mm; setSelecting(false); });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2); lp.rightMargin = dp(6);
            tabs.addView(t, lp);
        }
        trashNote.removeAllViews();
        if (mode == Gallery.TRASH && n[2] > 0) {
            trashNote.addView(text(L.s("Хранится 30 дней", "Kept for 30 days"), 12, DIM, false), new LinearLayout.LayoutParams(0, -2, 1));
            Button clr = button(L.s("Очистить", "Empty"), CARD, BAD); clr.setTextSize(12);
            clr.setOnClickListener(v -> confirm(L.s("Очистить корзину насовсем?", "Empty the trash for good?"), () -> { for (File f : Gallery.list(this, Gallery.TRASH)) Gallery.delete(f); fill(); }));
            trashNote.addView(clr, new LinearLayout.LayoutParams(-2, dp(34)));
        }
        File[] all = Gallery.list(this, mode);
        // model chips: the models that appear in this tab, with counts (shown when there is more than one)
        LinkedHashMap<String, Integer> byModel = new LinkedHashMap<>(); final String[] mn = new String[all.length];
        for (int i = 0; i < all.length; i++) { mn[i] = Gallery.meta(all[i]).optString("model", ""); if (mn[i].isEmpty()) mn[i] = "?"; byModel.merge(mn[i], 1, Integer::sum); }
        if (modelF != null && !byModel.containsKey(modelF)) modelF = null;
        chips.removeAllViews(); chipsScroll.setVisibility(byModel.size() > 1 ? View.VISIBLE : View.GONE);
        if (byModel.size() > 1) {
            addChip(L.s("Все модели", "All models") + "  " + all.length, null);
            for (Map.Entry<String, Integer> e : byModel.entrySet()) addChip(e.getKey() + "  " + e.getValue(), e.getKey());
        }
        if (modelF == null) cur = all;
        else { ArrayList<File> l = new ArrayList<>(); for (int i = 0; i < all.length; i++) if (mn[i].equals(modelF)) l.add(all[i]); cur = l.toArray(new File[0]); }
        grid.removeAllViews();
        if (cur.length == 0) {
            String[] em = {L.s("Пока пусто.\nКартинки появятся здесь, а в галерею телефона попадают только после «Скачать».", "Nothing here yet.\nPictures appear here; they reach the phone gallery only after Download."),
                L.s("Нет избранного.\nОтметьте картинку звёздочкой.", "No favourites.\nStar a picture to keep it here."), L.s("Корзина пуста.", "Trash is empty.")};
            TextView e = text(em[mode], 14, DIM, false); e.setGravity(Gravity.CENTER); e.setPadding(dp(32), dp(70), dp(32), 0);
            grid.addView(e);
        }
        LinearLayout[] colv = null; int[] colH = null; String lastDay = null;   // date sections, each a three-column masonry (tiles keep their picture's proportions)
        for (int i = 0; i < cur.length; i++) {
            String day = dayLabel(cur[i]);
            if (!day.equals(lastDay)) {
                lastDay = day;
                TextView dh = text(day.toUpperCase(), 11, DIM, true); dh.setLetterSpacing(.08f); dh.setPadding(dp(6), dp(16), 0, dp(6)); grid.addView(dh);
                LinearLayout rowc = new LinearLayout(this); grid.addView(rowc); colv = new LinearLayout[3]; colH = new int[3];
                for (int c = 0; c < 3; c++) { colv[c] = new LinearLayout(this); colv[c].setOrientation(LinearLayout.VERTICAL); rowc.addView(colv[c]); }
            }
            FrameLayout cv = makeCell(cur[i]); int best = 0; for (int c = 1; c < 3; c++) if (colH[c] < colH[best]) best = c;
            colv[best].addView(cv); colH[best] += cv.getLayoutParams().height + dp(6);
        }
        sel.retainAll(Arrays.asList(cur));
        if (selecting) updateBar();
    }

    String dayLabel(File f) {
        java.util.Calendar a = java.util.Calendar.getInstance(), b = java.util.Calendar.getInstance(); b.setTimeInMillis(f.lastModified());
        int d = (int) ((startOfDay(a) - startOfDay(b)) / 86400000L);
        if (d <= 0) return L.s("Сегодня", "Today"); if (d == 1) return L.s("Вчера", "Yesterday");
        return new java.text.SimpleDateFormat("d MMMM" + (a.get(java.util.Calendar.YEAR) != b.get(java.util.Calendar.YEAR) ? " yyyy" : ""), new java.util.Locale(L.lang)).format(new java.util.Date(f.lastModified()));
    }
    static long startOfDay(java.util.Calendar c) { java.util.Calendar x = (java.util.Calendar) c.clone(); x.set(java.util.Calendar.HOUR_OF_DAY, 0); x.set(java.util.Calendar.MINUTE, 0); x.set(java.util.Calendar.SECOND, 0); x.set(java.util.Calendar.MILLISECOND, 0); return x.getTimeInMillis(); }

    /** short model tag for the corner of a tile */
    static String shortModel(String n) {
        if (n.contains("MOSS")) return "MOSS"; if (n.contains("Qwen")) return n.contains("8 step") ? "Qwen 8" : "Qwen"; if (n.contains("Mage")) return "Mage";
        String s = n.contains("RealSkin") ? "Real" : n.contains("Harem") ? "Harem" : "";
        if (s.isEmpty()) return n.length() > 8 ? n.substring(0, 8) : n;
        return s + (n.contains("2.9B") ? " 2.9B" : n.contains("2B") || n.contains("1.3") || n.contains("LH3D") ? " 2B" : "") + (n.contains("(base)") ? " base" : "");
    }

    void addChip(String label, final String model) {
        boolean on = (model == null && modelF == null) || (model != null && model.equals(modelF));
        TextView t = text(label, 12, on ? ON_ACC : TXT, true);
        t.setPadding(dp(12), dp(8), dp(12), dp(8)); t.setBackground(bg(on ? ACC : CARD, 18));
        t.setOnClickListener(v -> { modelF = model; setSelecting(false); });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2); lp.rightMargin = dp(6);
        chips.addView(t, lp);
    }

    FrameLayout makeCell(final File f) {
        FrameLayout fl = new FrameLayout(this); fl.setTag(f);
        ImageView iv = new ImageView(this); iv.setScaleType(ImageView.ScaleType.CENTER_CROP); iv.setBackground(bg(CARD, 14)); iv.setClipToOutline(true);
        fl.addView(iv, new FrameLayout.LayoutParams(-1, -1));
        TextView star = text("★", 16, WARN, true); star.setShadowLayer(4, 0, 0, 0xFF000000);
        FrameLayout.LayoutParams sp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.END); sp.setMargins(0, dp(4), dp(8), 0);
        fl.addView(star, sp);
        TextView chk = text("✓", 16, ON_ACC, true); chk.setGravity(Gravity.CENTER); chk.setBackground(bg(ACC, 12));
        FrameLayout.LayoutParams cp = new FrameLayout.LayoutParams(dp(26), dp(26), Gravity.TOP | Gravity.START); cp.setMargins(dp(8), dp(8), 0, 0);
        fl.addView(chk, cp);
        org.json.JSONObject mo = Gallery.meta(f); String mname = mo.optString("model", "");
        if (!mname.isEmpty()) {
            TextView badge = text(shortModel(mname), 10, 0xFFFFFFFF, true); badge.setBackground(bg(0x99000000, 8)); badge.setPadding(dp(6), dp(2), dp(6), dp(2));
            FrameLayout.LayoutParams bp = new FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM | Gravity.START); bp.setMargins(dp(6), 0, 0, dp(6)); fl.addView(badge, bp);
        }
        float asp = 1.3f; int pw = mo.optInt("w"), ph2 = mo.optInt("h"); if (pw > 0 && ph2 > 0) asp = Math.max(0.7f, Math.min(1.6f, ph2 / (float) pw));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(cell - dp(6), (int) ((cell - dp(6)) * asp)); lp.setMargins(dp(3), dp(3), dp(3), dp(3));
        fl.setLayoutParams(lp);
        fl.setOnClickListener(v -> { if (selecting) { if (!sel.remove(f)) sel.add(f); refreshSel(); } else { int k = Arrays.asList(cur).indexOf(f); Viewer.open(this, cur, Math.max(0, k), mode, true); } });
        fl.setOnLongClickListener(v -> { if (!selecting) { setSelecting(true); sel.add(f); refreshSel(); } return true; });
        paintCell(fl);
        Bitmap bm = cache.get(f.getName());
        if (bm != null) iv.setImageBitmap(bm);
        else io.submit(() -> { Bitmap t = thumb(f, cell); if (t != null) { cache.put(f.getName(), t); ui.post(() -> iv.setImageBitmap(t)); } });
        return fl;
    }

    /** the viewer returns "reuse settings" (pass it on to the main screen) or just closes; its trash / favourite changes show in the grid */
    @Override protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == Viewer.REQ && res == RESULT_OK && data != null) { setResult(RESULT_OK, data); finish(); return; }
        fill();
    }

    void paintCell(FrameLayout fl) {
        File f = (File) fl.getTag();
        boolean s = sel.contains(f);
        fl.getChildAt(1).setVisibility(mode != Gallery.TRASH && Gallery.fav(f) ? View.VISIBLE : View.GONE);
        // while selecting every picture shows a ring in its corner: empty = not chosen, filled with a tick = chosen; a chosen picture also gets an accent frame, the others are dimmed
        TextView chk = (TextView) fl.getChildAt(2);
        chk.setVisibility(selecting ? View.VISIBLE : View.GONE);
        chk.setText(s ? "✓" : "");
        android.graphics.drawable.GradientDrawable d = bg(s ? ACC : 0x66000000, 14); d.setStroke(dp(2), 0xFFFFFFFF); chk.setBackground(d);
        fl.getChildAt(0).setAlpha(selecting && !s ? .45f : 1f);
        android.graphics.drawable.GradientDrawable fg = null;
        if (selecting && s) { fg = new android.graphics.drawable.GradientDrawable(); fg.setCornerRadius(dp(14)); fg.setStroke(dp(3), ACC); }
        fl.setForeground(fg);
    }

    static Bitmap thumb(File f, int target) {
        BitmapFactory.Options o = new BitmapFactory.Options(); o.inJustDecodeBounds = true; BitmapFactory.decodeFile(f.getPath(), o);
        int s = 1; while (o.outWidth / (s * 2) >= target) s *= 2;
        o = new BitmapFactory.Options(); o.inSampleSize = s; return BitmapFactory.decodeFile(f.getPath(), o);
    }

    void show(final File f) {
        final JSONObject m = Gallery.meta(f);
        LinearLayout box = new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setPadding(dp(16), dp(8), dp(16), dp(8));
        ImageView iv = new ImageView(this); iv.setAdjustViewBounds(true); iv.setImageBitmap(thumb(f, 1200));
        iv.setOnClickListener(v -> Gallery.view(this, f));
        box.addView(iv, new LinearLayout.LayoutParams(-1, -2));
        String when = new SimpleDateFormat("d MMM yyyy, HH:mm").format(new Date(Long.parseLong(f.getName().replace(".png", ""))));
        TextView info = text(m.optString("model", "") + "  ·  " + m.optInt("w") + "×" + m.optInt("h") + "  ·  seed " + m.optString("seed") + "\n" + when + (m.has("sec") ? L.f("  ·  %.1f с", "  ·  %.1f s", m.optDouble("sec")) : ""), 12, DIM, false);
        info.setPadding(0, dp(10), 0, dp(4)); box.addView(info);
        TextView pr = text(m.optString("prompt"), 14, TXT, false); pr.setTextIsSelectable(true); box.addView(pr);
        final AlertDialog[] d = new AlertDialog[1];
        if (mode == Gallery.TRASH) {
            row(box, btn(L.s("Восстановить", "Restore"), ACC, ON_ACC, () -> { Gallery.restore(this, f); d[0].dismiss(); fill(); }),
                btn(L.s("Удалить навсегда", "Delete forever"), CARD, BAD, () -> confirm(L.s("Удалить навсегда?", "Delete forever?"), () -> { Gallery.delete(f); d[0].dismiss(); fill(); })));
        } else {
            row(box, btn(L.s("Скачать", "Download"), ACC, ON_ACC, () -> { try { Gallery.export(this, f); Toast.makeText(this, L.s("Сохранено в галерею: Pictures/Anima", "Saved to gallery: Pictures/Anima"), Toast.LENGTH_SHORT).show(); } catch (Exception e) { Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show(); } }),
                btn(L.s("Поделиться", "Share"), CARD, TXT, () -> Gallery.share(this, f)));
            final Button favB = btn(Gallery.fav(f) ? "★ " + L.s("В избранном", "Favourite") : "☆ " + L.s("В избранное", "Favourite"), CARD, TXT, null);
            favB.setOnClickListener(v -> { boolean nv = !Gallery.fav(f); Gallery.setFav(f, nv); favB.setText(nv ? "★ " + L.s("В избранном", "Favourite") : "☆ " + L.s("В избранное", "Favourite")); fill(); });
            row(box, favB, btn(L.s("Взять настройки", "Reuse settings"), CARD, TXT, () -> { setResult(RESULT_OK, new Intent().putExtra("prompt", m.optString("prompt")).putExtra("seed", m.optString("seed")).putExtra("w", m.optInt("w", 512)).putExtra("h", m.optInt("h", 768))); d[0].dismiss(); finish(); }));
            row(box, btn(L.s("В корзину", "Move to trash"), CARD, BAD, () -> { Gallery.trash(this, f); d[0].dismiss(); fill(); }), null);
        }
        ScrollView sv = new ScrollView(this); sv.addView(box);
        d[0] = dialog(sv);
    }

    Button btn(String s, int color, int tc, Runnable r) { Button b = button(s, color, tc); b.setTextSize(13); if (r != null) b.setOnClickListener(v -> r.run()); return b; }

    void row(LinearLayout box, Button a, Button b) {
        LinearLayout r = new LinearLayout(this); r.setPadding(0, dp(8), 0, 0);
        LinearLayout.LayoutParams la = new LinearLayout.LayoutParams(0, dp(42), 1); la.rightMargin = b != null ? dp(8) : 0;
        r.addView(a, la);
        if (b != null) r.addView(b, new LinearLayout.LayoutParams(0, dp(42), 1));
        box.addView(r);
    }
}
