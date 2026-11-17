package com.hexanima.npu;

import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;
import java.io.FileOutputStream;
import java.io.InputStream;
import org.json.JSONObject;

/** the Upscale screen (main menu -> Upscale): a Image / Video switch, the source (own picture or one from the gallery), the engine and the scale; the work itself is UpscaleUi */
public class UpscaleActivity extends Base {
    static final int REQ = 21, REQ_V = 23;
    android.net.Uri vsrc; Rife.Info vinfo; String vname = ""; int factor = 2; TextView vhint, vsel;
    final UpscaleUi up = new UpscaleUi();
    boolean video; int engine, scale = 4;   // engine 0 Real-ESRGAN photo, 1 Real-ESRGAN anime, 2 SeedVR2
    File src; JSONObject meta = new JSONObject();
    LinearLayout body; ImageView srcView, resView; TextView srcInfo, hint; Button go;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout col = screen(L.s("Увеличение", "Upscale"), null);
        LinearLayout sw = new LinearLayout(this); sw.setPadding(0, dp(4), 0, dp(8));
        final TextView[] tabs = new TextView[2];
        for (int i = 0; i < 2; i++) {
            final int k = i; TextView t = text(i == 0 ? L.s("Картинка", "Image") : L.s("Видео", "Video"), 15, TXT, true); t.setGravity(Gravity.CENTER);
            t.setOnClickListener(v -> { video = k == 1; for (int j = 0; j < 2; j++) { tabs[j].setBackground(bg(j == k ? ACC : FIELD, 20)); tabs[j].setTextColor(j == k ? BG : TXT); } build(); });
            tabs[i] = t; LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(42), 1); lp.rightMargin = i == 0 ? dp(8) : 0; sw.addView(t, lp);
        }
        tabs[0].setBackground(bg(ACC, 20)); tabs[0].setTextColor(BG); tabs[1].setBackground(bg(FIELD, 20));
        col.addView(sw); body = new LinearLayout(this); body.setOrientation(LinearLayout.VERTICAL); col.addView(body);
        up.sh.env = new String[]{new File(getFilesDir(), "anima").getPath(), new File(getFilesDir(), "sd").getPath(), getApplicationInfo().nativeLibraryDir};
        build();
    }

    void build() {
        body.removeAllViews();
        if (video) { buildVideo(); return; }
        body.addView(label(L.s("Картинка", "Picture")));
        newSrcView(-1);
        srcInfo = text(L.s("Не выбрана", "Not chosen"), 13, DIM, false); srcInfo.setPadding(0, dp(6), 0, dp(6)); body.addView(srcInfo);
        Button pick = button(L.s("Выбрать на телефоне…", "Choose from the phone…"), FIELD, ACC);
        pick.setOnClickListener(v -> startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("image/*").addCategory(Intent.CATEGORY_OPENABLE), REQ));
        body.addView(pick, new LinearLayout.LayoutParams(-1, dp(44)));
        Button gal = button(L.s("Выбрать из галереи приложения", "Choose from the app gallery"), FIELD, ACC);
        gal.setOnClickListener(v -> pickGallery(false, f -> setSource(f, Gallery.meta(f))));
        LinearLayout.LayoutParams gl = new LinearLayout.LayoutParams(-1, dp(44)); gl.topMargin = dp(8); body.addView(gal, gl);
        body.addView(label(L.s("Метод", "Method")));
        String[] en = {"Real-ESRGAN · " + L.s("фото (быстро)", "photo (fast)"), "Real-ESRGAN · " + L.s("аниме (быстро)", "anime (fast)"), "SeedVR2 · " + L.s("диффузия, лучшее качество", "diffusion, best quality")};
        RadioGroup rg = new RadioGroup(this);
        for (int i = 0; i < en.length; i++) { RadioButton r = new RadioButton(this); r.setText(en[i]); r.setTextColor(TXT); r.setId(100 + i); r.setChecked(i == engine); rg.addView(r); }
        rg.setOnCheckedChangeListener((g, id) -> { engine = id - 100; if (engine == 2 && scale == 4 || engine < 2 && scale == 3) scale = engine == 2 ? 2 : 4; buildScale(); });
        body.addView(rg);
        body.addView(label(L.s("Во сколько раз", "Scale")));
        LinearLayout sc = new LinearLayout(this); sc.setTag("scale"); body.addView(sc);
        hint = text("", 13, DIM, false); hint.setPadding(0, dp(8), 0, dp(8)); body.addView(hint);
        go = button(L.s("Увеличить", "Upscale"), ACC, BG); go.setOnClickListener(v -> start()); body.addView(go, new LinearLayout.LayoutParams(-1, dp(48)));
        buildScale(); if (src != null) setSource(src, meta);
    }

    void buildScale() {
        LinearLayout sc = body.findViewWithTag("scale"); sc.removeAllViews();
        int[] opts = engine == 2 ? new int[]{2, 3, 4} : new int[]{2, 4};
        for (int s : opts) {
            TextView t = text("×" + s, 15, s == scale ? BG : TXT, true); t.setGravity(Gravity.CENTER); t.setBackground(bg(s == scale ? ACC : FIELD, 20));
            t.setOnClickListener(v -> { scale = s; buildScale(); });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(72), dp(40)); lp.rightMargin = dp(8); sc.addView(t, lp);
        }
        refreshHint();
    }

    void refreshHint() {
        if (hint == null) return;
        int w = meta.optInt("w", 0), h = meta.optInt("h", 0);
        if (src == null) { hint.setText(""); return; }
        String bad = engine == 2 ? SeedVr.sizeError(SeedVr.outSize(w, h, scale)[0], SeedVr.outSize(w, h, scale)[1]) : Upscale.sizeError(w, h, scale == 2);
        hint.setTextColor(bad != null ? WARN : DIM);
        int[] o = engine == 2 ? SeedVr.outSize(w, h, scale) : new int[]{w * scale, h * scale};
        hint.setText(bad != null ? bad : L.f("Результат: %d×%d", "Result: %d×%d", o[0], o[1]));
    }

    /** the picture slot: at the picture's own aspect ratio, full width (not taller than 3/4 of the screen); at is the index to put it at, -1 = the end */
    void newSrcView(int at) {
        ImageView v = new ImageView(this); v.setAdjustViewBounds(true); v.setScaleType(ImageView.ScaleType.FIT_CENTER); slot(v, at);
    }
    void slot(ImageView v, int at) {
        v.setAdjustViewBounds(true); v.setMaxHeight(getResources().getDisplayMetrics().heightPixels * 3 / 4);
        if (srcView != null && srcView.getParent() == body) { at = body.indexOfChild(srcView); body.removeView(srcView); }
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.gravity = Gravity.CENTER_HORIZONTAL; body.addView(v, at, lp); srcView = v;
    }

    void setSource(File f, JSONObject m) {
        BitmapFactory.Options o = new BitmapFactory.Options(); o.inJustDecodeBounds = true; BitmapFactory.decodeFile(f.getPath(), o);
        if (o.outWidth <= 0) { Toast.makeText(this, L.s("Не удалось открыть картинку", "Cannot open the picture"), Toast.LENGTH_SHORT).show(); return; }
        src = f; meta = new JSONObject(); try { meta.put("w", o.outWidth).put("h", o.outHeight).put("model", m.optString("model", "")); } catch (Exception ignored) { }
        if (srcView instanceof Compare) newSrcView(-1);
        srcView.setImageBitmap(History.thumb(f, 1600)); srcInfo.setText(L.f("%d×%d", "%d×%d", o.outWidth, o.outHeight)); refreshHint();
    }

    @Override protected void onActivityResult(int req, int res, Intent d) {
        if (req == REQ_V && res == RESULT_OK && d != null && d.getData() != null) { setVideo(d.getData(), d.getData().getLastPathSegment()); return; }
        if (req != REQ || res != RESULT_OK || d == null || d.getData() == null) return;
        try (InputStream in = getContentResolver().openInputStream(d.getData())) {
            Bitmap bm = BitmapFactory.decodeStream(in); if (bm == null) throw new Exception("not a picture");
            File dir = Upscale.dir(getFilesDir()); dir.mkdirs(); File f = new File(dir, "src.png");
            try (FileOutputStream o = new FileOutputStream(f)) { bm.compress(Bitmap.CompressFormat.PNG, 100, o); }
            bm.recycle(); setSource(f, new JSONObject().put("model", L.s("своя картинка", "own picture")));
        } catch (Exception e) { Toast.makeText(this, L.f("Не удалось открыть: %s", "Cannot open: %s", String.valueOf(e.getMessage())), Toast.LENGTH_LONG).show(); }
    }

    void start() {
        if (src == null) { Toast.makeText(this, L.s("Сначала выберите картинку", "Choose a picture first"), Toast.LENGTH_SHORT).show(); return; }
        final File before = src;
        java.util.function.Consumer<File> done = f -> {
            Bitmap a = History.thumb(before, 1600), b = History.thumb(f, 1600);
            if (a == null || b == null) return;
            Compare cv = new Compare(this, a, b); slot(cv, -1);
            cv.onTap = () -> { File[] all = Gallery.list(this, 0); int k = java.util.Arrays.asList(all).indexOf(f); if (k >= 0) Viewer.open(this, all, k, Gallery.ALL, false); };
            srcInfo.setText(L.f("%d×%d → %d×%d", "%d×%d → %d×%d", meta.optInt("w"), meta.optInt("h"), Gallery.meta(f).optInt("w"), Gallery.meta(f).optInt("h")));
            ((View) cv.getParent()).post(() -> cv.requestRectangleOnScreen(new android.graphics.Rect(0, 0, cv.getWidth(), cv.getHeight())));
        };
        if (engine == 2) { up.seedVr(this, src, meta, scale, done); return; }
        boolean an = engine == 1, hf = scale == 2;
        String bad = Upscale.sizeError(meta.optInt("w"), meta.optInt("h"), hf);
        if (bad != null) { Toast.makeText(this, bad, Toast.LENGTH_LONG).show(); return; }
        if (!Upscale.has(getFilesDir(), an)) up.askDownload(this, src, an, hf, done); else up.startUpscale(this, src, an, hf, done);
    }

    // ---- Video: frame interpolation (RIFE) ---------------------------------------------------------
    /** the whole app gallery as a grid of thumbnails (pictures, or only videos); a tap hands the entry to cb */
    void pickGallery(boolean videos, java.util.function.Consumer<File> cb) {
        ScrollView sv = new ScrollView(this); android.widget.GridLayout g = new android.widget.GridLayout(this); g.setColumnCount(3); sv.addView(g);
        final AlertDialog[] d = new AlertDialog[1]; int n = 0, w = (getResources().getDisplayMetrics().widthPixels - dp(80)) / 3;
        for (File f : Gallery.list(this, 0)) {
            File mp = Gallery.videoOf(f); if (videos ? (mp == null || !mp.exists()) : mp != null) continue; n++;
            ImageView iv = new ImageView(this); iv.setScaleType(ImageView.ScaleType.CENTER_CROP); iv.setBackground(bg(FIELD, 8));
            android.widget.GridLayout.LayoutParams lp = new android.widget.GridLayout.LayoutParams(); lp.width = w; lp.height = w; lp.setMargins(dp(2), dp(2), dp(2), dp(2)); g.addView(iv, lp);
            iv.setOnClickListener(v -> { d[0].dismiss(); cb.accept(f); });
            new Thread(() -> { Bitmap t = History.thumb(f, w); if (t != null) runOnUiThread(() -> iv.setImageBitmap(t)); }).start();
        }
        if (n == 0) { Toast.makeText(this, L.s("В галерее пока пусто", "The gallery is empty"), Toast.LENGTH_SHORT).show(); return; }
        d[0] = builder().setView(sv).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
    }


    /** before / after: the result with a vertical divider; the left of it shows the original. Dragging the handle (or a single finger when not zoomed) moves the divider, a pinch zooms both pictures together, one finger pans when zoomed, a plain tap opens the viewer */
    static final class Compare extends ImageView {
        final Bitmap a, b; float x = .5f, sc = 1, tx, ty, lx, ly, downX, downY; boolean moved, grab, pinch; Runnable onTap;
        final float d = getResources().getDisplayMetrics().density;
        final android.graphics.Paint p = new android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG), line = new android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG), tag = new android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG);
        final android.view.ScaleGestureDetector sg;
        Compare(Context c, Bitmap a, Bitmap b) {
            super(c); this.a = a; this.b = b; setImageBitmap(b); line.setColor(0xFFFFFFFF); tag.setTextSize(d * 12); tag.setColor(0xFFFFFFFF); tag.setShadowLayer(4, 0, 0, 0xFF000000);
            sg = new android.view.ScaleGestureDetector(c, new android.view.ScaleGestureDetector.SimpleOnScaleGestureListener() {
                @Override public boolean onScale(android.view.ScaleGestureDetector g) {
                    float n = Math.max(1, Math.min(12, sc * g.getScaleFactor())), k = n / sc; sc = n;
                    tx = g.getFocusX() - (g.getFocusX() - tx) * k; ty = g.getFocusY() - (g.getFocusY() - ty) * k; clamp(); invalidate(); return true;
                }
            });
        }
        void clamp() { float w = getWidth(), h = getHeight(); tx = Math.max(w - w * sc, Math.min(0, tx)); ty = Math.max(h - h * sc, Math.min(0, ty)); }
        @Override protected void onDraw(android.graphics.Canvas c) {
            int w = getWidth(), h = getHeight(), dx = (int) (w * x);
            android.graphics.Rect all = new android.graphics.Rect(0, 0, w, h);
            c.save(); c.translate(tx, ty); c.scale(sc, sc); c.drawBitmap(b, null, all, p); c.restore();
            c.save(); c.clipRect(0, 0, dx, h); c.translate(tx, ty); c.scale(sc, sc); c.drawBitmap(a, null, all, p); c.restore();
            line.setStrokeWidth(2 * d); c.drawLine(dx, 0, dx, h, line);
            float cy = h / 2f, r = 15 * d; line.setStyle(android.graphics.Paint.Style.FILL); c.drawCircle(dx, cy, r, line);
            line.setColor(0xFF000000); android.graphics.Path t = new android.graphics.Path(); float g = 3 * d, l = 6 * d, hh = 5 * d;
            t.moveTo(dx - g, cy - hh); t.lineTo(dx - g - l, cy); t.lineTo(dx - g, cy + hh); t.close();
            t.moveTo(dx + g, cy - hh); t.lineTo(dx + g + l, cy); t.lineTo(dx + g, cy + hh); t.close(); c.drawPath(t, line); line.setColor(0xFFFFFFFF);
            c.drawText(L.s("до", "before"), 8 * d, 18 * d, tag); String s = L.s("после", "after"); c.drawText(s, w - 8 * d - tag.measureText(s), 18 * d, tag);
            if (sc > 1.01f) c.drawText(String.format(java.util.Locale.US, "×%.1f", sc), 8 * d, h - 8 * d, tag);
        }
        @Override public boolean onTouchEvent(android.view.MotionEvent e) {
            sg.onTouchEvent(e); getParent().requestDisallowInterceptTouchEvent(true);
            switch (e.getActionMasked()) {
                case android.view.MotionEvent.ACTION_DOWN: downX = lx = e.getX(); downY = ly = e.getY(); moved = pinch = false; grab = sc <= 1.01f || Math.abs(downX - x * getWidth()) < 36 * d; return true;
                case android.view.MotionEvent.ACTION_POINTER_DOWN: pinch = true; moved = true; return true;
                case android.view.MotionEvent.ACTION_POINTER_UP: lx = e.getX(1 - e.getActionIndex()); ly = e.getY(1 - e.getActionIndex()); return true;
                case android.view.MotionEvent.ACTION_MOVE:
                    if (e.getPointerCount() > 1 || sg.isInProgress()) { moved = true; lx = e.getX(); ly = e.getY(); return true; }
                    if (pinch) { /* the finger left after a pinch: pan from here */ pinch = false; lx = e.getX(); ly = e.getY(); grab = false; }
                    if (Math.abs(e.getX() - downX) > 12 * d / 2 || Math.abs(e.getY() - downY) > 12 * d / 2) moved = true;
                    if (grab) x = Math.max(0, Math.min(1, e.getX() / getWidth()));
                    else { tx += e.getX() - lx; ty += e.getY() - ly; clamp(); }
                    lx = e.getX(); ly = e.getY(); invalidate(); return true;
                case android.view.MotionEvent.ACTION_UP: if (!moved && onTap != null) onTap.run(); return true;
            }
            return true;
        }
    }

    void buildVideo() {
        body.addView(label(L.s("Видео · больше кадров в секунду (RIFE)", "Video · more frames per second (RIFE)")));
        body.addView(text(L.s("Между каждой парой кадров NPU дорисовывает новые: плавнее движение. Звук сохраняется.", "The NPU draws new frames between every pair: smoother motion. The sound is kept."), 13, DIM, false));
        vsel = text(vsrc == null ? L.s("Видео не выбрано", "No video chosen") : vname, 13, DIM, false); vsel.setPadding(0, dp(8), 0, dp(6)); body.addView(vsel);
        Button pick = button(L.s("Выбрать видео…", "Choose a video…"), FIELD, ACC);
        pick.setOnClickListener(v -> startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("video/*").addCategory(Intent.CATEGORY_OPENABLE), REQ_V));
        body.addView(pick, new LinearLayout.LayoutParams(-1, dp(44)));
        Button gal = button(L.s("Выбрать видео из галереи приложения", "Choose a video from the app gallery"), FIELD, ACC);
        gal.setOnClickListener(v -> pickGallery(true, f -> { File mp = Gallery.videoOf(f); setVideo(android.net.Uri.fromFile(mp), mp.getName()); }));
        LinearLayout.LayoutParams gl = new LinearLayout.LayoutParams(-1, dp(44)); gl.topMargin = dp(8); body.addView(gal, gl);
        body.addView(label(L.s("Во сколько раз больше кадров", "Frame rate multiplier")));
        LinearLayout sc = new LinearLayout(this);
        for (final int s : new int[]{2, 4}) {
            TextView t = text("×" + s, 15, s == factor ? BG : TXT, true); t.setGravity(Gravity.CENTER); t.setBackground(bg(s == factor ? ACC : FIELD, 20));
            t.setOnClickListener(v -> { factor = s; build(); });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(72), dp(40)); lp.rightMargin = dp(8); sc.addView(t, lp);
        }
        body.addView(sc);
        vhint = text("", 13, DIM, false); vhint.setPadding(0, dp(8), 0, dp(8)); body.addView(vhint); refreshVideoHint();
        Button go = button(L.s("Добавить кадры", "Add frames"), ACC, BG); go.setOnClickListener(v -> startVideo()); body.addView(go, new LinearLayout.LayoutParams(-1, dp(48)));
        resView = new ImageView(this); resView.setAdjustViewBounds(true); LinearLayout.LayoutParams rl = new LinearLayout.LayoutParams(-1, -2); rl.topMargin = dp(14); body.addView(resView, rl);
    }

    void setVideo(android.net.Uri u, String name) {
        try { vinfo = Rife.probe(this, u); vsrc = u; vname = name == null ? "" : name; } catch (Exception e) { Toast.makeText(this, L.f("Не удалось открыть видео: %s", "Cannot open the video: %s", String.valueOf(e.getMessage())), Toast.LENGTH_LONG).show(); return; }
        if (vsel != null) vsel.setText(vname); refreshVideoHint();
    }

    void refreshVideoHint() {
        if (vhint == null) return;
        if (vinfo == null) { vhint.setText(""); return; }
        if (vinfo.ctx == null) { vhint.setTextColor(WARN); vhint.setText(L.f("%d×%d: больше 1280×768 (720p), не поддерживается", "%d×%d: larger than 1280x768 (720p), not supported", vinfo.w, vinfo.h)); return; }
        long frames = vinfo.durUs > 0 ? vinfo.durUs * vinfo.fps / 1000000L : 0; double sec = frames * (Rife.pairSec(vinfo.ctx) * (factor == 2 ? 1 : 3) + 0.03);
        vhint.setTextColor(DIM); vhint.setText(L.f("%d×%d, %d → %d кадров/с, %d с видео · примерно %d мин", "%d×%d, %d → %d fps, %d s of video · about %d min", vinfo.w, vinfo.h, vinfo.fps, vinfo.fps * factor, vinfo.durUs / 1000000, Math.max(1, Math.round(sec / 60))));
    }

    void startVideo() {
        if (vsrc == null || vinfo == null) { Toast.makeText(this, L.s("Сначала выберите видео", "Choose a video first"), Toast.LENGTH_SHORT).show(); return; }
        if (vinfo.ctx == null) { Toast.makeText(this, L.s("Видео слишком большое", "The video is too large"), Toast.LENGTH_LONG).show(); return; }
        if (up.busy) { Toast.makeText(this, L.s("Уже идёт обработка", "Something is already running"), Toast.LENGTH_SHORT).show(); return; }
        if (Rife.has(getFilesDir(), vinfo.ctx)) { runVideo(); return; }
        final int[] c = vinfo.ctx; up.busy = true;
        new Thread(() -> {
            long size = -1; try { size = Rife.entry(c).getLong("size"); } catch (Exception ignored) { }
            final long sz = size; runOnUiThread(() -> { up.busy = false;
                if (sz < 0) { Toast.makeText(this, L.s("Не удалось получить список файлов (нет сети?)", "Cannot reach the file list (no network?)"), Toast.LENGTH_LONG).show(); return; }
                builder().setTitle(L.s("Нужна загрузка", "Download needed")).setMessage(L.f("Сеть RIFE v4.6 для размера %d×%d (%d МБ, MIT) скачивается один раз с Hugging Face.", "The RIFE v4.6 network for %d×%d (%d MB, MIT) is downloaded once from Hugging Face.", c[1], c[0], sz / 1000000))
                    .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> downloadRife(c, sz)).setNegativeButton(L.s("Отмена", "Cancel"), null).show(); });
        }).start();
    }

    void downloadRife(final int[] c, final long sz) {
        final AtomicBoolean cancel = new AtomicBoolean(); up.busy = true;
        final android.app.AlertDialog dlg = builder().setTitle(L.s("Загрузка…", "Downloading…")).setMessage("0 %").setCancelable(false).setNegativeButton(L.s("Отмена", "Cancel"), (d, w) -> cancel.set(true)).create(); dlg.show();
        new Thread(() -> {
            String err = null; try { Rife.download(this, c, n -> runOnUiThread(() -> dlg.setMessage((100 * n / sz) + " %")), cancel); } catch (Throwable e) { err = cancel.get() ? "" : String.valueOf(e.getMessage() != null ? e.getMessage() : e); }
            final String er = err; runOnUiThread(() -> { dlg.dismiss(); up.busy = false; if (er != null) { if (!er.isEmpty()) Toast.makeText(this, L.f("Не удалось скачать: %s", "Download failed: %s", er), Toast.LENGTH_LONG).show(); return; } runVideo(); });
        }).start();
    }

    void runVideo() {
        final AtomicBoolean cancel = new AtomicBoolean(); up.busy = true; final long t0 = System.currentTimeMillis();
        final android.app.AlertDialog dlg = builder().setTitle("RIFE ×" + factor).setMessage(L.s("Подготовка…", "Preparing…")).setCancelable(false).setNegativeButton(L.s("Остановить", "Stop"), (d, w) -> cancel.set(true)).create(); dlg.show();
        new Thread(() -> {
            File res = null; String err = null;
            try { res = Rife.run(this, vsrc, vname, factor, (n, tot, m) -> { long s = (System.currentTimeMillis() - t0) / 1000; runOnUiThread(() -> dlg.setMessage(tot > 0 ? L.f("Кадр %d из ~%d · %d с", "Frame %d of ~%d · %d s", n, tot, s) : L.f("Кадр %d · %d с", "Frame %d · %d s", n, s))); }, cancel); }
            catch (Throwable e) { err = String.valueOf(e.getMessage() != null ? e.getMessage() : e); }
            final File f = res; final String er = err;
            runOnUiThread(() -> {
                dlg.dismiss(); up.busy = false;
                if (f == null) { if (er != null) Toast.makeText(this, L.f("Не удалось: %s", "Failed: %s", er), Toast.LENGTH_LONG).show(); return; }
                resView.setImageBitmap(History.thumb(f, 1600)); resView.setOnClickListener(v -> { File[] all = Gallery.list(this, 0); int i = java.util.Arrays.asList(all).indexOf(f); if (i >= 0) Viewer.open(this, all, i, Gallery.ALL, false); });
                Toast.makeText(this, L.s("Готово: новое видео в галерее", "Done: a new video in the gallery"), Toast.LENGTH_SHORT).show();
            });
        }).start();
    }
}
