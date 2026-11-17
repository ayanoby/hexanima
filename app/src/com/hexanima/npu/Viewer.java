package com.hexanima.npu;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.LruCache;
import android.view.GestureDetector;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.*;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** Full-screen picture viewer: swipe left/right between the pictures of the list it was opened with, pinch / double-tap to zoom, tap to hide the bars.
 *  Actions: download, share, favourite, reuse settings (when opened from History), move to trash / restore / delete forever. */
public class Viewer extends Base {
    static final int REQ = 11;
    final List<File> files = new ArrayList<>();
    int mode, idx;
    boolean reuse, changed;
    Pager pager;
    LinearLayout top, bottom, actions; ImageButton moreB;
    TextView counter, info, prompt;
    ImageButton favB;
    FrameLayout rootFl, videoBox; VideoView vv; SeekBar vSeek; TextView vCur, vTot; ImageView vPlay, vBig; View vBar, vClose; boolean vDrag, vCtl = true; final Handler vH = new Handler(Looper.getMainLooper());
    final Runnable vTick = new Runnable() { @Override public void run() {
        if (vv == null) return;
        try { if (!vDrag) { int p = vv.getCurrentPosition(); vSeek.setProgress(p); vCur.setText(clock(p)); } } catch (Exception ignored) { }
        vH.postDelayed(this, 200); } };
    final Runnable vHide = () -> showControls(false);
    static String clock(int ms) { int s = Math.max(0, ms) / 1000; return String.format("%d:%02d", s / 60, s % 60); }
    void showControls(boolean on) {
        vCtl = on; vH.removeCallbacks(vHide); if (videoBox == null) return;
        for (View v : new View[]{vPlay, vBar, vClose}) v.animate().alpha(on ? 1f : 0f).setDuration(180).withEndAction(() -> { }).start();
        vPlay.setClickable(on); vSeek.setEnabled(on); vClose.setClickable(on);
        if (on && vv != null && vv.isPlaying()) vH.postDelayed(vHide, 1000);
    }
    void vToggle() { if (vv == null) return; if (vv.isPlaying()) vv.pause(); else vv.start(); setBtnIcon(vPlay, vv.isPlaying() ? "ic_pause" : "ic_play"); showControls(true); }
    /** plays the clip of a video entry in this screen: big play / pause button in the middle, time and a seek bar at the bottom, tap shows / hides them (they hide by themselves while playing); loops; the cross or Back closes it */
    void playVideo(File png) {
        final File vid = Gallery.videoOf(png); if (vid == null || videoBox != null) return; if (vBig != null) vBig.setVisibility(View.GONE);
        videoBox = new FrameLayout(this); videoBox.setBackgroundColor(Color.BLACK); videoBox.setClickable(true);
        vv = new VideoView(this); vv.setVideoPath(vid.getPath());
        videoBox.addView(vv, new FrameLayout.LayoutParams(-1, -1, Gravity.CENTER));
        vPlay = circleBtn("ic_pause", 76); vPlay.setOnClickListener(v -> vToggle());
        videoBox.addView(vPlay, new FrameLayout.LayoutParams(dp(80), dp(80), Gravity.CENTER));
        LinearLayout bar = new LinearLayout(this); bar.setGravity(Gravity.CENTER_VERTICAL); bar.setPadding(dp(16), dp(10), dp(16), dp(10)); bar.setBackground(bg(0x66000000, 14)); vBar = bar;
        vCur = text("0:00", 13, Color.WHITE, false); vTot = text("0:00", 13, Color.WHITE, false);
        vSeek = new SeekBar(this); vSeek.setProgressTintList(android.content.res.ColorStateList.valueOf(Color.WHITE)); vSeek.setThumbTintList(android.content.res.ColorStateList.valueOf(Color.WHITE)); vSeek.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(0x66FFFFFF));
        vSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar sb, int p, boolean user) { if (user && vv != null) { vv.seekTo(p); vCur.setText(clock(p)); } }
            public void onStartTrackingTouch(SeekBar sb) { vDrag = true; vH.removeCallbacks(vHide); }
            public void onStopTrackingTouch(SeekBar sb) { vDrag = false; if (vv != null) { vv.seekTo(sb.getProgress()); } showControls(true); } });
        bar.addView(vCur); bar.addView(vSeek, new LinearLayout.LayoutParams(0, dp(36), 1)); bar.addView(vTot);
        FrameLayout.LayoutParams bp = new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM); bp.setMargins(dp(12), 0, dp(12), dp(36)); videoBox.addView(bar, bp);
        TextView x = text("✕", 22, Color.WHITE, false); x.setGravity(Gravity.CENTER); x.setBackground(bg(0x66000000, 24)); x.setOnClickListener(v -> closeVideo()); vClose = x;
        FrameLayout.LayoutParams xp = new FrameLayout.LayoutParams(dp(48), dp(48), Gravity.TOP | Gravity.END); xp.setMargins(0, dp(40), dp(12), 0); videoBox.addView(x, xp);
        videoBox.setOnClickListener(v -> showControls(!vCtl));
        vv.setOnPreparedListener(mp -> { mp.setLooping(true); vSeek.setMax(vv.getDuration()); vTot.setText(clock(vv.getDuration())); vv.start(); setBtnIcon(vPlay, "ic_pause"); showControls(true); vH.post(vTick); });
        vv.setOnErrorListener((mp, w, e) -> { Toast.makeText(this, L.s("Не удалось воспроизвести видео", "Cannot play the video"), Toast.LENGTH_SHORT).show(); closeVideo(); return true; });
        rootFl.addView(videoBox, new FrameLayout.LayoutParams(-1, -1)); vCtl = true;
    }
    void closeVideo() { if (videoBox == null) return; vH.removeCallbacksAndMessages(null); try { vv.stopPlayback(); } catch (Exception ignored) { } rootFl.removeView(videoBox); videoBox = null; vv = null; if (getIntent().getBooleanExtra("play", false)) finish(); else refresh(); }
    @Override public void onBackPressed() { if (videoBox != null) closeVideo(); else super.onBackPressed(); }
    @Override protected void onPause() { if (vv != null && vv.isPlaying()) { vv.pause(); if (vPlay != null) setBtnIcon(vPlay, "ic_play"); } super.onPause(); }

    /** mode = Gallery.ALL / FAV / TRASH of the list; reuse = show "Reuse settings" (the caller must handle RESULT_OK with prompt/seed/w/h) */
    static void open(Activity a, File[] list, int index, int mode, boolean reuse) {
        String[] p = new String[list.length]; for (int i = 0; i < p.length; i++) p[i] = list[i].getPath();
        a.startActivityForResult(new Intent(a, Viewer.class).putExtra("p", p).putExtra("i", index).putExtra("mode", mode).putExtra("reuse", reuse), REQ);
    }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().setStatusBarColor(Color.BLACK); getWindow().setNavigationBarColor(Color.BLACK);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        Intent in = getIntent(); String[] p = in.getStringArrayExtra("p");
        if (p != null) for (String s : p) { File f = new File(s); if (f.exists()) files.add(f); }
        mode = in.getIntExtra("mode", Gallery.ALL); idx = Math.max(0, Math.min(in.getIntExtra("i", 0), files.size() - 1)); reuse = in.getBooleanExtra("reuse", false);
        if (files.isEmpty()) { finish(); return; }

        FrameLayout root = rootFl = new FrameLayout(this); root.setBackgroundColor(Color.BLACK);
        pager = new Pager(this); root.addView(pager, new FrameLayout.LayoutParams(-1, -1));
        pager.onTap = this::toggleBars; pager.onPage = i -> { idx = i; refresh(); };

        top = new LinearLayout(this); top.setGravity(Gravity.CENTER_VERTICAL); top.setBackgroundColor(0x99000000); top.setPadding(dp(8), dp(6), dp(16), dp(6));
        Button close = button("✕", 0x00000000, Color.WHITE); close.setTextSize(20); close.setOnClickListener(v -> finish());
        top.addView(close, new LinearLayout.LayoutParams(dp(52), dp(44)));
        counter = text("", 15, Color.WHITE, true); top.addView(counter, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(top, new FrameLayout.LayoutParams(-1, -2, Gravity.TOP));

        bottom = new LinearLayout(this); bottom.setOrientation(LinearLayout.VERTICAL); bottom.setBackgroundColor(0xB3000000); bottom.setPadding(dp(14), dp(10), dp(14), dp(10));
        info = text("", 12, 0xFFB8C0D0, false); bottom.addView(info);
        prompt = text("", 13, Color.WHITE, false); prompt.setMaxLines(3); prompt.setPadding(0, dp(4), 0, dp(8)); bottom.addView(prompt);
        actions = new LinearLayout(this); bottom.addView(actions);
        root.addView(bottom, new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM));
        moreB = icon("ic_more", 0x33FFFFFF); moreB.setImageTintList(android.content.res.ColorStateList.valueOf(Color.WHITE)); moreB.setVisibility(View.GONE);
        FrameLayout.LayoutParams mlp = new FrameLayout.LayoutParams(dp(44), dp(44), Gravity.BOTTOM | Gravity.END); mlp.setMargins(0, 0, dp(10), dp(10)); root.addView(moreB, mlp);
        vBig = circleBtn("ic_play", 76); vBig.setVisibility(View.GONE); vBig.setOnClickListener(x -> playVideo(files.get(idx)));
        root.addView(vBig, new FrameLayout.LayoutParams(dp(76), dp(76), Gravity.CENTER));
        setContentView(root);
        pager.set(files, idx);
        refresh();
        if (in.getBooleanExtra("play", false) && Gallery.videoOf(files.get(idx)) != null) playVideo(files.get(idx));
    }

    void toggleBars() {
        boolean show = top.getVisibility() != View.VISIBLE;
        top.setVisibility(show ? View.VISIBLE : View.GONE); bottom.setVisibility(show ? View.VISIBLE : View.GONE); if (moreB != null && mode != Gallery.TRASH) moreB.setVisibility(show ? View.VISIBLE : View.GONE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | (show ? 0 : View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY));
    }

    /** the same icons as under a fresh picture on the main screen */
    ImageButton ico(String name, Runnable r) {
        ImageButton b = icon(name, 0x33FFFFFF); b.setOnClickListener(v -> r.run());
        if (!name.equals("ic_star") && !name.equals("ic_delete")) b.setImageTintList(android.content.res.ColorStateList.valueOf(Color.WHITE));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(48), 1); lp.setMargins(dp(3), 0, dp(3), 0); actions.addView(b, lp); return b;
    }
    void setFav(ImageButton b, boolean on) {
        b.setImageResource(getResources().getIdentifier(on ? "ic_star" : "ic_star_off", "drawable", getPackageName()));
        b.setImageTintList(on ? null : android.content.res.ColorStateList.valueOf(Color.WHITE));
    }

    Button act(String s, Runnable r) { final boolean trash = mode == Gallery.TRASH;
        Button b = button(s, 0x33FFFFFF, Color.WHITE); b.setTextSize(trash ? 13 : 22); b.setAllCaps(false); b.setPadding(0, 0, 0, 0); b.setOnClickListener(v -> r.run());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(48), 1); lp.setMargins(dp(3), 0, dp(3), 0); actions.addView(b, lp); return b;
    }

    /** texts and buttons for the current picture */
    void refresh() {
        final File f = files.get(idx); final JSONObject m = Gallery.meta(f);
        counter.setText((idx + 1) + " / " + files.size());
        String when = ""; try { when = new SimpleDateFormat("d MMM yyyy, HH:mm").format(new Date(Long.parseLong(f.getName().replace(".png", "")))); } catch (Exception ignored) { }
        info.setText(m.optString("model", "") + "  ·  " + m.optInt("w") + "×" + m.optInt("h") + "  ·  seed " + m.optString("seed") + "\n" + when + (m.has("sec") ? L.f("  ·  %.1f с", "  ·  %.1f s", m.optDouble("sec")) : ""));
        prompt.setText(m.optString("prompt"));
        actions.removeAllViews(); favB = null; moreB.setVisibility(View.GONE); bottom.setPadding(dp(14), dp(10), dp(14), dp(10));
        if (vBig != null) vBig.setVisibility(Gallery.videoOf(f) != null && videoBox == null ? View.VISIBLE : View.GONE);
        if (mode == Gallery.TRASH) {
            act(L.s("Восстановить", "Restore"), () -> { Gallery.restore(this, f); removeCurrent(); });
            act(L.s("Удалить навсегда", "Delete forever"), () -> builder().setMessage(L.s("Удалить навсегда?", "Delete forever?")).setPositiveButton(L.s("Да", "Yes"), (x, y) -> { Gallery.delete(f); removeCurrent(); }).setNegativeButton(L.s("Отмена", "Cancel"), null).show());
            return;
        }
        // one small "more" button in the bottom right corner: every action of the picture in a popup list next to it
        bottom.setPadding(dp(14), dp(10), dp(64), dp(10)); moreB.setVisibility(bottom.getVisibility());
        moreB.setOnClickListener(v -> {
            android.widget.PopupMenu pm = new android.widget.PopupMenu(this, moreB, Gravity.END);
            final List<Runnable> run = new ArrayList<>(); android.view.Menu mn = pm.getMenu();
            mn.add(0, run.size(), run.size(), L.s("Сохранить в галерею", "Save to gallery")); run.add(() -> { try { Gallery.export(this, f); Toast.makeText(this, L.s("Сохранено в галерею: Pictures/Anima", "Saved to gallery: Pictures/Anima"), Toast.LENGTH_SHORT).show(); } catch (Exception e) { Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show(); } });
            mn.add(0, run.size(), run.size(), L.s("Поделиться", "Share")); run.add(() -> Gallery.share(this, f));
            mn.add(0, run.size(), run.size(), Gallery.fav(f) ? L.s("Убрать из избранного", "Remove from favourites") : L.s("В избранное", "Add to favourites")); run.add(() -> { Gallery.setFav(f, !Gallery.fav(f)); changed = true; });
            if (reuse) { mn.add(0, run.size(), run.size(), L.s("Повторить (промпт и seed)", "Reuse (prompt and seed)")); run.add(() -> { setResult(RESULT_OK, new Intent().putExtra("prompt", m.optString("prompt")).putExtra("seed", m.optString("seed")).putExtra("w", m.optInt("w", 512)).putExtra("h", m.optInt("h", 768))); finish(); }); }
            if (Gallery.videoOf(f) == null) { mn.add(0, run.size(), run.size(), L.s("Увеличить…", "Upscale…")); run.add(() -> upscaleDialog(f, m)); }
            mn.add(0, run.size(), run.size(), L.s("Удалить", "Delete")); run.add(() -> { Gallery.trash(this, f); removeCurrent(); });
            pm.setOnMenuItemClickListener(it -> { run.get(it.getItemId()).run(); return true; });
            pm.show();
        });
    }

    final UpscaleUi up = new UpscaleUi();
    void upscaleDialog(final File f, final JSONObject m) { up.dialog(this, f, m, res -> { changed = true; files.add(idx + 1, res); idx++; pager.set(files, idx); refresh(); }); }

    /** the picture left this list (trash / restore / delete): show the next one, or close when it was the last */
    void removeCurrent() {
        changed = true; files.remove(idx);
        if (files.isEmpty()) { finish(); return; }
        idx = Math.min(idx, files.size() - 1); pager.set(files, idx); refresh();
    }

    /** three pages (previous, current, next) drawn by hand; pinch / double-tap zoom, pan, horizontal drag with a fling between pages */
    static final class Pager extends View {
        interface Page { void at(int i); }
        List<File> files = new ArrayList<>();
        int cur;
        float dragX, scale = 1, tx, ty;
        boolean flung, anim;
        Runnable onTap; Page onPage;
        final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        final LruCache<String, Bitmap> cache = new LruCache<>(5);
        final java.util.HashSet<String> loading = new java.util.HashSet<>();
        final ExecutorService io = Executors.newFixedThreadPool(2);
        final Handler ui = new Handler(Looper.getMainLooper());
        final ScaleGestureDetector sg; final GestureDetector gd;
        final float gap = 24 * getResources().getDisplayMetrics().density;

        Pager(Context c) {
            super(c);
            sg = new ScaleGestureDetector(c, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                @Override public boolean onScale(ScaleGestureDetector d) {
                    float ns = Math.max(1f, Math.min(6f, scale * d.getScaleFactor()));
                    float fx = d.getFocusX() - getWidth() / 2f, fy = d.getFocusY() - getHeight() / 2f;
                    tx = fx - (fx - tx) * ns / scale; ty = fy - (fy - ty) * ns / scale; scale = ns; clamp(); invalidate(); return true;
                }
            });
            gd = new GestureDetector(c, new GestureDetector.SimpleOnGestureListener() {
                @Override public boolean onDown(MotionEvent e) { return true; }
                @Override public boolean onSingleTapConfirmed(MotionEvent e) { if (onTap != null) onTap.run(); return true; }
                @Override public boolean onDoubleTap(MotionEvent e) { zoomTo(scale > 1.05f ? 1f : 2.5f, e.getX(), e.getY()); return true; }
                @Override public boolean onScroll(MotionEvent a, MotionEvent b, float dx, float dy) { if (!sg.isInProgress()) drag(dx, dy); return true; }
                @Override public boolean onFling(MotionEvent a, MotionEvent b, float vx, float vy) { flung = true; settle(vx); return true; }
            });
        }

        void set(List<File> l, int i) { files = l; cur = i; dragX = 0; scale = 1; tx = ty = 0; cache.evictAll(); invalidate(); }

        Bitmap get(int i) {
            if (i < 0 || i >= files.size()) return null;
            final File f = files.get(i); Bitmap b = cache.get(f.getPath());
            if (b != null || !loading.add(f.getPath())) return b;
            io.submit(() -> {
                BitmapFactory.Options o = new BitmapFactory.Options(); o.inJustDecodeBounds = true; BitmapFactory.decodeFile(f.getPath(), o);
                int s = 1; while (Math.max(o.outWidth, o.outHeight) / s > 3072) s *= 2;
                o = new BitmapFactory.Options(); o.inSampleSize = s; Bitmap d = BitmapFactory.decodeFile(f.getPath(), o);
                ui.post(() -> { loading.remove(f.getPath()); if (d != null) cache.put(f.getPath(), d); invalidate(); });
            });
            return null;
        }

        float fit(Bitmap b) { return Math.min((float) getWidth() / b.getWidth(), (float) getHeight() / b.getHeight()); }

        /** keep the zoomed picture covering the view (or centred when it is smaller) */
        void clamp() {
            Bitmap b = cache.get(files.get(cur).getPath()); if (b == null) return;
            float s = fit(b) * scale, mx = Math.max(0, (b.getWidth() * s - getWidth()) / 2), my = Math.max(0, (b.getHeight() * s - getHeight()) / 2);
            tx = Math.max(-mx, Math.min(mx, tx)); ty = Math.max(-my, Math.min(my, ty));
        }

        void drag(float dx, float dy) {
            if (scale > 1.01f && dragX == 0) {
                Bitmap b = cache.get(files.get(cur).getPath()); float mx = 0;
                if (b != null) mx = Math.max(0, (b.getWidth() * fit(b) * scale - getWidth()) / 2);
                tx -= dx; ty -= dy;
                if (tx > mx) { dragX += tx - mx; tx = mx; } else if (tx < -mx) { dragX += tx + mx; tx = -mx; }
                clamp();
            } else {
                float old = dragX, d = -dx;
                if ((cur == 0 && d > 0 && old >= 0) || (cur == files.size() - 1 && d < 0 && old <= 0)) d *= 0.35f;   // rubber band at both ends
                dragX += d;
                if (scale > 1.01f && old != 0 && old * dragX < 0) dragX = 0;   // dragged back past the picture's edge: pan again
            }
            invalidate();
        }

        void settle(float vx) {
            int W = getWidth(); float to = 0; int dir = 0;
            if ((dragX < -W / 4f || (vx < -1200 && (scale <= 1.01f || dragX != 0))) && cur < files.size() - 1) dir = 1;
            else if ((dragX > W / 4f || (vx > 1200 && (scale <= 1.01f || dragX != 0))) && cur > 0) dir = -1;
            if (dir != 0) to = -dir * (W + gap);
            final int d = dir; anim = true;
            ValueAnimator a = ValueAnimator.ofFloat(dragX, to); a.setDuration(d != 0 ? 180 : 140);
            a.addUpdateListener(v -> { dragX = (float) v.getAnimatedValue(); invalidate(); });
            a.addListener(new android.animation.AnimatorListenerAdapter() { @Override public void onAnimationEnd(android.animation.Animator x) {
                anim = false;
                if (d != 0) { cur += d; dragX = 0; scale = 1; tx = ty = 0; if (onPage != null) onPage.at(cur); } else dragX = 0;
                invalidate(); } });
            a.start();
        }

        void zoomTo(float ns, float px, float py) {
            final float s0 = scale, x0 = tx, y0 = ty; float fx = px - getWidth() / 2f, fy = py - getHeight() / 2f;
            final float tx1 = ns == 1f ? 0 : fx - (fx - tx) * ns / scale, ty1 = ns == 1f ? 0 : fy - (fy - ty) * ns / scale;
            ValueAnimator a = ValueAnimator.ofFloat(0, 1); a.setDuration(180);
            a.addUpdateListener(v -> { float t = (float) v.getAnimatedValue(); scale = s0 + (ns - s0) * t; tx = x0 + (tx1 - x0) * t; ty = y0 + (ty1 - y0) * t; clamp(); invalidate(); });
            a.start();
        }

        @Override public boolean onTouchEvent(MotionEvent e) {
            if (anim) return true;
            int act = e.getActionMasked();
            if (act == MotionEvent.ACTION_DOWN) flung = false;
            sg.onTouchEvent(e);
            if (!sg.isInProgress()) gd.onTouchEvent(e);
            if ((act == MotionEvent.ACTION_UP || act == MotionEvent.ACTION_CANCEL) && !flung) {
                if (scale < 1f) { scale = 1; }
                if (dragX != 0) settle(0);
            }
            return true;
        }

        @Override protected void onDraw(Canvas c) {
            int W = getWidth(), H = getHeight(); if (files.isEmpty()) return;
            for (int i = Math.max(0, cur - 1); i <= Math.min(files.size() - 1, cur + 1); i++) {
                Bitmap b = get(i); if (b == null) continue;
                float s = fit(b) * (i == cur ? scale : 1f);
                float x = (W - b.getWidth() * s) / 2 + (i == cur ? tx : 0) + (i - cur) * (W + gap) + dragX, y = (H - b.getHeight() * s) / 2 + (i == cur ? ty : 0);
                c.save(); c.translate(x, y); c.scale(s, s); c.drawBitmap(b, 0, 0, paint); c.restore();
            }
        }
    }
}
