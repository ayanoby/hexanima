package com.hexanima.npu;

import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.*;

/** Place the wallpaper: pinch to zoom, drag to move; what is shown is what every screen gets. Done saves, Reset goes back to "fill the screen". */
public class WallEdit extends Base {
    ScaleGestureDetector sg; float lx, ly; int lastN;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        final FrameLayout f = new FrameLayout(this); f.setFitsSystemWindows(true);
        final View v = new View(this);
        v.setBackground(new WallDrawable(this));
        sg = new ScaleGestureDetector(this, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector d) { wZoom = Math.max(1f, Math.min(4f, wZoom * d.getScaleFactor())); v.invalidate(); return true; }
        });
        v.setOnTouchListener((w, e) -> {
            sg.onTouchEvent(e);
            int n = e.getPointerCount();
            if (n == 1 && !sg.isInProgress() && wallBm != null) {
                if (e.getActionMasked() == MotionEvent.ACTION_MOVE && lastN == 1) {
                    float sc = wallScale(w.getWidth(), w.getHeight());
                    wX -= (e.getX() - lx) / sc / wallBm.getWidth(); wY -= (e.getY() - ly) / sc / wallBm.getHeight(); v.invalidate();
                }
                lx = e.getX(); ly = e.getY();
            }
            lastN = e.getActionMasked() == MotionEvent.ACTION_UP || e.getActionMasked() == MotionEvent.ACTION_CANCEL ? 0 : n;
            return true;
        });
        f.addView(v, new FrameLayout.LayoutParams(-1, -1));
        LinearLayout bar = new LinearLayout(this); bar.setOrientation(LinearLayout.VERTICAL); bar.setBackgroundColor(0xCC000000); bar.setPadding(dp(16), dp(10), dp(16), dp(12));
        TextView hint = text(L.s("Двумя пальцами — масштаб, одним — сдвиг", "Pinch to zoom, drag to move"), 13, 0xFFEDEDEF, false); hint.setGravity(Gravity.CENTER); bar.addView(hint, new LinearLayout.LayoutParams(-1, -2));
        LinearLayout row = new LinearLayout(this); row.setPadding(0, dp(8), 0, 0);
        Button cancel = button(L.s("Отмена", "Cancel"), 0x33FFFFFF, 0xFFEDEDEF), reset = button(L.s("Сбросить", "Reset"), 0x33FFFFFF, 0xFFEDEDEF), done = button(L.s("Готово", "Done"), 0xFFEDEDEF, 0xFF111113);
        cancel.setOnClickListener(x -> onBackPressed());
        reset.setOnClickListener(x -> { wZoom = 1f; wX = .5f; wY = .5f; v.invalidate(); });
        done.setOnClickListener(x -> { wallClamp(v.getWidth(), v.getHeight()); saveWallParams(this); finish(); });
        for (Button bt : new Button[]{cancel, reset, done}) { bt.setTextSize(14); LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(44), 1); lp.leftMargin = dp(4); lp.rightMargin = dp(4); row.addView(bt, lp); }
        bar.addView(row);
        f.addView(bar, new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM));
        setContentView(f);
    }

    @Override public void onBackPressed() { loadWallParams(this); super.onBackPressed(); }
}
