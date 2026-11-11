package com.hexanima.npu;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.res.Configuration;
import android.content.res.ColorStateList;
import android.view.View;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.view.ViewGroup;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.*;

/** Shared look for all screens: one neutral palette, flat surfaces, thin dividers, a single accent (the text colour); the UI is built in code, there are no layout files. */
abstract class Base extends Activity {
    /** palette: set by applyTheme() before any view is built. CARD and FIELD are the same flat surface (inputs, chips, the picture frame). */
    static int LINE, BG, CARD, FIELD, ACC, ACC2, ON_ACC, TXT, DIM, OK, BAD, WARN;
    static boolean dark = true;
    String uiKey = "";

    static boolean isDark(Context c) {
        String m = c.getSharedPreferences("anima", 0).getString("theme", "system");
        return m.equals("dark") || m.equals("amoled") || (m.equals("system") && (c.getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES);
    }

    static void applyTheme(Context c) {
        dark = isDark(c);
        if (dark) { BG = 0xFF111113; CARD = 0xFF1C1C1F; LINE = 0xFF36363C; TXT = 0xFFEDEDEF; DIM = 0xFF8E8E96; OK = 0xFF6BCB8B; BAD = 0xFFF07178; WARN = 0xFFE5B567; }
        else { BG = 0xFFFAFAFA; CARD = 0xFFF0F0F2; LINE = 0xFFE4E4E7; TXT = 0xFF18181B; DIM = 0xFF71717A; OK = 0xFF15803D; BAD = 0xFFDC2626; WARN = 0xFFB45309; }
        android.content.SharedPreferences p = c.getSharedPreferences("anima", 0);
        if (dark && p.getString("theme", "system").equals("amoled")) { BG = 0xFF000000; CARD = 0xFF151517; LINE = 0xFF303036; }   // true black for OLED screens
        FIELD = CARD; ACC = TXT; ACC2 = TXT; ON_ACC = BG;   // the accent is the text colour: a white (or black) primary button and selection
        if (android.os.Build.VERSION.SDK_INT >= 31 && p.getBoolean("dyn_accent", false)) {   // the system (wallpaper) accent colour
            ACC = c.getColor(dark ? android.R.color.system_accent1_200 : android.R.color.system_accent1_600); ACC2 = ACC; ON_ACC = dark ? 0xFF111113 : 0xFFFFFFFF;
        }
    }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        L.init(this);
        applyTheme(this);
        uiKey = themeKey();
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        if (!dark) getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
    }

    /** kept for callers: no effect (the UI stays still) */
    <T extends View> T press(T v) { return v; }
    @Override public void startActivity(android.content.Intent i) { super.startActivity(i); overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out); }
    @Override public void startActivityForResult(android.content.Intent i, int rc) { super.startActivityForResult(i, rc); overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out); }
    @Override public void finish() { super.finish(); overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out); }

    String themeKey() { android.content.SharedPreferences p = getSharedPreferences("anima", 0); return L.lang + isDark(this) + p.getString("theme", "system") + p.getBoolean("dyn_accent", false); }

    /** language or theme changed on another screen: rebuild this one */
    @Override protected void onResume() {
        super.onResume();
        L.init(this);
        if (!themeKey().equals(uiKey)) recreate();
    }

    /** dialogs follow the theme */
    AlertDialog.Builder builder() { return new AlertDialog.Builder(this, dark ? android.R.style.Theme_Material_Dialog_Alert : android.R.style.Theme_Material_Light_Dialog_Alert); }

    /** a dialog around a custom view, on the page colour */
    AlertDialog dialog(View v) { AlertDialog d = builder().setView(v).create(); d.show(); d.getWindow().setBackgroundDrawable(framed(20)); return d; }

    int dp(float v) { return (int) (v * getResources().getDisplayMetrics().density + .5f); }

    /** the page colour with the thin outline: dialogs and sheets */
    GradientDrawable framed(float radius) { GradientDrawable g = bg(BG, radius); g.setStroke(dp(1), LINE); return g; }
    GradientDrawable bg(int color, float radius) { GradientDrawable g = new GradientDrawable(); g.setColor(color); g.setCornerRadius(dp(radius)); if (color == CARD) g.setStroke(dp(1), LINE); return g; }   // surfaces carry a thin outline

    Drawable vec(String name, int tint, int sizeDp) { Drawable d = getResources().getDrawable(getResources().getIdentifier(name, "drawable", getPackageName()), getTheme()).mutate(); d.setBounds(0, 0, dp(sizeDp), dp(sizeDp)); d.setTint(tint); return d; }

    /** small status text (no frame) */
    TextView pill(String icon) { TextView t = text("", 12, DIM, false); t.setGravity(Gravity.CENTER_VERTICAL); return t; }
    void setPill(TextView t, String s, int color) { t.setText(s); t.setTextColor(color); android.graphics.drawable.Drawable[] d = t.getCompoundDrawablesRelative(); if (d[0] != null) d[0].setTint(color); }

    /** a list dialog anchored to the bottom edge */
    AlertDialog sheet(View content) {
        AlertDialog d = builder().setView(content).create(); d.show();
        android.view.Window w = d.getWindow();
        if (w != null) { w.setBackgroundDrawable(framed(20)); w.setGravity(Gravity.BOTTOM); w.setLayout(getResources().getDisplayMetrics().widthPixels - dp(16), ViewGroup.LayoutParams.WRAP_CONTENT); w.getAttributes().y = dp(8); w.setAttributes(w.getAttributes()); }
        return d;
    }

    TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(this); t.setText(s); t.setTextSize(sp); t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        return t;
    }

    Button button(String s, int color, int textColor) {
        Button b = new Button(this); b.setText(s); b.setAllCaps(false); b.setTextSize(14); b.setTextColor(textColor);
        b.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL)); b.setStateListAnimator(null); b.setBackground(bg(color, 12));
        b.setMinWidth(0); b.setMinimumWidth(0); b.setMinHeight(0); b.setMinimumHeight(0);
        return b;
    }

    /** plain icon from res/drawable (looked up by name: the build has no R class); the colour argument is ignored, icons have no frame */
    ImageButton icon(String name, int color) {
        ImageButton b = new ImageButton(this); b.setImageResource(getResources().getIdentifier(name, "drawable", getPackageName()));
        b.setBackground(null); b.setPadding(dp(10), dp(10), dp(10), dp(10)); b.setScaleType(ImageView.ScaleType.FIT_CENTER);
        if (!name.equals("ic_star") && !name.equals("ic_delete")) b.setImageTintList(ColorStateList.valueOf(TXT));   // star / trash keep their own colours
        return b;
    }

    EditText field(String hint) {
        EditText e = new EditText(this); e.setHint(hint); e.setHintTextColor(DIM); e.setTextColor(TXT); e.setTextSize(15);
        e.setBackground(bg(FIELD, 12)); e.setPadding(dp(14), dp(10), dp(14), dp(10));
        return e;
    }

    /** section caption above a group */
    TextView label(String s, int color) { TextView t = label(s); t.setTextColor(color); return t; }
    TextView label(String s) { TextView t = text(s, 13, DIM, false); t.setPadding(0, dp(20), 0, dp(8)); return t; }

    View divider() { View v = new View(this); v.setBackgroundColor(LINE); v.setLayoutParams(new LinearLayout.LayoutParams(-1, 1)); return v; }

    /** a flat section: content, then a thin line */
    LinearLayout card(LinearLayout parent) {
        LinearLayout c = new LinearLayout(this); c.setOrientation(LinearLayout.VERTICAL); c.setPadding(0, dp(14), 0, dp(14));
        parent.addView(c, new LinearLayout.LayoutParams(-1, -2)); parent.addView(divider());
        return c;
    }

    /** screen skeleton: ← title [extra view] on top, a scrolling column below; returns the column */
    LinearLayout screen(String title, android.view.View extra) {
        LinearLayout frame = new LinearLayout(this); frame.setOrientation(LinearLayout.VERTICAL); frame.setBackgroundColor(BG); frame.setFitsSystemWindows(true);
        LinearLayout head = new LinearLayout(this); head.setGravity(Gravity.CENTER_VERTICAL); head.setPadding(dp(8), dp(6), dp(16), dp(6));
        TextView back = text("←", 22, TXT, false); back.setGravity(Gravity.CENTER); back.setOnClickListener(v -> onBackPressed());
        head.addView(back, new LinearLayout.LayoutParams(dp(44), dp(44)));
        TextView t = text(title, 20, TXT, true); t.setPadding(dp(6), 0, 0, 0);
        head.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
        if (extra != null) head.addView(extra, new LinearLayout.LayoutParams(-2, dp(36)));
        frame.addView(head);
        ScrollView sv = new ScrollView(this); sv.setVerticalScrollBarEnabled(false);
        LinearLayout col = new LinearLayout(this); col.setOrientation(LinearLayout.VERTICAL); col.setPadding(dp(20), dp(4), dp(20), dp(24));
        sv.addView(col, new android.view.ViewGroup.LayoutParams(-1, -2));
        frame.addView(sv, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(frame);
        return col;
    }
}
