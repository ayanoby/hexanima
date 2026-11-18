package com.hexanima.npu;

import android.content.Context;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.TreeMap;
import org.json.JSONArray;
import org.json.JSONObject;

/** Real generation times of this phone (files/timings.json, the last 20 per model): every finished picture / video stores its seconds next to the formula's estimate, the settings it was made with
 *  (size, steps, frames, pictures to edit: "sig") and the token count. The estimate on the Generate button:
 *   1. the same settings were measured: the median of those runs (one run counts half, the rest comes from the formula),
 *   2. other sizes were measured: the formula times the measured / estimated ratio, interpolated between the measured token counts (log scale); beyond them, or with a single measured size, the ratio fades back to 1
 *      the farther the size is, because a phone's ratio at 512x768 says little about 1024x1536,
 *   3. nothing measured: the formula.
 *  The first run of a model after the app started is stored but ignored (the model load and cold files make it slower). */
final class Timings {
    private static final int KEEP = 20;
    private static final HashSet<String> warm = new HashSet<>();   // models already run since the app process started

    private static File file(Context c) { return new File(c.getFilesDir(), "timings.json"); }
    private static JSONObject read(Context c) { try { return new JSONObject(Models.read(file(c))); } catch (Exception e) { return new JSONObject(); } }

    static synchronized void record(Context c, String key, String sig, int tokens, double actual, double predicted) {
        if (key == null || key.isEmpty() || actual <= 0 || predicted <= 0 || tokens <= 0) return;
        boolean cold = warm.add(key);
        try {
            JSONObject all = read(c); JSONArray a = all.optJSONArray(key); if (a == null) a = new JSONArray();
            a.put(new JSONObject().put("a", Math.round(actual * 10) / 10.0).put("p", Math.round(predicted * 10) / 10.0).put("c", cold ? 1 : 0).put("s", sig).put("t", tokens));
            JSONArray cut = new JSONArray(); for (int i = Math.max(0, a.length() - KEEP); i < a.length(); i++) cut.put(a.get(i));
            all.put(key, cut);
            try (java.io.FileWriter w = new java.io.FileWriter(file(c))) { w.write(all.toString()); }
        } catch (Exception ignored) { }
    }

    private static double median(List<Double> v) { Collections.sort(v); int n = v.size(); return n % 2 == 1 ? v.get(n / 2) : (v.get(n / 2 - 1) + v.get(n / 2)) / 2; }

    /** seconds for these settings; formula = the formula's estimate for them */
    static synchronized double estimate(Context c, String key, String sig, int tokens, double formula) {
        try {
            JSONArray a = read(c).optJSONArray(key); if (a == null) return formula;
            List<Double> same = new ArrayList<>(); TreeMap<Integer, List<Double>> byTok = new TreeMap<>();
            for (int i = 0; i < a.length(); i++) {
                JSONObject e = a.optJSONObject(i); if (e == null || e.optInt("c") != 0 || e.optDouble("p") <= 0) continue;   // cold runs and old-format entries do not count
                if (sig.equals(e.optString("s"))) same.add(e.getDouble("a"));
                byTok.computeIfAbsent(e.getInt("t"), k -> new ArrayList<>()).add(e.getDouble("a") / e.getDouble("p"));
            }
            if (byTok.isEmpty()) return formula;
            // the ratio at this size from the other measurements
            double r;
            List<Integer> ts = new ArrayList<>(byTok.keySet()); List<Double> rs = new ArrayList<>(); for (int t : ts) rs.add(median(new ArrayList<>(byTok.get(t))));
            double lt = Math.log(tokens);
            if (ts.size() == 1) { double d = Math.abs(lt - Math.log(ts.get(0))); r = 1 + (rs.get(0) - 1) * Math.exp(-d / Math.log(4)); }   // 4x the tokens: a third of the correction is left
            else {
                int j = 0; while (j < ts.size() - 2 && lt > Math.log(ts.get(j + 1))) j++;   // the segment [j, j+1] that holds the size, or the nearest end segment
                double x0 = Math.log(ts.get(j)), x1 = Math.log(ts.get(j + 1)), y0 = rs.get(j), y1 = rs.get(j + 1), f = (lt - x0) / (x1 - x0);
                if (f >= 0 && f <= 1) r = y0 + (y1 - y0) * f;
                else { int k = f < 0 ? j : j + 1; double d = Math.abs(lt - Math.log(ts.get(k))); r = 1 + (rs.get(k) - 1) * Math.exp(-d / Math.log(4)); }   // outside the measured range: the nearest end, fading
            }
            r = Math.max(0.4, Math.min(2.5, r));
            double byFormula = formula * r;
            if (same.isEmpty()) return byFormula;
            double m = median(same);
            return same.size() == 1 ? (m + byFormula) / 2 : m;
        } catch (Exception e) { return formula; }
    }
}
