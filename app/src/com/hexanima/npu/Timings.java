package com.hexanima.npu;

import android.content.Context;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/** Real generation times of this phone: after every picture / video the measured time and the formula's estimate are stored (files/timings.json, the last 10 per model); the estimate on the Generate button is the formula times
 *  the median of measured / estimated, so it adapts to the phone (chip, thermal state, storage speed). The first run of a model after the app started is stored but not used (the model load and cold files make it slower). */
final class Timings {
    private static final int KEEP = 10, MIN = 3;
    private static final HashSet<String> warm = new HashSet<>();   // models already run since the app process started

    private static File file(Context c) { return new File(c.getFilesDir(), "timings.json"); }
    private static JSONObject read(Context c) { try { return new JSONObject(Models.read(file(c))); } catch (Exception e) { return new JSONObject(); } }

    /** @param actual seconds the run took, @param predicted what the formula said for the same settings */
    static synchronized void record(Context c, String key, double actual, double predicted) {
        if (key == null || key.isEmpty() || actual <= 0 || predicted <= 0) return;
        boolean cold = warm.add(key);
        try {
            JSONObject all = read(c); JSONArray a = all.optJSONArray(key); if (a == null) a = new JSONArray();
            a.put(new JSONArray().put(Math.round(actual * 10) / 10.0).put(Math.round(predicted * 10) / 10.0).put(cold ? 1 : 0));
            JSONArray cut = new JSONArray(); for (int i = Math.max(0, a.length() - KEEP); i < a.length(); i++) cut.put(a.get(i));
            all.put(key, cut);
            try (java.io.FileWriter w = new java.io.FileWriter(file(c))) { w.write(all.toString()); }
        } catch (Exception ignored) { }
    }

    /** measured / estimated for this model: the median of the warm runs, 1 until there are three of them; kept within 0.4..2.5 so one odd run cannot ruin the estimate */
    static synchronized double ratio(Context c, String key) {
        try {
            JSONArray a = read(c).optJSONArray(key); if (a == null) return 1;
            List<Double> r = new ArrayList<>();
            for (int i = 0; i < a.length(); i++) { JSONArray e = a.getJSONArray(i); if (e.getInt(2) == 0 && e.getDouble(1) > 0) r.add(e.getDouble(0) / e.getDouble(1)); }
            if (r.size() < MIN) return 1;
            Collections.sort(r); double m = r.size() % 2 == 1 ? r.get(r.size() / 2) : (r.get(r.size() / 2 - 1) + r.get(r.size() / 2)) / 2;
            return Math.max(0.4, Math.min(2.5, m));
        } catch (Exception e) { return 1; }
    }
}
