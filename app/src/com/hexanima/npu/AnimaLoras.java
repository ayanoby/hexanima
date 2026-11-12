package com.hexanima.npu;

import android.content.SharedPreferences;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** The Anima LoRA library: any number of files, each with its own switch and strength. The list lives in the prefs ("anima_loras", JSON), the files in files/anima/lora/lib/<id>.safetensors;
 *  all enabled ones go to the native side as LORA="file*strength|file*strength|..." and are folded into the block weights when the model loads (hexagon/mega/lora_fold.h). */
final class AnimaLoras {
    static final String KEY = "anima_loras";
    static final class Item { String id, name = "LoRA", info = ""; boolean on = true; float str = 1f; }

    static File dir(File home) { return new File(AnimaLora.dir(home), "lib"); }
    static File file(File home, String id) { return new File(dir(home), id + ".safetensors"); }

    /** the list; a single LoRA picked with an older version moves into the library the first time */
    static List<Item> load(SharedPreferences p, File home) {
        if (!p.contains(KEY)) {
            List<Item> l = new ArrayList<>();
            String oid = p.getString("anima_lora_id", "");
            if (!oid.isEmpty() && AnimaLora.src(home).exists()) {
                Item it = new Item(); it.id = "m" + oid; it.name = p.getString("anima_lora_name", "LoRA"); it.info = p.getString("anima_lora_info", ""); it.str = p.getFloat("anima_lora_str", 1f); it.on = it.str > 0;
                dir(home).mkdirs(); if (AnimaLora.src(home).renameTo(file(home, it.id))) l.add(it);
            }
            p.edit().remove("anima_lora_id").remove("anima_lora_name").remove("anima_lora_info").remove("anima_lora_str").apply();
            save(p, l); return l;
        }
        List<Item> out = new ArrayList<>();
        try {
            JSONArray a = new JSONArray(p.getString(KEY, "[]"));
            for (int i = 0; i < a.length(); i++) {
                JSONObject o = a.getJSONObject(i); Item it = new Item();
                it.id = o.getString("id"); it.name = o.optString("name", "LoRA"); it.info = o.optString("info", ""); it.on = o.optBoolean("on", true); it.str = (float) o.optDouble("str", 1.0);
                if (file(home, it.id).exists()) out.add(it);   // a file that is gone (cleared storage) drops out of the list
            }
        } catch (Exception ignored) { }
        return out;
    }

    static void save(SharedPreferences p, List<Item> l) {
        JSONArray a = new JSONArray();
        try { for (Item it : l) a.put(new JSONObject().put("id", it.id).put("name", it.name).put("info", it.info).put("on", it.on).put("str", (double) it.str)); } catch (Exception ignored) { }
        p.edit().putString(KEY, a.toString()).apply();
    }

    static boolean active(Item it) { return it.on && it.str > 0.001f; }

    /** the LORA= value: every active LoRA in list order, "" when none */
    static String env(SharedPreferences p, File home) {
        StringBuilder sb = new StringBuilder();
        for (Item it : load(p, home)) if (active(it)) { if (sb.length() > 0) sb.append('|'); sb.append(file(home, it.id).getPath()).append('*').append(String.format(Locale.US, "%.2f", it.str)); }
        return sb.toString();
    }

    /** "name 0.80, name2 1.00" for the picture's metadata */
    static String names(SharedPreferences p, File home) {
        StringBuilder sb = new StringBuilder();
        for (Item it : load(p, home)) if (active(it)) { if (sb.length() > 0) sb.append(", "); sb.append(it.name.replaceAll("(?i)\\.safetensors$", "")).append(' ').append(String.format(Locale.US, "%.2f", it.str)); }
        return sb.toString();
    }
}
