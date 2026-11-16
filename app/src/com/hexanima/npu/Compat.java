package com.hexanima.npu;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;
import java.util.ArrayList;
import java.util.List;

/** Can this phone run the models? only Hexagon V69 (Snapdragon 8 Gen 1 / 8+ Gen 1) is supported. */
final class Compat {
    static String soc() { return Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : Build.HARDWARE; }
    /** Hexagon generation of the NPU by SoC model ("" = unknown): the QNN contexts (text encoder, VAE) are compiled per generation */
    static String dspArch() {
        String s = soc();
        if (s.equals("SM8475") || s.equals("SM8450")) return "v69";
        return "";
    }
    static boolean chipOk() { return !dspArch().isEmpty(); }
    static String archLabel() { return "Hexagon " + dspArch().toUpperCase(); }
    static boolean chipAllowed(Context c) { return chipOk() || c.getSharedPreferences("anima", 0).getBoolean("chip_override", false); }
    static void allowChip(Context c) { c.getSharedPreferences("anima", 0).edit().putBoolean("chip_override", true).apply(); }

    static double ramGb(Context c) { ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo(); ((ActivityManager) c.getSystemService(Context.ACTIVITY_SERVICE)).getMemoryInfo(mi); return mi.totalMem / 1073741824.0; }
    /** RAM taken by the app and its helper processes (same uid): PSS of every process + the dma-buf buffers they hold (pinned model weights; counted once per buffer) */
    static double appRamGb() {
        long kb = 0; java.util.HashSet<String> seen = new java.util.HashSet<>(); int uid = android.os.Process.myUid();
        java.io.File[] ps = new java.io.File("/proc").listFiles();
        if (ps == null) return 0;
        for (java.io.File d : ps) {
            if (!d.getName().matches("\\d+")) continue;
            try {
                if (android.system.Os.stat(d.getPath()).st_uid != uid) continue;
                try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(new java.io.File(d, "smaps_rollup")))) {
                    for (String l; (l = r.readLine()) != null; ) if (l.startsWith("Pss:")) { kb += Long.parseLong(l.replaceAll("[^0-9]", "")); break; }
                } catch (Exception ignored) { }
                java.io.File[] fds = new java.io.File(d, "fd").listFiles();
                if (fds != null) for (java.io.File f : fds) {
                    try {
                        if (!android.system.Os.readlink(f.getPath()).contains("dmabuf")) continue;
                        long sz = 0; String ino = null;
                        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(new java.io.File(d, "fdinfo/" + f.getName())))) {
                            for (String l; (l = r.readLine()) != null; ) { if (l.startsWith("size:")) sz = Long.parseLong(l.replaceAll("[^0-9]", "")); else if (l.startsWith("ino:")) ino = l.substring(4).trim(); }
                        }
                        if (ino != null && seen.add(ino)) kb += sz / 1024;
                    } catch (Exception ignored) { }
                }
            } catch (Exception ignored) { }
        }
        return kb / 1048576.0;
    }
    static double freeGb(Context c) { return c.getFilesDir().getUsableSpace() / 1e9; }

    /** [chip, ram, storage] each as {ok?"1":"0", text} */
    static List<String[]> report(Context c) {
        List<String[]> r = new ArrayList<>();
        r.add(new String[]{chipOk() ? "1" : "0", chipOk() ? L.f("Чип %s: подходит (%s)", "Chip %s: supported (%s)", soc(), archLabel()) : L.f("Чип %s: не поддерживается, нужен Snapdragon 8 Gen 1 / 8+ Gen 1", "Chip %s: not supported, a Snapdragon 8 Gen 1 / 8+ Gen 1 is required", soc())});
        double ram = ramGb(c);
        r.add(new String[]{ram >= 7.5 ? "1" : "0", ram >= 7.5 ? L.f("ОЗУ %.1f ГБ: достаточно", "RAM %.1f GB: enough", ram) : L.f("ОЗУ %.1f ГБ: мало, нужно от 8 ГБ", "RAM %.1f GB: low, 8 GB or more is needed", ram)});
        double fr = freeGb(c);
        r.add(new String[]{fr >= 6 ? "1" : "0", fr >= 6 ? L.f("Свободно %.1f ГБ: достаточно", "Free space %.1f GB: enough", fr) : L.f("Свободно %.1f ГБ: мало, модель занимает до 5,4 ГБ", "Free space %.1f GB: low, a model needs up to 5.4 GB", fr)});
        return r;
    }
}
