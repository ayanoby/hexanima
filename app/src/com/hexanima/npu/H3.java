package com.hexanima.npu;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** MiniMax-H3-Image (33B DiT + 25B text encoder, INT8-ConvRot, 6 steps, no CFG) on the NPU: files under files/h3 (model/ te/ vae/ tok_*.txt), one picture per h3.sh call.
 *  Not in the catalog (MiniMax community licence: not for the US, EU, UK and Korea, needs the licence text with every copy): the pack is put into files/h3 by hand. */
final class H3 {
    static final String ID = "minimax-h3-image";
    static boolean is(String id) { return id != null && id.startsWith("minimax-h3"); }
    static File dir(File home) { return new File(home, "h3"); }
    static boolean installed(File home) { File d = dir(home); return new File(d, ".installed").exists() && new File(d, "model/blk49.bin").exists() && new File(d, "te/lay49.bin").exists() && new File(d, "vae/vae_h3.bin").exists(); }
    static final int STEPS = 6;
    /** the distillation LoRA (pDMD 2-NFE, rank 128, Apache-2.0, akhaliq/minimax-h3-image-pdmd-2step): files/h3/lora/loraNN.bin + lorar0/1.bin made by export_h3_lora.py; 2 steps, sigmas from shift 12, output heads 0 and 1 */
    static final int LORA_STEPS = 2, LORA_MAX_STEPS = 4;
    /** the grid of N evaluations: linspace(1, 0, N+1) pushed through the flow shift 12 (diffusers MiniMaxH3Scheduler); output head i = step i, as in the official loop */
    static String loraSigmas(int n) { StringBuilder sb = new StringBuilder(); for (int i = 0; i <= n; i++) { double b = 1.0 - (double) i / n, s = 12 * b / (1 + 11 * b); sb.append(i == 0 ? "" : ",").append(String.format(java.util.Locale.US, "%.7f", s)); } return sb.toString(); }
    /** n of the 6 native PDD steps (each has its own sigma and output head): step i of n is native step i*6/n, the last one runs to sigma 0 */
    static final double[] PDD_SIG = {1.0, 0.9675237, 0.8875485, 0.75, 0.5327739, 0.2320089};
    static String pddSigmas(int n) { StringBuilder sb = new StringBuilder(); for (int i = 0; i < n; i++) sb.append(String.format(java.util.Locale.US, "%.7f,", PDD_SIG[i * 6 / n])); return sb.append("0").toString(); }
    static String pddHeads(int n) { StringBuilder sb = new StringBuilder(); for (int i = 0; i < n; i++) sb.append(i == 0 ? "" : ",").append(i * 6 / n); return sb.toString(); }
    static String loraHeads(int n) { StringBuilder sb = new StringBuilder(); for (int i = 0; i < n; i++) sb.append(i == 0 ? "" : ",").append(i); return sb.toString(); }
    static File lora(File home) { return new File(dir(home), "lora"); }
    static boolean loraInstalled(File home) { File d = lora(home); return new File(d, "lora49.bin").exists() && new File(d, "lorar1.bin").exists(); }

    /** sizes: sides multiples of 32 (one token = 32x32 px), at least 256 (the VAE tile), the token count (W/32 * H/32) a multiple of 32 (whole 32-row slabs of image tokens) */
    static final int[][] PRESETS = {{512, 512}, {1024, 1024}, {512, 768}, {768, 512}, {768, 768}, {768, 1024}, {1024, 768}, {1024, 512}, {512, 1024}, {1024, 1536}, {1536, 1024}, {1536, 1536}, {2048, 1024}, {1024, 2048}, {2048, 2048}};
    static final int MAX_TOKENS = 4096;
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 32 != 0 || h % 32 != 0) return L.s("Ширина и высота должны быть кратны 32", "Width and height must be multiples of 32");
        int t = (w / 32) * (h / 32);
        if (t % 32 != 0) return L.f("Число токенов (%d) должно делиться на 32, например 512×512, 768×1024, 1024×1024", "Token count (%d) must be divisible by 32, e.g. 512×512, 768×1024, 1024×1024", t);
        if (t > MAX_TOKENS) return L.s("Слишком большой размер: максимум 4096 токенов (2048×2048)", "Too large: at most 4096 tokens (2048×2048)");
        return null;
    }
    /** rough seconds per picture: 50 layers of text encoder + 6 steps (flash read ~9 s + NPU compute ~3.8 s per 256 tokens) + VAE */
    static int estimate(int w, int h, int steps) { double tok = (w / 32.0) * (h / 32.0); return (int) Math.round(22 + steps * (9 + 3.8 * tok / 256) + 1.6 * tok / 256 + 2); }

    /** the prompt goes in verbatim: no chat template, no special tokens (the conditioning is the hidden state after decoder layer 50) */
    static int writeIds(File home, String prompt, File out) throws Exception {
        QwenTok t = new QwenTok(dir(home).getPath());
        int[] ids = t.encode(prompt == null || prompt.isEmpty() ? " " : prompt);
        if (ids.length > 480) ids = java.util.Arrays.copyOf(ids, 480);
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
        return ids.length;
    }
}
