package com.hexanima.npu;

import android.graphics.Bitmap;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** MOSS-Image (5B DiT + Qwen3.5-9B text encoder + FLUX.2 VAE) on the NPU: files under files/moss, one picture per moss.sh call. */
final class Moss {
    static final String PREFIX_ID = "moss";
    static boolean is(String id) { return id != null && id.startsWith(PREFIX_ID); }
    static File dir(File home) { return new File(home, "moss"); }
    /** a second MOSS entry: the 8-step DMD LoRA (lodestones) as an add-on to the installed pack: glue8.bin (time embedder, sampler tables) + lora8/ (overlays replacing the 4-step LoRA tiles of the DiT blocks), ~0.4 GB */
    static final String ID8 = "moss-sft-q5-8";
    static boolean is8(String id) { return ID8.equals(id); }
    static boolean installed8(File home) { return new File(dir(home), "lora8/blk41.ovl").exists() && new File(dir(home), "glue8.bin").exists(); }
    /** MiaoMiao RealSkin MOSS (a fine-tune of MOSS-Image-SFT by MIAOKA): its own DiT / refiner / glue under files/moss/rs, everything else (text encoder, embeddings, VAE, tokenizer) is the installed MOSS-Image pack's */
    static final String IDRS = "moss-realskin-1";
    static boolean isRs(String id) { return IDRS.equals(id); }
    static boolean installedRs(File home) { File r = new File(dir(home), "rs"); return new File(r, "dit/blk41.bin").exists() && new File(r, "ref/ref1.bin").exists() && new File(r, "glue.bin").exists(); }
    /** files/moss/vrs: the model dir mossrun is pointed at for RealSkin: links to the shared files of the base pack and to rs/{dit,ref,glue.bin}; its own text cache and log */
    static File variantRs(File home) throws IOException {
        File m = dir(home), v = new File(m, "vrs"); v.mkdirs();
        String[][] l = {{"te", "te"}, {"embd.q4k", "embd.q4k"}, {"vae", "vae"}, {"tok_merges.txt", "tok_merges.txt"}, {"tok_special.txt", "tok_special.txt"}, {"tok_vocab.txt", "tok_vocab.txt"}, {"glue8.bin", "glue8.bin"}, {"lora8", "lora8"}, {"dit", "rs/dit"}, {"ref", "rs/ref"}, {"glue.bin", "rs/glue.bin"}};
        for (String[] x : l) {
            File t = new File(m, x[1]), k = new File(v, x[0]);
            if (!t.exists()) continue;
            java.nio.file.Path p = k.toPath(); try { java.nio.file.Files.deleteIfExists(p); java.nio.file.Files.createSymbolicLink(p, t.toPath()); } catch (Exception e) { throw new IOException(e); }
        }
        return v;
    }
    static int steps(String id) { return is8(id) ? 8 : 4; }
    static boolean installed(File home) { return new File(dir(home), ".installed").exists(); }

    /** size presets (W×H): around 1 Mpx is what the 4-step adapter was trained on (1024², 896×1152, 832×1216, 768×1344 and their landscape twins); smaller ones are faster */
    static final int[][] PRESETS = {{512, 512}, {640, 640}, {768, 768}, {1024, 1024}, {512, 768}, {768, 512}, {576, 1024}, {1024, 576}, {640, 1152}, {1152, 640}, {768, 1024}, {1024, 768},
        {896, 1152}, {1152, 896}, {832, 1216}, {1216, 832}, {768, 1344}, {1344, 768}};
    /** any size with sides that are multiples of 16, at least 256, at most 6144 patches (the DiT attention must fit the 8 MB VTCM); null = fine */
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        if ((w / 16) * (h / 16) > 6144) return L.s("Слишком большой размер: максимум 6144 патча (например 1536×1024)", "Too large: at most 6144 patches (e.g. 1536×1024)");
        return null;
    }

    /** tokens of the prompt (chat template included) -> ids file for mossrun */
    static void writeIds(File home, String prompt, File out) throws Exception {
        QwenTok t = new QwenTok(dir(home).getPath());
        int[] ids = t.encodePrompt(prompt);
        if (ids.length > 480) ids = java.util.Arrays.copyOf(ids, 480);   // mossrun takes at most 512 tokens
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
    }

    /** raw fp32 RGB [3][h][w] in [-1,1] -> PNG bytes */
    static byte[] png(File raw, int w, int h) throws IOException {
        byte[] bytes = Models.readBytes(raw);
        if (bytes.length != 3 * w * h * 4) throw new IOException("decoder output has " + bytes.length + " bytes");
        java.nio.FloatBuffer f = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        int[] px = new int[w * h]; int n = w * h;
        float[][] ch = new float[3][n];   // 0..255, rounded and clamped below
        for (int k = 0; k < 3; k++) for (int i = 0; i < n; i++) ch[k][i] = (f.get(k * n + i) + 1f) * 127.5f;
        for (int i = 0; i < n; i++) px[i] = 0xFF000000 | (c8(ch[0][i]) << 16) | (c8(ch[1][i]) << 8) | c8(ch[2][i]);
        Bitmap bm = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888);
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        bm.compress(Bitmap.CompressFormat.PNG, 100, bo);
        return bo.toByteArray();
    }
    private static int c8(float v) { int x = Math.round(v); return x < 0 ? 0 : x > 255 ? 255 : x; }
    private static int c(float v) { int x = Math.round((Math.max(-1f, Math.min(1f, v)) + 1f) * 127.5f); return x < 0 ? 0 : x > 255 ? 255 : x; }
}
