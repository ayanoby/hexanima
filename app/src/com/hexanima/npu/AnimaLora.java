package com.hexanima.npu;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.regex.*;
import org.json.JSONObject;

/** User LoRA for the Anima models (2B and 2.9B). The picked .safetensors is copied to files/anima/lora/user.safetensors and checked here from its header only
 *  (key names, shapes, dtype); megarun folds it into the block weights when the model loads (LORA="file*strength", see phone/anima.sh). The text-encoder part is not applied. */
final class AnimaLora {
    static File dir(File home) { return new File(home, "anima/lora"); }
    static File src(File home) { return new File(dir(home), "user.safetensors"); }
    /** Lab: the turbo LoRA picked for a base pack (folded together with the user LoRA) */
    static File turboSrc(File home) { return new File(dir(home), "turbo.safetensors"); }

    static final class Check {
        boolean ok; String code = "", detail = ""; int blocks, of, rank; String format = ""; boolean te, adapter;
        String message() {
            if (ok) {
                StringBuilder s = new StringBuilder(L.f("Подходит: ранг %d, блоков %d из %d", "Compatible: rank %d, %d of %d blocks", rank, blocks, of)).append("  ·  ").append(format);
                if (te) s.append("\n").append(L.s("Часть для текстового энкодера не применяется", "The text-encoder part is not applied"));
                return s.toString();
            }
            switch (code) {
                case "OTHERMODEL": return L.s("Это LoRA для другой модели (FLUX, SDXL или похожей). Для Anima она не подходит.", "This LoRA is for another model (FLUX, SDXL or similar). It does not fit Anima.");
                case "NOTLORA": return L.s("Файл не похож на LoRA для Anima: не найдены нужные слои (blocks.N.*).", "This file does not look like an Anima LoRA: the expected layers (blocks.N.*) are missing.");
                case "TEXTONLY": return L.s("В файле только LoRA для текстового энкодера. Такие пока не поддерживаются.", "The file only has text-encoder LoRA weights. Those are not supported yet.");
                case "SHAPE": return L.f("Размеры слоёв не совпадают с Anima (%s). Скорее всего это LoRA для другой версии модели.", "Layer sizes do not match Anima (%s). It is probably a LoRA for another version of the model.", detail);
                case "LAYER": return L.f("В LoRA есть блоки, которых нет в Anima (%s).", "The LoRA has blocks that Anima does not have (%s).", detail);
                case "DTYPE": return L.s("Неподдерживаемый формат весов, нужны F32, F16 или BF16.", "Unsupported weight format: F32, F16 or BF16 are needed.");
                default: return L.f("Не удалось прочитать файл (%s). Нужен .safetensors.", "Cannot read the file (%s). A .safetensors file is needed.", detail);
            }
        }
    }

    private static final Pattern BLK = Pattern.compile("^(?:lora_unet_blocks_|diffusion_model\\.blocks\\.|lycoris_blocks_|transformer\\.blocks\\.|blocks\\.)(\\d+)[_.](.*)$");

    static Check check(File f) {
        Check c = new Check();
        try (RandomAccessFile r = new RandomAccessFile(f, "r")) {
            byte[] h8 = new byte[8]; r.readFully(h8); long n = ByteBuffer.wrap(h8).order(ByteOrder.LITTLE_ENDIAN).getLong();
            if (n <= 0 || n > (64L << 20) || n + 8 > r.length()) { c.code = "FILE"; c.detail = "header"; return c; }
            byte[] hb = new byte[(int) n]; r.readFully(hb);
            JSONObject h = new JSONObject(new String(hb, "UTF-8"));
            Set<Integer> blocks = new TreeSet<>(); int other = 0, teKeys = 0, rank = 0; boolean bad = false;
            for (Iterator<String> it = h.keys(); it.hasNext(); ) {
                String k = it.next(); if (k.equals("__metadata__")) continue;
                JSONObject t = h.getJSONObject(k); String dt = t.optString("dtype");
                if (k.startsWith("lora_te") || k.startsWith("text_encoder.")) { teKeys++; continue; }
                if (k.contains("llm_adapter")) { c.adapter = true; continue; }
                Matcher m = BLK.matcher(k);
                if (!m.matches()) { if (k.contains("double_blocks") || k.contains("single_blocks") || k.contains("input_blocks") || k.contains("transformer_blocks")) other++; continue; }
                if (!dt.equals("F32") && !dt.equals("F16") && !dt.equals("BF16")) bad = true;
                blocks.add(Integer.parseInt(m.group(1)));
                String rest = m.group(2);
                boolean down = k.contains("lora_down") || k.contains("lora_A");
                if (down && t.getJSONArray("shape").length() == 2) {
                    if (rank == 0) rank = t.getJSONArray("shape").getInt(0);
                    if (rest.startsWith("self_attn_q_proj") || rest.startsWith("self_attn.q_proj")) { if (t.getJSONArray("shape").getInt(1) != 2048) { c.code = "SHAPE"; c.detail = "q_proj in=" + t.getJSONArray("shape").getInt(1); return c; } }
                }
                if (rank == 0 && (k.contains("lokr_w1_a") || k.contains("lokr_w2_a") || k.contains("hada_w1_a")) && t.getJSONArray("shape").length() == 2) rank = t.getJSONArray("shape").getInt(1);
                if (k.contains("lokr_")) c.format = "LoKr"; else if (k.contains("hada_")) c.format = "LoHa";
                else if (c.format.isEmpty()) c.format = k.contains("lora_down") ? "kohya" : k.contains("lora_A") ? "diffusers" : "";
            }
            c.te = teKeys > 0;
            if (blocks.isEmpty()) { c.code = other > 0 ? "OTHERMODEL" : teKeys > 0 ? "TEXTONLY" : "NOTLORA"; return c; }
            int top = ((TreeSet<Integer>) blocks).last() + 1;
            if (top > 40) { c.code = "LAYER"; c.detail = "block " + (top - 1); return c; }
            if (bad) { c.code = "DTYPE"; return c; }
            c.ok = true; c.blocks = blocks.size(); c.of = top <= 28 ? 28 : 40; c.rank = rank; if (c.format.isEmpty()) c.format = "LyCORIS";
        } catch (Exception e) { c.code = "FILE"; c.detail = e.getMessage() == null ? e.toString() : e.getMessage(); }
        return c;
    }
}
