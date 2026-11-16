package com.hexanima.npu;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;

/** Gemma 4 BPE tokenizer of the LTX-2.5 text encoder (tok.bin made by anima/ltx/export_tok.py from the tokenizer.json inside the encoder file): spaces -> "▁", BPE over the whole string with the merge table,
 *  UTF-8 byte fallback "<0xXX>", BOS (id 2) first. Same ids as the HF tokenizers library. */
final class GemmaTok {
    static final int BOS = 2;
    private final HashMap<String, Integer> vocab = new HashMap<>(); private final int[] mergedOf; private final long[] keys; private final int[] rankOf;   // open-addressing table: (a << 32 | b) -> merge index (= rank)
    private final int mask;

    GemmaTok(File f) throws IOException {
        try (FileInputStream in = new FileInputStream(f); FileChannel ch = in.getChannel()) {
            ByteBuffer b = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN);
            final int nv = b.getInt(); byte[] tmp = new byte[256];
            for (int i = 0; i < nv; i++) { int n = b.getShort() & 0xFFFF; if (n > tmp.length) tmp = new byte[n]; b.get(tmp, 0, n); vocab.put(new String(tmp, 0, n, StandardCharsets.UTF_8), i); }
            final int nm = b.getInt(); mergedOf = new int[nm]; int cap = 1; while (cap < nm * 2) cap <<= 1; mask = cap - 1; keys = new long[cap]; rankOf = new int[cap]; java.util.Arrays.fill(rankOf, -1);
            for (int i = 0; i < nm; i++) {
                final int a = b.getInt(), c = b.getInt(), m = b.getInt(); mergedOf[i] = m; final long k = ((long) a << 32) | (c & 0xFFFFFFFFL);
                int h = hash(k) & mask; while (rankOf[h] >= 0) h = (h + 1) & mask; keys[h] = k; rankOf[h] = i;
            }
        }
    }
    private static int hash(long k) { k *= 0x9E3779B97F4A7C15L; return (int) (k >>> 33); }
    private int rank(int a, int c) { final long k = ((long) a << 32) | (c & 0xFFFFFFFFL); int h = hash(k) & mask; while (rankOf[h] >= 0) { if (keys[h] == k) return rankOf[h]; h = (h + 1) & mask; } return -1; }

    /** BOS + the BPE ids of the text */
    int[] encode(String text) {
        final String s = text.replace(' ', '▁'); final ArrayList<Integer> sym = new ArrayList<>();
        for (int i = 0; i < s.length(); ) {
            final int cp = s.codePointAt(i); i += Character.charCount(cp); final String ch = new String(Character.toChars(cp)); final Integer id = vocab.get(ch);
            if (id != null) sym.add(id); else for (byte x : ch.getBytes(StandardCharsets.UTF_8)) { final Integer bi = vocab.get(String.format("<0x%02X>", x & 0xFF)); sym.add(bi == null ? 3 : bi); }
        }
        while (sym.size() > 1) {   // lowest rank first, leftmost among equals
            int best = -1, bi = -1;
            for (int i = 0; i + 1 < sym.size(); i++) { final int r = rank(sym.get(i), sym.get(i + 1)); if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; } }
            if (best < 0) break;
            sym.set(bi, mergedOf[best]); sym.remove(bi + 1);
        }
        final int[] out = new int[sym.size() + 1]; out[0] = BOS; for (int i = 0; i < sym.size(); i++) out[i + 1] = sym.get(i); return out;
    }
}
