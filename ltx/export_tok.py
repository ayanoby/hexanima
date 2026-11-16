"""Gemma 4 tokenizer (BPE, from the tokenizer_json inside the LTX text-encoder file) -> compact binary for the app: python export_tok.py OUT.bin
format (little endian): int32 nvocab; nvocab x [uint16 len, utf8 bytes] (id = order); int32 nmerges; nmerges x [int32 a, int32 b, int32 merged id] in rank order"""
import sys, json, struct
tj = json.load(open('/tmp/gemma_tok.json')); m = tj['model']; vocab = m['vocab']; ids = {t: i for i, t in enumerate(vocab)} if isinstance(vocab, list) else vocab
inv = [None] * len(ids)
for t, i in ids.items(): inv[i] = t
out = bytearray(struct.pack('<i', len(inv)))
for t in inv: b = t.encode('utf-8'); out += struct.pack('<H', len(b)) + b
out += struct.pack('<i', len(m['merges']))
for mg in m['merges']:
    a, b = mg if isinstance(mg, list) else mg.split(' ')
    out += struct.pack('<iii', ids[a], ids[b], ids[a + b])
open(sys.argv[1], 'wb').write(out); print('tok.bin', len(out), 'bytes', len(inv), 'vocab', len(m['merges']), 'merges')
