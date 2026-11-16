# Mage-Flow kernels

The DSP kernels of Mage-Flow-Edit-Turbo live in the shared tree **`../../../flux3/hexagon/mega`** (the one tree that also builds Klein, LTX-2.5 and MiniMax-H3 Ref2VA). Mage uses:

| MODE | What | Where in `blk.c` |
|---|---|---|
| 53 | Mage-Flow double-stream DiT block (image rows with their own weights, text rows with theirs, joint attention); windowed runs through the flags `M_A4` bit 0 (only the keys/values of a window into the global K/V buffers, slab offset `M_A9`), bit 1 (the keys/values are there already, `M_A2` = key tokens), `M_BA` = image slabs in the window | `meta[M_MODE] == 53` |
| 55 | Qwen3-VL-4B vision layer (hidden 1024, 16 heads padded to 128) | `== 55` |
| 61 | generic GEMM for the VAE (K tiles `M_NCB`, N `M_BA`, optional tanh-GELU) | `== 61` |
| 22 | Qwen3-VL-4B text-encoder layer (hidden 2560), shared with Klein | `== 22` |

Build the skel with `flux3/hexagon/mega/build_skel.sh v69` (or `build.sh`); the runner (`../magerun`) is built by `../magerun/build.sh` against the same headers. The attention switches to the chunked version (keys streamed through VTCM) when the keys of a call do not fit in VTCM, which is what removes the old limit of about 6500 tokens.

The first Mage tree (own `blk.c`, modes 50/51/60, attention limited to ~6500 tokens) is in the git history (commit 5d05dd5).
