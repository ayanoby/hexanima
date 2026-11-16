# LTX-2.5 22B distilled (text/image to video) on the NPU

Notes: `NOTES.md` (architecture, kernel modes 90-92, measurements). The DSP kernels are in `../flux3/hexagon/mega` (one tree for Klein, LTX and Ref2VA).

| What | Source | Build |
|---|---|---|
| Runner (Gemma 4 12B text encoder, connectors, 48-block DiT, 8-step sampler) | `runner/gemma.inc` + `../flux3/runner/base.inc` | `runner/build.sh` -> `libltxrun.so` |
| VAE decoder / image encoder (QNN) | `vae/runner/lvae.cpp`, graphs `vae/` (`export_tile.py`, `export_enc.py`, `build_ctx.sh`) | `vae/runner/build.sh` -> `liblvae.so` |
| Pack conversion | `export_gemma.py`, `export_conn.py`, `export_dit.py`, `export_tok.py` | weights from `LTX_W` (default `ltx_weights`) |
| References | `ref_*.py`, `cmp_*.py`, `mk*_case.py` | PyTorch (transformers 5.x for Gemma 4) |

The converted packs are on Hugging Face as `ayanoby/hexanima-ltx-packs` (LTX-2 Community License and the Gemma terms, see `NOTICE.md` there).
