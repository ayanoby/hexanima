# FLUX.3 Klein, and the kernel tree shared by Klein, LTX-2.5 and MiniMax-H3 Ref2VA

This directory holds the sources of the DSP kernels and the phone runners behind **FLUX.3 Image Klein**. `hexagon/mega/` is the newest copy of the
Hexagon kernel tree (`hexagon/mega/` at the repository root is the older one used by Anima/H3-Image, `mage/hexagon/mega/` the one of Mage-Flow): its
`blk.c` holds every kernel mode used by Klein (60, 70), LTX-2.5 (90-92) and the Ref2VA video VAE (82), so **LTX (`../ltx`) and Ref2VA (`../h3`) are built with this tree too**.

| What | Source | Build | Ends up in the APK as |
|---|---|---|---|
| DSP skel (V69) | `hexagon/mega/*.c, *.S, mega.idl` | `hexagon/mega/build_skel.sh v69` (full Hexagon SDK in `~/hexsdk`) or `build.sh` (docker) | `libflux3skel.so`, `libltxskel.so`, `libr2vskel.so` (the same file, a name per model) |
| Klein runner | `runner/base.inc` + `runner/klein.inc` | `runner/build.sh` | `libflux3run.so` |
| Klein VAE decoder (QNN) | `runner/kvae.cpp`, graph `vae/` (`export_vae_klein.py`, `build_vae.sh`) | `runner/build_kvae.sh` | `libkvae.so` |
| Real-ESRGAN upscaler (viewer button) | `runner/kup.cpp` | `runner/build_kup.sh` | `libkup.so` |
| Ref2VA video VAE decoder | `hexagon/mega/vaerun.cpp` | `hexagon/mega/build_vaerun.sh` | `libr2vae.so` |

Order: `hexagon/mega/build.sh` (makes `out/mega_stub.c`, `out/mega.h` and the skel) -> the runners -> `tools/stage_jnilibs.sh` -> `app/build.sh`.
The Hexagon SDK and the QNN SDK are Qualcomm's and are not in this repository (`hexagon/mega/qhl_hmx` comes from the Hexagon SDK: `libs/qhl_hmx`).

Pack conversion (Klein weights from Black Forest Labs, 8-bit tiles for the HMX): `export_flux3.py` (DiT -> `img*.bin`, `txt*.bin`, `sgl*.bin`, `glue/`),
`mkblk_kl.py` (one block, for tests), `export_vae_klein.py` (VAE -> ONNX). The text encoder pack (`te/`) is the Qwen3-VL-4B pack of Mage-Flow (`mage/`).
`ref_*.py` and `cmp_*.py` are the PyTorch references the kernels were checked against. Paths of the weights come from `KLEIN_DIR` (default `klein`).
The Klein packs are published as `ayanoby/hexanima-klein-packs` on Hugging Face (FLUX Non-Commercial licence, see `NOTICE.md` there).
