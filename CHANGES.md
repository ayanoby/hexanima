# Changes

## 1.5.3

- **MiaoMiao Anima Harem MOSS 1.1** (MIAOKA, Civitai) as a model next to MOSS-Image and RealSkin MOSS (`files/moss/hm`). Fixed on the way: the checkpoint carries the fp16-fix scaling without a flag, `export_moss_orig.py` now takes `MOSS_FP16C`.
- **Download catalog:** RealSkin MOSS 1.0 and Harem MOSS 1.1 are listed in the new `catalog6.json` (the app tries it before `catalog5.json`; older apps keep reading `catalog5.json` and see nothing new). The packs are in `ayanoby/hexanima` on Hugging Face and need the MOSS-Image 8 step base (`moss-sft-q5`).
- RealSkin MOSS: "8 step · DMD" next to the 4 step rapid-moss in the Turbo LoRA row (uses the installed MOSS-Image 8 step add-on; the mode is the `rs_mode` preference).
- **MiniMax-H3-Image (33B) on the NPU, experimental, local files only.** INT8-ConvRot DiT (50 blocks), Qwen3-VL text encoder and the ViT VAE all run on the phone, the 70 GB of weights are streamed from flash (about 2.5 GB/s): from roughly 5 minutes at 1024x1024. The model shows up in the manager once `files/h3` is converted with `h3/export_h3*.py` (no download pack: the MiniMax license wants its text with every copy and excludes some regions). Sizes: multiples of 32 px, 256 to 2048.
  - Compact weight format (-12.5 % flash read): the per-row scale is stored once per tile column, no requantization.
  - **1 to 6 steps** by hand (the model's own 6-step schedule is sampled at the chosen number of steps).
  - **pDMD 2-step LoRA** (akhaliq / pdmd, 2-4 steps, rank 128) as a separate delta in the GEMM epilogue, picked in the Turbo LoRA row.
  - Known issue: with the pDMD LoRA below about 2 MP a fine grid/hatching texture can show up; the 4-step result with a lower shift looked better in a first test, not changed yet.
- `tools/make_catalog6.py` builds the catalog from the pack manifests.

## 1.5.2

- **Queue:** any number of pictures per run (up to 999; tap the number to type it), and the number can be raised while a batch is running: the new pictures join the queue.
- **Progress:** when the text stage was cached or skipped, the status no longer stays on "Encoding the text..." until the picture is ready; the first finished step switches it to "Step N/M" for Anima, Qwen-Image, Mage-Flow and MOSS-Image.
- **MiaoMiao RealSkin MOSS 1.0** (a fine-tune of MOSS-Image-SFT by MIAOKA, Civitai) as a separate model next to MOSS-Image: its own DiT / refiner / glue (`files/moss/rs`, about 5.9 GB), the text encoder, VAE and tokenizer are the MOSS-Image pack's. `moss/export_moss_orig.py` reads the single fp16 file (fused qkv) and undoes the "fp16 fix" scaling of such checkpoints, `moss/make_glue_ft.py` rebuilds glue.bin.

## 1.5.1

- Fixed: the live preview was missing in 1.5.0 (the four `preview_*.bin` files were left out of the build); it is back for all models.
- All new screens and the Info texts are translated into the 11 other languages (112 strings each); 1.5.0 showed them in English.
- MOSS-Image: the status line no longer sticks on "Preparing the text..." when the text was cached.

## 1.5.0

The model manager and the generation screen now follow one scheme for every model family: **you download a base model, and speed comes from a separate LoRA file that you pick**. Existing pictures, history and settings stay; the old merged "Turbo" checkpoints of Anima are no longer used (delete them in the manager to free about 2-4 GB each).

### Turbo LoRA from a file, steps by hand
- A **Turbo LoRA** row under the model: pick any compatible `.safetensors` file (Anima: kohya / diffusers / LyCORIS; Qwen-Image: Lightning and similar; MOSS-Image: rapid-moss 4 step, the DMD 8 step LoRA and others). The file is checked at once and the app says why if it does not fit. A strength slider goes with it. "None" runs the plain model.
- The **number of steps is entered by hand** for every model (up to 50 for Anima and Qwen-Image; MOSS-Image is limited to the 4 or 8 step codes its adapters were trained with; Mage-Flow stays at 4). Pick the steps the LoRA was made for, usually 4 or 8.
- Anima: a Turbo LoRA and a regular LoRA are folded into the weights together on load. The first picture with a new set (or strength) waits about 12-13 s ("Preparing the LoRA..."), the folded 8-bit blocks (about 2 GB) are cached for the last set. The fold runs in parallel and on NEON tiles (25 s before, byte-identical result).
- Qwen-Image: the LoRA is applied through the 4 spare K tiles of every weight panel (rank up to 128) as a small overlay (about 16 MB cache) instead of a second 6 GB checkpoint. The plain model is named **Qwen-Image-2.1-Turbo**; the old 4 step pack is only the Lightning LoRA on top of it.
- MOSS-Image: **8 step DMD** option (LoRA from lodestones) next to the 4 step one, as a small add-on (about 0.4 GB); a user LoRA replaces the Turbo slot.
- The model name no longer changes with the chosen LoRA or steps.

### Base models and models
- New Anima **base** models (Harem 2.9B / 1.6 2B / 3D Harem / RealSkin 2B and 2.9B) with CFG, negative prompt and 20-30 steps; the 2B turbo models are int8 (2.3 GB instead of 3.9 GB, about 11 s).
- MOSS-Image from the original OpenMOSS weights (the DiT is no longer taken from the Unsloth GGUF).
- Anima sizes up to 6912 patches (for example 1152x1536 in 77 s on the 2.9B model).
- Qwen-Image: much lower memory while editing (the 12 GB phone peaks at 1.9 GB instead of 5 GB with three pictures), sizes up to 6144 patches.

### Interface
- Old merged Turbo packs (Anima turbo checkpoints, the 4 step Qwen pack) are listed under "Old version" at the bottom of the manager; new users see only base models.
- Redesigned main screen: one column, model card, picture, prompt, size segments, parameters, RAM and temperature in the header; light, dark and AMOLED themes.
- New model manager: search, filters (installed / fits this phone), one entry per model with its variants, storage summary.
- Info screen rewritten for the new scheme.
- The first MOSS-Image 4-px grid filter (DeGrid) is removed; the LLM prompt enhancer is removed.

### Under the hood
- `hexagon/mega/lora_fold.h`: tiled 32x32 delta (NEON 4x16 register blocking), NEON dequant/requant of q8 tiles (`LORA_SCALAR=1` selects the scalar path), parallel block folding in `megarun` (3 workers).
- `qwen/qwenrun`: LoRA slots, sigma schedule resampled to any number of steps, `qwenfold` builds the overlay on the phone; `moss/mossrun`: `mode 4|8`, `lorafold replace`.
- `tools/build_catalog5.py` builds the new catalog (packs, models, modes).
