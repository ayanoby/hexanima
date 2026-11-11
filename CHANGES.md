# Changes

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
