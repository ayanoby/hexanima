# Changes

## 1.8.2

- **MiniMax-H3 Ref2VA video is in the download catalog** (`catalog10.json`, read by apps from 1.8.2; older apps keep `catalog9.json`; `tools/make_catalog10.py` builds it). It is listed under MiniMax-H3 on the Models screen, the pack (221 files, 37 GB, with `LICENSE` and `NOTICE`) is in `ayanoby/hexanima-r2v-packs` with a manifest of sizes and hashes, and the download asks for the MiniMax-H3 Community License first. MiniMax-H3-Image has to be installed before it (the video model uses its text encoder); the app says so. Copying the files by hand still works. Translations: the new strings in the 11 other languages.
- **SeedVR2 on the Models screen:** the Upscaler section has a SeedVR2 card (3.53 GB, Download / Delete, progress and cancel like the other downloads), next to Real-ESRGAN.

## 1.8.1

- **SeedVR2 downloads itself:** the first time you pick SeedVR2 in Upscale the app asks and downloads the model (3.6 GB, Apache 2.0) from `ayanoby/hexanima-upscale` (`seedvr2/manifest.json` lists sizes and hashes): resumable, verified, a free-space check first. No more copying files by hand. Translations: the 3 new strings in the 11 other languages.

## 1.8.0

A new **Upscale screen** (main menu ⋮ → Upscale) with pictures and video, a **picture-to-tags** button for Anima, **more frames per second** in a video (RIFE), and the SeedVR2 diffusion upscaler on the NPU.

- **Upscale screen:** an Image / Video switch on top. Image: your own picture (from the phone) or one from the app gallery (a grid of the whole gallery), the method (Real-ESRGAN photo or anime, ×2 / ×4, or SeedVR2 ×2 / ×3 / ×4) and the scale. The picture takes the whole width in its own aspect ratio; when the work is done the result appears **in its place with a before / after slider** (drag the handle, pinch to zoom into small details, one finger pans while zoomed); a tap opens the normal viewer with the whole gallery.
- **SeedVR2** (ByteDance Seed, Apache 2.0): a diffusion upscaler with the best quality, the DiT (int8) and the VAE tiles run on the NPU. The converted files are on Hugging Face (`ayanoby/hexanima-upscale`, `seedvr2/`) but are put on the phone by hand for now; the download from the Models screen comes later. Results up to about 4 megapixels.
- **More frames in a video (RIFE v4.6, hzwer, MIT):** Upscale → Video: pick a clip from the phone or the gallery, ×2 or ×4 frames per second; the network draws the missing frames on the NPU, the sound is kept, the result is an MP4 in the gallery. Clips up to 1280×768 (720p); a small network (12–40 MB, 15 sizes) is downloaded once per size from `ayanoby/hexanima-rife`. Hard cuts are not interpolated.
- **Picture to tags:** a photo button above the prompt field finds the Danbooru tags of a picture with the WD EVA02 Tagger v3 (SmilingWolf, Apache 2.0) on the NPU (about 3 seconds, a 674 MB context downloaded once from `ayanoby/hexanima-tagger`); the tags replace the prompt or are added to it.
- **Prompt format per model:** Templates now shows "Format for this model": the rule, a template and several examples of a good prompt for the selected model (tags for Anima, sentences for Qwen / Klein / Mage, the shot format for LTX, the H3 and Ref2VA formats).
- **Negative prompt for Turbo (NAG, experiment):** on Anima Turbo without CFG the text attention runs once more for the negative prompt (strength, limit tau, blend alpha); the kernel is in `hexagon/mega/blk.c` (scalar for now, about twice as slow as plain Turbo).
- **History:** the thumbnails no longer turn black after you come back from the viewer (the list is not rebuilt when nothing changed, and the thumbnail cache is limited by bytes).
- **Viewer:** the extra actions moved into one ⋮ menu.
- **Sources:** `seedvr2/` (reference, export, the `svsr` driver, notes), `flux3/hexagon/mega/svrun.cpp` (the DiT runner), `phone/svr.sh`; the RIFE and tagger conversion tools live in the research tree.
- **Licences:** the RIFE MIT text is bundled (Info → Licenses); SeedVR2 and the WD tagger (Apache 2.0) are listed in Info and `THIRD_PARTY.md`.
- Translations: the 82 new strings in the 11 other languages.

## 1.7.1

Two pictures of the same kind: **Mage-Flow edits with several photos** and **Anima pictures of any size** (up to 4096×4096) on the phone NPU.


- **Mage-Flow edits with up to three pictures** ("the fox from picture 1 sits in the car from picture 2"; the design was proposed in issue #6 by GrimmDev-Glitch): the "Pictures to edit" block of Qwen-Image works for Mage-Flow too. Every picture gets its own ViT features and DeepStack rows in the text encoder (`Image k: <|vision_start|>...`, interleaved mrope positions that continue after each picture), its own reference latent after the target tokens, and RoPE frame index k + 1. `mage/mk_edit_in.py` writes the same inputs for `magerun edit` on the command line.
- **Mage-Flow has no token limit any more** (the DiT runs in windows like LTX-2.5 and MiniMax-H3 Ref2VA, the attention streams the keys through VTCM): up to 24576 image tokens in all (the target and every picture, each picture has the size of the result), e.g. 2048x2048 (about 3.5 minutes) or 1024x768 with three pictures (2.3 minutes). The old limits (3200 patches for edits, 6144 for pictures) are gone; the time estimate on the button follows the measured step times.
- **Kernels:** the Mage modes (DiT 53, vision layer 55, VAE GEMM 61) moved into the shared kernel tree `flux3/hexagon/mega` (which also fixes a profiling array that was too small there); `mage/hexagon/mega` is now a README. The runner retries a call when the HMX self-test fails (rc -77) instead of stopping the picture.
- **Anima (2B and 2.9B) has no token limit any more** (it was 6912 patches, the attention kept all keys in the 8 MB VTCM): the attention streams the keys through VTCM in chunks (two passes: row max, then exp and P*V into fp32 accumulators, the port of the Klein/Mage kernel), and above 16384 tokens a block runs in windows of 4096 tokens (X, K and V hold all tokens, the other buffers one window: 16 KB per token instead of 48 KB; per block every window first makes its K and V, then runs the whole block). The app takes up to 65536 patches (4096×4096; the runner itself 98304); the time grows with the square of the token count: one block takes 2.35 s at 16384 tokens, 7.1 s at 32768 and 24 s at 65536. Measured end to end on the phone with the 2B model and a turbo LoRA (4 steps): 1536×1536 in 99 s (22 s per full step), 2560×2048 in 5.5 minutes (80 s per step); 2.9B at 1536×1536 in 2.2 minutes (31 s per step). Results are the same as before for the sizes that fitted (windows give bit-identical numbers, chunks differ by 4e-4 from the plain attention) and the time estimate on the button follows the square. Models are trained at about 1 Mpx, so very large pictures get soft and hazy (a property of the model, the same with and without windows). A runner that is done with a big picture frees its buffer so that only one of the chain's processes holds it. `hexagon/mega/blk.c` (attn_chunked, window meta M_W0/M_WN/M_PASS), `megarun.cpp`.
- **Picture area:** the box under the picture on the main screen takes exactly the shape of the picture (or of the chosen size before the run), also narrower than the screen for tall pictures, so there are no empty strips at the sides.
- Translations: the new strings in the 11 other languages (the size limit message now says 65536 patches).

## 1.7.0

- **FLUX.3 Image Klein and LTX-2.5 Video are in the download catalog** (`catalog9.json`, read by apps from 1.7.0; older apps keep `catalog8.json`). Klein (Black Forest Labs, 5.6B, Qwen3-VL-4B text encoder): photorealism and legible text, 16 steps without CFG, sizes 256 to 1536, about a minute for 512x512; a 9 GB pack in `ayanoby/hexanima-klein-packs`. LTX-2.5 (Lightricks, 22B distilled, Gemma 4 12B text encoder): text to video and image to video, 8 steps without CFG, 17-97 frames, no sound; a 31 GB pack in `ayanoby/hexanima-ltx-packs`. Each pack has `manifest.json` (size and sha256 of every file), `LICENSE` and `NOTICE.md`; the download asks for consent first (FLUX Kommunity License, non-commercial; LTX-2 Community License, Gemma terms). `tools/make_catalog9.py` builds the catalog. The Klein pack on Hugging Face used to lack 15 files of `glue/` (the model failed with "missing glue/x_img.f32"): they are uploaded and listed in the manifest.
- **LTX-2.5 video:** frames 17-97 and the playback frame rate of the MP4 are set on the main screen; sizes up to 24000 tokens (the activations live in phone memory in windows, only the keys and values of all tokens are mapped for the DSP). The number of steps works now (1-8, an even subset of the distilled schedule; before, 8 steps ran whatever was set). The result is an MP4 next to a preview frame in the history; a photo becomes the first frame.
- **MiniMax-H3 Ref2VA video** (33B, 4-step turbo LoRA, no CFG, 22-90 frames, steps 1-12): text to video and **reference photo to video** (the subject of the photo appears in the video). The video VAE and the reference encoder run on the NPU, the transformer runs in windows (no token limit; 1024x576 with 56 frames takes about 20 minutes). Not in the catalog: the converted files are put into `files/r2v` by hand (`phone/install_r2v.sh`), the text encoder is the one of MiniMax-H3-Image. MiniMax-H3 Community License.
- **Klein live preview:** the picture appears blurred after the first steps and gets sharper, like in the other models.
- **Video player:** under a finished video on the main screen a round play button, a pause button that fades out a second after the start (a tap on the video shows it again) and a full screen button at the bottom right. The history marks videos with a play mark; the viewer has the same player.
- **Upscaler:** Real-ESRGAN x4 and x2, photo and anime networks, on the NPU; a button in the picture viewer (a new picture next to the old one). The networks are two small downloads on the Models screen (BSD-3-Clause).
- **Main screen:** the picture area follows the shape of the picture it shows (a tall one is fitted into 70 % of the screen height).
- **Snapdragon 8 Elite (V79) code removed:** the experimental Klein path (skel, VAE context, QNN libs) is gone; the chip is not supported (see the README).
- **`app/build.sh` backup:** the copy of the in-app gallery and settings now goes to `/data/local/tmp/HexAnimaBackup` instead of `/sdcard`: the media scanner listed every backed-up picture and video in the phone's gallery.
- **Sources:** `flux3/` (the newest Hexagon kernel tree, shared by Klein, LTX and Ref2VA; the Klein runner and VAE; pack conversion), `ltx/` (runner, VAE, conversion, notes) and the Ref2VA runner, VAE and conversion in `h3/`, with READMEs and the paths of the build scripts relative to the repository, so that anyone can build the kernels (see `flux3/README.md`, `ltx/README.md`, `h3/NOTES_REF2VA.md`). `tools/stage_jnilibs.sh` stages the new binaries.
- **Licences:** the FLUX Kommunity License, the LTX-2 Community License Agreement, their notices and the Real-ESRGAN BSD-3 text are bundled in the app (Info -> Licenses) and listed in `THIRD_PARTY.md`.
- Translations: the 65 new strings in the 11 other languages.

## 1.6.0

- **MiniMax-H3-Image is in the download catalog** (`catalog8.json`, read by apps from 1.6.0; older apps keep `catalog7.json`). The 33B INT8-ConvRot model (6 steps without CFG, or 2 with the pDMD LoRA, sizes 512 to 2048) is a 48 GB pack in the public repository `ayanoby/hexanima-h3` with `manifest.json` (size and sha256 of every file), `LICENSE` and `NOTICE`. The download asks for consent first: MiniMax community licence, not for the US, EU, UK and South Korea, outputs must not be used to improve other AI models. `tools/make_catalog8.py` builds the catalog.
- **Updating the shared Anima files.** The catalog screen compares the shared files (text encoder, VAE, embeddings) with the catalog's manifest (sha256, kept in `common.json`; files installed before are hashed once) and shows an "Update" card with the size when something differs: it fixes installs that still have the 128 token text encoder (noise on the second picture with a long prompt, issue #3). The new files download beside the old ones (`name.new`, resumable) and replace them only when all are complete and verified, so an interrupted update loses nothing. A running Anima server restarts to pick them up.
- **Own wallpaper** (Settings): pick a picture, then place it (pinch to zoom, drag to move; "Position and scale" later), set the block transparency and the dimming; both sliders preview live. Every screen shows the wallpaper behind see-through blocks.
- **History: Pack all** puts every picture of the open tab (and model filter) into one ZIP (it opens everywhere; PNG is already compressed, so it costs almost nothing); the name is asked each time, the folder is chosen in Settings ("Where to save archives": Downloads or any folder), the archive can be shared at once; stopping removes the half-written file.
- **Prompt templates:** the history of recent prompts has no limit and "Clear all" (pictures stay); a star next to a recent prompt saves it (a filled star = saved); the negative prompt has its own templates; the buttons are bookmark icons.
- **Prompts are cleaned:** line breaks, tabs, no-break and zero-width spaces become one space and runs of spaces collapse, right in the field (a pasted prompt used to break at invisible line breaks and the rest never reached the encoder).
- **Main screen:** the picture area follows the shape of the chosen size; the token counters sit in the title row ("212 / 512"); pictures per run is one icon beside Generate (a tap opens − / + and a typed number); the model card says just "Unloaded" / "Loaded"; the "Settings" caption is gone; a faint "i" next to Steps, CFG, Sampler, Scheduler, Seed and LoRA opens a plain explanation (the same for every model); the CFG row is dimmed with a turbo model or LoRA (it still works, but turbo is made for CFG 1).
- **Info:** "About" is the first card.
- Translations: every new string in the 11 other languages.

## 1.5.6

- **Main screen layout:** the picture area is square and keeps its size whatever the resolution; the size row sits right under it (width ⇄ height, always editable, and a "Presets" button with the list of ready sizes, each group sorted from the smallest to the largest); the ratio segments and the "Size" caption are gone; the prompt field grows down with its text; the gaps between the blocks are smaller.
- **Token counters:** the positive and the negative prompt each have their own "Tokens: N / 512" line under their own field (they are encoded separately, so the numbers are not added up).
- **Anima LoRA list:** a pencil next to each LoRA renames it.
- **Model picker:** a pencil next to every model renames it (your own name, kept in `model_names.json`; Reset brings the original back; the two variants of an Anima model share the name).
- **Models screen:** the "Fits this phone" filter is gone; "Models on the phone" also counts the shared Anima files and the models converted on the phone; the catalog URL field (Source) opens right under the title; the own checkpoint button says BF16/FP16/FP32 and grows to two lines when needed.
- **History:** selecting shows a ring in the corner of every picture (empty = not chosen, ticked = chosen) and an accent frame on the chosen ones; fixed the crash on Select, Select all; the right column of tiles no longer sticks out of the screen.
- **Settings:** a "Keep metadata in the picture" switch (on by default; off = the PNG has no prompt/seed/model chunk, the app's gallery keeps its own copy) and a GitHub row (mark, repository, version).

## 1.5.5

- **Anima: any number of LoRA files at once.** The LoRA row is now a library: add as many `.safetensors` files as you need, each with its own switch, strength (0 to 2) and remove button (`AnimaLoras.java`). All active ones are folded into the blocks in list order when the model loads (the native side already took a `file*strength|file*strength` list); a new set or strength restarts the server and the first picture takes 10-15 seconds longer. A single LoRA picked with an older version moves into the list. MOSS-Image keeps its single LoRA.

## 1.5.4

- **Own Anima checkpoints (BF16 / FP16 / FP32 `.safetensors`) are converted on the phone** (`phone/animaconv.cpp`, 28 blocks in about 20 s, byte for byte the output of `export_all.py --q8`): button in the download catalog and in the model picker, the common files are offered for download first, "Own models" section in the catalog with details and Delete for each. 2B (28 blocks) and 2.9B (40 blocks) files, ComfyUI key prefixes (`net.`, `model.diffusion_model.`); fp8 files are not supported.
- Catalog: Qwen-Image 2.1 is shown as 7B (was 20B).
- **Anima: prompts up to 512 tokens run on the NPU.** The text encoder graphs were compiled for 128 tokens; a longer prompt fell back to the CPU encoder, whose weights file in the pack is a zero stub, so the picture ignored the text and the second one came out as noise. The slices are now built for 512 tokens (`export/export_te.py A B out.onnx 512`, +0.25 s per picture); `animad` picks 512 or 128 by the size of `te0.bin`, so old packs keep working (prompts over 128 tokens still fall back to the CPU).
- **Anima: the same prompt again is no longer encoded again.** A prompt over 128 tokens runs on the CPU text encoder (about 20 s) and then the LLM adapter (3 s); sd-cli and animad now keep the last 6 results, so another sampler, scheduler or seed for the same prompt starts at once (40 s -> about 12 s per picture).
- **Sampler, scheduler and CFG for every Anima model** (rows under Settings on the main screen; turbo models have their own CFG, default 1): euler (default), euler_a, er_sde, dpm++2m, dpm++2mv2, ipndm, lcm; scheduler default / simple / discrete / karras / exponential / sgm_uniform / bong_tangent (issue #3). Passed to the persistent sd-cli per picture (`sm=`, `sc=` job lines, `patches/sd-sampler.patch`; `SAMPLER`/`SCHED` env for `anima.sh`). Only single-pass samplers are offered (the NPU is asked once per step).
- **Common files with the 512 token text encoder:** new installs of the base files come from `common512/` through `catalog7.json` (about 0.9 GB, same size as before); the files in `common/` stay as they were for apps up to 1.5.3. An installation that already has the 128 token files keeps working (`animad` picks the size by `te0.bin`); delete and re-install the base files to get 512 tokens. `tools/make_catalog7.py`.
- Patches for sd.cpp: `patches/sd-sampler.patch` (sampler and scheduler per job), `patches/sd-te-cache.patch` (text encoder cache); `phone/build_conv.sh` builds the converter.
- Translations: every new string, and the 13 older ones that were still English (prompt enhancement, LoRA hint), in the 11 other languages.
- `app/build.sh` refuses to install without a device and copies the in-app gallery and settings to `/sdcard/HexAnimaBackup` first.

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
