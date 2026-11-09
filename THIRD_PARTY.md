# Third-party components

This repository contains only HexAnima's own source code (MIT, see `LICENSE`). Nothing below is stored here.

| Component | Used for | License |
|---|---|---|
| Anima (CircleStone Labs), built on NVIDIA Cosmos | base model | CircleStone Labs Non-Commercial License; NVIDIA Open Model License |
| Anima Turbo v1.1 Lora (somedoby) and TURBO for ANIMA V4-29B (n_Arno), both on Civitai | distillation LoRAs merged into the downloadable models (4-step generation); the LoRA files are not redistributed | Civitai terms of each model page (credit required for n_Arno's), plus the CircleStone Labs Non-Commercial License v1.2 for the Anima Turbo derivative |
| MiaoMiao Harem / 3D Harem / RealSkin (MIAOKA) | downloadable models | author's terms: non-commercial, no selling of images |
| Qwen3 0.6B text encoder, Qwen-Image VAE | text encoder, VAE | Apache 2.0 |
| MOSS-Image-SFT (OpenMOSS), Unsloth GGUF quantisation, rapid-moss 4-step adapter (darask0, on lodestones' 8-step DMD), Qwen3.5-9B (Qwen team) with Unsloth GGUF quantisation, FLUX.2 small decoder (Black Forest Labs) | the optional MOSS-Image pack (converted for the NPU, hosted in its own repository with `LICENSE` and `NOTICE.md`) | Apache 2.0 (each component) |
| Qwen-Image-2.1-Turbo (Qwen team) with the Unsloth Q5_K_S GGUF quantisation, Lightning v2-RL 4-step adapter (LightX2V), Qwen-Image-2.1 VAE | the optional Qwen-Image pack (generation and photo editing; converted for the NPU, hosted in its own repository with `LICENSE` and `NOTICE.md`) | Qwen Research License Agreement (non-commercial only) |
| Qwen3-VL-8B-Instruct (Qwen team) with Unsloth GGUF quantisation (text encoder and vision tower of the Qwen-Image pack) | same pack | Apache 2.0 |
| Mage-Flow-Edit-Turbo transformer and Mage-VAE (Microsoft) | the optional Mage-Flow pack (text-to-image and photo editing; converted for the NPU, hosted in its own repository with `LICENSE` and `NOTICE.md`) | MIT |
| Qwen3-VL-4B (Qwen team): text encoder and vision tower of the Mage-Flow pack | same pack | Apache 2.0 |
| stable-diffusion.cpp | sampler and CPU parts (`patches/` are applied to it) | MIT |
| TAEHV (`taew2_1`) | preview decoder | see github.com/madebyollin/taehv |
| Danbooru tag list (DominikDoom/a1111-sd-webui-tagcomplete) | prompt suggestions | see that repository |
| Qualcomm AI Engine Direct (QNN) runtime libraries | bundled in the APK | Qualcomm AI Stack License |
| Qualcomm Hexagon SDK / QHL (HMX matrix library) | building the DSP kernels | Qualcomm license, not redistributed here |

## MOSS-Image notice

All MOSS-Image components are under the Apache License 2.0. The pack is converted and quantised (8-bit tiles for the Hexagon matrix unit, the rapid-moss LoRA kept unmerged), see the pack's `NOTICE.md`.
The Apache License 2.0 text and that notice are bundled in the app (`app/assets/licenses/apache-2.0.txt`, `moss_notice.txt`) and shown under Info -> Licenses.
A user-supplied LoRA is a separate work under its author's license; HexAnima only folds it with the adapter on the phone (nothing is uploaded or redistributed).

## Qwen-Image notice

Built with Qwen. The Lightning v2-RL adapter's repository (lightx2v/Qwen-Image-2.1-Lightning) declares no license; it is an adapter of Qwen-Image-2.1 and is treated as a derivative of the Qwen Materials (Qwen Research License, non-commercial).

Qwen-Image-2.1-Turbo, the Lightning v2-RL adapter and the Qwen-Image-2.1 VAE are licensed under the Qwen RESEARCH LICENSE AGREEMENT (Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd.): non-commercial use only.
The pack is converted and quantised (8-bit and lossless 4-bit tiles for the Hexagon matrix unit, the Lightning LoRA kept unmerged, the vision tower's heads zero-padded, the VAE re-written as fp16 Hexagon contexts); the changes are listed in the pack's `NOTICE.md` and in `app/assets/licenses/qwen_notice.txt`.
The license text is bundled in the app (`qwen_research_license.txt`) and shown under Info -> Licenses. Sources of the conversion are in `qwen/`.

## Mage-Flow notice

Mage-Flow-Edit-Turbo and its VAE are Copyright (c) Microsoft Corporation, licensed under the MIT License; the Qwen3-VL-4B text encoder and vision tower that ship with it are under the Apache License 2.0.
The pack is converted and quantised (8-bit tiles for the Hexagon matrix unit, the modulation of the fixed 4-step schedule computed in advance, the VAE re-ordered for the app's kernels); the changes are listed in the pack's `NOTICE.md` and in `app/assets/licenses/mage_notice.txt`.
The MIT license text and that notice are bundled in the app and shown under Info -> Licenses. The official pipeline's content filter and watermark are not part of the app's runner.

## CircleStone Model notice

The downloadable model files are modified versions of the CircleStone Model (Anima): MIAOKA's fine-tunes with a turbo LoRA merged in, converted to HexAnima's block layout.
They are not an official product of CircleStone Labs and have not been endorsed by it. The Attribution Notice, the modification statement and the license text are published with the model packs
at https://huggingface.co/ayanoby/anima-npu-packs (`NOTICE.txt`, `LICENSE.md`).

## Qualcomm components

- The QNN runtime libraries (`libQnnHtp*.so`, `libQnnSystem.so`) are **not** in this repository and must never be added to it or published on their own:
  the Qualcomm AI Stack License allows distributing them only in object code and only as part of your own application (not standalone).
  `tools/stage_jnilibs.sh` copies them from a QNN SDK that you download from Qualcomm and accept the license for.
  The license text and notices are shown in the app (Info -> Licenses) and live in `app/assets/licenses/`.
- `hexagon/mega/build.sh` needs the Hexagon SDK and the QHL HMX sources (`hexagon/mega/qhl_hmx`) plus `syslib/libcdsprpc.so` from a device.
  They are Qualcomm proprietary files and are not included; get them from the SDK you are licensed for.
