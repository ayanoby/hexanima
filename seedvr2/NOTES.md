# SeedVR2-3B on the phone NPU (notes)

Source: github.com/ByteDance-Seed/SeedVR (cloned in ../seedvr2-code), weights HF ByteDance-Seed/SeedVR2-3B (Apache 2.0), copy on the external disk
/run/media/artem/32363EC6363E8ABD/seedvr2/3b (seedvr2_ema_3b.pth 13.5 GB fp32, ema_vae.pth 1 GB, pos_emb.pt / neg_emb.pt = fixed text embeddings, no text encoder needed).
Python venv with torch (CPU) on the same disk: .../seedvr2/venv.

## One-step image SR flow (projects/inference_seedvr2_3b.py)
LR image -> resize to target area, divisible by 16 -> VAE encode -> cond latent c (16 ch, 8x down) -> x_T = randn; condition = [x_T (16), c (16), mask (1)] = 33 channels
-> ONE DiT pass (t = 1000, cfg 1, text = pos_emb.pt) -> x0 = x_T - v -> VAE decode -> wavelet colour fix against the input.

## DiT (NaDiT, configs_3b/main.yaml)
32 blocks, dim 2560, 20 heads x 128, SwiGLU expand 4 (hidden 6912 (int(2*2560*4/3)=6826 rounded up to 256)), RMSNorm (fusedrms eps 1e-5), qk RMSNorm (per head, affine), no qkv bias, patch (1,2,2) on the latent, in 33*4=132 -> 2560, out 2560 -> 64.
First 10 blocks have separate vid / txt weights (mm_layers 10), the other 22 share weights between vid and txt. Text: 58 tokens(?) x 5120 -> Linear -> 2560.
AdaSingle: one global emb = TimeEmbedding(t) -> 6*dim; per block + site (attn, mlp): shift/scale/gate = emb part + learned per-block vectors (shift, scale+..., gate). Applied as h*(scaleA+scaleB)+(shiftA+shiftB) after the (non-affine) norm, out = h*(gateA+gateB) before the residual.
Attention: windowed (window (4,3,3), methods alternate "720pwin_by_size_bysize" / "720pswin" = shifted by half a window): window size in tokens = ceil(sqrt(3600/(h*w))*h/3) x ceil(...*w/3), about 20x20 = 400 tokens whatever the picture size. Per window q/k/v of the window tokens + the text tokens (repeated per window), mmrope3d, text outputs of all windows are averaged ("text pooling").
Last block: txt has no MLP / ada (vid_only). vid_out_norm (fusedrms, affine) + ada "out" (shift/scale) then Linear to 64.
Cost: ~105 M params per block (3.4 B total), int8 pack ~3.5 GB; 1024x1024 output = 4096 tokens; attention is local so cheap.

## VAE (video_vae_v3, s8 c16 t4, SD3-like inflated causal Conv3d)
channels 128/256/512/512, 2 res blocks per level, groupnorm 32, mid attention, latent 16, no quant convs. For one frame the causal conv3d reduces to a 2D conv (check which temporal slice(s) count).

## Findings while reading the code
- SwiGLU hidden = int(2*dim*4/3) rounded up to 256 = 6912 (not 10240): ~81 M params per block.
- The text stream (58 tokens, 5120 -> 2560) is processed by ALL blocks; per window the text q/k/v are concatenated with the window's q/k/v, text attention outputs are averaged over the windows ("text pooling"). The out ada (vid_out_ada) reuses the cached attn-site emb (cache key emb_repeat_0_vid collides) -> out shift/scale = global emb chunk 0 + learned.
- rope: 126 of 128 head dims rotated, 3 x 42, interleaved pairs; T index = text length (58), H/W = position INSIDE the window; text rows: position i on all 3 axes.
- VAE for one frame: Conv3d with replicate-padded first frame == Conv2d with the weight summed over time (verified: encode+decode of a 512x768 picture gives PSNR 37.9 dB, ref_vae.py). Decoder upsampler keeps the LAST temporal copy of the (x y z c) pixel shuffle.

## NPU plan
- Window order: the host builds, per layer, x_win = for each window [vid tokens | 58 text tokens | zero pad to a 32 multiple]; every window is slab aligned, attention per window = existing attn_run(nq = window slabs, nkv = same, nvalid = n + 58) with trailing pad only.
- Two DSP calls per layer: A (RMS+ada, qkv with qk RMS + rope in the epilogue, attention, out proj + residual) -> host: text rows := mean over windows; B (RMS+ada, SwiGLU MLP, residual). Text attention pooling is linear in proj_out so the mean can be taken after the residual.
- Between layers the host permutes the vid rows (alternating unshifted / shifted partition).
- int8 pack like mage/export_mage.py (q8 tiles [N/32][K/32+4]), kernel in a copy of the flux3 mega tree with DIM = 2560 (NCB 80), RMSNorm variant of k_lnmod, SwiGLU epilogue.
- VAE: ONNX -> QNN like the Real-ESRGAN / Klein VAE, tiles with overlap.

## Status 2026-11-17: working end to end on the phone (app: viewer -> HQ -> "x2 / x3 / x4 SeedVR2")
- DSP: MODE 100 in flux3/hexagon/mega/blk.c (+ kern.c k_pool_acc/k_pool_fin, attn_win, p_rms2); host driver svrun.cpp (window layouts, rope tables, token permutation between layers, vid_in / final projection on the CPU);
  orchestrator svsr.cpp (bicubic resize, VAE encoder tiles 512 px / overlap 128, DiT via svrun, decoder tiles, blending); VAE tiles are QNN graphs (export_vae_svr.py + build_vae_qnn.sh; the decoder upsampler uses PixelShuffle, a 7-D reshape broke the QNN graph).
- Accuracy: per layer 0.3-0.4 % vs the q8 PyTorch reference (layers 0-9), relative error grows smoothly to 12 % at layer 31 (corr 0.995 for the final velocity); the pictures are indistinguishable from the CPU reference.
- Speed (phone): 512x768 -> 1024x1536 (x2): 30 s; 1024x1024 -> 2048x2048: 69 s (encode 10, DiT 37, decode 21). Limit 16384 tokens (about 4.2 MP): FastRPC mapping.
- Files in the app: files/seedvr2/pack/blkNN.bin + glue/, vae/svr_enc_t64.bin, svr_dec_t64.bin (3.6 GB); not in the catalog yet (publishing needs the user's OK).
