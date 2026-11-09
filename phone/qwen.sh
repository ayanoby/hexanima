#!/system/bin/sh
# Qwen-Image-2.1 on the NPU (Qwen3-VL text encoder, DiT, VAE; editing adds the ViT and the VAE encoder): one picture per call, weights are streamed from flash.
#   qwen.sh IDS DROP W H SEED STEPS OUT_RAW [EDIT_DIR]    tokens (int32, DROP leading system tokens are removed) -> raw fp32 RGB [3][H][W] in [-1,1]; exit 0 on success
#   qwen.sh stop                                           kill the helpers
# EDIT_DIR (editing; further condition images in EDIT_DIR/1 and EDIT_DIR/2): meta.txt "CW CH", rgba.f32 (RGBA [4][CH][CW] in [-1,1]), px.f32 (ViT patches); ids already contain the expanded image slots
# env: Q (model dir, files/qwen: glue.bin embd.q4k te/ dit/ vit/ vae/{vae_t32.bin,enc_t32.bin} tok_*.txt), BIN (the app's native lib dir)
# The DSP skel has the same file name as Anima's (libmega_skel.so): it is copied to $Q/dsp and that dir comes first in ADSP_LIBRARY_PATH.
if [ "$1" = stop ]; then for p in libqwenrun.so libvaeenc.so libqvaetile.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; pkill -f 'tail -f .*vae.cmd'; exit 0; fi
IDS=$1 DROP=$2 W=$3 H=$4 SEED=$5 STEPS=$6 OUT=$7 ED=$8
LIBS=$BIN:/system/lib64:/vendor/lib64; ADSPP="$Q/dsp;$BIN"
RUNX='export LD_LIBRARY_PATH="$LIBS" ADSP_LIBRARY_PATH="$ADSPP"; exec "$@"'
mkdir -p $Q/dsp && cp -f $BIN/libqwenskel.so $Q/dsp/libmega_skel.so
mkdir -p $Q/cache   # text-encoder cache: the same prompt (and photo) with another seed skips the ViT and the encoder
rm -f $Q/qwen.log $Q/z.raw $Q/prev.bin $OUT
if [ -n "$ED" ]; then   # the condition images (up to 3: $ED, $ED/1, $ED/2): VAE encoder tiles on the NPU -> raw latents (kept while the photo and size stay the same; the app deletes them when it rewrites the photo files)
  : > $Q/venc.cmd
  for d in $ED $ED/1 $ED/2; do
    [ -s $d/meta.txt ] || continue
    [ -s $d/cond_raw.f32 ] && continue
    read CW CH < $d/meta.txt; printf 'run %s/rgba.f32 %s %s %s/cond_raw.f32\n' "$d" "$CW" "$CH" "$d" >> $Q/venc.cmd
  done
  if [ -s $Q/venc.cmd ]; then
    echo quit >> $Q/venc.cmd
    LIBS=$LIBS ADSPP=$ADSPP sh -c "$RUNX" sh $BIN/libvaeenc.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $Q/vae/enc_t32.bin < $Q/venc.cmd > $Q/qwen.log 2>&1
  fi
  for d in $ED $ED/1 $ED/2; do [ -s $d/meta.txt ] && { [ -s $d/cond_raw.f32 ] || exit 1; }; done
fi
# the decoder starts now and loads its 0.5 GB context while the transformer runs; the "run" command is appended to its command file afterwards
VAE=${VAE:-vae_t32.bin}; rm -f $Q/vae.cmd $Q/vae.log; : > $Q/vae.cmd
(tail -f $Q/vae.cmd | LIBS=$LIBS ADSPP=$ADSPP sh -c "$RUNX" sh $BIN/libqvaetile.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $Q/vae/$VAE > $Q/vae.log 2>&1 &)
printf 'load %s\ngen %s %s %s %s %s %s %s/z.raw\nquit\n' "$Q" "$IDS" "$DROP" "$H" "$W" "$SEED" "$STEPS" "$Q" | LIBS=$LIBS ADSPP=$ADSPP QWEN_EDIT="$ED" QWEN_CACHE=$Q/cache QWEN_PREVIEW=$Q/prev.bin sh -c "$RUNX" sh $BIN/libqwenrun.so >> $Q/qwen.log 2>&1
if [ -s $Q/z.raw ]; then printf 'run %s/z.raw %s %s %s\nquit\n' "$Q" "$W" "$H" "$OUT" >> $Q/vae.cmd; else printf 'quit\n' >> $Q/vae.cmd; fi
n=0; while pidof libqvaetile.so >/dev/null && [ $n -lt 900 ]; do sleep 0.2; n=$((n+1)); done
pkill -f "tail -f $Q/vae.cmd"; cat $Q/vae.log >> $Q/qwen.log 2>/dev/null
[ -s $OUT ]
