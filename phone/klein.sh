#!/system/bin/sh
# FLUX.3 Image Klein on the NPU (Qwen3-VL-4B text encoder, 5.6B DiT, QNN VAE): one picture per call, the packs are streamed from the page cache / flash.
#   klein.sh IDS_FILE W H SEED STEPS OUT_RAW      tokens (int32) -> raw fp32 RGB [3][H][W] in [-1,1]; exit 0 on success
#   klein.sh stop                                  kill the helpers
# env: KM (model dir, files/klein: te/ img*.bin txt*.bin sgl*.bin glue/ vae/kvae_t32.bin), BIN (the app's native lib dir)
# The DSP skel has the same file name as the other models' (libmega_skel.so): it is copied to $KM/dsp and that dir comes first in ADSP_LIBRARY_PATH.
if [ "$1" = stop ]; then for p in libflux3run.so libkvae.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; exit 0; fi
IDS=$1 W=$2 H=$3 SEED=$4 STEPS=$5 OUT=$6
LIBS=$BIN:/system/lib64:/vendor/lib64; ADSPP="$KM/dsp;$BIN"
SK=libflux3skel.so
mkdir -p $KM/dsp && cp -f $BIN/$SK $KM/dsp/libmega_skel.so
rm -f $KM/klein.log $KM/z.f32 $OUT
NT=$(( $(wc -c < $IDS) / 4 ))
printf 'load %s\nglue %s/glue\nte %s %s/feat.f32\ndit %s/feat.f32 %s %s %s %s %s %s/z.f32\nquit\n' "$KM" "$KM" "$IDS" "$KM" "$KM" "$NT" "$((H / 16))" "$((W / 16))" "$SEED" "$STEPS" "$KM" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libflux3run.so > $KM/klein.log 2>&1
[ -s $KM/z.f32 ] || exit 1
printf 'runtok %s/z.f32 %s %s %s\nquit\n' "$KM" "$W" "$H" "$OUT" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libkvae.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $KM/vae/kvae_t32.bin $KM/glue/vae_bn.f32 >> $KM/klein.log 2>&1
[ -s $OUT ]
