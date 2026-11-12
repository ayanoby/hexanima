#!/system/bin/sh
# MiniMax-H3-Image on the NPU (text encoder, DiT, VAE): one picture per call, the 70 GB-class weights are streamed from flash.
#   h3.sh IDS_FILE W H SEED OUT_RAW      tokens (int32) -> raw fp32 RGB [3][H][W] in [-1,1]; exit 0 on success
#   h3.sh stop                           kill the helpers
# env: HM (model dir, files/h3: model/ te/ vae/vae_h3.bin), BIN (the app's native lib dir)
# The DSP skel has the same file name as the other models' (libmega_skel.so): it is copied to $HM/dsp and that dir comes first in ADSP_LIBRARY_PATH.
if [ "$1" = stop ]; then for p in libh3run.so libh3vae.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; exit 0; fi
IDS=$1 W=$2 H=$3 SEED=$4 OUT=$5
LIBS=$BIN:/system/lib64:/vendor/lib64; ADSPP="$HM/dsp;$BIN"
mkdir -p $HM/dsp && cp -f $BIN/libh3skel.so $HM/dsp/libmega_skel.so
rm -f $HM/h3.log $HM/z.f32 $OUT
LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP H3_TE=$HM/te H3_DENORM=1 $BIN/libh3run.so $HM/model $((H / 16)) $((W / 16)) seed:$SEED $IDS 0 $HM/z.f32 > $HM/h3.log 2>&1
[ -s $HM/z.f32 ] || exit 1
printf 'run %s/z.f32 %s %s %s\nquit\n' "$HM" "$W" "$H" "$OUT" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libh3vae.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $HM/vae/vae_h3.bin >> $HM/h3.log 2>&1
[ -s $OUT ]
