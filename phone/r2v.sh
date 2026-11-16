#!/system/bin/sh
# MiniMax-H3 Ref2VA text (+ reference photo) to VIDEO on the NPU: image encoder (QNN) -> text encoder + 50-block DiT (4 steps, turbo LoRA, windowed) -> video VAE decoder; the packs are streamed from flash (32 GB of DiT weights per step).
#   r2v.sh IDS_FILE T LH LW SEED OUT_RAW [IMG IH IW]   tokens (int32) -> raw fp32 video [F][3][16*LH][16*LW] in [-1,1] (F = 17*(T-2)/5 + 5 frames), exit 0 on success
#        T = latent frames (5n + 2), LH/LW = pixels / 16; IMG (optional): raw fp32 reference photo [3][IH][IW] ImageNet-normalised (IH, IW multiples of 16, >= 256)
#   r2v.sh stop                                 kill the helpers
# env: LM (model dir, files/r2v), BIN (the app's native lib dir), TE (the H3 text-encoder dir, files/h3/te). The DSP skel has the file name libmega_skel.so: it is copied to $LM/dsp, which comes first in ADSP_LIBRARY_PATH.
if [ "$1" = stop ]; then for p in libr2vrun.so libr2vae.so libr2enc.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; exit 0; fi
IDS=$1 T=$2 LH=$3 LW=$4 SEED=$5 OUT=$6 IMG=$7 IH=$8 IW=$9
LIBS=$BIN:/system/lib64:/vendor/lib64; ADSPP="$LM/dsp;$BIN"
mkdir -p $LM/dsp && cp -f $BIN/libr2vskel.so $LM/dsp/libmega_skel.so
rm -f $LM/r2v.log $LM/ref.f32 $LM/lat.f32 $LM/lat.f32.ckpt* $OUT
REF=
if [ -n "$IMG" ]; then
  printf 'enc %s %s %s %s/ref.f32 %s/vae/glue\nquit\n' "$IMG" "$IH" "$IW" "$LM" "$LM" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libr2enc.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $LM/enc/henc.bin 2>&1 | sed 's/^henc:/renc:/' > $LM/r2v.log
  [ -s $LM/ref.f32 ] || exit 1
  REF="$LM/ref.f32:$((IH / 16)):$((IW / 16))"
fi
R2V_REF=$REF H3_LORA=$LM/lora R2V_STEPS=${STEPS:-4} H3_TE=$TE LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libr2vrun.so $LM $LH $LW $T seed:$SEED $IDS 0 0 $LM/lat.f32 50 >> $LM/r2v.log 2>&1
[ -s $LM/lat.f32 ] || exit 1
VAE_PM1=1 LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libr2vae.so $LM/vae $T $LH $LW $LM/lat.f32 $OUT >> $LM/r2v.log 2>&1
rm -f $LM/lat.f32 $LM/lat.f32.ckpt* $LM/ref.f32
[ -s $OUT ]
