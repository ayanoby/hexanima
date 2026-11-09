#!/system/bin/sh
# Mage-Flow-Edit-Turbo on the NPU (Qwen3-VL-4B text encoder, 12-block DiT, Mage-VAE; editing adds the ViT and the VAE encoder): one picture per call.
#   mage.sh IDS DROP W H SEED STEPS OUT_RAW [EDIT_DIR]   tokens (int32) -> raw fp32 RGB [H][W][3] in [-1,1]; exit 0 on success
#   mage.sh stop
# EDIT_DIR: meta.txt "gh gw h w", px.f32, ref.rgb (see Mage.java). env: M (model dir, files/mage), BIN (native lib dir)
if [ "$1" = stop ]; then pidof libmagerun.so >/dev/null && kill -9 $(pidof libmagerun.so); exit 0; fi
IDS=$1 DROP=$2 W=$3 H=$4 SEED=$5 STEPS=$6 OUT=$7 ED=$8
mkdir -p $M/dsp $M/cache; cp -f $BIN/libmageskel.so $M/dsp/libmega_skel.so
rm -f $M/mage.log $M/z.f32 $M/prev.bin $OUT
if [ -n "$ED" ]; then CMD="edit $ED $DROP $SEED $STEPS $M/z.f32"; else CMD="genp $IDS $DROP $H $W $SEED $STEPS $M/z.f32"; fi
printf 'load %s\n%s\nquit\n' "$M" "$CMD" | LD_LIBRARY_PATH=$BIN:/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH="$M/dsp;$BIN" MAGE_CACHE=$M/cache MAGE_PREVIEW=$M/prev.bin MAGE_RGB=$OUT $BIN/libmagerun.so > $M/mage.log 2>&1
[ -s $OUT ]
