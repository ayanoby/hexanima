#!/system/bin/sh
# LTX-2.5 22B distilled text-to-VIDEO on the NPU: Gemma 4 12B text encoder + connector, 48-block video DiT (8 steps), QNN VAE; the packs are streamed from flash (12.9 GB of DiT weights per step).
#   ltx.sh IDS_FILE FL HL WL SEED OUT_RAW [IMG] tokens (int32, BOS first) -> raw fp32 video [PF][3][32*HL][32*WL] in [-1,1] (PF = 8*(FL-1)+1 frames); exit 0 on success
#                                                IMG (optional): raw fp32 image [3][32*HL][32*WL] in [-1,1] = the first frame (image to video; vae/etile.bin encodes it, the DiT runs frame 0 at timestep 0)
#   ltx.sh stop                                kill the helpers
# env: STEPS (1..8, default 8: an even subset of the distilled schedule), LM (model dir, files/ltx: te/ conn/ glue/ dit/ vae/vtile.bin tok.bin), BIN (the app's native lib dir). The DSP skel has the file name libmega_skel.so: it is copied to $LM/dsp, which comes first in ADSP_LIBRARY_PATH.
if [ "$1" = stop ]; then for p in libltxrun.so liblvae.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; exit 0; fi
IDS=$1 FL=$2 HL=$3 WL=$4 SEED=$5 OUT=$6 IMG=$7
LIBS=$BIN:/system/lib64:/vendor/lib64; ADSPP="$LM/dsp;$BIN"
mkdir -p $LM/dsp && cp -f $BIN/libltxskel.so $LM/dsp/libmega_skel.so
rm -f $LM/ltx.log $LM/z.f32 $LM/ctx.vc.f32 $LM/cond.f32 $OUT
COND=
if [ -n "$IMG" ]; then
  printf 'enc %s %s %s %s/cond.f32\nquit\n' "$IMG" "$HL" "$WL" "$LM" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/liblvae.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $LM/vae/etile.bin 2>&1 | sed 's/^lvae:/lenc:/' > $LM/ltx.log
  [ -s $LM/cond.f32 ] || exit 1
  COND="- $LM/cond.f32"
fi
printf 'load %s\nte %s %s/ctx\ndit %s/ctx.vc.f32 %s %s %s %s %s %s/z.f32 %s\nquit\n' "$LM" "$IDS" "$LM" "$LM" "$FL" "$HL" "$WL" "$SEED" "${STEPS:-8}" "$LM" "$COND" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/libltxrun.so >> $LM/ltx.log 2>&1
[ -s $LM/z.f32 ] || exit 1
printf 'dec %s/z.f32 %s %s %s %s\nquit\n' "$LM" "$FL" "$HL" "$WL" "$OUT" | LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=$ADSPP $BIN/liblvae.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $LM/vae/vtile.bin >> $LM/ltx.log 2>&1
[ -s $OUT ]
