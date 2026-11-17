#!/system/bin/sh
# SeedVR2-3B (one-step diffusion super-resolution, Apache-2.0) on the NPU: one picture per call.
#   svr.sh IN_RGB W0 H0 OUT_RGB W H [SEED]    IN: uint8 RGB [H0][W0][3] (resized bicubically to W x H inside); OUT: uint8 RGB [H][W][3]; exit 0 on success
#   svr.sh stop
# env: S (dir files/seedvr2: pack/ vae/ dsp/ work/), BIN (the app's native lib dir). Progress lines ("stage encode 1 2", "stage dit 1 1", "stage decode 3 6") go to $S/svr.log.
if [ "$1" = stop ]; then for p in libsvsr.so libsvrun.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; exit 0; fi
IN=$1 W0=$2 H0=$3 OUT=$4 W=$5 H=$6 SEED=${7:-666}
mkdir -p $S/dsp $S/work; cp -f $BIN/libsvrskel.so $S/dsp/libmega_skel.so
rm -f $OUT $S/svr.log
SV_LIB=1 SV_BIN=$BIN LD_LIBRARY_PATH=$BIN:/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH="$S/dsp;$BIN" $BIN/libsvsr.so $S/work $S/pack $S/vae $IN $W0 $H0 $OUT $W $H $SEED > $S/svr.log 2>&1
[ -s $OUT ]
