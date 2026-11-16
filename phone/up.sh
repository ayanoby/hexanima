#!/system/bin/sh
# Real-ESRGAN x4 (photo or anime variant) on the NPU through a QNN context: one picture per call.
#   up.sh IN_RGB W H OUT_RGB KIND HALF      IN: uint8 RGB [H][W][3]; OUT: uint8 RGB [4H][4W][3] (HALF=1: [2H][2W][3], the x4 result averaged 2x2); KIND = photo | anime; exit 0 on success
#   up.sh stop
# env: UM (dir files/upscale with kup_photo_t128.bin, kup_anime_t128.bin), BIN (the app's native lib dir)
if [ "$1" = stop ]; then pidof libkup.so >/dev/null && kill -9 $(pidof libkup.so); exit 0; fi
IN=$1 W=$2 H=$3 OUT=$4 KIND=$5 HALF=$6
rm -f $OUT
printf 'run %s %s %s %s %s\nquit\n' "$IN" "$W" "$H" "$OUT" "$HALF" | LD_LIBRARY_PATH=$BIN:/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH="$BIN" $BIN/libkup.so $BIN/libQnnHtp.so $BIN/libQnnSystem.so $UM/kup_${KIND}_t128.bin > $UM/up.log 2>&1
[ -s $OUT ]
