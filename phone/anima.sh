#!/system/bin/sh
# Anima 2B on the phone, no laptop: text encoder + sampler + decoder (sd.cpp, CPU) and DiT blocks/VAE (NPU) via the animad server.
#   anima.sh "prompt" [seed] [out.png]   generate (starts the server on first use, ~5 s one-time load)
#   anima.sh load | status | stop         load the NPU model / print loaded|unloaded / unload
#   anima.sh stop                         stop the server and free the NPU
# Stop button: the app touches $A/cancel and runs "stop"; the waiting loops below see the marker and exit 1.
# IDLE=<s>: unload the model after that many idle seconds.
# env: STEPS (default 4), PREFIX (quality tags), W/H (default 512x768; the NPU blocks are compiled for this size only)
# env A (work dir), S (model dir), BIN (dir with libanimad.so/libsdcli.so... = the app's native lib dir; unset = ./animad and ./sd-cli as root setup)
A=${A:-/data/local/tmp/anima}; S=${S:-/data/local/tmp/sd}; ROOT=$A/req; SQ=$A/sdq
if [ -n "$BIN" ]; then NA=libanimad.so NS=libsdcli.so; NH="libmegarun.so libvaerun.so libterun.so"; ANIMAD=$BIN/libanimad.so; SDCLI=$BIN/libsdcli.so; export ANIMAD_HOME=$A ANIMAD_BIN=$BIN LIBS=$BIN:/system/lib64:/vendor/lib64 ANIMAD_NOMLOCK=1; else NA=animad NS=sd-cli; NH="megarun vaerun terun"; ANIMAD=./animad; SDCLI=./sd-cli; LIBS=$A; fi
if [ -n "$MDIR" ]; then   # app mode: MID/MDIR come from the model manager (files/anima/MDIR/{small.bin,blkNN.bin}, stub files/sd/MID.safetensors)
  MODEL=$MID; DM=$MID.safetensors; SM=$MDIR/small.bin; export ANIMAD_HMXDIR=$MDIR
else
case "$MODEL" in   # 2.9B models only as q8 (29q, r29q): int8 block weights, 3.3 GB (fp16 needed 5.5 GB and thrashed RAM). 2 (default) = 3D Harem 2B, 29q = Harem 2.9B, r2 = Realskin 2B, r29q = Realskin 2.9B; switching needs "stop" first
  29q) DM=miaomiaoHarem_29BBETA11_turbo4_f16.safetensors; SM=small29.bin; export ANIMAD_HMXDIR=hmx29q;;
  r2) DM=miaomiaoRealskin_anima13_turbo11_f16.safetensors; SM=smallr2.bin; export ANIMAD_HMXDIR=hmxr2;;
  r29q) DM=miaomiaoRealskin_anima29B10_turbo4_f16.safetensors; SM=smallr29.bin; export ANIMAD_HMXDIR=hmxr29q;;
  *) DM=miaomiao3DHarem_LH3D11_turbo11_f16.safetensors; SM=small.bin;;
esac
fi
# not exported: system tools (setsid, grep...) must not see /vendor/lib64 (its old libc++ breaks them on newer Android); only our binaries get it, with /system/lib64 listed first so the system libc++ wins over the older vendor copy
RUNX='export LD_LIBRARY_PATH="$LIBS" ADSP_LIBRARY_PATH="$ADSPP"; exec "$@"'   # "env VAR=.. cmd" is not used: an app path contains "=" and env would take it for an assignment
ADSPP=${BIN:-$A}
if [ "$1" = stop ]; then touch $ROOT/quit; if pidof $NS >/dev/null; then printf quit > $SQ/job.tmp; mv $SQ/job.tmp $SQ/job; fi; pidof $NS >/dev/null && kill $(pidof $NS); n=0; while pidof $NA >/dev/null && [ $n -lt 20 ]; do sleep 0.25; n=$((n+1)); done
  # animad stuck in an aborted request never polls "quit": kill it and its children so the next load does not wait forever
  if pidof $NA >/dev/null; then kill -9 $(pidof $NA); fi; for p in $NH; do pidof $p >/dev/null && kill -9 $(pidof $p); done; rm -f $ROOT/quit; exit 0; fi
if [ "$1" = status ]; then if pidof $NA >/dev/null && grep -q serving $A/animad.log; then echo loaded; else echo unloaded; fi; exit 0; fi
# LORA=<file*strength|...>: Anima LoRAs (kohya, diffusers/PEFT, LyCORIS LoKr/LoHa) folded into the block weights when megarun loads them (hexagon/mega/lora_fold.h);
# their LLM-adapter and text-encoder parts are applied by animad (phone/lora_side.h)
KEY="${MODEL:-2}${MEGA_STREAM:+s}${LORA:+|$LORA|${LORA_S:-1}}"   # the low-memory (streaming) mode and the LoRAs are part of the key: changing them restarts the server
if pidof $NA >/dev/null && [ "$(cat $A/model 2>/dev/null)" != "$KEY" ]; then   # loaded server holds another model (or runs in the other memory mode): restart it
  sh $0 stop; while pidof $NA >/dev/null; do sleep 0.3; done
fi
if ! pidof $NA >/dev/null; then
  echo $KEY > $A/model
  rm -rf $ROOT; mkdir -p $ROOT
  # lcache: the folded blocks of the last LoRA set (q8); the next load with the same set reads them instead of folding again
  # tecache: the text encoder with the LoRAs' text-encoder weights merged in (F16; animad writes it and names it in $A/te_llm, sd.cpp then encodes on the CPU)
  rm -f $A/te_llm
  if [ -n "$LORA" ]; then export MEGA_LORA="$LORA" MEGA_LORA_S="${LORA_S:-1}" MEGA_LORA_CACHE=$A/lcache ANIMA_TE=$S/miaomiao3DHarem_animaLH3D10_txt.safetensors ANIMA_TE_CACHE=$A/tecache
  else unset MEGA_LORA MEGA_LORA_S MEGA_LORA_CACHE ANIMA_TE ANIMA_TE_CACHE; fi
  cd $A && LIBS=$LIBS ADSPP=$ADSPP setsid nohup sh -c "$RUNX" sh $ANIMAD $SM --serve $ROOT < /dev/null > $A/animad.log 2>&1 &
  while ! grep -q serving $A/animad.log 2>/dev/null && [ ! -f $A/cancel ]; do sleep 0.3; done
fi
# persistent sd-cli (text tokenizer/sampler/PNG writer stay loaded: saves ~0.7 s per image); ANIMA_NOSERVE=1 starts one per image as before
if [ -z "$ANIMA_NOSERVE" ] && ! pidof $NS >/dev/null; then
  rm -rf $SQ; mkdir -p $SQ; chmod 777 $SQ
  LLM=miaomiao3DHarem_animaLH3D10_txt.safetensors; [ -s $A/te_llm ] && LLM=$(cat $A/te_llm)
  cd $S && SD_SERVE=$SQ LIBS=$LIBS ADSPP=$ADSPP setsid nohup sh -c "$RUNX" sh $SDCLI --diffusion-model $DM --llm "$LLM" \
    -p x --sampling-method euler --cfg-scale 1 --steps 4 -W 512 -H 768 ${VAEARG:---taesd taew2_1.safetensors} --mmap -t ${T:-4} -o /sdcard/Download/anima_srv.png < /dev/null > $A/sdcli.log 2>&1 &
  while [ ! -f $SQ/ready ] && [ ! -f $A/cancel ]; do sleep 0.1; done
fi
if [ -f $A/cancel ]; then rm -f $A/cancel; exit 1; fi
[ "$1" = load ] && exit 0
touch $A/last_use
# idle unload: with IDLE=<seconds> a background watchdog stops the server after that long without a request (it frees the RAM; the next picture reloads, ~5 s)
if [ "${IDLE:-0}" -gt 0 ] && ! kill -0 "$(cat $A/wd.pid 2>/dev/null)" 2>/dev/null; then
  NA=$NA A=$A S=$S IDLE=$IDLE SELF=$A/anima.sh setsid nohup sh -c 'echo $$ > "$A/wd.pid"; while pidof $NA >/dev/null; do sleep 15; l=$(stat -c %Y "$A/last_use" 2>/dev/null || date +%s); if [ $(( $(date +%s) - l )) -ge "$IDLE" ]; then sh "$SELF" stop; break; fi; done; rm -f "$A/wd.pid"' < /dev/null > /dev/null 2>&1 &
fi
ID=r$(date +%s)$$
OUT=${3:-/sdcard/Download/anima_$ID.png}
mkdir -p $ROOT/$ID/vae && chmod -R 777 $ROOT/$ID && echo ${STEPS:-4} > $ROOT/$ID/start
V=${V:-1}   # the NPU VAE graph is 512x768; animad tiles larger latents through it
if [ -z "$ANIMA_NOSERVE" ]; then
  printf 'p=%s\ns=%s\nn=%s\no=%s\nd=%s\nw=%s\nh=%s\nv=%s\n' "${PREFIX:-best quality, score_7, score_9, very aesthetic, ultra detailed, fair skin, high contrast, anime coloring, }$1" "${2:--1}" "${STEPS:-4}" "$OUT" "$ROOT/$ID" "${W:-512}" "${H:-768}" "$V" > $SQ/job.tmp
  mv $SQ/job.tmp $SQ/job
  while [ ! -f $ROOT/$ID/sd_done ] && [ ! -f $ROOT/$ID/sd_failed ] && [ ! -f $A/cancel ]; do sleep 0.05; done
else
cd $S
LD_LIBRARY_PATH=$LIBS ADSP_LIBRARY_PATH=${BIN:-$A} SD_NPU_DIR=$ROOT/$ID SD_NPU_MODE=extern SD_VAE_DIR=$ROOT/$ID/vae SD_VAE_MODE=extern SD_TE_DIR=$ROOT/$ID SD_TE_MODE=extern $SDCLI \
  --diffusion-model $DM --llm miaomiao3DHarem_animaLH3D10_txt.safetensors \
  -p "${PREFIX:-best quality, score_7, score_9, very aesthetic, ultra detailed, fair skin, high contrast, anime coloring, }$1" \
  --sampling-method euler --cfg-scale 1 --steps ${STEPS:-4} -W ${W:-512} -H ${H:-768} ${VAEARG:---taesd taew2_1.safetensors} --mmap -t ${T:-4} -s ${2:--1} -o $OUT 2>&1 | grep -E "${LOGRE:-completed in|rror}"
fi
while [ ! -f $ROOT/$ID/finished ] && [ ! -f $A/cancel ]; do sleep 0.1; done
if [ -f $A/cancel ]; then rm -f $A/cancel; exit 1; fi   # the app stopped the server (Stop button): no picture
rm -rf $ROOT/$ID; touch $A/last_use
echo $OUT
