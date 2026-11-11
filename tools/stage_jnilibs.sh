#!/bin/bash
# copies the native binaries the APK bundles into app/jnilibs/lib/arm64-v8a (executables are named lib*.so: an app may only exec from its native lib dir)
set -e; cd "$(dirname "$0")/.."; J=app/jnilibs/lib/arm64-v8a; Q=~/anima-npu/sdk/qairt/2.28.0.241029/lib; mkdir -p $J
cp phone/animad $J/libanimad.so; cp hexagon/mega/out/megarun $J/libmegarun.so; cp hexagon/mega/out/libmega_skel.so $J/
cp chainrun/vaerun $J/libvaerun.so; cp chainrun/terun $J/libterun.so; cp android/build/bin/sd-cli.stripped $J/libsdcli.so   # llvm-strip of android/build/bin/sd-cli
cp $Q/aarch64-android/{libQnnHtp.so,libQnnSystem.so,libQnnHtpV69Stub.so} $J/; cp $Q/hexagon-v69/unsigned/libQnnHtpV69Skel.so $J/; cp $Q/aarch64-android/libQnnHtpV79Stub.so $J/; cp $Q/hexagon-v79/unsigned/libQnnHtpV79Skel.so $J/   # v79 = Snapdragon 8 Elite
# MOSS-Image (optional, from the research tree): the runner and its DSP skel (same file name as Anima's, so it is shipped as libmossskel.so and copied to files/moss/dsp/libmega_skel.so by moss.sh)
MOSS=${MOSS_DIR:-moss}; [ -f $MOSS/mossrun/out/mossrun ] && cp $MOSS/mossrun/out/mossrun $J/libmossrun.so && cp $MOSS/hexagon/mega/out/libmega_skel.so $J/libmossskel.so && cp $MOSS/mossrun/out/vaetile $J/libvaetile.so && cp $MOSS/mossrun/out/lorafold $J/liblorafold.so
# Qwen-Image-2.1 (optional, from the research tree): runner, its DSP skel (libqwenskel.so -> files/qwen/dsp/libmega_skel.so), the VAE decoder tiler and the VAE encoder tiler
QW=${QWEN_DIR:-qwen}; [ -f $QW/qwenrun/out/qwenrun ] && cp $QW/qwenrun/out/qwenrun $J/libqwenrun.so && cp $QW/hexagon/mega/out/libmega_skel.so $J/libqwenskel.so && cp $QW/qwenrun/out/vaetile $J/libqvaetile.so && cp $QW/qwenrun/out/vaeenc $J/libvaeenc.so && cp $QW/qwenrun/out/qwenfold $J/libqwenfold.so
ls -la $J
