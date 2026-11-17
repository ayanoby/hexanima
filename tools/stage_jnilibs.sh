#!/bin/bash
# copies the native binaries the APK bundles into app/jnilibs/lib/arm64-v8a (executables are named lib*.so: an app may only exec from its native lib dir)
set -e; cd "$(dirname "$0")/.."; J=app/jnilibs/lib/arm64-v8a; Q=~/anima-npu/sdk/qairt/2.28.0.241029/lib; mkdir -p $J
cp phone/animad $J/libanimad.so; cp hexagon/mega/out/megarun $J/libmegarun.so; cp hexagon/mega/out/libmega_skel.so $J/
cp chainrun/vaerun $J/libvaerun.so; cp chainrun/terun $J/libterun.so; cp android/build/bin/sd-cli.stripped $J/libsdcli.so   # llvm-strip of android/build/bin/sd-cli
cp $Q/aarch64-android/{libQnnHtp.so,libQnnSystem.so,libQnnHtpV69Stub.so} $J/; cp $Q/hexagon-v69/unsigned/libQnnHtpV69Skel.so $J/;
# MOSS-Image (optional, from the research tree): the runner and its DSP skel (same file name as Anima's, so it is shipped as libmossskel.so and copied to files/moss/dsp/libmega_skel.so by moss.sh)
MOSS=${MOSS_DIR:-moss}; [ -f $MOSS/mossrun/out/mossrun ] && cp $MOSS/mossrun/out/mossrun $J/libmossrun.so && cp $MOSS/hexagon/mega/out/libmega_skel.so $J/libmossskel.so && cp $MOSS/mossrun/out/vaetile $J/libvaetile.so && cp $MOSS/mossrun/out/lorafold $J/liblorafold.so
# Qwen-Image-2.1 (optional, from the research tree): runner, its DSP skel (libqwenskel.so -> files/qwen/dsp/libmega_skel.so), the VAE decoder tiler and the VAE encoder tiler
QW=${QWEN_DIR:-qwen}; [ -f $QW/qwenrun/out/qwenrun ] && cp $QW/qwenrun/out/qwenrun $J/libqwenrun.so && cp $QW/hexagon/mega/out/libmega_skel.so $J/libqwenskel.so && cp $QW/qwenrun/out/vaetile $J/libqvaetile.so && cp $QW/qwenrun/out/vaeenc $J/libvaeenc.so && cp $QW/qwenrun/out/qwenfold $J/libqwenfold.so
# FLUX.3 Klein, LTX-2.5 and MiniMax-H3 Ref2VA (optional, from flux3/, ltx/, h3/): runners, the shared DSP skel (built by flux3/hexagon/mega/build_skel.sh, shipped under one name per model: each model copies its own to files/<model>/dsp/libmega_skel.so), the VAE helpers
F3=${FLUX3_DIR:-flux3}; SK=$F3/hexagon/mega/bld/v69/libmega_skel.so
[ -f $F3/runner/out/flux3run ] && cp $F3/runner/out/flux3run $J/libflux3run.so && cp $SK $J/libflux3skel.so && cp $F3/runner/out/kup $J/libkup.so && cp $F3/runner/out/kvae $J/libkvae.so
# Mage-Flow-Edit-Turbo (optional): the runner (mage/magerun, build.sh) and the shared DSP skel (modes 53 / 55 / 61 / 22)
[ -f mage/magerun/out/magerun ] && cp mage/magerun/out/magerun $J/libmagerun.so && cp $SK $J/libmageskel.so
[ -f ltx/runner/out/ltxrun ] && cp ltx/runner/out/ltxrun $J/libltxrun.so && cp $SK $J/libltxskel.so && cp ltx/vae/runner/out/lvae $J/liblvae.so
[ -f h3/hexagon/mega/out/r2vrun ] && cp h3/hexagon/mega/out/r2vrun $J/libr2vrun.so && cp $SK $J/libr2vskel.so && cp $F3/hexagon/mega/out/vaerun $J/libr2vae.so && cp h3/vae/out/henc $J/libr2enc.so
# SeedVR2 (optional): the driver svsr, the DiT runner svrun and the shared DSP skel (MODE 100) under its own name; its VAE tiles run in libvaerun.so
[ -f seedvr2/svsr_s ] && cp seedvr2/svsr_s $J/libsvsr.so && cp flux3/hexagon/mega/out/svrun_s $J/libsvrun.so && cp $SK $J/libsvrskel.so
ls -la $J
