#!/bin/bash
# build ltxrun for Android arm64 (needs ../../flux3/hexagon/mega/out/{mega_stub.c,mega.h}); usage: ./build.sh
cd "$(dirname "$0")"; cat ../../flux3/runner/base.inc gemma.inc > ltxrun.cpp
docker run --rm --platform linux/amd64 -v $PWD/../..:/work ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
S=/opt/hexagon/6.7.0.0; NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; W=/work; M=$W/flux3/hexagon/mega
INC="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I$M/out -I$M"
mkdir -p $W/ltx/runner/out
$NDK/aarch64-linux-android31-clang -O2 $INC -c $M/out/mega_stub.c -o $W/ltx/runner/out/mega_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $INC $W/ltx/runner/ltxrun.cpp $W/ltx/runner/out/mega_stub.o -L$M/syslib -lcdsprpc -llog -static-libstdc++ -o $W/ltx/runner/out/ltxrun 2>&1 | grep -E "error" -A3 | head -30'
ls -la out/ltxrun
