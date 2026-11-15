#!/bin/bash
# build megarun (needs out/mega_stub.c + mega.h from build.sh, and syslib/libcdsprpc.so pulled from the phone)
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD:/work ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
mkdir -p /work/out_arm; S=/opt/hexagon/6.7.0.0; NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin
mkdir -p /work/out_arm; INC="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I/work/out"
$NDK/aarch64-linux-android31-clang -O2 $INC -c /work/out/mega_stub.c -o /work/out_arm/mega_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $INC /work/megarun.cpp /work/out_arm/mega_stub.o -L/work/syslib -lcdsprpc -llog -static-libstdc++ -o /work/out/megarun 2>&1 | grep -E "error" -A3 | head -30'
ls -la out/megarun
