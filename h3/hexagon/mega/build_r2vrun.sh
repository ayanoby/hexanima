#!/bin/bash
# build r2vrun (the MiniMax-H3 Ref2VA sampler) for Android arm64; it shares the kernel headers and the FastRPC stub with flux3/hexagon/mega
# (needs flux3/hexagon/mega/out/{mega_stub.c,mega.h} from its build.sh and flux3/hexagon/mega/syslib/libcdsprpc.so)  ->  out/r2vrun
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD/../../..:/work ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
M=/work/flux3/hexagon/mega; O=/work/h3/hexagon/mega/out; mkdir -p $O; S=/opt/hexagon/6.7.0.0; NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin
INC="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I$M/out -I$M"
$NDK/aarch64-linux-android31-clang -O2 $INC -c $M/out/mega_stub.c -o $O/mega_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $INC /work/h3/hexagon/mega/r2vrun.cpp $O/mega_stub.o -L$M/syslib -lcdsprpc -llog -static-libstdc++ -o $O/r2vrun 2>&1 | grep -E "error" -A3 | head -30'
ls -la out/r2vrun
