#!/bin/bash
# build magerun for Android arm64: ./build.sh -> out/magerun. It needs ../../flux3/hexagon/mega/{out/mega_stub.c, out/mega.h (from its build.sh), syslib/libcdsprpc.so (Hexagon SDK)}; blk_meta.h is the one of the shared kernel tree.
# magerun.cpp = base.inc + mage.inc (edit those, then: cat base.inc mage.inc > magerun.cpp)
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD/../..:/work ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
S=/opt/hexagon/6.7.0.0; NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; W=/work; F=$W/flux3/hexagon/mega
INC="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I$F/out -I$F"
mkdir -p $W/mage/magerun/out
$NDK/aarch64-linux-android31-clang -O2 $INC -c $F/out/mega_stub.c -o $W/mage/magerun/out/mega_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $INC $W/mage/magerun/magerun.cpp $W/mage/magerun/out/mega_stub.o -L$F/syslib -lcdsprpc -llog -static-libstdc++ -o $W/mage/magerun/out/magerun 2>&1 | grep -E "error" -A3 | head -30'
ls -la out/magerun
