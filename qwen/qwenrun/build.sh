#!/bin/bash
# build qwenrun for Android arm64 (needs ../hexagon/mega/out/{mega_stub.c,mega.h} from the skel build)
cd "$(dirname "$0")"; M=$PWD/../hexagon/mega
docker run --rm --platform linux/amd64 -v $PWD/..:/work -v ~/anima-npu/sdk/qairt/2.28.0.241029:/qairt:ro ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
S=/opt/hexagon/6.7.0.0; NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; W=/work
INC="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I$W/hexagon/mega/out -I$W/hexagon/mega"
mkdir -p $W/qwenrun/out
$NDK/aarch64-linux-android31-clang -O2 $INC -c $W/hexagon/mega/out/mega_stub.c -o $W/qwenrun/out/mega_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $INC $W/qwenrun/qwenrun.cpp $W/qwenrun/out/mega_stub.o -L$W/hexagon/mega/syslib -lcdsprpc -llog -static-libstdc++ -o $W/qwenrun/out/qwenrun 2>&1 | grep -E "error" -A3 | head -30
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O2 -march=armv8.2-a+fp16 $W/qwenrun/qwenfold.cpp -o $W/qwenrun/out/qwenfold -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -20
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O2 -I/qairt/include/QNN $W/qwenrun/vaetile.cpp -o $W/qwenrun/out/vaetile -ldl -llog -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -20
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O2 -I/qairt/include/QNN $W/qwenrun/vaeenc.cpp -o $W/qwenrun/out/vaeenc -ldl -llog -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -20'
ls -la out/qwenrun out/vaetile out/vaeenc out/qwenfold

