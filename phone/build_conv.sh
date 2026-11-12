#!/bin/bash
# build animaconv (the on-phone Anima checkpoint converter) for Android arm64 and stage it as libanimaconv.so
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD:/repo ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin
$NDK/aarch64-linux-android29-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod -pthread /repo/animaconv.cpp -o /repo/animaconv -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -30'
ls -la animaconv && cp animaconv ../app/jnilibs/lib/arm64-v8a/libanimaconv.so
