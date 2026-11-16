#!/bin/bash
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD:/work -v ~/anima-npu/sdk/qairt/2.28.0.241029:/qairt:ro ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O2 -march=armv8.2-a+fp16 -I/qairt/include/QNN /work/h3vae.cpp -o /work/out/h3vae -ldl -llog -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -20'
ls -la out/h3vae
