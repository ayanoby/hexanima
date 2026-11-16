#!/bin/bash
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD/..:/work -v ~/anima-npu/sdk/qairt/2.28.0.241029:/qairt:ro ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; W=/work; mkdir -p $W/runner/out
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O2 -I/qairt/include/QNN $W/runner/kup.cpp -o $W/runner/out/kup -ldl -llog -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -20'
ls -la out/kup
