#!/bin/bash
# build henc for Android arm64 (needs the QNN SDK mounted at /qairt): ./build_henc.sh -> out/henc
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v $PWD:/work -v ~/anima-npu/sdk/qairt/2.28.0.241029:/qairt:ro ghcr.io/snapdragon-toolchain/arm64-android:v0.8 bash -c '
NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; mkdir -p /work/out
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O2 -I/qairt/include/QNN /work/henc.cpp -o /work/out/henc -ldl -llog -static-libstdc++ 2>&1 | grep -E "error" -A3 | head -20'
ls -la out/henc
