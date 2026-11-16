#!/bin/bash
# context binary for another Hexagon generation from an existing NAME.dlc: ./build_ctx_arch.sh kvae_t32 v79 69  ->  out/kvae_t32_v79.bin   (v69 = soc_model 36, v79 = 69; compile only, runs on that chip)
set -e
cd "$(dirname "$0")"; source ~/anima-npu/work/env.sh
N=${1:-kvae_t32}; A=${2:-v79}; SOC=${3:-69}; T=${N}_$A
echo "{\"graphs\":[{\"vtcm_mb\":8,\"graph_names\":[\"$N\"],\"O\":3.0,\"fp16_relaxed_precision\":1}],\"devices\":[{\"dsp_arch\":\"$A\",\"soc_model\":$SOC,\"cores\":[{\"perf_profile\":\"burst\",\"rpc_control_latency\":100}]}]}" > htp_config_$T.json
echo "{\"backend_extensions\":{\"shared_library_path\":\"libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"$PWD/htp_config_$T.json\"}}" > htp_backend_$T.json
mkdir -p out; rm -f out/$T.bin
qnn-context-binary-generator --dlc_path $N.dlc --model $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --output_dir out --binary_file $T --config_file htp_backend_$T.json --log_level warn 2>&1 | grep -E "ERROR|rror" | head -8 || true
ls -la out/$T.bin
