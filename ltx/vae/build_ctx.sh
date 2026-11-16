#!/bin/bash
# ONNX -> QNN HTP context binary (fp16): ./build_ctx.sh NAME [ARCH SOC]   (NAME.onnx in cwd) -> out/NAME[_ARCH].bin ; default v69 / soc 36
cd "$(dirname "$0")"; source ~/anima-npu/work/env.sh
N=${1:?name}; A=${2:-v69}; SOC=${3:-36}; T=$N; [ "$A" != v69 ] && T=${N}_$A
echo "{\"graphs\":[{\"vtcm_mb\":8,\"graph_names\":[\"$N\"],\"O\":3.0,\"fp16_relaxed_precision\":1}],\"devices\":[{\"dsp_arch\":\"$A\",\"soc_model\":$SOC,\"cores\":[{\"perf_profile\":\"burst\",\"rpc_control_latency\":100}]}]}" > htp_config_$T.json
echo "{\"backend_extensions\":{\"shared_library_path\":\"libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"$PWD/htp_config_$T.json\"}}" > htp_backend_$T.json
qairt-converter --onnx_no_simplification --input_network $N.onnx --output_path $N.dlc --float_bitwidth 16 2>&1 | grep -iE "error|unsupported|fail" | head -8
mkdir -p out; rm -f out/$T.bin
qnn-context-binary-generator --dlc_path $N.dlc --model $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --output_dir out --binary_file $T --config_file htp_backend_$T.json --log_level warn 2>&1 | grep -E "ERROR|rror|not supported|Unsupported" | head -10
ls -la out/$T.bin 2>&1 | tail -n 1
