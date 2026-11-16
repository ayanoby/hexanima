#!/bin/bash
# DSP skeleton for one Hexagon generation with the full SDK in ~/hexsdk (no docker): ./build_skel.sh v79  ->  bld/v79/libmega_skel.so (v69 default; the phone build of v69 stays build.sh)
set -e
A=${1:-v69}; S=~/hexsdk/Hexagon_SDK/6.7.0.0; T=$S/tools/HEXAGON_Tools/21.0.04/Tools; Q=$S/rtos/qurt/compute$A; cd "$(dirname "$0")"; O=bld/$A; mkdir -p $O
INC="-I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$Q/include/qurt -I$Q/include/posix -Iout -Iqhl_hmx/inc/qhblas_hmx -I$S/libs/qhl_hvx/inc/internal -I$S/utils/examples"
HF="-G0 -m$A -mhvx -mhvx-length=128B -mhmx -O2 -fPIC -DUSE_OS_QURT -D_HAS_EXCEPTIONS=0 -w $INC"
for f in mega_imp.c gemm.c stream.c copybw.c powermax.c kern.c ktest.c pool.c gemm2.c gtest.c blk.c out/mega_skel.c qhl_hmx/src/qhblas_hmx/*.c qhl_hmx/src/qhblas_hmx/*.S hmx8.S hmxacc.S hmxv73.c; do $T/bin/hexagon-clang $HF -c $f -o $O/$(basename $f).o; done
$T/bin/hexagon-clang -G0 -m$A -shared -fPIC -Wl,-Bsymbolic -nodefaultlibs -o $O/libmega_skel.so $O/*.o -Wl,--start-group $T/target/hexagon/lib/$A/G0/pic/libgcc.a -Wl,--end-group
ls -la $O/libmega_skel.so
