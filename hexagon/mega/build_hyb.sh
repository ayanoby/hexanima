#!/bin/bash
# libmega_skel.so for Hexagon v73+ chips (Snapdragon 8 Gen 2 / 3 / Elite: V73, V75, V79): all kernel code is compiled for v69 exactly like the v69 skel (the v79 compiler changes the qfloat code and breaks softmax, the v69 code runs unchanged on these DSPs),
# only the HMX drain (hmxacc.S, hmxv73.c: bias/scale block, instructions of v73) is assembled for v73. Needs the full Hexagon SDK (~/hexsdk) and out/ from build.sh (mega_skel.c, mega.h). Output: out_hyb/libmega_skel.so
set -e
S=~/hexsdk/Hexagon_SDK/6.7.0.0; T=$S/tools/HEXAGON_Tools/21.0.04/Tools; cd "$(dirname "$0")"; mkdir -p out_hyb
INC="-I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/rtos/qurt/computev69/include/qurt -I$S/rtos/qurt/computev69/include/posix -Iout -Iqhl_hmx/inc/qhblas_hmx -I$S/libs/qhl_hvx/inc/internal -I$S/utils/examples"
for f in mega_imp.c gemm.c stream.c copybw.c powermax.c kern.c ktest.c pool.c gemm2.c gtest.c blk.c out/mega_skel.c qhl_hmx/src/qhblas_hmx/*.c qhl_hmx/src/qhblas_hmx/*.S hmx8.S hmxacc.S hmxv73.c; do
  A=69; case $f in hmxacc.S|hmxv73.c) A=73;; esac
  $T/bin/hexagon-clang -G0 -mv$A -mhvx -mhvx-length=128B -mhmx -O2 -fPIC -DUSE_OS_QURT -D_HAS_EXCEPTIONS=0 -w $INC -c $f -o out_hyb/$(basename $f).o
done
$T/bin/hexagon-clang -G0 -mv69 -shared -fPIC -Wl,-Bsymbolic -nodefaultlibs -o out_hyb/libmega_skel.so out_hyb/*.o -Wl,--start-group $T/target/hexagon/lib/v69/G0/pic/libgcc.a -Wl,--end-group
ls -la out_hyb/libmega_skel.so
