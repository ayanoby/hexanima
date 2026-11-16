#include <stdio.h>
#include <stdlib.h>
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), nw = atoi(argv[4]);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) return 1;
  long long us; float err; int rc; r = mega_gtest(h, M, K, N, nw, &us, &err, &rc);
  printf("gemm M=%d K=%d N=%d nw=%d r=0x%x rc=%d %.3f ms  %.2f TFLOPS  relerr=%g\n", M, K, N, nw, r, rc, us / 1e3, 2.0 * M * K * N / us / 1e6, err);
  mega_close(h);
}
