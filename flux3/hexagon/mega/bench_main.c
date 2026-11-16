#include <stdio.h>
#include <stdlib.h>
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) { printf("open 0x%x\n", r); return 1; }
  long long us; float err; int rc;
  r = mega_bench(h, M, K, N, &us, &err, &rc);
  double fl = 2.0 * M * K * N;
  printf("rc=0x%x inner=%d M=%d K=%d N=%d  %.3f ms  %.2f TFLOPS  maxerr=%g\n", r, rc, M, K, N, us / 1e3, fl / us / 1e6, err);
  mega_close(h);
}
