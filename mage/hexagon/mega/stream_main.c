#include <stdio.h>
#include <stdlib.h>
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), mode = atoi(argv[4]);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) { printf("open 0x%x\n", r); return 1; }
  long long us, cus; int rc;
  r = mega_stream(h, M, K, N, mode, &us, &cus, &rc);
  double bytes = (double)K * N * 2;
  printf("mode=%d r=0x%x inner=%d M=%d K=%d N=%d  total %.3f ms  (copy-only %.3f ms)  weights %.1f MB -> %.1f GB/s eff; %.2f TFLOPS\n", mode, r, rc, M, K, N, us / 1e3, cus / 1e3, bytes / 1e6, bytes / us / 1e3, 2.0 * M * K * N / us / 1e6);
  mega_close(h);
}
