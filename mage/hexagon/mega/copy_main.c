#include <stdio.h>
#include <stdlib.h>
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  int T = atoi(argv[1]), mb = atoi(argv[2]);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) return 1;
  long long us; r = mega_copybw(h, T, mb, &us);
  printf("threads=%d r=%d %d MB in %.2f ms = %.1f GB/s (includes thread create)\n", T, r, mb, us / 1e3, mb * 1.048576 / us * 1e3);
  mega_close(h);
}
