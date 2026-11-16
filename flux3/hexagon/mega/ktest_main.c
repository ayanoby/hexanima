#include <stdio.h>
#include <stdlib.h>
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  int w = atoi(argv[1]);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) return 1;
  float err; long long us, cyc; r = mega_ktest(h, w, &err, &us, &cyc);
  printf("ktest %d r=0x%x maxerr=%g  %lld us  %lld cyc\n", w, r, err, us, cyc);
  mega_close(h);
}
