/* dump_main in.bin out.bin: in = int32 N, int32 mode, int32 alen, int32 wlen, int32 olen, then N x (act[alen], wgt[wlen]); out = N x olen */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); int hd[5]; if (fread(hd, 4, 5, f) != 5) return 2;
  int N = hd[0], mode = hd[1], al = hd[2], wl = hd[3], ol = hd[4];
  uint8_t *a = malloc(al), *w = malloc(wl), *o = malloc(ol); FILE *g = fopen(argv[2], "wb");
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; if (mega_open(mega_URI CDSP_DOMAIN, &h)) return 1;
  for (int i = 0; i < N; i++) {
    if (fread(a, 1, al, f) != al || fread(w, 1, wl, f) != wl) return 3;
    int rc = 0; int r = mega_dump(h, mode, a, al, w, wl, o, ol, &rc);
    if (r || rc) { fprintf(stderr, "exp %d r=0x%x rc=%d\n", i, r, rc); return 4; }
    fwrite(o, 1, ol, g);
  }
  fclose(g); mega_close(h); return 0;
}
