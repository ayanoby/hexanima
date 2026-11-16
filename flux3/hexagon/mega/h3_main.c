/* H3 block test: h3blk DIR NTOK NIMG NVALID [REPS] [STOP]; DIR has wts.bin x32.bin mod.bin cos.bin sin.bin hn.bin; writes out_x.bin (+ out_h/q/k/v/o/t/mid/hr.bin) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/ioctl.h>
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
#include "blk_meta.h"
static int rd(const char *dir, const char *name, void *dst, size_t want) {
  char p[512]; snprintf(p, sizeof p, "%s/%s", dir, name); FILE *f = fopen(p, "rb"); if (!f) { printf("missing %s\n", p); return -1; }
  size_t n = fread(dst, 1, want, f); fclose(f); if (n != want) { printf("%s: got %zu want %zu\n", name, n, want); return -1; } return 0;
}
static void wr(const char *dir, const char *name, const void *src, size_t n) { char p[512]; snprintf(p, sizeof p, "%s/%s", dir, name); FILE *f = fopen(p, "wb"); fwrite(src, 1, n, f); fclose(f); }
static size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }
int main(int argc, char **argv) {
  const char *dir = argv[1]; const int ntok = atoi(argv[2]), nimg = atoi(argv[3]), nvalid = atoi(argv[4]), ns = ntok / 32;
  const int reps = argc > 5 ? atoi(argv[5]) : 1, stop = argc > 6 ? atoi(argv[6]) : 0;
  int meta[M_NMETA]; memset(meta, 0, sizeof meta); setvbuf(stdout, 0, _IONBF, 0);
  const int ncb = getenv("NCB") ? atoi(getenv("NCB")) : 168, mode = getenv("H3MODE") ? atoi(getenv("H3MODE")) : 50; const size_t HID = ncb * 32, QW = 8192, FF = 25600;
  struct { int id; size_t bytes; const char *name; } ent[] = {
    {M_X, (size_t)ntok * HID * 4, "x32.bin"}, {M_COS, (size_t)ns * 8192, "cos.bin"}, {M_SIN, (size_t)ns * 8192, "sin.bin"}, {M_HN, 131072, "hn.bin"},
    {M_H, (size_t)ntok * HID * 2, "h.bin"}, {M_HR, (size_t)ntok * FF * 2, "hr.bin"}, {M_Q, (size_t)ntok * QW * 2, "q.bin"}, {M_K, (size_t)ntok * QW * 2, "k.bin"}, {M_V, (size_t)ntok * QW * 2, "v.bin"},
    {M_O, (size_t)ntok * QW * 2, "o.bin"}, {M_T, (size_t)ntok * HID * 2, "t.bin"}, {M_MID, (size_t)ntok * FF * 2, "mid.bin"},
    {M_LQ, (size_t)ns * 12 * 2048, "lq.bin"}, {M_LO, (size_t)ns * 4 * 2048, "lo.bin"}, {M_LU, (size_t)ns * 4 * 2048, "lu.bin"}, {M_LD, (size_t)ns * 4 * 2048, "ld.bin"}, {M_DLT, (size_t)ntok * 28672 * 2, "dlt.bin"}};
  size_t off = 0; const int ne = sizeof ent / sizeof ent[0];
  for (int i = 0; i < ne; i++) { meta[ent[i].id] = (int)off; off += up(ent[i].bytes); }
  meta[M_MODE] = mode; const int lora = getenv("H3LORA") ? 1 : 0; meta[M_LORA] = lora; meta[M_NCB] = ncb; meta[M_NTOK] = ntok; meta[M_NVALID] = nvalid; meta[M_NIMG] = nimg; meta[M_STOP] = stop; meta[M_NW] = 3; meta[M_REL] = 1; meta[M_FIRST] = 1;
  rpcmem_init();
  const size_t WT = (getenv("WTSIZE") ? (size_t)atoll(getenv("WTSIZE")) : 385660928u) + (lora ? 26607616u : 0), MODB = 258048;
  uint8_t *wts = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WT);
  uint8_t *blob = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, off);
  if (!wts || !blob) { printf("rpcmem_alloc failed (%zu + %zu)\n", WT, off); return 1; }
  memset(blob, 0, off);
  if (rd(dir, "wts.bin", wts, WT)) return 1;
  for (int i = 0; i < 4; i++) if (rd(dir, ent[i].name, blob + meta[ent[i].id], ent[i].bytes)) return 1;
  uint8_t *mod = (uint8_t *)malloc(MODB); if (rd(dir, "mod.bin", mod, MODB)) return 1;
  uint8_t *x0 = (uint8_t *)malloc(ent[0].bytes); memcpy(x0, blob + meta[M_X], ent[0].bytes);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) { printf("open 0x%x\n", r); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(wts), wts, 0, WT, FASTRPC_MAP_FD)) { printf("wts mmap failed\n"); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(blob), blob, 0, off, FASTRPC_MAP_FD)) { printf("blob mmap failed\n"); return 1; }
  typedef struct { unsigned long long f; } S_;
  for (int rep = 0; rep < reps; rep++) {
    memcpy(blob + meta[M_X], x0, ent[0].bytes);
    { S_ sy = {(1ull << 2) | 3}; ioctl(rpcmem_to_fd(blob), _IOW('b', 0, S_), &sy); ioctl(rpcmem_to_fd(wts), _IOW('b', 0, S_), &sy); }
    int times[32]; int rc; struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wts), rpcmem_to_fd(blob), mod, MODB, times, 32, &rc);
    clock_gettime(CLOCK_MONOTONIC, &b);
    { S_ sy = {3}; ioctl(rpcmem_to_fd(blob), _IOW('b', 0, S_), &sy); }
    printf("rep %d r=0x%x rc=%d total %.2f ms | stages(us):", rep, r, rc, (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6);
    for (int i = 0; i < 24; i++) printf(" %d", times[i]); printf("\n");
  }
  wr(dir, "out_x.bin", blob + meta[M_X], ent[0].bytes);
  for (int i = 4; i < ne; i++) { char n[64]; snprintf(n, sizeof n, "out_%s", ent[i].name); wr(dir, n, blob + meta[ent[i].id], ent[i].bytes); }
  mega_close(h); return 0;
}
