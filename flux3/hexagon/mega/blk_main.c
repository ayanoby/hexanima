/* MOSS block test: blk DIR NTOK NVALID [STOP] [NW] [REPS]; DIR has wts.bin x32.bin mod.bin cos.bin sin.bin; writes out_*.bin */
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
  const char *dir = argv[1]; const int ntok = atoi(argv[2]), nvalid = atoi(argv[3]), ns = ntok / 32;
  const char *pat = argc > 7 ? argv[7] : 0; int nblk = argc > 8 ? atoi(argv[8]) : 1;
  int stop = argc > 4 ? atoi(argv[4]) : 0, nw = argc > 5 ? atoi(argv[5]) : 3, reps = argc > 6 ? atoi(argv[6]) : 1;
  const int npre = getenv("NPRE") ? atoi(getenv("NPRE")) : 0;
  int meta[M_NMETA]; memset(meta, 0, sizeof meta); setvbuf(stdout, 0, _IONBF, 0);
  const int ncb = getenv("NCB") ? atoi(getenv("NCB")) : 96, hid = ncb * 32, qh = getenv("QH") ? atoi(getenv("QH")) : hid;   /* QH: width of q/k/v/o (2048 for the padded ViT heads) */
  struct { int id; size_t bytes; const char *name; } ent[] = {
    {M_X, (size_t)ntok * hid * 4, "x32.bin"}, {M_COS, (size_t)ns * 8192, "cos.bin"}, {M_SIN, (size_t)ns * 8192, "sin.bin"},
    {M_H, (size_t)ntok * (hid > 4096 ? hid : 4096) * 2, "h.bin"}, {M_Q, (size_t)ntok * qh * 2, "q.bin"}, {M_K, (size_t)(ntok + 32 * npre) * qh * 2, "k.bin"}, {M_V, (size_t)(ntok + 32 * npre) * qh * 2, "v.bin"},
    {M_O, (size_t)ntok * qh * 2, "o.bin"}, {M_T, (size_t)ntok * hid * 2, "t.bin"}, {M_MID, (size_t)ntok * 24576 * 2, "mid.bin"},
    {M_ROWS, (size_t)ntok * 25000 * 4, "rows.bin"}, {M_CS, (size_t)ntok * 128 * 4, "cs.bin"}, {M_LQ, (size_t)ns * 12 * 2048, "lq.bin"}, {M_LO, (size_t)ns * 4 * 2048, "lo.bin"}, {M_LU, (size_t)ns * 4 * 2048, "lu.bin"}, {M_LD, (size_t)ns * 4 * 2048, "ld.bin"}};
  size_t off = 0; int ne = sizeof ent / sizeof ent[0];
  for (int i = 0; i < ne; i++) { meta[ent[i].id] = (int)off; off += up(ent[i].bytes); }
  meta[M_BA] = getenv("BA") ? atoi(getenv("BA")) : 0; meta[M_BB] = getenv("BB") ? atoi(getenv("BB")) : 0; meta[M_NPRE] = npre; meta[M_CAUS] = getenv("CAUS") ? atoi(getenv("CAUS")) : 0; meta[M_NCB] = ncb; meta[M_FMT] = getenv("FMT") ? atoi(getenv("FMT")) : 0; meta[M_NTOK] = ntok; meta[M_NVALID] = nvalid; meta[M_STOP] = stop; meta[M_NW] = nw; meta[M_REL] = 1; meta[M_FIRST] = 1;
  rpcmem_init();
  const size_t WT = getenv("WTSIZE") ? (size_t)atoll(getenv("WTSIZE")) : 135397376u;
  uint8_t *wts = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, WT);
  uint8_t *blob = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, off);
  if (!wts || !blob) { printf("rpcmem_alloc failed (%zu)\n", off); return 1; }
  memset(blob, 0, off);
  if (!pat && rd(dir, "wts.bin", wts, WT)) return 1;
  for (int i = 0; i < 3; i++) if (rd(dir, ent[i].name, blob + meta[ent[i].id], ent[i].bytes)) return 1;
  for (int i = 3; i < ne; i++) { char p[512]; snprintf(p, sizeof p, "%s/in_%s", dir, ent[i].name); FILE *f = fopen(p, "rb"); if (f) { size_t n = fread(blob + meta[ent[i].id], 1, ent[i].bytes, f); fclose(f); printf("loaded %s (%zu)\n", p, n); } }
  meta[M_MODE] = getenv("MODE") ? atoi(getenv("MODE")) : 0;
  uint8_t *mod = (uint8_t *)malloc(73728); if (rd(dir, "mod.bin", mod, 73728)) return 1;
  uint8_t *x0 = (uint8_t *)malloc(ent[0].bytes); memcpy(x0, blob + meta[M_X], ent[0].bytes);
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h); if (r) { printf("open 0x%x\n", r); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(wts), wts, 0, WT, FASTRPC_MAP_FD)) { printf("wts mmap failed\n"); return 1; }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, rpcmem_to_fd(blob), blob, 0, off, FASTRPC_MAP_FD)) { printf("blob mmap failed\n"); return 1; }
  typedef struct { unsigned long long f; } S_;
  if (pat) {   /* all blocks in one process: pattern like /data/local/tmp/moss/packs/blk%02d.bin; x stays in the blob; writes out_x.bin */
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a); double tl = 0;
    for (int bi = 0; bi < nblk; bi++) {
      char pn[512]; snprintf(pn, sizeof pn, pat, bi); struct timespec l0, l1; clock_gettime(CLOCK_MONOTONIC, &l0);
      FILE *f = fopen(pn, "rb"); if (!f) { printf("missing %s\n", pn); return 1; } fseek(f, 0, SEEK_END); size_t fsz_ = ftell(f); fseek(f, 0, SEEK_SET); if (fsz_ > WT || fread(wts, 1, fsz_, f) != fsz_) { printf("bad size %s\n", pn); return 1; } fclose(f);
      if (getenv("TE")) meta[M_MODE] = bi == nblk - 1 && getenv("TEFINAL") ? 30 : (bi % 4 == 3 ? 21 : 20);
      clock_gettime(CLOCK_MONOTONIC, &l1); tl += (l1.tv_sec - l0.tv_sec) + (l1.tv_nsec - l0.tv_nsec) / 1e9;
      { S_ sy = {(1ull << 2) | 3}; ioctl(rpcmem_to_fd(wts), _IOW('b', 0, S_), &sy); }
      meta[M_FIRST] = bi == 0; meta[M_REL] = bi == nblk - 1;
      { S_ sy = {(1ull << 2) | 3}; ioctl(rpcmem_to_fd(blob), _IOW('b', 0, S_), &sy); }
      int times[32]; int rc; r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wts), rpcmem_to_fd(blob), mod, 73728, times, 32, &rc);
      if (r || rc) { printf("block %d r=0x%x rc=%d\n", bi, r, rc); return 1; }
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    printf("all %d blocks %.2f s (file reads %.2f s)\n", nblk, (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9, tl);
    wr(dir, "out_x.bin", blob + meta[M_X], ent[0].bytes); wr(dir, "out_h.bin", blob + meta[M_H], ent[3].bytes); mega_close(h); return 0;
  }
  for (int rep = 0; rep < reps; rep++) {
    memcpy(blob + meta[M_X], x0, ent[0].bytes);
    { S_ sy = {(1ull << 2) | 3}; ioctl(rpcmem_to_fd(blob), _IOW('b', 0, S_), &sy); }
    int times[32]; int rc; struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    r = mega_block(h, meta, M_NMETA, rpcmem_to_fd(wts), rpcmem_to_fd(blob), mod, 73728, times, 32, &rc);
    clock_gettime(CLOCK_MONOTONIC, &b);
    { S_ sy = {3}; ioctl(rpcmem_to_fd(blob), _IOW('b', 0, S_), &sy); }
    printf("rep %d r=0x%x rc=%d total %.2f ms | stages(us):", rep, r, rc, (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6);
    for (int i = 0; i < 24; i++) printf(" %d", times[i]); printf("\n");
  }
  wr(dir, "out_x.bin", blob + meta[M_X], ent[0].bytes);
  for (int i = 3; i < ne; i++) { char n[64]; snprintf(n, sizeof n, "out_%s", ent[i].name); wr(dir, n, blob + meta[ent[i].id], ent[i].bytes); }
  mega_close(h); return 0;
}
