/* HMX drain compatibility: on v73+ hmx_store goes through a bias/scale block in VTCM; hmx_bias_setup(vt) must run once after the VTCM is acquired (4 KB at vt) */
#include <stdint.h>
#include <string.h>
#include "pool.h"
#if __HEXAGON_ARCH__ >= 73
void hmx_store_cvt(uint8_t *o, uint32_t c, uint8_t *bias);
static uint8_t *g_hmx_bias;
void hmx_bias_setup(uint8_t *b) {
  memset(b, 0, 4096);
  for (int ch = 0; ch < 32; ch++) ((uint32_t *)b)[ch] = 0x00003C00u;   /* output bias 0, scale 1.0 (fp16); the input-bias half stays 0 */
  g_hmx_bias = b;
}
void hmx_store(uint8_t *o, uint32_t c) { hmx_store_cvt(o, c, g_hmx_bias); }
#else
void hmx_bias_setup(uint8_t *b) { (void)b; }
#endif
