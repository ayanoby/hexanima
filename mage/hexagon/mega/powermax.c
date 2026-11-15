#include <string.h>
#include "HAP_power.h"
static int g_hmx_rc = -1;
int power_hmx_rc(void) { return g_hmx_rc; }
static int getenv_hmx_clk(void) { return 1; }
/* Pin DSP core/bus clocks to max corners and disable sleep; without this the clock depends on whatever ran before (2.5x swings seen). */
void power_max(void *ctx) {
  HAP_power_request_t r;
  HAP_power_set_dcvs_v3_init(&r);
  r.dcvs_v3.dcvs_enable = FALSE;
  r.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  r.dcvs_v3.latency = 100;
  r.dcvs_v3.core_params.min_corner = r.dcvs_v3.core_params.max_corner = r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
  r.dcvs_v3.bus_params.min_corner = r.dcvs_v3.bus_params.max_corner = r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
  r.dcvs_v3.sleep_disable = 1;
  HAP_power_set(ctx, &r);
  if (getenv_hmx_clk()) {   /* without set_clock the HMX clock stays at its LOWEST frequency */
    HAP_power_request_t h; memset(&h, 0, sizeof h);
    h.type = HAP_power_set_HMX_v2;
    h.hmx_v2.set_power = TRUE; h.hmx_v2.power_up = TRUE; h.hmx_v2.set_clock = TRUE; h.hmx_v2.pick_default = FALSE;
    h.hmx_v2.target_corner = h.hmx_v2.min_corner = h.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    h.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    g_hmx_rc = HAP_power_set(ctx, &h);
  }
}
