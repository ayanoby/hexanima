int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: vaerun BACKEND SYSTEM CONTEXT.bin\n"); return 2; }
  void* hb = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* hs = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!hb || !hs) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
  typedef Qnn_ErrorHandle_t (*GetP)(const QnnInterface_t***, uint32_t*);
  typedef Qnn_ErrorHandle_t (*GetSP)(const QnnSystemInterface_t***, uint32_t*);
  const QnnInterface_t** provs = nullptr; uint32_t np = 0;
  if (!check(((GetP)dlsym(hb, "QnnInterface_getProviders"))(&provs, &np), "getProviders")) return 1;
  qi = provs[0]->QNN_INTERFACE_VER_NAME;
  const QnnSystemInterface_t** sprovs = nullptr; uint32_t nsp = 0;
  if (!check(((GetSP)dlsym(hs, "QnnSystemInterface_getProviders"))(&sprovs, &nsp), "getSystemProviders")) return 1;
  qsi = sprovs[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  Qnn_BackendHandle_t backend = nullptr;
  if (!check(qi.backendCreate(nullptr, nullptr, &backend), "backendCreate")) return 1;
  Qnn_DeviceHandle_t device = nullptr;
  if (!check(qi.deviceCreate(nullptr, nullptr, &device), "deviceCreate")) return 1;
    // burst power profile (what qnn-net-run --perf_profile burst does)
  QnnDevice_Infrastructure_t infra = nullptr;
  if (qi.deviceGetInfrastructure && qi.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
    auto* hi = (QnnHtpDevice_Infrastructure_t*)infra;
    uint32_t pid = 0;
    if (hi->perfInfra.createPowerConfigId(0, 0, &pid) == QNN_SUCCESS) {
      QnnHtpPerfInfrastructure_PowerConfig_t dc = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
      dc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
      auto& d = dc.dcvsV3Config;
      d.contextId = pid; d.setDcvsEnable = 1; d.dcvsEnable = 0;
      d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
      d.setSleepLatency = 1; d.sleepLatency = 40; d.setSleepDisable = 1; d.sleepDisable = 1;
      d.setBusParams = 1; d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      d.setCoreParams = 1; d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      QnnHtpPerfInfrastructure_PowerConfig_t rc = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
      rc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
      rc.rpcControlLatencyConfig = 100;
      const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dc, &rc, nullptr};
      hi->perfInfra.setPowerConfig(pid, cfgs);
    }
  }
