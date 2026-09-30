#pragma once
#include <agent_types.h>
#include <Common_Util.h>
#include "pv_core.hpp"
#include "pv_delivery.hpp"
#include <Itron/AgentApi.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <deque>
struct PVMetrologyCache {
    double watts = 0, vars = 0;
    bool p_valid = false, q_valid = false;
    int64_t previous = -1;
    double phase_volts[3] = {}, phase_watts[3] = {};
    uint8_t diagnostic_valid = 0;
};
pv::MeterSample DecodePVLids(LidList, int64_t utc, PVMetrologyCache &);
int SubscribePV();
void UnsubscribePV();
class CAgentFeature {
  public:
    CAgentFeature();
    ~CAgentFeature();
    void InitializeConfig();
    void UpdateAgentFeatureConfig(stGeneralConfig, stAgentFeatureConfig);
    void StartReplay();
    void StopReplay();
    bool Enabled();
    void Enqueue(const pv::MeterSample &);

  private:
    std::mutex mutex_;
    pv::Config config_;
    bool enabled_ = false, started_ = false;
    uint64_t generation_ = 0;
    std::deque<pv::MeterSample> queue_;
    std::atomic<bool> stopping_, cancel_;
    std::atomic<unsigned> dropped_;
    std::thread worker_, configuration_worker_;
    void Worker();
#ifdef DI_EMULATOR_REPLAY
    void LabWorker();
#endif
    void ConfigurationWorker();
};
CAgentFeature &GetAgentFeature();
