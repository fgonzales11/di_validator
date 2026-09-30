#pragma once
#include <agent_types.h>
#include <Common_Util.h>
#include "fault_core.hpp"
#include <atomic>
#include <mutex>
#include <thread>
class CAgentFeature {
  public:
    CAgentFeature();
    ~CAgentFeature();
    void InitializeConfig();
    void UpdateAgentFeatureConfig(const stGeneralConfig general,
                                  const stAgentFeatureConfig feature);
    void StartReplay();
    void StopReplay();
    bool Enabled();

  private:
    std::mutex mutex_;
    fault::Config config_;
    bool enabled_, started_;
    std::atomic<bool> stopping_, cancel_;
    std::thread worker_;
    std::thread configuration_worker_;
    void ReplayWorker();
#ifdef DI_EMULATOR_REPLAY
    void LabWorker();
#endif
    void ConfigurationWorker();
    void Publish(const fault::Result &result, const fault::Config &config,
                 const std::string &source, const std::string &id);
};
CAgentFeature &GetAgentFeature();
