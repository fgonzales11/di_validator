#include <agent_metrology.h>
#include <agent_feature.h>
#include <agent_config.h>
common::ELogLevel metrolLogLevel = kDefaultLogLevel;
// The offline pipeline requires six waveforms, not one-second RMS LIDs.
void ProcessLids(const stMetrologyLids &) {}
void SubscribeLids() {}
bool IsAgentFeatureEnabled() {
    return GetAgentFeature().Enabled();
}
void SubscribeToMetrologyData() {
    GetAgentFeature().StartReplay();
}
void UnSubscribeFromMetrologyData() {
    GetAgentFeature().StopReplay();
}
