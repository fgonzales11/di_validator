#include "agent_feature.h"
#include <agent_config.h>
#include <agent_constants.h>
#include <agent_metrology.h>
#include <common_metrology.h>
#include <apiWrapper.h>
#include <cmath>
#include <ctime>
#include <mutex>
#include <cstdlib>
using namespace common;
common::ELogLevel metrolLogLevel = kDefaultLogLevel;
bitMask bmMetLids = 0;
static PVMetrologyCache live_cache;
static std::mutex cache_mutex;
pv::MeterSample DecodePVLids(LidList list, int64_t utc, PVMetrologyCache &cache) {
    if (cache.previous >= 0 && (utc <= cache.previous || utc - cache.previous > 2)) {
        cache.p_valid = cache.q_valid = false;
        cache.diagnostic_valid = 0;
    }
    cache.previous = utc;
    ApiLidListResetIteration(list);
    uint32_t lid = 0;
    ApiVariableType value = {};
    while (AgentLidListGetNext(list, &lid, &value) == API_SUCCESS) {
        bool valid = value.type == AGENT_API_DOUBLE && std::isfinite(value.value.Double);
        if (lid != 18153531 && lid != 18153532) {
            int index = lid >= 18153520 && lid <= 18153522   ? static_cast<int>(lid - 18153520)
                        : lid >= 18153658 && lid <= 18153660 ? static_cast<int>(lid - 18153658 + 3)
                                                             : -1;
            if (index >= 0) {
                const uint8_t bit = static_cast<uint8_t>(1u << index);
                cache.diagnostic_valid = static_cast<uint8_t>(cache.diagnostic_valid & ~bit);
                if (valid) {
                    cache.diagnostic_valid |= bit;
                    if (index < 3)
                        cache.phase_volts[index] = value.value.Double;
                    else
                        cache.phase_watts[index - 3] = value.value.Double;
                }
            }
            continue;
        }
        if (lid == 18153531) {
            cache.p_valid = valid;
            if (valid)
                cache.watts = value.value.Double;
        } else {
            cache.q_valid = valid;
            if (valid)
                cache.vars = value.value.Double;
        }
    }
    pv::MeterSample sample;
    sample.utc = utc;
    sample.watts = cache.watts;
    sample.vars = cache.vars;
    sample.p_valid = cache.p_valid;
    sample.q_valid = cache.q_valid;
    sample.diagnostic_valid = cache.diagnostic_valid;
    for (unsigned i = 0; i < 3; ++i) {
        sample.phase_volts[i] = cache.phase_volts[i];
        sample.phase_watts[i] = cache.phase_watts[i];
    }
    sample.time_valid = utc >= 946684800 && utc <= 2145916799;
    return sample;
}
static void MeterCallback(LidList list) {
#ifdef NODAEMON
    // Host fixture ingress uses the same typed LID decoder, with fixture timestamps.
    if (std::getenv("PV_REPLAY_DIR"))
        return;
#endif
    std::lock_guard<std::mutex> lock(cache_mutex);
    GetAgentFeature().Enqueue(DecodePVLids(list, static_cast<int64_t>(std::time(0)), live_cache));
}
int SubscribePV() {
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        live_cache = PVMetrologyCache();
    }
    LidList lids = AgentLidListCreate();
    if (!lids)
        return -1;
    AgentLidListAddLid(lids, 18153531);
    AgentLidListAddLid(lids, 18153532);
    for (uint32_t lid : {18153520u, 18153521u, 18153522u, 18153658u, 18153659u, 18153660u})
        AgentLidListAddLid(lids, lid);
    AgentApiReturnType result = ApiSubscribeToPeriodicData(MeterCallback, eOnPeriodOnly, 1, lids);
    AgentLidListFree(lids);
    return static_cast<int>(result);
}
void UnsubscribePV() {
    ApiUnsubscribeFromPeriodicData();
}
void ProcessLids(const stMetrologyLids &) {}
bool IsAgentFeatureEnabled() {
    return GetAgentFeature().Enabled();
}
void SubscribeToMetrologyData() {
    GetAgentFeature().StartReplay();
}
void UnSubscribeFromMetrologyData() {
    GetAgentFeature().StopReplay();
}
void SubscribeLids() {} // Direct typed aggregate-P/Q subscription owns metrology.
