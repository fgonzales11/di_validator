#pragma once
// Compiled only in the SDK host executable. Uses the SDK's own LidList implementation.
#include <Itron/LidList.h>
#include <limits>
#include <stdexcept>
static void metrology_contracts(const std::string &path) {
    unsigned checks = 0;
    auto require = [&](bool ok) {
        if (!ok)
            throw std::runtime_error("SDK typed metrology contract failed");
        ++checks;
    };
    auto put = [](Itron::AgentSDK::AgentLidList &list, uint32_t lid, double x) {
        ApiVariableType v = {};
        v.type = AGENT_API_DOUBLE;
        v.value.Double = x;
        list.AddLidPair(lid, v);
    };
    PVMetrologyCache cache;
    int64_t t = 1735689600;
    Itron::AgentSDK::AgentLidList empty, sp, pp, phase_only, invalid;
    auto sample = DecodePVLids(&empty, t, cache);
    require(!sample.p_valid && !sample.q_valid);
    put(phase_only, 18153658, 5000);
    sample = DecodePVLids(&phase_only, ++t, cache);
    require(!sample.p_valid && sample.diagnostic_valid == 8);
    put(sp, 18153531, -600);
    put(sp, 18153532, 200);
    put(sp, 18153520, 240);
    sample = DecodePVLids(&sp, ++t, cache);
    require(sample.p_valid && sample.q_valid && sample.watts == -600 && sample.vars == 200);
    require(sample.phase_volts[0] == 240 && (sample.diagnostic_valid & 1));
    sample = DecodePVLids(&empty, ++t, cache);
    require(sample.p_valid && sample.q_valid && sample.watts == -600);
    t += 5;
    sample = DecodePVLids(&empty, t, cache);
    require(!sample.p_valid && !sample.q_valid && sample.diagnostic_valid == 0);
    put(pp, 18153531, -500);
    put(pp, 18153532, 300);
    put(pp, 18153658, 900);
    put(pp, 18153659, -700);
    put(pp, 18153660, -700);
    for (uint32_t lid : {18153520u, 18153521u, 18153522u})
        put(pp, lid, 120);
    sample = DecodePVLids(&pp, ++t, cache);
    require(sample.p_valid && sample.watts == -500 && sample.diagnostic_valid == 63);
    put(invalid, 18153531, std::numeric_limits<double>::quiet_NaN());
    ApiVariableType wrong = {};
    wrong.type = AGENT_API_UINT32;
    invalid.AddLidPair(18153532, wrong);
    sample = DecodePVLids(&invalid, ++t, cache);
    require(!sample.p_valid && !sample.q_valid);
    sample = DecodePVLids(&empty, ++t, cache);
    require(!sample.p_valid && !sample.q_valid);
    pv::atomic_write(
        path, "{\"status\":\"PASS\",\"checks\":" + std::to_string(checks) +
                  ",\"forms\":[\"singlephase\",\"polyphase\"],\"physical_meter_tested\":false}");
}
