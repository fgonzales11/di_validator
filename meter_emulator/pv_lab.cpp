#include "agent_feature.h"
#include "agent_constants.h"
#include "lab_stream.hpp"
#include <apiWrapper.h>
#include <Itron/LidList.h>
#include <limits>
#include <chrono>
#include <cmath>
#include <ctime>
using namespace common;
void CAgentFeature::LabWorker() {
    try {
        while (!Enabled() && !stopping_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        pv::Config cfg; { std::lock_guard<std::mutex> lock(mutex_); cfg = config_; }
        pv::Engine engine(cfg); pv::Delivery delivery; PVMetrologyCache cache;
        const std::string dir = lab::directory();
        const std::string state = "/usr/share/flash/587530241/587333633/pv";
        if (std::ifstream((dir + "/seed.pvs").c_str()).good()) engine.restore_state(pv::read_state_file(dir + "/seed.pvs"));
        std::ofstream evidence((dir + "/diagnostics.jsonl").c_str());
        auto send = [&](const std::string &p, bool event, bool alarm) {
            const int result = static_cast<int>(event ? apiLogHistEvt(2, p.c_str(), static_cast<uint32_t>(p.size()), kuiAgentFeatureID, alarm)
                : apiUpstreamForAgentData(kuiAgentFeatureID, p.c_str(), static_cast<uint32_t>(p.size())));
            lab::submitted(kuiAgentFeatureID, p, event, result); return result;
        };
        auto checkpoint = [&]() {
            pv::atomic_write(state + "/delivery.pvs", delivery.serialize());
            pv::atomic_write(state + "/analytics.pvs", engine.serialize_state());
        };
        auto publish = [&]() {
            auto results = engine.take_results();
            for (const auto &r : results) { evidence << pv::result_json(r) << '\n'; delivery.add(r, engine.detection()); }
            evidence.flush(); delivery.drain(send, std::time(0));
            lab::wait_output(8, &stopping_);
            if (!results.empty()) checkpoint();
        };
        size_t count = 0; unsigned sequence = 0; double clock = 0; int64_t last = -1;
        lab::Frame frame;
        while (!stopping_.load()) {
            if (!Enabled()) { cancel_.store(true); break; }
            if (!lab::read(frame, sequence, 9)) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            size_t expanded = 0;
            for (const auto &row : frame.rows) {
                if (stopping_.load() || !Enabled()) break;
                const unsigned offset = frame.kind == "CONST" ? 1 : 0;
                const double length = offset ? row[1] : 1;
                if (!std::isfinite(length) || length < 1 || length > 1024 || std::floor(length) != length)
                    throw std::runtime_error("Invalid constant-sample block");
                const unsigned seconds = static_cast<unsigned>(length);
                expanded += seconds; if (expanded > 1024) throw std::runtime_error("Expanded frame exceeds 1024 samples");
                if (!std::isfinite(row[0]) || row[0] < 0 || row[0] > 4102444800.0 || std::floor(row[0]) != row[0])
                    throw std::runtime_error("Invalid one-second UTC timestamp");
                for (unsigned j = 3 + offset; j <= 5 + offset; ++j) if (row[j] != 0 && row[j] != 1) throw std::runtime_error("Invalid field validity");
                LidList lids = AgentLidListCreate(); if (!lids) throw std::runtime_error("LidList allocation failed");
                LidList unchanged = AgentLidListCreate(); if (!unchanged) { AgentLidListFree(lids); throw std::runtime_error("LidList allocation failed"); }
                ApiVariableType v = {}; v.type = AGENT_API_DOUBLE;
                for (unsigned i = 0; i < 5; ++i) {
                    v.value.Double = i < 2 ? (row[3 + i + offset] ? row[1 + i + offset] : std::numeric_limits<double>::quiet_NaN()) : row[4 + i + offset];
                    const uint32_t lid = i < 2 ? 18153531u + i : 18153520u + i - 2;
                    // The ARM SDK exports the C LidList API but hides the C++
                    // AddLidPair symbol. Populate its documented typed map.
                    static_cast<Itron::AgentSDK::AgentLidList *>(lids)->mLidValuePairList[lid] = v;
                }
                for (unsigned second = 0; second < seconds; ++second) {
                    if (stopping_.load() || !Enabled()) break;
                    // Explicitly constant synthetic samples exercise unchanged-LID
                    // callback semantics without reparsing identical decimal text.
                    auto sample = DecodePVLids(second ? unchanged : lids, static_cast<int64_t>(row[0]) + second, cache);
                    sample.time_valid = row[5 + offset] != 0; Enqueue(sample);
                    { std::lock_guard<std::mutex> lock(mutex_);
                      if (queue_.empty()) throw std::runtime_error("Lab sample rejected by feature queue");
                      sample = queue_.front(); queue_.pop_front(); }
                    engine.ingest(sample); ++count; clock = static_cast<double>(sample.utc); if (sample.time_valid) last = sample.utc;
                }
                AgentLidListFree(lids); AgentLidListFree(unchanged);
            }
            if (frame.kind == "END" && last >= 0) engine.advance(last + 1);
            publish();
            if (frame.kind == "END") { delivery.flush_hour(); delivery.drain(send, std::time(0), true); delivery.drain(send, std::time(0), true); checkpoint(); }
            std::ostringstream details; details << ",\"pending\":" << delivery.pending() << ",\"rejected\":" << delivery.rejected()
                << ",\"dropped\":" << delivery.dropped() << ",\"queue\":0,\"detection\":" << pv::detection_json(engine.detection());
            lab::acknowledge(frame, count, count, clock, details.str()); ++sequence;
            if (frame.kind == "END") break;
        }
        delivery.flush_hour(); delivery.drain(send, std::time(0), true); delivery.drain(send, std::time(0), true); checkpoint();
    } catch (const std::exception &e) { lab::error(e.what()); }
    lab::atomic(lab::directory() + "/worker-finished.json", "{\"finished\":true}");
}
