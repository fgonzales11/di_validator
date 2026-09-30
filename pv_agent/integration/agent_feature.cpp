#include "agent_feature.h"
#include <agent_constants.h>
#include <apiWrapper.h>
#include <CoutLogger.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#ifdef DI_EMULATOR_REPLAY
#include "lab_stream.hpp"
#endif
#ifdef NODAEMON
#include "replay_io.hpp"
#include <Itron/LidList.h>
#include "metrology_checks.hpp"
#include <dirent.h>
#include <algorithm>
bool ApplyPVHostConfiguration(const std::string &);
#endif
using namespace common;
template <typename T> static std::string integer_text(T value) {
    std::ostringstream out;
    out << value;
    return out.str();
}
static void log_pv(const std::string &s) {
    logMessage("INFO", s.c_str(), static_cast<uint32_t>(s.size()));
}
static std::string folder() {
#ifdef NODAEMON
    const char *p = std::getenv("PV_STATE_DIR");
    if (p)
        return p;
#endif
    return "/usr/share/flash/587530241/587333633/pv";
}
CAgentFeature::CAgentFeature() : stopping_(false), cancel_(false), dropped_(0) {}
CAgentFeature::~CAgentFeature() {
    StopReplay();
}
void CAgentFeature::InitializeConfig() {}
bool CAgentFeature::Enabled() {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}
void CAgentFeature::UpdateAgentFeatureConfig(stGeneralConfig general,
                                             stAgentFeatureConfig feature) {
    bool valid = false;
    std::string reason;
    try {
        pv::Config cfg;
        for (const auto &entry : feature.parameters)
            cfg.set(entry.first, entry.second);
        cfg.validate();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            config_ = cfg;
            enabled_ = general.uiFeatureEnabled;
            ++generation_;
            if (!enabled_) {
                cancel_.store(true);
                queue_.clear();
            }
        }
        valid = true;
        log_pv("PV_CONFIG_APPLIED " + pv::hex64(cfg.id()));
    } catch (const std::exception &e) {
        reason = e.what();
        log_pv("PV_CONFIG_REJECTED " + reason);
    }
#ifdef NODAEMON
    const char *path = std::getenv("PV_REPLAY_DIR");
    if (path) {
        std::ostringstream s;
        s << "{\"valid\":" << (valid ? "true" : "false")
          << ",\"enabled\":" << (Enabled() ? "true" : "false")
          << ",\"reason\":" << pv::quote(reason) << '}';
        pv::atomic_write(std::string(path) + "/configuration.json", s.str());
    }
#else
    (void)valid;
#endif
}
void CAgentFeature::StartReplay() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_)
        return;
    started_ = true;
    stopping_.store(false);
    cancel_.store(false);
    worker_ = std::thread(&CAgentFeature::Worker, this);
#ifdef NODAEMON
    configuration_worker_ = std::thread(&CAgentFeature::ConfigurationWorker, this);
    const char *p = std::getenv("PV_REPLAY_DIR");
    if (p)
        pv::atomic_write(std::string(p) + "/lifecycle.json", "{\"running\":true}");
#endif
}
void CAgentFeature::StopReplay() {
    stopping_.store(true);
    cancel_.store(true);
    if (worker_.joinable())
        worker_.join();
    if (configuration_worker_.joinable())
        configuration_worker_.join();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = false;
        queue_.clear();
    }
#ifdef NODAEMON
    const char *p = std::getenv("PV_REPLAY_DIR");
    if (p)
        pv::atomic_write(std::string(p) + "/lifecycle.json", "{\"running\":false}");
#endif
}
void CAgentFeature::Enqueue(const pv::MeterSample &sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || stopping_.load())
        return;
    if (queue_.size() >= 4096) {
        ++dropped_;
        return;
    }
    queue_.push_back(sample);
}
void CAgentFeature::ConfigurationWorker() {
#ifdef NODAEMON
    const char *p = std::getenv("PV_REPLAY_DIR");
    if (!p)
        return;
    std::string dir(p);
    while (!stopping_.load()) {
        std::string name = dir + "/configuration.request.xml";
        if (std::ifstream(name.c_str()).good()) {
            std::ifstream in(name.c_str());
            std::ostringstream xml;
            xml << in.rdbuf();
            ApplyPVHostConfiguration(xml.str());
            ::rename(name.c_str(), (dir + "/configuration.applied.xml").c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#endif
}
void CAgentFeature::Worker() {
#ifdef DI_EMULATOR_REPLAY
    if (std::getenv("METER_LAB_DIR")) { LabWorker(); return; }
#endif
    pv::Config initial;
    uint64_t seen;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initial = config_;
        seen = generation_;
    }
    pv::Engine engine(initial);
    pv::Delivery delivery;
#ifdef DI_EMULATOR_REPLAY
    const char *metrology_path = std::getenv("METER_LAB_METROLOGY_DIR");
    std::ofstream metrology_evidence, metrology_observations;
    size_t metrology_processed = 0;
    if (metrology_path) {
        metrology_evidence.open((std::string(metrology_path) + "/diagnostics.jsonl").c_str());
        metrology_observations.open((std::string(metrology_path) + "/observations.csv").c_str());
    }
#endif
    int64_t last_diagnostic_hour = -1;
    bool subscribed = false, attempted = false, was_enabled = false;
    std::string dir = folder();
    ::mkdir(dir.c_str(), 0750);
    auto checkpoint = [&]() {
        // Delivery first: recovered interval IDs suppress repeats if the core file
        // still describes the preceding interval after a power interruption.
        pv::atomic_write(dir + "/delivery.pvs", delivery.serialize());
        pv::atomic_write(dir + "/analytics.pvs", engine.serialize_state());
    };
    try {
        engine.restore_state(pv::read_state_file(dir + "/analytics.pvs"));
        log_pv("PV_ANALYTICS_RESTORED");
    } catch (const std::exception &e) {
        log_pv(std::string("PV_WARMUP ") + e.what());
    }
    try {
        delivery.restore(pv::read_state_file(dir + "/delivery.pvs"));
    } catch (const std::exception &e) {
        log_pv(std::string("PV_DELIVERY_NEW ") + e.what());
    }
    auto sync_config = [&]() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (generation_ != seen) {
            engine.configure(config_);
            seen = generation_;
        }
    };
    auto send = [&](const std::string &payload, bool event, bool alarm) {
        int rc = static_cast<int>(
            event ? apiLogHistEvt(2, payload.c_str(), static_cast<uint32_t>(payload.size()),
                                  kuiAgentFeatureID, alarm)
                  : apiUpstreamForAgentData(kuiAgentFeatureID, payload.c_str(),
                                            static_cast<uint32_t>(payload.size())));
        if (rc)
            log_pv("PV_WRITE_REJECTED " + integer_text(rc));
#ifdef DI_EMULATOR_REPLAY
        lab::submitted(kuiAgentFeatureID, payload, event, rc);
#endif
        return rc;
    };
    auto publish = [&](std::ostream *evidence) {
        auto rows = engine.take_results();
        for (const auto &r : rows) {
            delivery.add(r, engine.detection());
#ifdef DI_EMULATOR_REPLAY
            if (metrology_path) { metrology_evidence << pv::result_json(r) << '\n'; metrology_evidence.flush(); }
#endif
            if (evidence)
                *evidence << pv::result_json(r) << '\n';
        }
        if (!rows.empty()) {
            checkpoint();
            delivery.drain(send, std::time(0));
            pv::atomic_write(dir + "/delivery.pvs", delivery.serialize());
        }
    };
    try {
#ifdef NODAEMON
        if (std::getenv("PV_TEST_METROLOGY")) {
            metrology_contracts(dir + "/metrology-checks.json");
            const char probe[] = "PV unauthorized feature probe";
            int rc =
                static_cast<int>(apiUpstreamForAgentData(0x2303ffff, probe, sizeof(probe) - 1));
            std::string large(1025, 'x');
            int size_rc = static_cast<int>(apiUpstreamForAgentData(
                kuiAgentFeatureID, large.c_str(), static_cast<uint32_t>(large.size())));
            pv::atomic_write(dir + "/policy-check.json",
                             "{\"unregistered_feature_status\":" + integer_text(rc) +
                                 ",\"oversize_status\":" + integer_text(size_rc) + "}");
        }
#endif
        while (!stopping_.load()) {
#ifdef DI_EMULATOR_REPLAY
            if (metrology_path && std::ifstream((std::string(metrology_path) + "/finish.request").c_str()).good())
                break;
#endif
            sync_config();
            bool enabled = Enabled();
            if (!enabled && was_enabled) {
                delivery.flush_hour();
                delivery.drain(send, std::time(0), true);
                delivery.drain(send, std::time(0), true);
                checkpoint();
            }
            was_enabled = enabled;
            if (!enabled) {
                if (subscribed)
                    UnsubscribePV();
                subscribed = false;
                attempted = false;
            }
            if (enabled && !attempted) {
                int rc = SubscribePV();
                subscribed = rc == 0;
                attempted = true;
                log_pv("PV_SUBSCRIPTION_RESULT " + integer_text(rc));
            }
            if (enabled) {
                std::deque<pv::MeterSample> work;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    size_t count = std::min<size_t>(256, queue_.size());
                    while (count--) {
                        work.push_back(queue_.front());
                        queue_.pop_front();
                    }
                }
                unsigned lost = dropped_.exchange(0);
                if (lost)
                    log_pv("PV_QUEUE_DROPPED " + integer_text(lost));
                for (auto sample : work) {
                    if (lost) {
                        sample.flags |= pv::QUEUE_LOSS;
                        lost = 0;
                    }
                    if (sample.diagnostic_valid && sample.utc / 3600 != last_diagnostic_hour) {
                        std::ostringstream info;
                        info << "PV_PHASE_DIAGNOSTIC utc=" << sample.utc
                             << " mask=" << static_cast<unsigned>(sample.diagnostic_valid);
                        for (unsigned i = 0; i < 3; ++i)
                            info << " V" << i << '=' << sample.phase_volts[i] << " W" << i << '='
                                 << sample.phase_watts[i];
                        log_pv(info.str());
                        last_diagnostic_hour = sample.utc / 3600;
                    }
                    engine.ingest(sample);
#ifdef DI_EMULATOR_REPLAY
                    if (metrology_path) {
                        ++metrology_processed;
                        metrology_observations << std::setprecision(17) << sample.utc << ',' << sample.watts << ',' << sample.vars << ','
                            << sample.p_valid << ',' << sample.q_valid << ',' << sample.time_valid << '\n'; metrology_observations.flush();
                        std::ostringstream progress;
                        progress << "{\"processed\":" << metrology_processed << ",\"scenario_time\":" << sample.utc
                            << ",\"pending\":" << delivery.pending() << ",\"rejected\":" << delivery.rejected()
                            << ",\"detection\":" << pv::detection_json(engine.detection()) << '}';
                        lab::atomic(std::string(metrology_path) + "/metrology-progress.json", progress.str());
                    }
#endif
                    publish(0);
                }
            }
#ifdef NODAEMON
            const char *spool = std::getenv("PV_REPLAY_DIR");
            if (spool && enabled) {
                std::vector<std::string> names;
                DIR *list = ::opendir(spool);
                if (list) {
                    dirent *e;
                    while ((e = ::readdir(list))) {
                        std::string n(e->d_name);
                        if (n.size() > 8 && n.substr(n.size() - 8) == ".request")
                            names.push_back(n);
                    }
                    ::closedir(list);
                }
                std::sort(names.begin(), names.end());
                for (const auto &name : names) {
                    if (!Enabled() || stopping_.load())
                        break;
                    std::string id = name.substr(0, name.size() - 8);
                    if (id.size() > 40 ||
                        id.find_first_not_of("0123456789abcdef-") != std::string::npos)
                        continue;
                    std::string completion = std::string(spool) + "/" + id + ".done.json";
                    if (std::ifstream(completion.c_str()).good())
                        continue;
                    cancel_.store(false);
                    std::ostringstream answer;
                    try {
                        pv::atomic_write(std::string(spool) + "/" + id + ".running", "processing");
                        std::ifstream request((std::string(spool) + "/" + name).c_str());
                        std::string path;
                        std::getline(request, path);
                        std::ifstream input(path.c_str());
                        pv::read_header(input); // Active SDK config is authoritative.
                        std::ofstream evidence(
                            (std::string(spool) + "/" + id + ".intervals.jsonl").c_str());
                        PVMetrologyCache cache;
                        int64_t end = -1;
                        size_t samples = 0;
                        pv::replay(
                            input,
                            [&](const pv::MeterSample &original) {
                                if (++samples % 257 == 0) {
                                    sync_config();
                                    if (!Enabled() || stopping_.load())
                                        throw std::runtime_error("cancelled");
                                }
                                LidList lids = AgentLidListCreate();
                                ApiVariableType value = {};
                                value.type = AGENT_API_DOUBLE;
                                value.value.Double = original.p_valid
                                                         ? original.watts
                                                         : std::numeric_limits<double>::quiet_NaN();
                                static_cast<Itron::AgentSDK::AgentLidList *>(lids)->AddLidPair(
                                    18153531, value);
                                value.value.Double = original.q_valid
                                                         ? original.vars
                                                         : std::numeric_limits<double>::quiet_NaN();
                                static_cast<Itron::AgentSDK::AgentLidList *>(lids)->AddLidPair(
                                    18153532, value);
                                auto sample = DecodePVLids(lids, original.utc, cache);
                                AgentLidListFree(lids);
                                sample.time_valid = original.time_valid;
                                engine.ingest(sample);
                                if (sample.time_valid)
                                    end = std::max(end, sample.utc + 1);
                                publish(&evidence);
                            },
                            997, &cancel_);
                        if (end >= 0)
                            engine.advance(end);
                        publish(&evidence);
                        checkpoint();
                        delivery.drain(send, std::time(0));
                        checkpoint();
                        answer << "{\"status\":\"complete\",\"samples\":" << samples
                               << ",\"detection\":" << pv::detection_json(engine.detection())
                               << ",\"pending\":" << delivery.pending()
                               << ",\"rejected\":" << delivery.rejected()
                               << ",\"dropped\":" << delivery.dropped() << '}';
                    } catch (const std::exception &e) {
                        answer << "{\"status\":\"error\",\"reason\":" << pv::quote(e.what()) << '}';
                    }
                    pv::atomic_write(completion, answer.str());
                }
            }
#endif
#ifdef NODAEMON
            if (!std::getenv("PV_REPLAY_DIR"))
#endif
            {
                if (enabled) {
                    engine.advance(std::time(0));
                    publish(0);
                }
            }
            unsigned before_bytes = delivery.sent_bytes(), before_rejected = delivery.rejected(),
                     before_dropped = delivery.dropped();
            size_t before_pending = delivery.pending();
            delivery.drain(send, std::time(0));
            if (before_bytes != delivery.sent_bytes() || before_rejected != delivery.rejected() ||
                before_dropped != delivery.dropped() || before_pending != delivery.pending())
                pv::atomic_write(dir + "/delivery.pvs", delivery.serialize());
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        delivery.flush_hour();
        delivery.drain(send, std::time(0), true);
        delivery.drain(send, std::time(0), true);
        checkpoint();
    } catch (const std::exception &e) {
        log_pv(std::string("PV_WORKER_FAILURE ") + e.what());
    }
    if (subscribed)
        UnsubscribePV();
#ifdef DI_EMULATOR_REPLAY
    if (metrology_path)
        lab::atomic(std::string(metrology_path) + "/worker-finished.json", "{\"finished\":true}");
#endif
}
