#include "agent_feature.h"
#ifdef DI_EMULATOR_REPLAY
#include "lab_stream.hpp"
#endif
#include "agent_config.h"
#include "agent_constants.h"
#include "fault_wire.hpp"
#include <Itron/AgentApi.h>
#include <apiWrapper.h>
#include <CoutLogger.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#ifdef NODAEMON
#include "replay_io.hpp"
#include <dirent.h>
#include <algorithm>
#include <cstdio>
bool ApplyFaultHostConfiguration(const std::string &xml);
#endif
using namespace common;
CAgentFeature::CAgentFeature()
    : enabled_(false), started_(false), stopping_(false), cancel_(false) {}
CAgentFeature::~CAgentFeature() {
    StopReplay();
}
void CAgentFeature::InitializeConfig() {}
bool CAgentFeature::Enabled() {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}
void CAgentFeature::UpdateAgentFeatureConfig(const stGeneralConfig general,
                                             const stAgentFeatureConfig feature) {
    try {
        fault::Config next;
        for (std::map<std::string, std::string>::const_iterator it = feature.parameters.begin();
             it != feature.parameters.end(); ++it)
            next.set(it->first, it->second);
        next.validate();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            config_ = next;
            enabled_ = general.uiFeatureEnabled;
            if (!enabled_)
                cancel_.store(true);
        }
        const std::string message = "FAULT_CONFIG_APPLIED " + fault::config_id(next) +
                                    (general.uiFeatureEnabled ? " enabled" : " disabled");
        logMessage("INFO", message.c_str(), message.size());
#ifdef NODAEMON
        const char *folder = std::getenv("FAULT_REPLAY_DIR");
        if (folder) {
            std::ofstream out((std::string(folder) + "/configuration.json").c_str());
            out << "{\"valid\":true,\"enabled\":" << (general.uiFeatureEnabled ? "true" : "false")
                << ",\"config\":" << fault::quote(fault::config_id(next)) << '}';
        }
#endif
    } catch (const std::exception &error) {
        const std::string message = "FAULT_CONFIG_REJECTED " + std::string(error.what());
        logMessage("ERROR", message.c_str(), message.size());
#ifdef NODAEMON
        const char *folder = std::getenv("FAULT_REPLAY_DIR");
        if (folder) {
            std::ofstream out((std::string(folder) + "/configuration.json").c_str());
            out << "{\"valid\":false,\"reason\":" << fault::quote(error.what()) << '}';
        }
#endif
    }
}
void CAgentFeature::StartReplay() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_)
        return;
    started_ = true;
#ifdef DI_EMULATOR_REPLAY
    if (std::getenv("METER_LAB_DIR")) {
        stopping_.store(false); cancel_.store(false);
        worker_ = std::thread(&CAgentFeature::LabWorker, this); return;
    }
#endif
#ifdef NODAEMON
    const char *folder = std::getenv("FAULT_REPLAY_DIR");
    if (folder) {
        stopping_.store(false);
        worker_ = std::thread(&CAgentFeature::ReplayWorker, this);
        configuration_worker_ = std::thread(&CAgentFeature::ConfigurationWorker, this);
        std::ofstream state((std::string(folder) + "/lifecycle.json").c_str());
        state << "{\"running\":true}";
    }
#else
    const char *message =
        "FaultLocationAgent: live waveform adapter is not configured; no metrology subscription.";
    logMessage("INFO", message, std::strlen(message));
#endif
}
void CAgentFeature::StopReplay() {
    stopping_.store(true);
    cancel_.store(true);
    if (worker_.joinable())
        worker_.join();
    if (configuration_worker_.joinable())
        configuration_worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
#ifdef NODAEMON
    const char *folder = std::getenv("FAULT_REPLAY_DIR");
    if (folder) {
        std::ofstream state((std::string(folder) + "/lifecycle.json").c_str());
        state << "{\"running\":false}";
    }
#endif
}
void CAgentFeature::Publish(const fault::Result &result, const fault::Config &cfg,
                            const std::string &source, const std::string &id) {
    const fault::Payloads payload = fault::make_payloads(result, cfg, source, id);
    if (payload.summary.empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || stopping_.load() || cancel_.load())
        throw std::runtime_error("cancelled");
    // No application retries. The SDK may enqueue writes internally; the lab
    // observes their actual completion separately from this wrapper's return.
    const AgentApiReturnType data = apiUpstreamForAgentData(
        kuiAgentFeatureID, payload.summary.c_str(), static_cast<uint32_t>(payload.summary.size()));
#ifdef DI_EMULATOR_REPLAY
    lab::submitted(kuiAgentFeatureID, payload.summary, false, static_cast<int>(data));
#endif
    if (data != API_SUCCESS) {
        std::ostringstream error;
        error << "Summary write rejected by SDK: " << data;
        throw std::runtime_error(error.str());
    }
    if (!payload.alarm.empty()) {
        const AgentApiReturnType alarm =
            apiLogHistEvt(1, payload.alarm.c_str(), static_cast<uint32_t>(payload.alarm.size()),
                          kuiAgentFeatureID, true);
#ifdef DI_EMULATOR_REPLAY
        lab::submitted(kuiAgentFeatureID, payload.alarm, true, static_cast<int>(alarm));
#endif
        if (alarm != API_SUCCESS) {
            std::ostringstream error;
            error << "Alarm write rejected by SDK after summary: " << alarm;
            throw std::runtime_error(error.str());
        }
    }
}
void CAgentFeature::ConfigurationWorker() {
#ifdef NODAEMON
    const std::string directory(std::getenv("FAULT_REPLAY_DIR"));
    while (!stopping_.load()) {
        // Host-only ingress exercises the same SDK XML parser and registered callbacks.
        const std::string update = directory + "/configuration.request.xml";
        if (std::ifstream(update.c_str()).good()) {
            std::ifstream input(update.c_str());
            std::ostringstream xml;
            xml << input.rdbuf();
            ApplyFaultHostConfiguration(xml.str());
            std::rename(update.c_str(), (directory + "/configuration.applied.xml").c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#endif
}
void CAgentFeature::ReplayWorker() {
#ifdef NODAEMON
    const std::string directory(std::getenv("FAULT_REPLAY_DIR"));
    while (!stopping_.load()) {
        if (!Enabled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        std::vector<std::string> requests;
        DIR *dir = opendir(directory.c_str());
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != 0) {
                std::string name(entry->d_name);
                if (name.size() > 8 && name.substr(name.size() - 8) == ".request")
                    requests.push_back(name);
            }
            closedir(dir);
        }
        std::sort(requests.begin(), requests.end());
        for (std::size_t index = 0; index < requests.size() && !stopping_.load(); ++index) {
            const std::string request = directory + "/" + requests[index],
                              id = requests[index].substr(0, requests[index].size() - 8);
            const std::string completion = directory + "/" + id + ".done.json";
            if (std::ifstream(completion.c_str()).good())
                continue;
            if (id.size() > 40 || id.find_first_not_of("0123456789abcdef-") != std::string::npos)
                continue;
            std::ofstream result((completion + ".tmp").c_str());
            try {
                fault::Config cfg;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!enabled_)
                        continue;
                    cfg = config_;
                    cancel_.store(false);
                }
                {
                    std::ofstream running((directory + "/" + id + ".running").c_str());
                    running << "processing\n";
                }
                std::ifstream file(request.c_str());
                std::string path;
                std::getline(file, path);
                fault::Recording recording = fault::read_recording(path);
                const std::vector<fault::Segment> segments =
                    fault::split_record(recording.input, cfg);
                std::ostringstream output;
                output << "{\"status\":\"complete\",\"segments\":[";
                for (std::size_t i = 0; i < segments.size(); ++i) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        cfg = config_;
                        if (!enabled_ || stopping_.load())
                            throw std::runtime_error("cancelled");
                    }
                    fault::Result analysis =
                        fault::analyze_segment(segments[i], cfg, static_cast<int>(i), 0, &cancel_);
                    std::ostringstream event;
                    event << id << '-' << i;
                    Publish(analysis, cfg, recording.source_id, event.str());
                    if (i) {
                        output << ',';
                    }
                    fault::write_json(output, analysis);
                }
                output << "]}";
                result << output.str();
            } catch (const std::exception &error) {
                result << "{\"status\":\"error\",\"reason\":" << fault::quote(error.what()) << '}';
            }
            result.close();
            if (std::rename((completion + ".tmp").c_str(), completion.c_str()) != 0)
                logMessage("ERROR", "Cannot publish replay completion", 32);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#endif
}
