#include "agent_feature.h"
#include "lab_stream.hpp"
#include "replay_io.hpp"
#include <chrono>
#include <cmath>
void CAgentFeature::LabWorker() {
    try {
        while (!Enabled() && !stopping_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        fault::Segment recording;
        std::ifstream metadata((lab::directory() + "/waveform.meta").c_str()); metadata >> recording.fs >> recording.f0;
        if (!metadata || recording.fs <= 0 || recording.f0 <= 0) throw std::runtime_error("Missing waveform metadata");
        std::ofstream evidence((lab::directory() + "/diagnostics.jsonl").c_str());
        size_t received = 0, processed = 0; unsigned sequence = 0, rejected = 0; lab::Frame frame;
        while (!stopping_.load()) {
            if (!Enabled()) { cancel_.store(true); break; }
            if (!lab::read(frame, sequence, 7)) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            for (const auto &row : frame.rows) {
                if (++received > 200001) throw std::runtime_error("Fault recording exceeds 200001 sample limit");
                if (!std::isfinite(row[0])) throw std::runtime_error("Invalid waveform timestamp");
                recording.time.push_back(row[0]); for (unsigned c = 0; c < 6; ++c) recording.samples[c].push_back(row[c + 1]);
            }
            if (frame.kind == "END") {
                fault::Config cfg; { std::lock_guard<std::mutex> lock(mutex_); cfg = config_; cancel_.store(false); }
                auto segments = fault::split_record(recording, cfg);
                for (size_t i = 0; i < segments.size(); ++i) {
                    { std::lock_guard<std::mutex> lock(mutex_); cfg = config_; }
                    fault::Diagnostics diagnostics;
                    auto result = fault::analyze_segment(segments[i], cfg, static_cast<int>(i), &diagnostics, &cancel_);
                    fault::write_json(evidence, result, &diagnostics); evidence << '\n'; evidence.flush();
                    std::ostringstream id; id << frame.run << '-' << i;
                    try { Publish(result, cfg, "meter-lab", id.str()); }
                    catch (const std::exception &e) {
                        if (cancel_.load() || stopping_.load()) throw;
                        ++rejected; std::ofstream errors((lab::directory() + "/delivery.log").c_str(), std::ios::app); errors << e.what() << '\n';
                    }
                    lab::wait_output(8, &stopping_);
                    processed += segments[i].time.size();
                    std::ostringstream progress;
                    progress << "{\"processed\":" << processed << ",\"segments\":" << (i + 1) << '}';
                    lab::atomic(lab::directory() + "/analysis-progress.json", progress.str());
                }
                processed = received;
            }
            std::ostringstream extra; extra << ",\"queue\":0,\"pending\":0,\"rejected\":" << rejected;
            lab::acknowledge(frame, received, processed, recording.time.empty() ? 0 : recording.time.back(), extra.str()); ++sequence;
            if (frame.kind == "END") break;
        }
    } catch (const std::exception &e) { lab::error(e.what()); }
    lab::atomic(lab::directory() + "/worker-finished.json", "{\"finished\":true}");
}
