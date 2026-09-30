#include "replay_io.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::runtime_error(
                "Usage: fault_replay recording.wave [--diagnostics] [--distance] [--chunk N]");
        bool diagnostics = false, distance = false;
        std::size_t chunk = 256;
        for (int i = 2; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--diagnostics")
                diagnostics = true;
            else if (arg == "--distance")
                distance = true;
            else if (arg == "--chunk" && i + 1 < argc)
                chunk = static_cast<std::size_t>(std::stoul(argv[++i]));
            else
                throw std::runtime_error("Unknown replay option");
        }
        const fault::Recording recording = fault::read_recording(argv[1], chunk);
        if (distance) {
            fault::Phases values = recording.input.samples;
            for (std::size_t c = 0; c < 6; ++c)
                for (std::size_t i = 0; i < values[c].size(); ++i)
                    values[c][i] /= 1000;
            fault::write_distance(
                std::cout, fault::estimate_distance(values, recording.input.fs, recording.input.f0,
                                                    0, static_cast<int>(values[0].size()),
                                                    recording.config, recording.config.fault_loop));
        } else {
            const std::vector<fault::Segment> segments =
                fault::split_record(recording.input, recording.config);
            std::cout << "{\"config_id\":" << fault::quote(fault::config_id(recording.config))
                      << ",\"segments\":[";
            for (std::size_t i = 0; i < segments.size(); ++i) {
                if (i)
                    std::cout << ',';
                fault::Diagnostics arrays;
                const fault::Result result = fault::analyze_segment(
                    segments[i], recording.config, static_cast<int>(i), diagnostics ? &arrays : 0);
                fault::write_json(std::cout, result, diagnostics ? &arrays : 0);
            }
            std::cout << "]}";
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
