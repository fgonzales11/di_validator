#include "replay_io.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::invalid_argument(
                "pv_replay fixture [--chunk N] [--state-in FILE] [--state-out FILE]");
        size_t chunk = 997;
        std::string in, out;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 == argc)
                throw std::invalid_argument("Missing option value");
            std::string option(argv[i]);
            if (option == "--chunk")
                chunk = std::stoul(argv[i + 1]);
            else if (option == "--state-in")
                in = argv[i + 1];
            else if (option == "--state-out")
                out = argv[i + 1];
            else
                throw std::invalid_argument("Unknown option");
        }
        std::ifstream input(argv[1]);
        pv::Engine engine(pv::read_header(input));
        if (!in.empty())
            engine.restore_state(pv::read_state_file(in));
        bool first = true;
        int64_t end = -1;
        std::cout << "{\"intervals\":[";
        auto drain = [&]() {
            for (const auto &r : engine.take_results()) {
                if (!first)
                    std::cout << ',';
                first = false;
                std::cout << pv::result_json(r);
            }
        };
        pv::replay(
            input,
            [&](const pv::MeterSample &s) {
                engine.ingest(s);
                if (s.time_valid)
                    end = std::max(end, s.utc + 1);
                drain();
            },
            chunk);
        if (end >= 0)
            engine.advance(end);
        drain();
        std::cout << "],\"detection\":" << pv::detection_json(engine.detection())
                  << ",\"unassigned_import_kwh\":" << engine.unassigned_import()
                  << ",\"unassigned_export_kwh\":" << engine.unassigned_export() << "}\n";
        if (!out.empty())
            pv::atomic_write(out, engine.serialize_state());
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
