#pragma once
#include "fault_core.hpp"
#include <iosfwd>
namespace fault {
struct Recording {
    Segment input;
    Config config;
    std::string source_id;
};
Recording read_recording(const std::string &file, std::size_t chunk = 256);
void write_json(std::ostream &out, const Result &result, const Diagnostics *diagnostics = 0);
void write_distance(std::ostream &out, const Distance &value);
std::string quote(const std::string &value);
} // namespace fault
