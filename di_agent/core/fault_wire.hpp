#pragma once
#include "fault_core.hpp"
namespace fault {
struct Payloads {
    std::string summary, alarm;
};
std::string quote(const std::string &value);
Payloads make_payloads(const Result &result, const Config &config, const std::string &source,
                       const std::string &correlation);
} // namespace fault
