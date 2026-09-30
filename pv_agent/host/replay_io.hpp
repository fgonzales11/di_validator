#pragma once
#include "pv_core.hpp"
#include <atomic>
#include <functional>
#include <istream>
namespace pv {
Config read_header(std::istream &);
void replay(std::istream &, const std::function<void(const MeterSample &)> &, size_t chunk = 997,
            const std::atomic<bool> *cancel = 0);
} // namespace pv
