#include "replay_io.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
namespace pv {
static bool line(std::istream &s, std::string &v) {
    if (!std::getline(s, v))
        return false;
    if (!v.empty() && v.back() == '\r')
        v.pop_back();
    return true;
}
Config read_header(std::istream &s) {
    std::string text;
    if (!line(s, text) || text != "PVR1")
        throw std::invalid_argument("Expected PVR1");
    Config c;
    while (line(s, text) && text != "DATA") {
        size_t p = text.find('=');
        if (p == std::string::npos)
            throw std::invalid_argument("Invalid metadata");
        c.set(text.substr(0, p), text.substr(p + 1));
    }
    if (!line(s, text) || text != "utc_start,seconds,watts,vars,p_valid,q_valid,time_valid")
        throw std::invalid_argument("Invalid columns/units");
    c.validate();
    return c;
}
void replay(std::istream &s, const std::function<void(const MeterSample &)> &ingest, size_t chunk,
            const std::atomic<bool> *cancel) {
    if (!chunk || chunk > 100000)
        throw std::invalid_argument("Invalid chunk size");
    std::string text;
    uint64_t total = 0;
    while (line(s, text)) {
        if (text.empty())
            continue;
        for (char &c : text)
            if (c == ',')
                c = ' ';
        std::istringstream row(text);
        MeterSample sample;
        int64_t count;
        int p, q, t;
        std::string extra;
        if (!(row >> sample.utc >> count >> sample.watts >> sample.vars >> p >> q >> t) ||
            row >> extra || count < 1 || count > 86400 || (p != 0 && p != 1) ||
            (q != 0 && q != 1) || (t != 0 && t != 1) || !std::isfinite(sample.watts) ||
            !std::isfinite(sample.vars))
            throw std::invalid_argument("Invalid replay row");
        sample.p_valid = p;
        sample.q_valid = q;
        sample.time_valid = t;
        if (total + count > 100ULL * DAY)
            throw std::invalid_argument("Replay limit 100 days");
        for (int64_t i = 0; i < count; ++i) {
            if (total % chunk == 0 && cancel && cancel->load())
                throw std::runtime_error("cancelled");
            ingest(sample);
            ++sample.utc;
            ++total;
        }
    }
}
} // namespace pv
