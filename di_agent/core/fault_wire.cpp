#include "fault_wire.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace fault {
std::string quote(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (std::size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c == '"' || c == '\\')
            out << '\\' << value[i];
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<unsigned>(c) << std::dec;
        else
            out << value[i];
    }
    out << '"';
    return out.str();
}
Payloads make_payloads(const Result &r, const Config &cfg, const std::string &source,
                       const std::string &correlation) {
    Payloads p;
    if (r.status != "analyzed")
        return p;
    std::ostringstream out;
    out << std::setprecision(12) << "{\"v\":1,\"id\":" << quote(correlation)
        << ",\"source\":" << quote(source)
        << ",\"pipeline\":\"notebook-reactance-2\",\"config\":" << quote(config_id(cfg))
        << ",\"clock\":\"recording_relative\",\"onset_s\":" << r.onset
        << ",\"available_s\":" << r.emitted_at << ",\"inception\":" << quote(r.inception_source)
        << ",\"type\":" << quote(r.kind) << ",\"loop\":" << quote(r.distance.loop)
        << ",\"distance_status\":" << quote(r.distance.status) << ",\"distance_km\":";
    if (r.distance.status == "estimated")
        out << r.distance.km;
    else
        out << "null";
    out << ",\"reason\":" << quote(r.distance.reason)
        << ",\"within_line\":" << (r.distance.within_line ? "true" : "false")
        << ",\"ground_compensated\":" << (r.distance.ground_compensated ? "true" : "false")
        << ",\"uncompensated_ground\":" << (r.distance.uncompensated ? "true" : "false")
        << ",\"line_verified\":" << (cfg.line_parameters_verified ? "true" : "false")
        << ",\"cycles_used\":" << r.distance.cycles.size()
        << ",\"cycles_skipped\":" << r.distance.skipped << ",\"min_km\":" << r.distance.minimum
        << ",\"max_km\":" << r.distance.maximum
        << ",\"measurement\":" << quote(r.distance.measurement)
        << ",\"r_ohm\":" << r.distance.resistance << ",\"x_ohm\":" << r.distance.reactance << '}';
    p.summary = out.str();
    if (r.inception_source == "detected") {
        std::ostringstream alarm;
        alarm << std::setprecision(12) << "98#FAULT#1#" << correlation << '#' << config_id(cfg)
              << '#' << r.onset << '#' << r.kind << '#' << r.distance.loop << '#';
        // Withheld results name their status (unavailable, out_of_range, behind_relay, inconsistent).
        if (r.distance.status == "estimated")
            alarm << r.distance.km;
        else
            alarm << r.distance.status;
        alarm << '#' << (cfg.line_parameters_verified ? "verified" : "unverified") << '#'
              << (r.distance.uncompensated ? "uncompensated" : "compensated_or_phase");
        p.alarm = alarm.str();
    }
    if (p.summary.size() > 1024 || p.alarm.size() > 256)
        throw std::runtime_error("DI payload exceeds byte limit");
    return p;
}
} // namespace fault
