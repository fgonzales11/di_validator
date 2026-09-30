#include "pv_core.hpp"
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace pv {
std::string quote(const std::string &text) {
    std::string s = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            s += '\\';
            s += static_cast<char>(c);
        } else if (c == '\n')
            s += "\\n";
        else if (c >= 32)
            s += static_cast<char>(c);
    }
    return s + '"';
}
std::string detection_json(const Detection &d) {
    std::ostringstream s;
    s << std::setprecision(17) << "{\"state\":" << quote(state_name(d.state))
      << ",\"candidate\":" << quote(state_name(d.candidate)) << ",\"streak\":" << d.streak
      << ",\"valid_days\":" << d.valid_days << ",\"export_days\":" << d.export_days
      << ",\"profile_days\":" << d.profile_days << ",\"export_fraction\":" << d.export_fraction
      << ",\"valley_fraction\":" << d.valley_fraction << ",\"valley_median\":" << d.valley_median
      << ",\"night_export\":" << (d.night_export ? "true" : "false")
      << ",\"model\":{\"valid\":" << (d.model.valid ? "true" : "false")
      << ",\"reason\":" << quote(reason_name(d.model.reason)) << ",\"b0\":" << d.model.b0
      << ",\"b1\":" << d.model.b1 << ",\"q_min\":" << d.model.q_min
      << ",\"q_max\":" << d.model.q_max << ",\"residual95\":" << d.model.residual95
      << ",\"r2\":" << d.model.r2 << ",\"nrmse\":" << d.model.nrmse << "}}";
    return s.str();
}
std::string result_json(const Result &r) {
    std::ostringstream s;
    s << std::setprecision(17) << "{\"start\":" << r.measured.start
      << ",\"state\":" << quote(state_name(r.state))
      << ",\"reason\":" << quote(reason_name(r.reason)) << ",\"flags\":" << r.flags
      << ",\"p_seconds\":" << r.measured.p_count << ",\"q_seconds\":" << r.measured.q_count
      << ",\"net_kw\":" << r.measured.p() << ",\"reactive_kvar\":" << r.measured.q()
      << ",\"import_kwh\":" << r.measured.import_kwh << ",\"export_kwh\":" << r.measured.export_kwh
      << ",\"available\":" << (r.available ? "true" : "false") << ",\"generation_kw\":";
    if (r.available)
        s << r.generation_kw;
    else
        s << "null";
    s << ",\"low_kw\":";
    if (r.available)
        s << r.low_kw;
    else
        s << "null";
    s << ",\"high_kw\":";
    if (r.available)
        s << r.high_kw;
    else
        s << "null";
    s << ",\"generation_kwh\":";
    if (r.available)
        s << r.generation_kwh;
    else
        s << "null";
    s << ",\"estimated_seconds\":" << r.estimated_seconds
      << ",\"state_changed\":" << (r.state_changed ? "true" : "false") << '}';
    return s.str();
}
static void integer(std::string &s, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i)
        s += static_cast<char>(value >> (8 * i));
}
static void floating(std::string &s, double v) {
    float f = static_cast<float>(v);
    uint32_t b;
    std::memcpy(&b, &f, 4);
    integer(s, b, 4);
}
static std::string base64(const std::string &data) {
    static const char *alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < data.size(); i += 3) {
        unsigned a = static_cast<unsigned char>(data[i]),
                 b = i + 1 < data.size() ? static_cast<unsigned char>(data[i + 1]) : 0;
        unsigned c = i + 2 < data.size() ? static_cast<unsigned char>(data[i + 2]) : 0;
        out += alphabet[a >> 2];
        out += alphabet[((a & 3) << 4) | (b >> 4)];
        out += i + 1 < data.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=';
        out += i + 2 < data.size() ? alphabet[c & 63] : '=';
    }
    return out;
}
std::string hourly_payload(const std::vector<Result> &rows) {
    if (rows.empty() || rows.size() > 4)
        throw std::invalid_argument("Hourly batch size");
    std::string b;
    integer(b, rows.front().measured.start, 8);
    integer(b, rows.front().config_id, 8);
    integer(b, rows.front().model_id, 8);
    integer(b, rows.size(), 1);
    double missing = std::numeric_limits<double>::quiet_NaN();
    for (const auto &r : rows) {
        if (r.config_id != rows.front().config_id || r.model_id != rows.front().model_id ||
            r.measured.start < rows.front().measured.start ||
            r.measured.start - rows.front().measured.start > 2700)
            throw std::invalid_argument("Mixed hourly provenance/window");
        integer(b, (r.measured.start - rows.front().measured.start) / 900, 1);
        floating(b, r.measured.p_count ? r.measured.p() : missing);
        floating(b, r.measured.import_kwh);
        floating(b, r.measured.export_kwh);
        floating(b, r.available ? r.generation_kw : missing);
        floating(b, r.available ? r.low_kw : missing);
        floating(b, r.available ? r.high_kw : missing);
        floating(b, r.available ? r.generation_kwh : missing);
        integer(b, r.measured.p_count, 2);
        integer(b, r.estimated_seconds, 2);
        integer(b, r.flags, 4);
        integer(b, r.state, 1);
        integer(b, r.reason, 1);
    }
    std::string payload = "PVH1:" + base64(b);
    if (payload.size() > 1024)
        throw std::length_error("Hourly payload too large");
    return payload;
}
std::string transition_payload(const Result &r) {
    std::ostringstream s;
    s << "98#PV#1#" << (r.measured.start + INTERVAL) << '#' << state_name(r.state) << '#'
      << hex64(r.config_id) << '#' << hex64(r.model_id) << "#screening_unvalidated";
    if (s.str().size() > 256)
        throw std::length_error("Event payload too large");
    return s.str();
}
} // namespace pv
