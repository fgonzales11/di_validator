#include "replay_io.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fault {
static double read_number(const std::string &text) {
    char *end = 0;
    double x = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end)
        throw std::runtime_error("Malformed waveform number");
    return x;
}
static bool read_line(std::istream &input, std::string &line) {
    if (!std::getline(input, line))
        return false;
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    return true;
}
Recording read_recording(const std::string &file, std::size_t chunk) {
    if (chunk == 0 || chunk > 200001)
        throw std::runtime_error("Invalid replay chunk size");
    std::ifstream input(file.c_str());
    if (!input)
        throw std::runtime_error("Cannot open waveform: " + file);
    std::string line;
    read_line(input, line);
    if (line != "FLA1")
        throw std::runtime_error("Expected FLA1 waveform format");
    Recording out;
    while (read_line(input, line) && line != "DATA") {
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos)
            throw std::runtime_error("Invalid waveform metadata");
        const std::string key = line.substr(0, separator), value = line.substr(separator + 1);
        if (key == "fs")
            out.input.fs = read_number(value);
        else if (key == "f0")
            out.input.f0 = read_number(value);
        else if (key == "source_id")
            out.source_id = value;
        else
            out.config.set(key, value);
    }
    read_line(input, line);
    if (line != "offset_seconds,IA_A,IB_A,IC_A,UA_V,UB_V,UC_V")
        throw std::runtime_error("Six calibrated waveform channels in A/V required");
    if (out.source_id.empty() || out.source_id.size() > 64 ||
        out.source_id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
            std::string::npos)
        throw std::runtime_error("Invalid source identifier");
    bool eof = false;
    while (!eof) {
        // Parse in bounded input batches; boundaries do not define electrical segments.
        for (std::size_t row = 0; row < chunk; ++row) {
            if (!read_line(input, line)) {
                eof = true;
                break;
            }
            if (line.empty())
                continue;
            if (out.input.time.size() >= 200001)
                throw std::runtime_error("Waveform sample limit exceeded");
            std::istringstream fields(line);
            std::array<double, 7> values;
            for (std::size_t c = 0; c < 7; ++c) {
                std::string item;
                if (!std::getline(fields, item, ','))
                    throw std::runtime_error("Missing waveform channel");
                values[c] = read_number(item);
            }
            std::string extra;
            if (std::getline(fields, extra, ','))
                throw std::runtime_error("Extra waveform column");
            out.input.time.push_back(values[0]);
            for (std::size_t c = 0; c < 6; ++c)
                out.input.samples[c].push_back(values[c + 1]);
        }
    }
    out.config.validate();
    return out;
}
static void vector_json(std::ostream &out, const Signal &values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i)
            out << ',';
        if (std::isfinite(values[i]))
            out << values[i];
        else
            out << "null";
    }
    out << ']';
}
template <class T> static void matrix_json(std::ostream &out, const T &values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i)
            out << ',';
        vector_json(out, values[i]);
    }
    out << ']';
}
void write_distance(std::ostream &out, const Distance &d) {
    out << std::setprecision(17) << "{\"status\":" << quote(d.status);
    if (d.status == "unavailable") {
        out << ",\"reason\":" << quote(d.reason) << '}';
        return;
    }
    const bool estimated = d.status == "estimated";
    if (!estimated)
        out << ",\"reason\":" << quote(d.reason);
    out << ",\"method\":\"single-ended simple reactance\",\"measurement\":" << quote(d.measurement)
        << ",\"fault_loop\":" << quote(d.loop)
        << (estimated ? ",\"estimated_distance_km\":" : ",\"apparent_distance_km\":") << d.km
        << ",\"distance_percent_of_line\":" << d.percent
        << ",\"within_line\":" << (d.within_line ? "true" : "false")
        << ",\"ground_compensated\":" << (d.ground_compensated ? "true" : "false")
        << ",\"uncompensated_ground_estimate\":" << (d.uncompensated ? "true" : "false")
        << ",\"k0_real\":" << d.k0_real << ",\"k0_imag\":" << d.k0_imag
        << ",\"r_apparent_ohm\":" << d.resistance << ",\"x_apparent_ohm\":" << d.reactance
        << ",\"cycle_min_km\":" << d.minimum << ",\"cycle_max_km\":" << d.maximum
        << ",\"cycles_used\":" << d.cycles.size() << ",\"cycles_skipped_low_current\":" << d.skipped
        << ",\"cycles_skipped_decayed\":" << d.decayed << ",\"cycles\":[";
    for (std::size_t i = 0; i < d.cycles.size(); ++i) {
        const Cycle &c = d.cycles[i];
        if (i)
            out << ',';
        out << "{\"cycle\":" << c.cycle << ",\"start_sample\":" << c.start_sample
            << ",\"stop_sample\":" << c.stop_sample
            << ",\"start_after_fault_ms\":" << c.start_after_fault_ms
            << ",\"stop_after_fault_ms\":" << c.stop_after_fault_ms
            << ",\"loop_current_rms_ka\":" << c.current << ",\"loop_voltage_rms_kv\":" << c.voltage
            << ",\"r_apparent_ohm\":" << c.resistance << ",\"x_apparent_ohm\":" << c.reactance
            << ",\"distance_km\":" << c.distance << '}';
    }
    out << ']';
    if (d.absolute_error >= 0)
        out << ",\"absolute_error_km\":" << d.absolute_error;
    out << '}';
}
void write_json(std::ostream &out, const Result &r, const Diagnostics *d) {
    out << std::setprecision(17) << "{\"segment\":" << r.segment << ",\"start\":" << r.start
        << ",\"end\":" << r.end << ",\"measured_samples\":" << r.measured_samples
        << ",\"status\":" << quote(r.status)
        << ",\"reason\":" << (r.reason.empty() ? "null" : quote(r.reason));
    if (r.status == "analyzed") {
        out << ",\"onset\":" << r.onset << ",\"inception_source\":" << quote(r.inception_source)
            << ",\"detected_sample\":";
        if (r.detected_sample < 0)
            out << "null";
        else
            out << r.detected_sample;
        out << ",\"center_sample\":" << r.center_sample
            << ",\"fault_type_heuristic\":" << quote(r.kind)
            << ",\"sequence_ratios\":{\"negative_positive\":" << r.ratio21
            << ",\"zero_positive\":" << r.ratio01
            << "},\"loop_selection\":" << quote(r.loop_selection) << ",\"distance\":";
        write_distance(out, r.distance);
        out << ",\"padding_left_samples\":" << r.padding_left
            << ",\"padding_right_samples\":" << r.padding_right
            << ",\"pre_window_samples\":" << r.npre << ",\"post_window_samples\":" << r.npost
            << ",\"emitted_at\":" << r.emitted_at
            << ",\"decision_mode\":\"offline; full segment available\",\"tensor_shape\":[12,"
            << r.npre + r.npost << ']';
        if (d) {
            out << ",\"diagnostics\":{\"filtered_ka_kv\":";
            matrix_json(out, d->filtered);
            out << ",\"sequence_peak_ka_kv\":";
            matrix_json(out, d->sequences);
            out << ",\"current_ratio\":";
            vector_json(out, d->current_ratio);
            out << ",\"voltage_ratio\":";
            vector_json(out, d->voltage_ratio);
            out << ",\"X\":";
            matrix_json(out, d->tensor);
            out << '}';
        }
    }
    out << '}';
}
} // namespace fault
