#pragma once
#include <array>
#include <atomic>
#include <complex>
#include <map>
#include <string>
#include <vector>

namespace fault {
typedef std::vector<double> Signal;
typedef std::array<Signal, 6> Phases;
struct Config {
    double start = 0, end = -1, manual_onset = -1;
    double eta_i = .5, eta_u = .85, pre_seconds = .05, post_seconds = .15;
    double line_length_km = 50, nominal_voltage_kv = 110, base_power_mva = 100;
    double r1_ohm_km = .1, x1_ohm_km = .4, r0_ohm_km = -1, x0_ohm_km = -1;
    double settle_cycles = 1, min_current_ka = .001, known_distance_km = -1;
    double fault_current_fraction = .8, max_cycle_spread = .5;
    int average_cycles = 3;
    bool line_parameters_verified = false;
    std::string fault_loop = "AUTO";
    void validate() const;
    void set(const std::string &key, const std::string &value);
    std::map<std::string, std::string> values() const;
};
struct Segment {
    Signal time;
    Phases samples;
    double fs = 0, f0 = 0;
};
struct Cycle {
    int cycle = 0, start_sample = 0, stop_sample = 0;
    double start_after_fault_ms = 0, stop_after_fault_ms = 0;
    double current = 0, voltage = 0, resistance = 0, reactance = 0, distance = 0;
};
struct Distance {
    std::string status = "unavailable", reason, loop, measurement;
    bool within_line = false, ground_compensated = false, uncompensated = false;
    double km = 0, percent = 0, k0_real = 0, k0_imag = 0, minimum = 0, maximum = 0;
    double resistance = 0, reactance = 0, absolute_error = -1;
    int skipped = 0, decayed = 0;
    std::vector<Cycle> cycles;
};
struct Result {
    int segment = 0, measured_samples = 0, detected_sample = -1, center_sample = -1;
    int padding_left = 0, padding_right = 0, npre = 0, npost = 0;
    double start = 0, end = 0, onset = 0, emitted_at = 0, ratio21 = 0, ratio01 = 0;
    std::string status = "no_fault", reason = "No fault inception met the combined RMS criterion";
    std::string inception_source, kind, loop_selection;
    Distance distance;
};
struct Diagnostics {
    Phases filtered, sequences;
    std::vector<Signal> tensor;
    Signal current_ratio, voltage_ratio;
};
long python_round(double x);
std::string classify(double i1, double i2, double i0);
std::vector<std::complex<double>> fit_phasors(const std::vector<Signal> &channels, double fs,
                                              double f0);
Distance estimate_distance(const Phases &windows, double fs, double f0, int center,
                           int measured_stop, const Config &cfg, const std::string &loop);
Distance estimate_peak_distance(const Phases &raw, double fs, int center, const Config &cfg,
                                const std::string &loop);
Result analyze_segment(const Segment &segment, const Config &cfg, int number = 0,
                       Diagnostics *diagnostics = 0, const std::atomic<bool> *cancel = 0);
std::vector<Segment> split_record(const Segment &input, const Config &cfg);
std::string config_id(const Config &cfg);
} // namespace fault
