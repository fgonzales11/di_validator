#include "fault_core.hpp"
#include "fault_wire.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
static void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
int main() {
    using namespace fault;
    try {
        check(python_round(2.5) == 2 && python_round(3.5) == 4 && python_round(-2.5) == -2,
              "Python rounding");
        const double base = 1 + 1e-10;
        check(classify(1, .149 * base, .149 * base) == "3ph", "3ph interior");
        check(classify(1, .15 * base, 0) == "undefined", "strict .15 boundary");
        check(classify(1, .701 * base, .249 * base) == "2ph", "2ph interior");
        check(classify(1, .7 * base, 0) == "undefined", "strict .7 boundary");
        check(classify(1, .701 * base, .701 * base) == "1ph-G", "ground interior");
        check(classify(1, .5 * base, .5 * base) == "2ph-G", "2ph ground interior");
        check(classify(1, .25 * base, .5 * base) == "undefined", "strict .25 boundary");
        Config cfg;
        bool rejected = false;
        try {
            cfg.set("average_cycles", "1.5");
        } catch (const std::exception &) {
            rejected = true;
        }
        check(rejected, "integer validation");
        rejected = false;
        try {
            cfg.set("eta_i", "nan");
        } catch (const std::exception &) {
            rejected = true;
        }
        check(rejected, "NaN configuration");
        cfg.r0_ohm_km = .3;
        rejected = false;
        try {
            cfg.validate();
        } catch (const std::exception &) {
            rejected = true;
        }
        check(rejected, "partial ground compensation");
        cfg = Config();
        Phases phases;
        for (std::size_t c = 0; c < 6; ++c)
            phases[c].assign(80, 0);
        rejected = false;
        try {
            estimate_distance(phases, 2000, 50, 0, 81, cfg, "AG");
        } catch (const std::exception &) {
            rejected = true;
        }
        check(rejected, "measured bounds");
        Result r;
        r.status = "analyzed";
        r.inception_source = "detected";
        r.kind = "1ph-G";
        r.distance.loop = "AG";
        r.distance.reason = "Every selected cycle has insufficient loop current for division.";
        Payloads payload =
            make_payloads(r, cfg, std::string(64, 'a'), std::string(40, 'f') + "-200000");
        check(payload.alarm.size() <= 256 && payload.summary.size() <= 1024, "wire size bounds");
        check(payload.alarm.find("unavailable") != std::string::npos, "unavailable-distance alarm");
        r.distance.status = "out_of_range";
        payload = make_payloads(r, cfg, "test", "id");
        check(payload.alarm.find("#AG#out_of_range#") != std::string::npos &&
                  payload.summary.find("\"distance_km\":null") != std::string::npos,
              "withheld distance names its status");
        r.inception_source = "manual";
        check(make_payloads(r, cfg, "test", "id").alarm.empty(), "manual onset must not alarm");
        // 4 samples/cycle AB fault at 1.5 km that clears after two cycles: the peak
        // sample pairs, not the post-clearing samples, set the distance.
        const std::complex<double> z1(.2, .3), fault_current = std::polar(3.0, -1.2);
        Phases raw;
        for (int k = 0; k < 48; ++k) {
            const bool faulted = k >= 20 && k < 28;
            const std::complex<double> turn = std::polar(1.0, 3.14159265358979323846 / 2 * k);
            const std::complex<double> ia = faulted ? fault_current : .1,
                                       ua = faulted ? fault_current * z1 * 1.5 : 7.6;
            const double values[6] = {(ia * turn).real(), (-ia * turn).real(), 0,
                                      (ua * turn).real(), (-ua * turn).real(), 0};
            for (std::size_t c = 0; c < 6; ++c)
                raw[c].push_back(values[c]);
        }
        cfg = Config();
        cfg.line_length_km = 5;
        cfg.r1_ohm_km = .2;
        cfg.x1_ohm_km = .3;
        Distance peak = estimate_peak_distance(raw, 240, 24, cfg, "AB");
        check(peak.status == "estimated" && peak.measurement == "peak-sample-pair" &&
                  std::abs(peak.km - 1.5) < 1e-9 && peak.cycles.size() >= 7,
              "4 samples/cycle peak distance");
        cfg.line_length_km = 1;
        peak = estimate_peak_distance(raw, 240, 24, cfg, "AB");
        check(peak.status == "out_of_range" && std::abs(peak.km - 1.5) < 1e-9 &&
                  std::abs(peak.reactance - .45) < 1e-9,
              "beyond-line distance withheld with R/X kept");
        for (std::size_t c = 3; c < 6; ++c)
            for (std::size_t k = 0; k < raw[c].size(); ++k)
                raw[c][k] = -raw[c][k];
        peak = estimate_peak_distance(raw, 240, 24, cfg, "AB");
        check(peak.status == "behind_relay", "negative reactance is behind the relay");
        std::cout << "PASS: rounding, classification boundaries, invalid configuration/bounds, "
                     "payload limits, withheld-status alarms, manual alarm suppression, "
                     "4 samples/cycle peak distance and plausibility gates\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
