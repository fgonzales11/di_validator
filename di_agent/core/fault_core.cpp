#include "fault_core.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <cstdint>

namespace fault {
static const double pi = 3.1415926535897932384626433832795;
static const double peak_current_fraction = .9;
// Apparent distances at or below this are not positive: resistive V/I rounds either side of zero.
static const double min_distance_km = 1e-6;
static void require(bool ok, const char *why) {
    if (!ok)
        throw std::invalid_argument(why);
}
static void cancelled(const std::atomic<bool> *stop) {
    if (stop && stop->load())
        throw std::runtime_error("cancelled");
}
static double number(const std::string &text) {
    char *end = 0;
    errno = 0;
    const double x = std::strtod(text.c_str(), &end);
    require(end != text.c_str() && *end == '\0' && errno != ERANGE && std::isfinite(x),
            "Invalid finite decimal parameter");
    return x;
}
long python_round(double x) {
    const double lower = std::floor(x), fraction = x - lower;
    return static_cast<long>(fraction < .5   ? lower
                             : fraction > .5 ? lower + 1
                                             : (std::fmod(lower, 2) == 0 ? lower : lower + 1));
}
std::map<std::string, std::string> Config::values() const {
    std::map<std::string, std::string> out;
#define FIELD(name)                                                                                \
    {                                                                                              \
        std::ostringstream s;                                                                      \
        s << std::setprecision(17) << name;                                                        \
        out[#name] = s.str();                                                                      \
    }
    FIELD(start)
    FIELD(end) FIELD(manual_onset) FIELD(eta_i) FIELD(eta_u) FIELD(pre_seconds) FIELD(post_seconds)
        FIELD(line_length_km) FIELD(nominal_voltage_kv) FIELD(base_power_mva) FIELD(r1_ohm_km)
            FIELD(x1_ohm_km) FIELD(r0_ohm_km) FIELD(x0_ohm_km) FIELD(settle_cycles)
                FIELD(average_cycles) FIELD(min_current_ka) FIELD(fault_current_fraction)
                    FIELD(max_cycle_spread) FIELD(known_distance_km)
                    FIELD(line_parameters_verified) FIELD(fault_loop)
#undef FIELD
                        return out;
}
void Config::set(const std::string &key, const std::string &text) {
    if (key == "fault_loop") {
        fault_loop = text;
        return;
    }
    const double v = number(text);
#define FIELD(name)                                                                                \
    if (key == #name) {                                                                            \
        name = v;                                                                                  \
        return;                                                                                    \
    }
    FIELD(start)
    FIELD(end) FIELD(manual_onset) FIELD(eta_i) FIELD(eta_u) FIELD(pre_seconds) FIELD(post_seconds)
        FIELD(line_length_km) FIELD(nominal_voltage_kv) FIELD(base_power_mva) FIELD(r1_ohm_km)
            FIELD(x1_ohm_km) FIELD(r0_ohm_km) FIELD(x0_ohm_km) FIELD(settle_cycles)
                FIELD(min_current_ka) FIELD(fault_current_fraction) FIELD(max_cycle_spread)
                    FIELD(known_distance_km)
#undef FIELD
                    if (key == "average_cycles") {
        require(v >= 1 && v <= 50 && std::floor(v) == v, "Invalid average_cycles");
        average_cycles = static_cast<int>(v);
        return;
    }
    if (key == "line_parameters_verified") {
        require(v == 0 || v == 1, "Invalid verified flag");
        line_parameters_verified = v != 0;
        return;
    }
    throw std::invalid_argument("Unknown configuration parameter: " + key);
}
void Config::validate() const {
    const std::map<std::string, std::string> all = values();
    for (std::map<std::string, std::string>::const_iterator it = all.begin(); it != all.end(); ++it)
        if (it->first != "fault_loop")
            number(it->second);
    require(start >= 0 && (end == -1 || end > start), "Invalid analysis bounds");
    require(manual_onset == -1 || (manual_onset >= start && (end == -1 || manual_onset <= end)),
            "Invalid manual onset");
    require(eta_i > 0 && eta_u > 0 && eta_u < 1, "Invalid inception thresholds");
    require(pre_seconds > 0 && pre_seconds <= 2 && post_seconds > 0 && post_seconds <= 2,
            "Invalid pre/post windows");
    require(line_length_km > 0 && nominal_voltage_kv > 0 && base_power_mva > 0 && r1_ohm_km >= 0 &&
                x1_ohm_km > 0,
            "Invalid line parameters");
    require((r0_ohm_km == -1 && x0_ohm_km == -1) || (r0_ohm_km >= 0 && x0_ohm_km > 0),
            "Ground compensation requires both r0 and x0");
    require(settle_cycles >= 0 && settle_cycles <= 20 && average_cycles >= 1 &&
                average_cycles <= 50 && min_current_ka > 0 && fault_current_fraction >= 0 &&
                fault_current_fraction < 1 && max_cycle_spread > 0,
            "Invalid distance settings");
    require(known_distance_km == -1 ||
                (known_distance_km >= 0 && known_distance_km <= line_length_km),
            "Known distance must lie within line");
    require(fault_loop == "AUTO" || fault_loop == "AG" || fault_loop == "BG" ||
                fault_loop == "CG" || fault_loop == "AB" || fault_loop == "BC" ||
                fault_loop == "CA" || fault_loop == "POS",
            "Invalid fault loop");
}
std::string config_id(const Config &cfg) {
    // Reproducible non-security identity; fixture manifests additionally carry SHA-256.
    uint64_t hash = UINT64_C(14695981039346656037);
    const std::map<std::string, std::string> values = cfg.values();
    for (std::map<std::string, std::string>::const_iterator it = values.begin(); it != values.end();
         ++it) {
        const std::string text = it->first + "=" + it->second + "\n";
        for (std::size_t i = 0; i < text.size(); ++i) {
            hash ^= static_cast<unsigned char>(text[i]);
            hash *= UINT64_C(1099511628211);
        }
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}
static double rms(const Signal &x, int start, int stop) {
    double sum = 0;
    for (int i = start; i < stop; ++i)
        sum += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    return std::sqrt(sum / static_cast<double>(stop - start));
}
static Signal extract(const Signal &x, int center, int before, int after) {
    Signal out(static_cast<std::size_t>(before + after), 0);
    for (int j = 0; j < before + after; ++j) {
        int i = center - before + j;
        if (i >= 0 && i < static_cast<int>(x.size()))
            out[static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(i)];
    }
    return out;
}
std::string classify(double i1, double i2, double i0) {
    double r21 = i2 / (i1 + 1e-10), r01 = i0 / (i1 + 1e-10);
    if (r21 < .15 && r01 < .15)
        return "3ph";
    if (r21 > .7 && r01 < .25)
        return "2ph";
    if (r21 > .7 && r01 > .7)
        return "1ph-G";
    if (r21 > .25 && r21 < .7 && r01 > .25 && r01 < .7)
        return "2ph-G";
    return "undefined";
}
std::vector<std::complex<double>> fit_phasors(const std::vector<Signal> &channels, double fs,
                                              double f0) {
    require(!channels.empty() && channels[0].size() >= 3 && std::isfinite(fs) &&
                std::isfinite(f0) && f0 > 0 && fs > 2 * f0,
            "Invalid phasor inputs");
    const std::size_t n = channels[0].size();
    std::vector<std::array<double, 3>> a(n);
    std::vector<Signal> rhs = channels;
    for (std::size_t c = 0; c < rhs.size(); ++c) {
        require(rhs[c].size() == n, "Mismatched phasor channels");
        for (std::size_t i = 0; i < n; ++i)
            require(std::isfinite(rhs[c][i]), "Nonfinite phasor input");
    }
    for (std::size_t i = 0; i < n; ++i) {
        double angle = 2 * pi * f0 * static_cast<double>(i) / fs;
        a[i] = {{std::cos(angle), std::sin(angle), 1}};
    }
    // Householder QR: avoid normal equations and their squared condition number.
    for (std::size_t k = 0; k < 3; ++k) {
        double norm = 0;
        for (std::size_t i = k; i < n; ++i)
            norm = ::hypot(norm, a[i][k]);
        require(norm > std::numeric_limits<double>::epsilon() * static_cast<double>(n) *
                           std::sqrt(static_cast<double>(n)),
                "The selected samples cannot resolve a fundamental phasor");
        double alpha = -::copysign(norm, a[k][k]);
        Signal v(n - k);
        for (std::size_t i = k; i < n; ++i)
            v[i - k] = a[i][k];
        v[0] -= alpha;
        double vv = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
        for (std::size_t j = k; j < 3; ++j) {
            double dot = 0;
            for (std::size_t i = k; i < n; ++i)
                dot += v[i - k] * a[i][j];
            for (std::size_t i = k; i < n; ++i)
                a[i][j] -= 2 * v[i - k] * dot / vv;
        }
        for (std::size_t c = 0; c < rhs.size(); ++c) {
            double dot = 0;
            for (std::size_t i = k; i < n; ++i)
                dot += v[i - k] * rhs[c][i];
            for (std::size_t i = k; i < n; ++i)
                rhs[c][i] -= 2 * v[i - k] * dot / vv;
        }
    }
    std::vector<std::complex<double>> out;
    for (std::size_t c = 0; c < rhs.size(); ++c) {
        double coef[3] = {0, 0, 0};
        for (int i = 2; i >= 0; --i) {
            double value = rhs[c][static_cast<std::size_t>(i)];
            for (int j = i + 1; j < 3; ++j)
                value -= a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * coef[j];
            coef[i] = value / a[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
        }
        out.push_back(std::complex<double>(coef[0], -coef[1]) / std::sqrt(2.0));
    }
    return out;
}
typedef std::complex<double> Complex;
static void loop_pair(const Complex *p, const std::string &loop, Complex k0, Complex &current,
                      Complex &voltage) {
    if (loop.size() == 2 && loop[1] == 'G') {
        const std::size_t phase = static_cast<std::size_t>(loop[0] - 'A');
        current = p[phase] + k0 * (p[0] + p[1] + p[2]);
        voltage = p[phase + 3];
    } else if (loop == "POS") {
        const Complex a = std::polar(1.0, 2 * pi / 3);
        current = (p[0] + a * p[1] + a * a * p[2]) / 3.0;
        voltage = (p[3] + a * p[4] + a * a * p[5]) / 3.0;
    } else {
        const std::size_t left = static_cast<std::size_t>(loop[0] - 'A'),
                          right = static_cast<std::size_t>(loop[1] - 'A');
        current = p[left] - p[right];
        voltage = p[left + 3] - p[right + 3];
    }
}
// Validates inputs and fills the loop/compensation fields; false when no loop is selected.
static bool prepare(Distance &out, const Phases &w, const Config &cfg, const std::string &loop,
                    Complex &k0) {
    cfg.validate();
    for (std::size_t c = 0; c < 6; ++c) {
        require(w[c].size() == w[0].size(), "Six aligned distance channels required");
        for (std::size_t i = 0; i < w[c].size(); ++i)
            require(std::isfinite(w[c][i]), "Distance waveforms must contain only finite values");
    }
    require(loop == "AUTO" || loop == "AG" || loop == "BG" || loop == "CG" || loop == "AB" ||
                loop == "BC" || loop == "CA" || loop == "POS",
            "Invalid distance loop");
    out.loop = loop;
    if (loop == "AUTO") {
        out.reason = "Fault type is uncertain; select the fault loop explicitly";
        return false;
    }
    const bool ground = loop.size() == 2 && loop[1] == 'G';
    const Complex z1(cfg.r1_ohm_km, cfg.x1_ohm_km);
    out.ground_compensated = ground && cfg.r0_ohm_km >= 0;
    out.uncompensated = ground && !out.ground_compensated;
    k0 = out.ground_compensated ? (Complex(cfg.r0_ohm_km, cfg.x0_ohm_km) - z1) / (3.0 * z1)
                                : Complex(0, 0);
    out.k0_real = k0.real();
    out.k0_imag = k0.imag();
    return true;
}
static Cycle make_row(int number, int start, int stop, int center, double fs, Complex current,
                      Complex voltage, const Config &cfg) {
    const Complex z = voltage / current;
    Cycle row;
    row.cycle = number;
    row.start_sample = start;
    row.stop_sample = stop;
    row.start_after_fault_ms = static_cast<double>(start - center) / fs * 1000;
    row.stop_after_fault_ms = static_cast<double>(stop - center) / fs * 1000;
    row.current = std::abs(current);
    row.voltage = std::abs(voltage);
    row.resistance = z.real();
    row.reactance = z.imag();
    row.distance = z.imag() / cfg.x1_ohm_km;
    require(std::isfinite(row.distance), "Nonfinite distance");
    return row;
}
static double median(Signal values) {
    std::sort(values.begin(), values.end());
    const std::size_t m = values.size() / 2;
    return values.size() % 2 ? values[m] : (values[m - 1] + values[m]) / 2;
}
// Median estimate plus plausibility gate; implausible results keep R/X but carry no estimate.
static void summarize(Distance &out, const Config &cfg, const char *measurement, bool spread_gate) {
    Signal distances, resistances, reactances;
    for (std::size_t i = 0; i < out.cycles.size(); ++i) {
        distances.push_back(out.cycles[i].distance);
        resistances.push_back(out.cycles[i].resistance);
        reactances.push_back(out.cycles[i].reactance);
    }
    out.measurement = measurement;
    out.km = median(distances);
    out.resistance = median(resistances);
    out.reactance = median(reactances);
    out.minimum = *std::min_element(distances.begin(), distances.end());
    out.maximum = *std::max_element(distances.begin(), distances.end());
    out.percent = 100 * out.km / cfg.line_length_km;
    out.within_line = out.km >= 0 && out.km <= cfg.line_length_km;
    if (out.km <= min_distance_km) {
        out.status = "behind_relay";
        out.reason = "Measured loop reactance is not positive; the fault may be behind the relay.";
    } else if (out.km > cfg.line_length_km) {
        out.status = "out_of_range";
        out.reason = "Apparent distance exceeds the configured line length.";
    } else if (spread_gate && out.maximum - out.minimum > cfg.max_cycle_spread * out.km) {
        out.status = "inconsistent";
        out.reason = "Post-fault cycle estimates disagree by more than max_cycle_spread of the median.";
    } else {
        out.status = "estimated";
        out.reason.clear();
    }
}
Distance estimate_distance(const Phases &w, double fs, double f0, int center, int measured_stop,
                           const Config &cfg, const std::string &loop) {
    require(std::isfinite(fs) && std::isfinite(f0) && f0 > 0 && fs > 2 * f0,
            "Invalid distance sampling frequency");
    require(center >= 0 && center < measured_stop && measured_stop <= static_cast<int>(w[0].size()),
            "Measured post-fault bounds must lie inside phase window");
    Distance out;
    Complex k0;
    if (!prepare(out, w, cfg, loop, k0))
        return out;
    const int nc = static_cast<int>(std::ceil(fs / f0));
    const int first = center + static_cast<int>(std::ceil(cfg.settle_cycles * fs / f0));
    const int count = std::min(cfg.average_cycles, std::max(0, (measured_stop - first) / nc));
    if (count == 0) {
        out.reason = "Less than one measured post-fault cycle remains after settling.";
        return out;
    }
    const auto cycle_loop = [&](int start, Complex &current, Complex &voltage) {
        std::vector<Signal> slice;
        for (std::size_t c = 0; c < 6; ++c)
            slice.push_back(Signal(w[c].begin() + start, w[c].begin() + start + nc));
        const std::vector<Complex> p = fit_phasors(slice, fs, f0);
        loop_pair(p.data(), loop, k0, current, voltage);
    };
    std::vector<Complex> currents(static_cast<std::size_t>(count)),
        voltages(static_cast<std::size_t>(count));
    double peak = 0;
    for (int cycle = 0; cycle < count; ++cycle) {
        const std::size_t i = static_cast<std::size_t>(cycle);
        cycle_loop(first + cycle * nc, currents[i], voltages[i]);
        peak = std::max(peak, std::abs(currents[i]));
    }
    // Largest whole-cycle loop current after inception; later cycles well below it follow breaker opening.
    for (int start = center; start + nc <= measured_stop; start += nc) {
        Complex current, voltage;
        cycle_loop(start, current, voltage);
        peak = std::max(peak, std::abs(current));
    }
    for (int cycle = 0; cycle < count; ++cycle) {
        const std::size_t i = static_cast<std::size_t>(cycle);
        if (std::abs(currents[i]) < cfg.min_current_ka) {
            ++out.skipped;
            continue;
        }
        if (std::abs(currents[i]) < cfg.fault_current_fraction * peak) {
            ++out.decayed;
            continue;
        }
        const int start = first + cycle * nc;
        out.cycles.push_back(
            make_row(cycle + 1, start, start + nc, center, fs, currents[i], voltages[i], cfg));
    }
    if (out.cycles.empty()) {
        out.reason = out.decayed
                         ? "Fault current decayed below fault_current_fraction of its peak in every "
                           "selected cycle (breaker opened); reduce settle_cycles."
                         : "Every selected cycle has insufficient loop current for division.";
        return out;
    }
    summarize(out, cfg, "cycle-fit", true);
    return out;
}
Distance estimate_peak_distance(const Phases &raw, double fs, int center, const Config &cfg,
                                const std::string &loop) {
    require(std::isfinite(fs) && fs > 0, "Invalid distance sampling frequency");
    Distance out;
    Complex k0;
    if (!prepare(out, raw, cfg, loop, k0))
        return out;
    // Fundamental-filtered 4 samples/cycle reports: x[k] + j*x[k-1] is the phasor at k.
    const std::size_t n = raw[0].size();
    std::vector<Complex> currents, voltages;
    double largest = 0;
    for (std::size_t k = 1; k < n; ++k) {
        Complex p[6], current, voltage;
        for (std::size_t c = 0; c < 6; ++c)
            p[c] = Complex(raw[c][k], raw[c][k - 1]);
        loop_pair(p, loop, k0, current, voltage);
        currents.push_back(current);
        voltages.push_back(voltage);
        largest = std::max(largest, std::abs(current));
    }
    if (currents.empty() || largest < cfg.min_current_ka) {
        out.reason = "No sample carries the minimum loop current";
        return out;
    }
    const double threshold = std::max(peak_current_fraction * largest, cfg.min_current_ka);
    for (std::size_t m = 0; m < currents.size(); ++m)
        if (std::abs(currents[m]) >= threshold) {
            const int sample = static_cast<int>(m) + 1;
            out.cycles.push_back(make_row(static_cast<int>(out.cycles.size()) + 1, sample, sample + 1,
                                          center, fs, currents[m], voltages[m], cfg));
        }
    summarize(out, cfg, "peak-sample-pair", false);
    return out;
}
Result analyze_segment(const Segment &input, const Config &cfg, int number,
                       Diagnostics *diagnostics, const std::atomic<bool> *cancel) {
    cfg.validate();
    const int n = static_cast<int>(input.time.size());
    require(n > 0 && n <= 200001 && input.f0 > 0 && input.fs >= 4 * input.f0 &&
                std::isfinite(input.fs) && std::isfinite(input.f0),
            "Invalid waveform shape or sampling frequencies");
    require(input.fs / input.f0 <= 10000 &&
                (cfg.pre_seconds + cfg.post_seconds) * input.fs <= 200000,
            "Feature window or electrical period too large");
    for (std::size_t c = 0; c < 6; ++c) {
        require(input.samples[c].size() == input.time.size(), "Six aligned channels required");
        for (int i = 0; i < n; ++i)
            require(std::isfinite(input.samples[c][static_cast<std::size_t>(i)]),
                    "Nonfinite segment sample");
    }
    const int period = static_cast<int>(python_round(input.fs / input.f0)), half = period / 2,
              k = static_cast<int>(input.fs / (4 * input.f0));
    Result result;
    result.segment = number;
    result.measured_samples = n;
    result.start = input.time.front();
    result.end = input.time.back();
    if (n < 2 * period) {
        result.status = "insufficient_data";
        result.reason = "Need at least two contiguous electrical cycles";
        return result;
    }
    Phases filtered;
    for (std::size_t c = 0; c < 6; ++c) {
        cancelled(cancel);
        Signal values(static_cast<std::size_t>(n)), mean(static_cast<std::size_t>(n));
        double base = 0;
        for (int i = 0; i < period; ++i)
            base += input.samples[c][static_cast<std::size_t>(i)] / 1000;
        base /= period;
        for (int i = 0; i < n; ++i)
            values[static_cast<std::size_t>(i)] =
                input.samples[c][static_cast<std::size_t>(i)] / 1000 - base;
        for (int i = 0; i < n; ++i) {
            if (i % 64 == 0)
                cancelled(cancel);
            int lo = std::max(0, i - half), hi = std::min(n, i - half + period);
            double sum = 0;
            for (int j = lo; j < hi; ++j)
                sum += values[static_cast<std::size_t>(j)];
            mean[static_cast<std::size_t>(i)] = sum / (hi - lo);
        }
        double left = 0, right = 0;
        for (int i = half; i < period; ++i)
            left += mean[static_cast<std::size_t>(i)];
        left /= (period - half);
        for (int i = n - period; i < n - half; ++i) {
            right += mean[static_cast<std::size_t>(i)];
        }
        right /= (period - half);
        for (int i = 0; i < half; ++i)
            mean[static_cast<std::size_t>(i)] = left;
        for (int i = n - half; i < n; ++i)
            mean[static_cast<std::size_t>(i)] = right;
        filtered[c].resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            filtered[c][static_cast<std::size_t>(i)] =
                values[static_cast<std::size_t>(i)] - mean[static_cast<std::size_t>(i)];
    }
    Signal ir(static_cast<std::size_t>(n), 1), ur(static_cast<std::size_t>(n), 1);
    for (int i = k; i < n - k; ++i) {
        if (i % 64 == 0)
            cancelled(cancel);
        double ipre = 0, ipost = 0, upre = std::numeric_limits<double>::infinity(), upost = upre;
        for (std::size_t c = 0; c < 3; ++c) {
            ipre = std::max(ipre, rms(filtered[c], i - k, i));
            ipost = std::max(ipost, rms(filtered[c], i, i + k));
            upre = std::min(upre, rms(filtered[c + 3], i - k, i));
            upost = std::min(upost, rms(filtered[c + 3], i, i + k));
        }
        ir[static_cast<std::size_t>(i)] = ipre > 1e-6 ? ipost / (ipre + 1e-9) : 1;
        ur[static_cast<std::size_t>(i)] = upre > 1e-6 ? upost / (upre + 1e-9) : 1;
        if (result.detected_sample < 0 && i >= 2 * k &&
            ir[static_cast<std::size_t>(i)] > 1 + cfg.eta_i &&
            ur[static_cast<std::size_t>(i)] < cfg.eta_u)
            result.detected_sample = i;
    }
    bool manual = cfg.manual_onset >= result.start && cfg.manual_onset <= result.end;
    if (cfg.manual_onset >= 0 && !manual) {
        result.status = "outside_manual_selection";
        result.reason = "Manual onset belongs to another segment";
        return result;
    }
    if (result.detected_sample < 0 && !manual)
        return result;
    int center = result.detected_sample;
    if (manual) {
        center = 0;
        for (int i = 1; i < n; ++i)
            if (std::abs(input.time[static_cast<std::size_t>(i)] - cfg.manual_onset) <
                std::abs(input.time[static_cast<std::size_t>(center)] - cfg.manual_onset))
                center = i;
    }
    const int npre = static_cast<int>(python_round(cfg.pre_seconds * input.fs)),
              npost = static_cast<int>(python_round(cfg.post_seconds * input.fs));
    require(npre > 0 && npost > 0, "Pre/post windows must each contain a sample");
    Phases windows, seq;
    for (std::size_t c = 0; c < 6; ++c) {
        windows[c] = extract(filtered[c], center, npre, npost);
        seq[c].resize(static_cast<std::size_t>(n));
    }
    const int nc = std::max(3, period);
    std::vector<std::complex<double>> reference(static_cast<std::size_t>(nc));
    Signal offsets(static_cast<std::size_t>(nc));
    const std::complex<double> a = std::polar(1.0, 2 * pi / 3);
    for (int j = 0; j < nc; ++j) {
        const int step = j - nc / 2;
        offsets[static_cast<std::size_t>(j)] = step * input.fs / (input.f0 * nc);
        reference[static_cast<std::size_t>(j)] = std::polar(1.0, -2 * pi * step / nc);
    }
    for (int i = 0; i < n; ++i) {
        if (i % 32 == 0)
            cancelled(cancel);
        std::complex<double> p[6];
        for (std::size_t c = 0; c < 6; ++c) {
            for (int j = 0; j < nc; ++j) {
                const double position =
                    std::max(0.0, std::min(static_cast<double>(n - 1),
                                           i + offsets[static_cast<std::size_t>(j)]));
                const int lo = static_cast<int>(position), hi = std::min(n - 1, lo + 1);
                const double weight = position - lo;
                p[c] += (filtered[c][static_cast<std::size_t>(lo)] * (1 - weight) +
                         filtered[c][static_cast<std::size_t>(hi)] * weight) *
                        reference[static_cast<std::size_t>(j)];
            }
            p[c] *= 2.0 / nc;
        }
        for (std::size_t c = 0; c < 6; c += 3) {
            seq[c][static_cast<std::size_t>(i)] = std::abs((p[c] + p[c + 1] + p[c + 2]) / 3.0);
            seq[c + 1][static_cast<std::size_t>(i)] =
                std::abs((p[c] + a * p[c + 1] + a * a * p[c + 2]) / 3.0);
            seq[c + 2][static_cast<std::size_t>(i)] =
                std::abs((p[c] + a * a * p[c + 1] + a * p[c + 2]) / 3.0);
        }
    }
    Phases seq_windows;
    for (std::size_t c = 0; c < 6; ++c)
        seq_windows[c] = extract(seq[c], center, npre, npost);
    const int stop = std::min(npre + npost, npre + n - center);
    double i0 = rms(seq_windows[0], npre, stop) / std::sqrt(2.0),
           i1 = rms(seq_windows[1], npre, stop) / std::sqrt(2.0),
           i2 = rms(seq_windows[2], npre, stop) / std::sqrt(2.0);
    result.kind = classify(i1, i2, i0);
    result.ratio21 = i2 / (i1 + 1e-10);
    result.ratio01 = i0 / (i1 + 1e-10);
    std::string loop = cfg.fault_loop;
    std::array<double, 3> phase_rms = {
        {rms(windows[0], npre, stop), rms(windows[1], npre, stop), rms(windows[2], npre, stop)}};
    if (loop == "AUTO") {
        if (result.kind == "1ph-G") {
            loop = std::string(1, static_cast<char>(
                                      'A' + std::distance(phase_rms.begin(),
                                                          std::max_element(phase_rms.begin(),
                                                                           phase_rms.end())))) +
                   "G";
        } else if (result.kind == "2ph" || result.kind == "2ph-G") {
            std::array<int, 3> indices = {{0, 1, 2}};
            std::stable_sort(indices.begin(), indices.end(), [&phase_rms](int l, int r) {
                return phase_rms[static_cast<std::size_t>(l)] <
                       phase_rms[static_cast<std::size_t>(r)];
            });
            const int omitted = indices[0];
            loop = omitted == 0 ? "BC" : omitted == 1 ? "CA" : "AB";
        } else if (result.kind == "3ph")
            loop = "POS";
    }
    if (period == 4) {
        // Relay reports at 4 samples/cycle: onset detection lags the fault, so measure at the
        // fault-current peak of the whole raw segment instead.
        Phases raw;
        for (std::size_t c = 0; c < 6; ++c) {
            raw[c].resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
                raw[c][static_cast<std::size_t>(i)] = input.samples[c][static_cast<std::size_t>(i)] / 1000;
        }
        result.distance = estimate_peak_distance(raw, input.fs, center, cfg, loop);
    } else
        result.distance = estimate_distance(windows, input.fs, input.f0, npre, stop, cfg, loop);
    if (result.distance.status == "estimated" && cfg.known_distance_km >= 0)
        result.distance.absolute_error = std::abs(result.distance.km - cfg.known_distance_km);
    result.status = "analyzed";
    result.reason.clear();
    result.center_sample = center;
    result.onset = input.time[static_cast<std::size_t>(center)];
    result.inception_source = manual ? "manual" : "detected";
    result.loop_selection = cfg.fault_loop == "AUTO" ? "heuristic" : "manual";
    result.npre = npre;
    result.npost = npost;
    result.padding_left = std::max(0, npre - center);
    result.padding_right = std::max(0, center + npost - n);
    result.emitted_at = result.end + 1 / input.fs;
    if (diagnostics) {
        diagnostics->filtered = filtered;
        diagnostics->sequences = seq;
        diagnostics->current_ratio = ir;
        diagnostics->voltage_ratio = ur;
        const double base_i = cfg.base_power_mva / (std::sqrt(3.0) * cfg.nominal_voltage_kv),
                     base_u = cfg.nominal_voltage_kv / std::sqrt(3.0);
        const int reorder[6] = {1, 2, 0, 4, 5, 3};
        diagnostics->tensor.resize(12);
        for (std::size_t c = 0; c < 12; ++c) {
            const Signal &source =
                c < 6 ? windows[c] : seq_windows[static_cast<std::size_t>(reorder[c - 6])];
            Signal &target = diagnostics->tensor[c];
            target.resize(source.size());
            const double base = (c < 3 || (c >= 6 && c < 9)) ? base_i : base_u;
            for (std::size_t j = 0; j < source.size(); ++j)
                target[j] = static_cast<float>(source[j] / base);
        }
    }
    return result;
}
std::vector<Segment> split_record(const Segment &input, const Config &cfg) {
    cfg.validate();
    require(std::isfinite(input.fs) && std::isfinite(input.f0) && input.f0 > 0 &&
                input.fs >= 4 * input.f0,
            "Waveforms require at least four samples per cycle");
    require(!input.time.empty() && input.time.size() <= 200001,
            "Analysis window must contain 1-200001 native samples");
    for (std::size_t c = 0; c < 6; ++c)
        require(input.samples[c].size() == input.time.size(), "Six aligned channels required");
    std::vector<Segment> out;
    Segment part;
    part.fs = input.fs;
    part.f0 = input.f0;
    double previous = -1;
    bool manual_valid = cfg.manual_onset < 0;
    for (std::size_t i = 0; i < input.time.size(); ++i) {
        const double t = input.time[i];
        require(std::isfinite(t) && t >= 0 && (i == 0 || t > previous),
                "Sample timestamps must be finite, nonnegative, and strictly increasing");
        const double steps = i ? (t - previous) * input.fs : 1;
        require(std::abs(steps - static_cast<double>(python_round(steps))) <= .01,
                "Samples must lie on declared sampling grid; no interpolation");
        const bool gap = i && steps > 1.5;
        previous = t;
        if (t < cfg.start || (cfg.end >= 0 && t > cfg.end))
            continue;
        bool finite = true;
        for (std::size_t c = 0; c < 6; ++c)
            finite = finite && std::isfinite(input.samples[c][i]);
        if (gap || !finite) {
            if (!part.time.empty()) {
                out.push_back(part);
                part.time.clear();
                for (std::size_t c = 0; c < 6; ++c)
                    part.samples[c].clear();
            }
        }
        if (finite) {
            part.time.push_back(t);
            for (std::size_t c = 0; c < 6; ++c)
                part.samples[c].push_back(input.samples[c][i]);
        }
    }
    if (!part.time.empty())
        out.push_back(part);
    for (std::size_t i = 0; i < out.size(); ++i)
        if (cfg.manual_onset >= out[i].time.front() && cfg.manual_onset <= out[i].time.back())
            manual_valid = true;
    require(manual_valid, "Manual onset falls in a data gap or outside observed samples");
    require(!out.empty(), "No finite samples in selected window");
    require((out.back().time.back() - out.front().time.front()) * input.fs <= 200000,
            "Analysis window exceeds 200000 native samples");
    return out;
}
} // namespace fault
