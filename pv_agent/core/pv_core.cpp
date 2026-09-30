#include "pv_core.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace pv {
const char *state_name(State x) {
    static const char *names[] = {"learning", "likely_pv", "no_evidence", "ambiguous"};
    return names[static_cast<unsigned>(x)];
}
const char *reason_name(Reason x) {
    static const char *names[] = {
        "available", "warmup",           "coordinates",  "coverage",         "no_pv_evidence",
        "other_der", "night_export",     "model_rank",   "model_validation", "q_range",
        "twilight",  "reactive_missing", "invalid_time", "model_invalid"};
    return names[static_cast<unsigned>(x)];
}
static std::string number(double value) {
    std::ostringstream s;
    s << std::setprecision(17) << value;
    return s.str();
}
uint64_t hash64(const std::string &s) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (unsigned char c : s) {
        h ^= c;
        h *= UINT64_C(1099511628211);
    }
    return h;
}
std::string hex64(uint64_t value) {
    std::ostringstream s;
    s << std::hex << std::setw(16) << std::setfill('0') << value;
    return s.str();
}
std::map<std::string, std::string> Config::values() const {
    std::map<std::string, std::string> v;
#define VALUE(x) v[#x] = number(x)
    VALUE(latitude);
    VALUE(longitude);
    VALUE(polarity);
    VALUE(power_scale);
    VALUE(reactive_scale);
    VALUE(min_coverage);
    VALUE(export_threshold_kw);
    VALUE(export_fraction);
    VALUE(valley_threshold);
    VALUE(valley_fraction);
    VALUE(valley_median);
    VALUE(min_r2);
    VALUE(max_nrmse);
    VALUE(q_margin);
    VALUE(storage);
    VALUE(other_generation);
#undef VALUE
    return v;
}
void Config::set(const std::string &key, const std::string &text) {
    char *end = 0;
    double x = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end || !std::isfinite(x))
        throw std::invalid_argument("Invalid decimal: " + key);
#define SET(xname)                                                                                 \
    if (key == #xname) {                                                                           \
        xname = x;                                                                                 \
        return;                                                                                    \
    }
    SET(latitude);
    SET(longitude);
    SET(polarity);
    SET(power_scale);
    SET(reactive_scale);
    SET(min_coverage);
    SET(export_threshold_kw);
    SET(export_fraction);
    SET(valley_threshold);
    SET(valley_fraction);
    SET(valley_median);
    SET(min_r2);
    SET(max_nrmse);
    SET(q_margin);
#undef SET
    if ((key == "storage" || key == "other_generation") && (x == 0 || x == 1)) {
        if (key == "storage")
            storage = x != 0;
        else
            other_generation = x != 0;
        return;
    }
    throw std::invalid_argument("Unknown or invalid parameter: " + key);
}
void Config::validate() const {
    for (const auto &entry : values()) {
        if (!std::isfinite(std::strtod(entry.second.c_str(), 0)))
            throw std::invalid_argument("Nonfinite config");
    }
    if (!((latitude == 999 && longitude == 999) ||
          (std::abs(latitude) <= 90 && std::abs(longitude) <= 180)))
        throw std::invalid_argument("Provide both coordinates or use 999/999");
    if ((polarity != -1 && polarity != 1) || power_scale <= 0 || power_scale > 1e6 ||
        reactive_scale <= 0 || reactive_scale > 1e6)
        throw std::invalid_argument("Invalid polarity/scaling");
    if (min_coverage < .9 || min_coverage > 1 || export_threshold_kw < 0 ||
        export_threshold_kw > 1e6 || export_fraction <= 0 || export_fraction > 1 ||
        valley_threshold <= 0 || valley_threshold > 1 || valley_fraction <= 0 ||
        valley_fraction > 1 || valley_median <= 0 || valley_median > 1 || min_r2 < 0 ||
        min_r2 > 1 || max_nrmse <= 0 || max_nrmse > 1 || q_margin < 0 || q_margin > 1)
        throw std::invalid_argument("Invalid analytical threshold");
}
std::string Config::identity() const {
    return std::string(ALGORITHM) + ":" + number(latitude) + ":" + number(longitude) + ":" +
           number(polarity) + ":" + number(power_scale) + ":" + number(reactive_scale);
}
uint64_t Config::id() const {
    std::string s(ALGORITHM);
    for (const auto &e : values())
        s += "\n" + e.first + "=" + e.second;
    return hash64(s);
}
Solar solar(int64_t utc, double latitude, double longitude) {
    const double pi = 3.14159265358979323846;
    time_t t = static_cast<time_t>(utc);
    struct tm b;
    if (!gmtime_r(&t, &b))
        throw std::invalid_argument("UTC outside supported clock range");
    int year = b.tm_year + 1900;
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    double hour = b.tm_hour + b.tm_min / 60.0 + b.tm_sec / 3600.0;
    double gamma = 2 * pi / (leap ? 366 : 365) * (b.tm_yday + (hour - 12) / 24);
    double eq = 229.18 * (.000075 + .001868 * std::cos(gamma) - .032077 * std::sin(gamma) -
                          .014615 * std::cos(2 * gamma) - .040849 * std::sin(2 * gamma));
    double decl = .006918 - .399912 * std::cos(gamma) + .070257 * std::sin(gamma) -
                  .006758 * std::cos(2 * gamma) + .000907 * std::sin(2 * gamma) -
                  .002697 * std::cos(3 * gamma) + .00148 * std::sin(3 * gamma);
    double sh = std::fmod(hour + (eq + 4 * longitude) / 60 + 48, 24.0);
    double lat = latitude * pi / 180, ha = (sh * 15 - 180) * pi / 180;
    double sinel = std::sin(lat) * std::sin(decl) + std::cos(lat) * std::cos(decl) * std::cos(ha);
    return Solar{std::asin(std::max(-1.0, std::min(1.0, sinel))) * 180 / pi, sh};
}
// Treat roundoff-scale differences as equality for screening/model thresholds.
// This is six orders tighter than the analytical parity tolerance.
static bool below(double value, double threshold) {
    return value <
           threshold - 1e-12 * std::max(1.0, std::max(std::abs(value), std::abs(threshold)));
}
static double quantile(std::vector<double> x, double q) {
    if (x.empty())
        return 0;
    std::sort(x.begin(), x.end());
    double p = static_cast<double>(x.size() - 1) * q;
    size_t i = static_cast<size_t>(p), j = std::min(i + 1, x.size() - 1);
    return x[i] + (x[j] - x[i]) * (p - static_cast<double>(i));
}
// Weighted, centered two-column QR. Orthogonalizing Q against the intercept
// avoids normal equations and rejects the rank-deficient constant-Q case.
static bool qr(const std::vector<Interval> &rows, const std::vector<double> &w, double &a,
               double &b) {
    double sw = std::accumulate(w.begin(), w.end(), 0.0), mq = 0, mp = 0;
    if (rows.size() < 20 || sw <= 0)
        return false;
    for (size_t i = 0; i < rows.size(); ++i) {
        mq += w[i] * rows[i].q();
        mp += w[i] * rows[i].p();
    }
    mq /= sw;
    mp /= sw;
    double norm = 0;
    for (size_t i = 0; i < rows.size(); ++i)
        norm = ::hypot(norm, std::sqrt(w[i]) * (rows[i].q() - mq));
    if (norm <= 1e-10 * std::max(1.0, std::abs(mq)) * std::sqrt(sw))
        return false;
    double dot = 0;
    for (size_t i = 0; i < rows.size(); ++i)
        dot += std::sqrt(w[i]) * (rows[i].q() - mq) / norm * std::sqrt(w[i]) * (rows[i].p() - mp);
    b = dot / norm;
    a = mp - b * mq;
    return std::isfinite(a) && std::isfinite(b);
}
static bool robust(const std::vector<Interval> &rows, double &a, double &b) {
    std::vector<double> w(rows.size(), 1), residual(rows.size());
    for (unsigned iteration = 0; iteration < 10; ++iteration) {
        if (!qr(rows, w, a, b))
            return false;
        for (size_t i = 0; i < rows.size(); ++i)
            residual[i] = rows[i].p() - a - b * rows[i].q();
        double center = quantile(residual, .5);
        std::vector<double> deviations;
        deviations.reserve(rows.size());
        for (double e : residual)
            deviations.push_back(std::abs(e - center));
        double cutoff = 1.345 * std::max(.01, 1.4826 * quantile(deviations, .5));
        for (size_t i = 0; i < rows.size(); ++i)
            w[i] = std::min(1.0, cutoff / std::max(1e-30, std::abs(residual[i])));
    }
    return qr(rows, w, a, b);
}
Model fit_model(const std::vector<Interval> &train, const std::vector<Interval> &test,
                const std::vector<Interval> &all, const Config &cfg) {
    Model m;
    double a = 0, b = 0;
    if (!robust(train, a, b) || test.size() < 20)
        return m;
    double mean = 0, sse = 0, sst = 0, squares = 0;
    for (const auto &r : test)
        mean += r.p();
    mean /= static_cast<double>(test.size());
    std::vector<double> residual;
    for (const auto &r : test) {
        double error = r.p() - std::max(0.0, a + b * r.q());
        sse += error * error;
        sst += (r.p() - mean) * (r.p() - mean);
        squares += r.p() * r.p();
        residual.push_back(std::abs(error));
    }
    m.r2 = sst > 1e-12 ? 1 - sse / sst : 0;
    m.nrmse = squares > 1e-12 ? std::sqrt(sse / squares) : 1;
    m.residual95 = quantile(residual, .95);
    m.reason = MODEL_VALIDATION;
    if (below(m.r2, cfg.min_r2) || below(cfg.max_nrmse, m.nrmse))
        return m;
    if (!robust(all, m.b0, m.b1)) {
        m.reason = MODEL_RANK;
        return m;
    }
    m.q_min = all.front().q();
    m.q_max = m.q_min;
    for (const auto &r : all) {
        m.q_min = std::min(m.q_min, r.q());
        m.q_max = std::max(m.q_max, r.q());
    }
    m.valid = true;
    m.reason = AVAILABLE;
    // Stable provenance rounds only the identifier material, never the calculation.
    std::ostringstream id;
    id << std::fixed << std::setprecision(9) << m.b0 << ',' << m.b1 << ',' << m.q_min << ','
       << m.q_max << ',' << m.residual95 << ',' << all.back().start;
    m.id = hash64(id.str());
    return m;
}
Engine::Engine(const Config &cfg) : config_(cfg), pending_(cfg) {
    config_.validate();
}
void Engine::reset() {
    history_.clear();
    ready_.clear();
    detection_ = Detection();
    current_ = Interval();
    active_ = false;
    last_sample_ = -1;
    unassigned_import_ = unassigned_export_ = 0;
}
void Engine::configure(const Config &cfg) {
    cfg.validate();
    pending_ = cfg;
    has_pending_ = true;
    if (!active_)
        commit();
}
void Engine::commit() {
    if (!has_pending_)
        return;
    if (config_.identity() != pending_.identity()) {
        if (detection_.state != LEARNING)
            current_.flags |= STATE_RESET;
        history_.clear();
        detection_ = Detection();
        current_.flags |= CONFIG_RESET;
    } else if (config_.id() != pending_.id()) {
        // Old coefficients must not bypass a tighter newly accepted quality gate.
        detection_.model = Model();
    }
    config_ = pending_;
    has_pending_ = false;
}
void Engine::ingest(const MeterSample &sample) {
    const bool finite_p =
        sample.p_valid && std::isfinite(sample.watts) && std::abs(sample.watts) <= 1e12;
    if (!sample.time_valid || sample.utc < 946684800 || sample.utc > 2145916799) {
        if (finite_p) {
            double p = sample.watts * config_.polarity * config_.power_scale / 1000;
            unassigned_import_ += std::max(0.0, p) / 3600;
            unassigned_export_ += std::max(0.0, -p) / 3600;
        }
        return;
    }
    if (sample.utc <= last_sample_) {
        if (active_)
            current_.flags |= TIME_ORDER;
        return;
    }
    advance(sample.utc);
    if (!active_) {
        commit();
        current_ = Interval();
        current_.start = sample.utc / INTERVAL * INTERVAL;
        active_ = true;
    }
    if (last_sample_ >= 0 && sample.utc > last_sample_ + 1)
        current_.flags |= GAP;
    last_sample_ = sample.utc;
    current_.flags |= sample.flags;
    if (finite_p) {
        double p = sample.watts * config_.polarity * config_.power_scale / 1000;
        current_.p_sum += p;
        ++current_.p_count;
        current_.import_kwh += std::max(0.0, p) / 3600;
        current_.export_kwh += std::max(0.0, -p) / 3600;
    }
    if (sample.q_valid && std::isfinite(sample.vars) && std::abs(sample.vars) <= 1e12) {
        current_.q_sum += sample.vars * config_.reactive_scale / 1000;
        ++current_.q_count;
    }
}
void Engine::advance(int64_t end) {
    if (!active_ || end < current_.start + INTERVAL)
        return;
    // A prolonged outage expires history without allocating unbounded empty records.
    if (end - current_.start > 60LL * DAY) {
        finalize_interval();
        const bool was_established = detection_.state != LEARNING;
        history_.clear();
        detection_ = Detection();
        current_ = Interval();
        current_.start = end / INTERVAL * INTERVAL;
        current_.flags = GAP | (was_established ? static_cast<uint32_t>(STATE_RESET) : 0u);
        commit();
        return;
    }
    while (end >= current_.start + INTERVAL) {
        finalize_interval();
        if (ready_.size() > 5761)
            throw std::runtime_error("Drain completed intervals before continuing");
    }
}
Result Engine::estimate(const Interval &r) const {
    Result out;
    out.measured = r;
    out.state = detection_.state;
    out.config_id = config_.id();
    out.model_id = detection_.model.id;
    out.flags |= r.flags;
    if (r.p_count < INTERVAL)
        out.flags |= PARTIAL;
    if (r.q_count < INTERVAL * config_.min_coverage)
        out.flags |= Q_MISSING;
    auto unavailable = [&](Reason reason) {
        out.reason = reason;
        return out;
    };
    if (config_.storage || config_.other_generation) {
        out.flags |= AMBIGUITY;
        return unavailable(OTHER_DER);
    }
    if (!config_.located()) {
        out.flags |= MISSING_LOCATION;
        return unavailable(COORDINATES);
    }
    if (detection_.night_export) {
        out.flags |= AMBIGUITY;
        return unavailable(NIGHT_EXPORT);
    }
    if (detection_.valid_days < 30 || detection_.state == LEARNING) {
        out.flags |= LEARNING_FLAG;
        return unavailable(WARMUP);
    }
    if (detection_.state != LIKELY_PV)
        return unavailable(NO_PV_EVIDENCE);
    if (r.p_count < INTERVAL * config_.min_coverage)
        return unavailable(COVERAGE);
    if (r.q_count < INTERVAL * config_.min_coverage)
        return unavailable(REACTIVE_MISSING);
    const Model &m = detection_.model;
    if (!m.valid)
        return unavailable(m.reason);
    Solar sun = solar(r.start + INTERVAL / 2, config_.latitude, config_.longitude);
    if (sun.elevation >= -6 && sun.elevation <= 10)
        return unavailable(TWILIGHT);
    if (sun.elevation > 10) {
        double margin = (m.q_max - m.q_min) * config_.q_margin;
        if (below(r.q(), m.q_min - margin) || below(m.q_max + margin, r.q()))
            return unavailable(Q_RANGE);
        double load = std::max(0.0, m.b0 + m.b1 * r.q());
        if (!std::isfinite(load))
            return unavailable(MODEL_INVALID);
        double floor = r.export_kwh * 3600 / r.p_count, raw = load - r.p();
        out.generation_kw = std::max(floor, std::max(0.0, raw));
        out.low_kw = std::max(floor, std::max(0.0, raw - m.residual95));
        out.high_kw = std::max(out.generation_kw, raw + m.residual95);
    }
    out.available = true;
    out.reason = AVAILABLE;
    out.flags &= ~static_cast<uint32_t>(UNAVAILABLE);
    out.estimated_seconds = r.p_count;
    out.generation_kwh = out.generation_kw * r.p_count / 3600;
    return out;
}
Result Engine::finalize_interval() {
    if (!active_)
        throw std::logic_error("No active interval");
    Result out = estimate(current_);
    history_.push_back(current_);
    while (!history_.empty() && history_.front().start < current_.start + INTERVAL - 60LL * DAY)
        history_.pop_front();
    State old = detection_.state;
    if ((current_.start + INTERVAL) % DAY == 0)
        evaluate_history();
    out.state_changed = old != detection_.state || (current_.flags & STATE_RESET);
    out.event_state = detection_.state;
    out.event_config_id = config_.id();
    out.event_model_id = detection_.model.id;
    // Estimate provenance describes the model at interval start. Transition metadata
    // is published separately using the new detection snapshot.
    ready_.push_back(out);
    const int64_t next = current_.start + INTERVAL;
    current_ = Interval();
    current_.start = next;
    commit();
    return out;
}
const Detection &Engine::evaluate_history() {
    if (history_.empty())
        return detection_;
    int64_t complete_day = (history_.back().start + INTERVAL) / DAY - 1;
    if (complete_day <= detection_.evaluated_day)
        return detection_;
    detection_.evaluated_day = complete_day;
    detection_.model = Model();
    detection_.valid_days = detection_.export_days = detection_.profile_days = 0;
    detection_.export_fraction = detection_.valley_fraction = detection_.valley_median = 0;
    detection_.night_export = false;
    if (!config_.located()) {
        detection_.state = LEARNING;
        detection_.candidate = LEARNING;
        detection_.streak = 0;
        return detection_;
    }
    struct Daily {
        std::vector<Interval> rows;
        int covered = 0, light = 0, dark = 0;
    };
    std::map<int64_t, Daily> days;
    for (const auto &r : history_) {
        if (r.start / DAY > complete_day)
            continue;
        Daily &d = days[r.start / DAY];
        d.covered += r.p_count;
        if (r.p_count >= INTERVAL * config_.min_coverage) {
            Solar sun = solar(r.start + INTERVAL / 2, config_.latitude, config_.longitude);
            d.light += sun.elevation > 10;
            d.dark += sun.elevation < -6;
            d.rows.push_back(r);
        }
    }
    std::vector<Daily> selected;
    for (const auto &e : days)
        if (e.second.covered >= DAY * config_.min_coverage && e.second.light >= 4 &&
            e.second.dark >= 4)
            selected.push_back(e.second);
    if (selected.size() > 30)
        selected.erase(selected.begin(), selected.end() - 30);
    detection_.valid_days = static_cast<unsigned>(selected.size());
    if (selected.size() < 30) {
        detection_.state = LEARNING;
        detection_.candidate = LEARNING;
        detection_.streak = 0;
        return detection_;
    }
    std::vector<double> valleys, all_abs;
    for (const auto &d : selected)
        for (const auto &r : d.rows)
            all_abs.push_back(std::abs(r.p()));
    double minimum_shoulder = std::max(.1 * quantile(all_abs, .5), 1e-6);
    unsigned daylight = 0, exports = 0, nights_exporting = 0;
    std::vector<Interval> train, test, all;
    for (size_t i = 0; i < selected.size(); ++i) {
        double morning = 0, midday = 0, evening = 0;
        unsigned nm = 0, nd = 0, ne = 0;
        bool day_export = false, night_export = false;
        for (const auto &r : selected[i].rows) {
            Solar s = solar(r.start + INTERVAL / 2, config_.latitude, config_.longitude);
            if (s.elevation > 10) {
                ++daylight;
                if (below(r.p(), -config_.export_threshold_kw)) {
                    ++exports;
                    day_export = true;
                }
            }
            if (s.elevation < -6) {
                night_export |= below(r.p(), -config_.export_threshold_kw);
                if (r.q_count >= INTERVAL * config_.min_coverage) {
                    all.push_back(r);
                    (i < 25 ? train : test).push_back(r);
                }
            }
            if (s.hour >= 6 && s.hour < 10) {
                morning += r.p();
                ++nm;
            }
            if (s.hour >= 10 && s.hour < 16) {
                midday += r.p();
                ++nd;
            }
            if (s.hour >= 17 && s.hour < 21) {
                evening += r.p();
                ++ne;
            }
        }
        detection_.export_days += day_export;
        nights_exporting += night_export;
        if (nm >= 12 && nd >= 20 && ne >= 12) {
            double shoulder = std::min(morning / nm, evening / ne);
            if (shoulder > minimum_shoulder)
                valleys.push_back(midday / nd / shoulder);
        }
    }
    detection_.night_export = nights_exporting >= 3;
    detection_.export_fraction = daylight ? static_cast<double>(exports) / daylight : 0;
    detection_.profile_days = static_cast<unsigned>(valleys.size());
    detection_.valley_median = quantile(valleys, .5);
    unsigned low = 0;
    for (double v : valleys)
        low += below(v, config_.valley_threshold);
    detection_.valley_fraction =
        valleys.empty() ? 0 : static_cast<double>(low) / static_cast<double>(valleys.size());
    detection_.model = fit_model(train, test, all, config_);
    State candidate = NO_EVIDENCE;
    if (config_.storage || config_.other_generation || detection_.night_export)
        candidate = AMBIGUOUS;
    else if ((detection_.export_fraction >= config_.export_fraction &&
              detection_.export_days >= 5) ||
             (valleys.size() >= 15 && detection_.valley_fraction >= config_.valley_fraction &&
              below(detection_.valley_median, config_.valley_median)))
        candidate = LIKELY_PV;
    if (candidate == detection_.candidate)
        ++detection_.streak;
    else {
        detection_.candidate = candidate;
        detection_.streak = 1;
    }
    detection_.streak = std::min(3u, detection_.streak);
    if (detection_.streak >= 3)
        detection_.state = candidate;
    return detection_;
}
std::vector<Result> Engine::take_results() {
    std::vector<Result> r;
    r.swap(ready_);
    return r;
}
} // namespace pv
