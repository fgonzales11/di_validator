#pragma once
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace pv {
static const char *const ALGORITHM = "pv-meter-only-1";
static const int INTERVAL = 900;
static const int DAY = 86400;
enum State { LEARNING, LIKELY_PV, NO_EVIDENCE, AMBIGUOUS };
enum Reason {
    AVAILABLE,
    WARMUP,
    COORDINATES,
    COVERAGE,
    NO_PV_EVIDENCE,
    OTHER_DER,
    NIGHT_EXPORT,
    MODEL_RANK,
    MODEL_VALIDATION,
    Q_RANGE,
    TWILIGHT,
    REACTIVE_MISSING,
    INVALID_TIME,
    MODEL_INVALID
};
enum Flag : uint32_t {
    GAP = 1,
    TIME_ORDER = 2,
    PARTIAL = 4,
    Q_MISSING = 8,
    MODEL_ASSUMPTIONS = 16,
    UNCALIBRATED = 32,
    AMBIGUITY = 64,
    MISSING_LOCATION = 128,
    LEARNING_FLAG = 256,
    UNAVAILABLE = 512,
    QUEUE_LOSS = 1024,
    CONFIG_RESET = 2048,
    STATE_RESET = 4096
};
const char *state_name(State value);
const char *reason_name(Reason value);
struct Config {
    double latitude = 999, longitude = 999;
    double polarity = 1, power_scale = 1, reactive_scale = 1;
    double min_coverage = .9, export_threshold_kw = .05, export_fraction = .05;
    double valley_threshold = .55, valley_fraction = .35, valley_median = .8;
    double min_r2 = .5, max_nrmse = .25, q_margin = .1;
    bool storage = false, other_generation = false;
    void validate() const;
    void set(const std::string &, const std::string &);
    std::map<std::string, std::string> values() const;
    bool located() const {
        return latitude != 999 && longitude != 999;
    }
    std::string identity() const;
    uint64_t id() const;
};
struct Solar {
    double elevation, hour;
};
Solar solar(int64_t utc, double latitude, double longitude);
uint64_t hash64(const std::string &);
std::string hex64(uint64_t);
struct MeterSample {
    int64_t utc = 0; // Start of one measured second, UTC epoch seconds.
    double watts = 0, vars = 0;
    bool p_valid = true, q_valid = true, time_valid = true;
    uint32_t flags = 0;
    double phase_volts[3] = {}, phase_watts[3] = {}; // Optional diagnostics only.
    uint8_t diagnostic_valid = 0;                    // Bits 0..2 voltage, 3..5 phase W.
};
struct Interval {
    int64_t start = 0;
    uint16_t p_count = 0, q_count = 0;
    double p_sum = 0, q_sum = 0, import_kwh = 0, export_kwh = 0;
    uint32_t flags = 0;
    double p() const {
        return p_count ? p_sum / p_count : 0;
    }
    double q() const {
        return q_count ? q_sum / q_count : 0;
    }
};
struct Model {
    bool valid = false;
    Reason reason = MODEL_RANK;
    double b0 = 0, b1 = 0, q_min = 0, q_max = 0, residual95 = 0, r2 = 0, nrmse = 0;
    uint64_t id = 0;
};
struct Detection {
    State state = LEARNING, candidate = LEARNING;
    unsigned streak = 0, valid_days = 0, export_days = 0, profile_days = 0;
    double export_fraction = 0, valley_fraction = 0, valley_median = 0;
    bool night_export = false;
    int64_t evaluated_day = -1;
    Model model;
};
struct Result {
    Interval measured;
    State state = LEARNING;
    Reason reason = WARMUP;
    uint32_t flags = MODEL_ASSUMPTIONS | UNCALIBRATED | UNAVAILABLE;
    uint64_t config_id = 0, model_id = 0;
    bool available = false, state_changed = false;
    State event_state = LEARNING;
    uint64_t event_config_id = 0, event_model_id = 0;
    double generation_kw = 0, low_kw = 0, high_kw = 0, generation_kwh = 0;
    uint16_t estimated_seconds = 0;
};
class Engine {
  public:
    explicit Engine(const Config & = Config());
    void configure(const Config &); // Commit at the next interval boundary.
    void ingest(const MeterSample &);
    void advance(int64_t exclusive_end); // Complete elapsed intervals, never fill missing seconds.
    Result finalize_interval();
    const Detection &evaluate_history();
    std::vector<Result> take_results();
    const Detection &detection() const {
        return detection_;
    }
    const Config &config() const {
        return config_;
    }
    const std::deque<Interval> &history() const {
        return history_;
    }
    double unassigned_import() const {
        return unassigned_import_;
    }
    double unassigned_export() const {
        return unassigned_export_;
    }
    std::string serialize_state() const;
    void restore_state(const std::string &);
    void reset();

  private:
    Config config_, pending_;
    bool has_pending_ = false, active_ = false;
    int64_t last_sample_ = -1;
    Interval current_;
    std::deque<Interval> history_;
    std::vector<Result> ready_;
    Detection detection_;
    double unassigned_import_ = 0, unassigned_export_ = 0;
    void commit();
    Result estimate(const Interval &) const;
};
Model fit_model(const std::vector<Interval> &train, const std::vector<Interval> &test,
                const std::vector<Interval> &all, const Config &);
std::string result_json(const Result &);
std::string detection_json(const Detection &);
std::string quote(const std::string &);
std::string hourly_payload(const std::vector<Result> &);
std::string transition_payload(const Result &);
void atomic_write(const std::string &path, const std::string &bytes);
std::string read_state_file(const std::string &path);
} // namespace pv
