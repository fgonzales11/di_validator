#pragma once
#include "pv_core.hpp"
#include <functional>
namespace pv {
struct Message {
    bool event = false, alarm = false;
    unsigned attempts = 0;
    int64_t retry_after = 0;
    std::string payload;
};
class Delivery {
  public:
    typedef std::function<int(const std::string &, bool, bool)> Sink;
    void add(const Result &, const Detection &);
    void flush_hour();
    void drain(const Sink &, int64_t delivery_utc, bool finish = false);
    std::string serialize() const;
    void restore(const std::string &);
    size_t pending() const {
        return outbox_.size();
    }
    unsigned rejected() const {
        return rejected_;
    }
    unsigned dropped() const {
        return dropped_;
    }
    unsigned sent_bytes() const {
        return sent_bytes_;
    }

  private:
    std::vector<Result> batch_;
    std::deque<Message> outbox_;
    int64_t last_interval_ = -1, day_ = -1, budget_day_ = -1;
    double imported_ = 0, exported_ = 0, generated_ = 0;
    uint32_t measured_seconds_ = 0, estimated_seconds_ = 0;
    unsigned sent_bytes_ = 0, rejected_ = 0, dropped_ = 0;
    uint64_t day_config_ = 0, day_model_ = 0;
    bool mixed_provenance_ = false;
    void enqueue(const std::string &, bool event = false, bool alarm = false);
    void daily();
};
} // namespace pv
