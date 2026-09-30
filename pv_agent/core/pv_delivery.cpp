#include "pv_delivery.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace pv {
void Delivery::enqueue(const std::string &payload, bool event, bool alarm) {
    if (payload.size() > (event ? 256u : 1024u))
        throw std::length_error("DI output size");
    if (outbox_.size() >= 32) {
        ++dropped_;
        return;
    }
    Message m;
    m.event = event;
    m.alarm = alarm;
    m.payload = payload;
    outbox_.push_back(m);
}
void Delivery::flush_hour() {
    if (!batch_.empty()) {
        enqueue(hourly_payload(batch_));
        batch_.clear();
    }
}
void Delivery::daily() {
    if (day_ < 0)
        return;
    std::ostringstream s;
    s << std::setprecision(9) << "PVD1:{\"day\":" << day_ * DAY << ",\"import_kwh\":" << imported_
      << ",\"export_kwh\":" << exported_ << ",\"generation_kwh\":";
    if (estimated_seconds_)
        s << generated_;
    else
        s << "null";
    s << ",\"measured_s\":" << measured_seconds_ << ",\"estimated_s\":" << estimated_seconds_
      << ",\"config\":" << quote(hex64(day_config_)) << ",\"model\":" << quote(hex64(day_model_))
      << ",\"mixed\":" << (mixed_provenance_ ? "true" : "false")
      << ",\"flags\":48,\"dropped\":" << dropped_ << ",\"rejected\":" << rejected_ << '}';
    enqueue(s.str());
}
void Delivery::add(const Result &r, const Detection &d) {
    (void)d; // Transition provenance is frozen in the completed result.
    if (r.measured.start <= last_interval_)
        return;
    last_interval_ = r.measured.start;
    if (day_ != r.measured.start / DAY) {
        if (day_ >= 0) {
            flush_hour();
            daily();
        }
        day_ = r.measured.start / DAY;
        imported_ = exported_ = generated_ = 0;
        measured_seconds_ = estimated_seconds_ = 0;
        mixed_provenance_ = false;
        day_config_ = r.config_id;
        day_model_ = r.model_id;
    }
    mixed_provenance_ |= day_config_ != r.config_id || day_model_ != r.model_id;
    imported_ += r.measured.import_kwh;
    exported_ += r.measured.export_kwh;
    generated_ += r.generation_kwh;
    measured_seconds_ += r.measured.p_count;
    estimated_seconds_ += r.estimated_seconds;
    if (!batch_.empty() &&
        (batch_.front().config_id != r.config_id || batch_.front().model_id != r.model_id ||
         batch_.front().measured.start / 3600 != r.measured.start / 3600))
        flush_hour();
    batch_.push_back(r);
    if (batch_.size() == 4 || (r.measured.start + INTERVAL) % 3600 == 0)
        flush_hour();
    if (r.state_changed) {
        Result event = r;
        event.state = r.event_state;
        event.config_id = r.event_config_id;
        event.model_id = r.event_model_id;
        enqueue(transition_payload(event), true, r.event_state == LIKELY_PV);
    }
    if ((r.measured.start + INTERVAL) % DAY == 0) {
        daily();
        day_ = -1;
    }
}
void Delivery::drain(const Sink &sink, int64_t utc, bool finish) {
    if (budget_day_ != utc / DAY) {
        budget_day_ = utc / DAY;
        sent_bytes_ = 0;
    }
    while (!outbox_.empty()) {
        Message &m = outbox_.front();
        if (!finish && m.retry_after > utc)
            break;
        if (!m.event && sent_bytes_ + m.payload.size() > 8192) {
            ++dropped_;
            outbox_.pop_front();
            continue;
        }
        // Count attempted bytes conservatively; a transport error can leave delivery uncertain.
        if (!m.event)
            sent_bytes_ += static_cast<unsigned>(m.payload.size());
        ++m.attempts;
        int status = sink(m.payload, m.event, m.alarm);
        if (status == 0) {
            outbox_.pop_front();
            continue;
        }
        ++rejected_;
        if (m.attempts >= 2) {
            ++dropped_;
            outbox_.pop_front();
        } else {
            m.retry_after = utc + 60;
            if (!finish)
                break;
        }
    }
}
std::string Delivery::serialize() const {
    std::ostringstream s;
    s << std::setprecision(17);
    s << last_interval_ << ' ' << day_ << ' ' << budget_day_ << ' ' << imported_ << ' ' << exported_
      << ' ' << generated_ << ' ' << measured_seconds_ << ' ' << estimated_seconds_ << ' '
      << sent_bytes_ << ' ' << rejected_ << ' ' << dropped_ << ' ' << day_config_ << ' '
      << day_model_ << ' ' << mixed_provenance_ << '\n';
    s << batch_.size() << '\n';
    for (const auto &r : batch_) {
        const auto &m = r.measured;
        s << m.start << ' ' << m.p_count << ' ' << m.q_count << ' ' << m.p_sum << ' ' << m.q_sum
          << ' ' << m.import_kwh << ' ' << m.export_kwh << ' ' << m.flags << ' ' << r.state << ' '
          << r.reason << ' ' << r.flags << ' ' << r.config_id << ' ' << r.model_id << ' '
          << r.available << ' ' << r.state_changed << ' ' << r.generation_kw << ' ' << r.low_kw
          << ' ' << r.high_kw << ' ' << r.generation_kwh << ' ' << r.estimated_seconds << '\n';
    }
    s << outbox_.size() << '\n';
    for (const auto &m : outbox_)
        s << m.event << ' ' << m.alarm << ' ' << m.attempts << ' ' << m.retry_after << '\n'
          << m.payload << '\n';
    return "PVDL1" + hex64(hash64(s.str())) + s.str();
}
void Delivery::restore(const std::string &bytes) {
    if (bytes.size() < 21 || bytes.size() > 65536 || bytes.substr(0, 5) != "PVDL1" ||
        bytes.substr(5, 16) != hex64(hash64(bytes.substr(21))))
        throw std::runtime_error("Delivery checkpoint checksum");
    Delivery d;
    std::istringstream s(bytes.substr(21));
    s >> d.last_interval_ >> d.day_ >> d.budget_day_ >> d.imported_ >> d.exported_ >>
        d.generated_ >> d.measured_seconds_ >> d.estimated_seconds_ >> d.sent_bytes_ >>
        d.rejected_ >> d.dropped_ >> d.day_config_ >> d.day_model_ >> d.mixed_provenance_;
    unsigned count = 0;
    s >> count;
    if (count > 4)
        throw std::runtime_error("Delivery batch size");
    for (unsigned i = 0; i < count; ++i) {
        Result r;
        auto &m = r.measured;
        unsigned state = 0, reason = 0;
        s >> m.start >> m.p_count >> m.q_count >> m.p_sum >> m.q_sum >> m.import_kwh >>
            m.export_kwh >> m.flags >> state >> reason >> r.flags >> r.config_id >> r.model_id >>
            r.available >> r.state_changed >> r.generation_kw >> r.low_kw >> r.high_kw >>
            r.generation_kwh >> r.estimated_seconds;
        if (state > AMBIGUOUS || reason > MODEL_INVALID || m.p_count > 900 || m.q_count > 900 ||
            r.estimated_seconds > 900)
            throw std::runtime_error("Delivery result bounds");
        r.state = static_cast<State>(state);
        r.reason = static_cast<Reason>(reason);
        d.batch_.push_back(r);
    }
    s >> count;
    if (count > 32)
        throw std::runtime_error("Outbox bound");
    for (unsigned i = 0; i < count; ++i) {
        Message m;
        s >> m.event >> m.alarm >> m.attempts >> m.retry_after;
        s.ignore(1);
        std::getline(s, m.payload);
        if (m.attempts > 2 || m.payload.size() > (m.event ? 256u : 1024u))
            throw std::runtime_error("Outbox record bounds");
        d.outbox_.push_back(m);
    }
    if (!s || d.sent_bytes_ > 8192)
        throw std::runtime_error("Invalid delivery checkpoint");
    *this = d;
}
} // namespace pv
