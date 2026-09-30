#include "pv_core.hpp"
#include "pv_delivery.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
int main() {
    try {
        pv::Config cfg;
        cfg.latitude = 34;
        cfg.longitude = -118;
        pv::Engine e(cfg);
        int64_t start = 1735689600;
        for (int i = 0; i < 900; ++i) {
            pv::MeterSample s;
            s.utc = start + i;
            s.watts = i < 450 ? 1000 : -1000;
            s.vars = 300;
            e.ingest(s);
        }
        e.advance(start + 900);
        auto out = e.take_results();
        check(out.size() == 1, "interval count");
        check(std::abs(out[0].measured.p()) < 1e-12, "net average");
        check(std::abs(out[0].measured.export_kwh - .125) < 1e-12, "export before average");
        check(!out[0].available, "cold start cannot estimate");
        auto saved = e.serialize_state();
        pv::Engine restored(cfg);
        restored.restore_state(saved);
        check(restored.serialize_state() == saved, "state roundtrip");
        saved.back() ^= 1;
        bool rejected = false;
        try {
            restored.restore_state(saved);
        } catch (...) {
            rejected = true;
        }
        check(rejected, "corrupt state accepted");
        auto payload = pv::hourly_payload(out);
        check(payload.size() < 256, "wire limit");
        pv::MeterSample duplicate;
        duplicate.utc = start + 899;
        duplicate.watts = -1000;
        restored.ingest(duplicate);
        check(restored.history().size() == 1, "duplicate sample");
        pv::Config changed = cfg;
        changed.polarity = -1;
        restored.configure(changed);
        restored.advance(start + 1800);
        check(restored.history().empty(), "identity reset");
        auto s = pv::solar(1710936000, 0, 0);
        check(s.elevation > 85, "equinox noon");
        // Configuration is frozen until the following boundary; invalid updates are atomic.
        pv::Engine configured(cfg);
        pv::MeterSample one;
        one.utc = start;
        one.watts = 1000;
        configured.ingest(one);
        configured.configure(changed);
        one.utc++;
        configured.ingest(one);
        auto first = configured.finalize_interval();
        check(first.measured.p() == 1 && first.config_id == cfg.id(),
              "interval configuration freeze");
        one.utc = start + 900;
        configured.ingest(one);
        auto second = configured.finalize_interval();
        check(second.measured.p() == -1 && second.config_id == changed.id(),
              "next interval config");
        bool invalid = false;
        changed.min_coverage = .89;
        try {
            configured.configure(changed);
        } catch (const std::invalid_argument &) {
            invalid = true;
        }
        check(invalid && configured.config().min_coverage == .9, "invalid update changed config");
        // Exercise bounded policy retries and crash/restart preservation, not a mock SDK success.
        pv::Delivery pending;
        pending.add(first, configured.detection());
        pending.flush_hour();
        unsigned calls = 0;
        auto reject = [&](const std::string &, bool, bool) {
            ++calls;
            return -1;
        };
        pending.drain(reject, start);
        check(calls == 1 && pending.pending() == 1, "first policy rejection");
        pv::Delivery recovered;
        recovered.restore(pending.serialize());
        recovered.drain(reject, start + 59);
        check(calls == 1, "retry backoff");
        recovered.drain(reject, start + 60);
        check(calls == 2 && recovered.pending() == 0 && recovered.dropped() == 1,
              "bounded retries");
        recovered.add(first, configured.detection());
        recovered.flush_hour();
        check(recovered.pending() == 0, "restart duplicate interval");
        // Multiple historical days replayed today must not bypass today's byte allowance.
        pv::Delivery budget;
        size_t bytes = 0;
        auto sink = [&](const std::string &p, bool event, bool alarm) {
            check(p.size() <= (event ? 256 : 1024), "payload limit");
            check(!alarm, "routine estimate alarm");
            if (!event)
                bytes += p.size();
            return 0;
        };
        for (unsigned i = 0; i < 96 * 4; ++i) {
            pv::Result r = first;
            r.measured.start = start + i * 900;
            budget.add(r, configured.detection());
            budget.drain(sink, start);
        }
        check(bytes <= 8192 && budget.dropped() > 0, "daily output budget");
        pv::Delivery transition;
        pv::Result event = first;
        event.state_changed = true;
        event.event_state = pv::LIKELY_PV;
        event.event_config_id = 17;
        event.event_model_id = 23;
        transition.add(event, pv::Detection()); // Later detector snapshots must not rewrite events.
        bool alarm = false;
        transition.drain(
            [&](const std::string &text, bool is_event, bool requested) {
                alarm |=
                    is_event && requested &&
                    text.find("likely_pv#0000000000000011#0000000000000017") != std::string::npos;
                return 0;
            },
            start);
        check(alarm, "transition provenance frozen at evaluation");
        // Analytic held-out moments: variance(P)=5/12, error variance=1/6,
        // so R2=0.6 and normalized RMSE=0.25 exactly at both quality gates.
        std::vector<pv::Interval> train, heldout;
        for (int i = 0; i < 40; ++i) {
            pv::Interval r;
            r.p_count = r.q_count = 900;
            r.q_sum = (i % 2) * 900;
            r.p_sum = (1 + i % 2) * 900;
            train.push_back(r);
            r.p_sum += (i % 4 < 2 ? -1 : 1) * std::sqrt(1.0 / 6) * 900;
            heldout.push_back(r);
        }
        pv::Config quality = cfg;
        quality.min_r2 = .6;
        auto model = pv::fit_model(train, heldout, train, quality);
        check(model.valid && std::abs(model.r2 - .6) < 1e-12 && std::abs(model.nrmse - .25) < 1e-12,
              "inclusive model quality thresholds");
        quality.max_nrmse = .249999;
        check(!pv::fit_model(train, heldout, train, quality).valid,
              "model quality rejection boundary");
        pv::Delivery queue;
        for (unsigned i = 0; i < 200; ++i) {
            pv::Result r = first;
            r.measured.start = start + i * 900;
            queue.add(r, configured.detection());
            queue.flush_hour();
        }
        check(queue.pending() == 32 && queue.dropped() > 0, "outbox cap");
        auto checkpoint = queue.serialize();
        checkpoint.back() ^= 1;
        bool corrupt = false;
        try {
            recovered.restore(checkpoint);
        } catch (const std::runtime_error &) {
            corrupt = true;
        }
        check(corrupt, "delivery corruption");
        pv::Engine bounded(cfg);
        for (int day = 0; day < 62; ++day) {
            for (int quarter = 0; quarter < 96; ++quarter) {
                one.utc = start + day * 86400 + quarter * 900;
                bounded.ingest(one);
                bounded.finalize_interval();
                bounded.take_results();
            }
        }
        check(bounded.history().size() == 60 * 96, "60 calendar day history cap");
        pv::Engine restore_bounded(cfg);
        restore_bounded.restore_state(bounded.serialize_state());
        check(restore_bounded.history().size() == 60 * 96, "full history restore");
        check(bounded.serialize_state().size() < 300000, "compact persistent history");
        one.utc += 100LL * 86400;
        bounded.ingest(one);
        check(bounded.history().empty() && bounded.detection().state == pv::LEARNING,
              "long clock jump expires model/history");
        pv::Engine established(cfg);
        one.watts = 1000;
        for (int day = 0; day < 32; ++day) {
            for (int second = 0; second < 86400; ++second) {
                one.utc = start + day * 86400 + second;
                established.ingest(one);
            }
            established.advance(start + (day + 1) * 86400);
            established.take_results();
        }
        check(established.detection().state == pv::NO_EVIDENCE, "stable no-evidence before outage");
        one.utc = start + 132LL * 86400;
        established.ingest(one);
        established.advance(one.utc + 900);
        auto expired = established.take_results();
        check(expired.back().state_changed && expired.back().event_state == pv::LEARNING,
              "long outage retains a learning-reset transition");
        std::cout << "PASS: energy sign integration, learning, checksums, identity reset, solar "
                     "geometry, interval config, bounded retries/outbox, wire/daily limits\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
