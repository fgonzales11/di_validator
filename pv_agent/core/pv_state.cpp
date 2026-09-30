#include "pv_core.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>

namespace pv {
namespace {
struct Writer {
    std::string bytes;
    void u(uint64_t x, unsigned n = 8) {
        for (unsigned i = 0; i < n; ++i)
            bytes.push_back(static_cast<char>(x >> (i * 8)));
    }
    void d(double x) {
        uint64_t b;
        std::memcpy(&b, &x, 8);
        u(b);
    }
    void s(const std::string &v) {
        u(v.size(), 4);
        bytes += v;
    }
    void config(const Config &c) {
        for (const auto &v : c.values()) {
            s(v.first);
            s(v.second);
        }
        s("");
    }
    void row(const Interval &r) {
        u(r.start);
        u(r.p_count, 2);
        u(r.q_count, 2);
        d(r.p_sum);
        d(r.q_sum);
        d(r.import_kwh);
        d(r.export_kwh);
        u(r.flags, 4);
    }
};
struct Reader {
    const std::string &b;
    size_t at = 0;
    explicit Reader(const std::string &v) : b(v) {}
    uint64_t u(unsigned n = 8) {
        if (at + n > b.size())
            throw std::runtime_error("Truncated checkpoint");
        uint64_t x = 0;
        for (unsigned i = 0; i < n; ++i)
            x |= static_cast<uint64_t>(static_cast<unsigned char>(b[at++])) << (8 * i);
        return x;
    }
    double d() {
        uint64_t v = u();
        double x;
        std::memcpy(&x, &v, 8);
        if (!std::isfinite(x))
            throw std::runtime_error("Nonfinite checkpoint");
        return x;
    }
    std::string s() {
        size_t n = static_cast<size_t>(u(4));
        if (n > 4096 || at + n > b.size())
            throw std::runtime_error("Bad checkpoint string");
        std::string v = b.substr(at, n);
        at += n;
        return v;
    }
    Config config() {
        Config c;
        unsigned count = 0;
        for (;;) {
            auto k = s();
            if (k.empty())
                break;
            auto v = s();
            c.set(k, v);
            if (++count > 32)
                throw std::runtime_error("Bad config count");
        }
        c.validate();
        return c;
    }
    Interval row() {
        Interval r;
        r.start = static_cast<int64_t>(u());
        r.p_count = static_cast<uint16_t>(u(2));
        r.q_count = static_cast<uint16_t>(u(2));
        r.p_sum = d();
        r.q_sum = d();
        r.import_kwh = d();
        r.export_kwh = d();
        r.flags = static_cast<uint32_t>(u(4));
        if (r.p_count > 900 || r.q_count > 900 || r.start % 900 || r.import_kwh < 0 ||
            r.export_kwh < 0)
            throw std::runtime_error("Invalid checkpoint interval");
        return r;
    }
};
} // namespace
std::string Engine::serialize_state() const {
    Writer w;
    w.s(ALGORITHM);
    w.config(config_);
    w.config(pending_);
    w.u(has_pending_, 1);
    w.u(active_, 1);
    w.u(static_cast<uint64_t>(last_sample_));
    w.row(current_);
    w.d(unassigned_import_);
    w.d(unassigned_export_);
    w.u(history_.size(), 4);
    for (const auto &r : history_)
        w.row(r);
    const Detection &v = detection_;
    w.u(v.state, 1);
    w.u(v.candidate, 1);
    w.u(v.streak, 1);
    w.u(v.valid_days, 1);
    w.u(v.export_days, 1);
    w.u(v.profile_days, 1);
    w.d(v.export_fraction);
    w.d(v.valley_fraction);
    w.d(v.valley_median);
    w.u(v.night_export, 1);
    w.u(static_cast<uint64_t>(v.evaluated_day));
    const Model &m = v.model;
    w.u(m.valid, 1);
    w.u(m.reason, 1);
    w.d(m.b0);
    w.d(m.b1);
    w.d(m.q_min);
    w.d(m.q_max);
    w.d(m.residual95);
    w.d(m.r2);
    w.d(m.nrmse);
    w.u(m.id);
    return "PVS1" + hex64(hash64(w.bytes)) + w.bytes;
}
void Engine::restore_state(const std::string &bytes) {
    if (bytes.size() < 20 || bytes.size() > 1048576 || bytes.substr(0, 4) != "PVS1" ||
        bytes.substr(4, 16) != hex64(hash64(bytes.substr(20))))
        throw std::runtime_error("Checkpoint checksum/version failed");
    std::string body = bytes.substr(20);
    Reader r(body);
    if (r.s() != ALGORITHM)
        throw std::runtime_error("Checkpoint algorithm mismatch");
    Config stored = r.config();
    Engine next(stored);
    next.pending_ = r.config();
    next.has_pending_ = r.u(1) != 0;
    next.active_ = r.u(1) != 0;
    next.last_sample_ = static_cast<int64_t>(r.u());
    next.current_ = r.row();
    next.unassigned_import_ = r.d();
    next.unassigned_export_ = r.d();
    size_t count = static_cast<size_t>(r.u(4));
    if (count > 5760)
        throw std::runtime_error("Oversized history");
    for (size_t i = 0; i < count; ++i) {
        Interval row = r.row();
        if (i && row.start <= next.history_.back().start)
            throw std::runtime_error("Unordered history");
        next.history_.push_back(row);
    }
    Detection &v = next.detection_;
    unsigned state = static_cast<unsigned>(r.u(1)), candidate = static_cast<unsigned>(r.u(1));
    if (state > AMBIGUOUS || candidate > AMBIGUOUS)
        throw std::runtime_error("Invalid detection state");
    v.state = static_cast<State>(state);
    v.candidate = static_cast<State>(candidate);
    v.streak = static_cast<unsigned>(r.u(1));
    v.valid_days = static_cast<unsigned>(r.u(1));
    v.export_days = static_cast<unsigned>(r.u(1));
    v.profile_days = static_cast<unsigned>(r.u(1));
    if (v.streak > 3 || v.valid_days > 30 || v.export_days > 30 || v.profile_days > 30)
        throw std::runtime_error("Invalid detection counts");
    v.export_fraction = r.d();
    v.valley_fraction = r.d();
    v.valley_median = r.d();
    v.night_export = r.u(1) != 0;
    v.evaluated_day = static_cast<int64_t>(r.u());
    Model &m = v.model;
    m.valid = r.u(1) != 0;
    unsigned reason = static_cast<unsigned>(r.u(1));
    if (reason > MODEL_INVALID)
        throw std::runtime_error("Invalid model reason");
    m.reason = static_cast<Reason>(reason);
    m.b0 = r.d();
    m.b1 = r.d();
    m.q_min = r.d();
    m.q_max = r.d();
    m.residual95 = r.d();
    m.r2 = r.d();
    m.nrmse = r.d();
    m.id = r.u();
    if (r.at != body.size())
        throw std::runtime_error("Trailing checkpoint bytes");
    if (stored.identity() != config_.identity())
        throw std::runtime_error("Checkpoint identity mismatch; warmup required");
    Config desired = config_;
    *this = next;
    if (desired.id() != config_.id())
        configure(desired);
}
void atomic_write(const std::string &path, const std::string &bytes) {
    if (bytes.size() > 1048576)
        throw std::runtime_error("Checkpoint size bound");
    std::string tmp = path + ".tmp";
    int fd = ::open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (fd < 0)
        throw std::runtime_error("Cannot create checkpoint");
    size_t done = 0;
    bool ok = true;
    while (done < bytes.size()) {
        ssize_t n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0) {
            ok = false;
            break;
        }
        done += static_cast<size_t>(n);
    }
    if (::fsync(fd) != 0)
        ok = false;
    if (::close(fd) != 0)
        ok = false;
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        throw std::runtime_error("Checkpoint commit failed");
    }
    std::string directory = path.substr(0, path.find_last_of('/'));
    int dir = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (dir >= 0) {
        ::fsync(dir);
        ::close(dir);
    }
}
std::string read_state_file(const std::string &path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
        throw std::runtime_error("Checkpoint absent");
    input.seekg(0, std::ios::end);
    auto size = input.tellg();
    if (size < 0 || size > 1048576)
        throw std::runtime_error("Checkpoint size invalid");
    input.seekg(0);
    std::string value(static_cast<size_t>(size), '\0');
    if (!value.empty())
        input.read(&value[0], static_cast<std::streamsize>(size));
    if (!input)
        throw std::runtime_error("Checkpoint read failed");
    return value;
}
} // namespace pv
