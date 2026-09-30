#pragma once
#ifndef DI_EMULATOR_REPLAY
#error Meter Lab transport must never enter a production build
#endif
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <stdint.h>
#include <mutex>
#include <set>
#include <atomic>
#include <chrono>
#include <thread>
#include <unistd.h>
namespace lab {
inline std::string directory() {
    const char *p = std::getenv("METER_LAB_DIR");
    if (!p) p = std::getenv("METER_LAB_METROLOGY_DIR");
    return p ? p : "";
}
inline void submitted(uint32_t feature, const std::string &payload, bool event, int result) {
    if (directory().empty()) return;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char c : payload) { hash ^= c; hash *= UINT64_C(1099511628211); }
    std::ostringstream key; key << std::hex << hash << (event ? "e" : "d");
    static std::mutex lock; std::lock_guard<std::mutex> guard(lock);
    std::ofstream out((directory() + "/submissions.jsonl").c_str(), std::ios::app);
    out << "{\"key\":\"" << key.str() << "\",\"feature\":" << feature << ",\"return_code\":" << result << "}\n";
}
inline size_t pending_output() {
    std::set<std::string> pending;
    for (unsigned pass=0;pass<2;++pass) {
        std::ifstream in((directory() + (pass ? "/sdk-completions.jsonl" : "/submissions.jsonl")).c_str());
        std::string line;
        while(std::getline(in,line)) {
            if(!pass && line.find("\"return_code\":0}")==std::string::npos) continue;
            const size_t begin=line.find("\"key\":\"");
            if(begin==std::string::npos) continue;
            const size_t end=line.find('"',begin+7);
            if(end==std::string::npos) continue;
            const std::string key=line.substr(begin+7,end-begin-7);
            if(pass) pending.erase(key); else pending.insert(key);
        }
    }
    return pending.size();
}
inline void wait_output(size_t maximum, std::atomic<bool> *cancel) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    while(pending_output()>maximum) {
        if(cancel && cancel->load()) return;
        if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("SDK delivery acknowledgement timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
inline void atomic(const std::string &path, const std::string &value) {
    std::ofstream out((path + ".tmp").c_str()); out << value; out.close();
    if (!out || std::rename((path + ".tmp").c_str(), path.c_str())) throw std::runtime_error("Lab atomic write failed");
}
struct Frame { std::string run, kind; unsigned sequence = 0; std::vector<std::vector<double>> rows; };
inline bool read(Frame &frame, unsigned expected, unsigned width) {
    std::ifstream input((directory() + "/frame.ready").c_str()); if (!input) return false;
    std::string version; unsigned count = 0;
    input >> version >> frame.run >> frame.sequence >> frame.kind >> count;
    if (version != "ML1" || frame.run.size() != 32 || frame.run.find_first_not_of("0123456789abcdef") != std::string::npos ||
        frame.sequence != expected || count > 1024 || (frame.kind != "DATA" && frame.kind != "END" && !(width == 9 && frame.kind == "CONST")))
        throw std::runtime_error("Invalid ML1 frame header/sequence");
    if (frame.kind == "END" && count) throw std::runtime_error("Nonempty END frame");
    std::ifstream identity((directory() + "/identity").c_str()); std::string run; identity >> run;
    if (!identity || run != frame.run) throw std::runtime_error("ML1 run identity mismatch");
    frame.rows.clear();
    for (unsigned i = 0; i < count; ++i) {
        std::vector<double> row;
        for (unsigned j = 0; j < width + (frame.kind == "CONST" ? 1 : 0); ++j) {
            std::string token; input >> token; char *end = 0;
            double value = std::strtod(token.c_str(), &end);
            if (!input || end == token.c_str() || *end) throw std::runtime_error("Invalid ML1 numeric field");
            row.push_back(value);
        }
        frame.rows.push_back(row);
    }
    std::string extra; if (input >> extra) throw std::runtime_error("Extra ML1 data");
    return true;
}
inline void acknowledge(const Frame &f, size_t received, size_t processed, double clock, const std::string &extra = "") {
    std::ostringstream out; out << std::setprecision(17) << "{\"version\":1,\"run_id\":\"" << f.run
        << "\",\"sequence\":" << f.sequence << ",\"received\":" << received << ",\"processed\":" << processed
        << ",\"scenario_time\":" << clock << ",\"complete\":" << (f.kind == "END" ? "true" : "false") << extra << '}';
    // Consumer owns ready removal; producer waits for this sequence's acknowledgement.
    std::remove((directory() + "/frame.ready").c_str());
    atomic(directory() + "/ack.json", out.str());
    std::ofstream history((directory() + "/acknowledgements.jsonl").c_str(), std::ios::app);
    history << out.str() << '\n';
}
inline void error(const std::string &message) {
    std::string safe; for (char c : message) safe += (c == '"' || c == '\\' || c < 32) ? ' ' : c;
    atomic(directory() + "/error.json", "{\"error\":\"" + safe + "\"}");
}
} // namespace lab
