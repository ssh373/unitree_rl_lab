// Lightweight, lock-free-ish CSV telemetry logger for sim2sim / sim2real debugging.
//
// Usage: set the env var RL_LOG_CSV=/path/to/run.csv before starting the controller.
// Rows are buffered in RAM at the FSM rate (1 kHz) and flushed to disk, so the control
// loop never touches the filesystem.
//
//   RL_LOG_CSV      output path; unset disables the logger entirely
//   RL_LOG_SECONDS  stop recording and flush after this many seconds (default: unlimited)
//
// The file is written when the state exits, when RL_LOG_SECONDS elapses, or on SIGINT,
// so a run always produces a file even when it is stopped with Ctrl+C.
#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <spdlog/spdlog.h>

class TelemetryLogger
{
public:
    // 1 kHz * 600 s. ~117 floats/row * 600k rows would be too much, so cap at 300 s.
    static constexpr size_t MAX_ROWS = 300 * 1000;

    void open(const std::vector<std::string> & columns)
    {
        const char * path = std::getenv("RL_LOG_CSV");
        if (path == nullptr) return;

        path_ = path;
        columns_ = columns;
        buffer_.clear();
        buffer_.reserve(MAX_ROWS * columns_.size());
        rows_ = 0;
        const char * secs = std::getenv("RL_LOG_SECONDS");
        max_seconds_ = secs ? std::atof(secs) : 0.0;
        t0_ = std::chrono::steady_clock::now();
        enabled_ = true;
        install_signal_handler();
        active_ = this;
        spdlog::info("TelemetryLogger: recording {} columns to {}{}", columns_.size(), path_,
                     max_seconds_ > 0 ? fmt::format(" for {:g}s", max_seconds_) : "");
    }

    bool enabled() const { return enabled_; }

    // `values` must have columns_.size() - 1 entries (time is prepended).
    void append(const std::vector<float> & values)
    {
        if (!enabled_ || rows_ >= MAX_ROWS) return;
        const auto now = std::chrono::steady_clock::now();
        const float t = std::chrono::duration<float>(now - t0_).count();
        if (max_seconds_ > 0.0 && t > max_seconds_) {
            // Hand the file write to a detached thread: the caller is the 1 kHz control loop.
            enabled_ = false;
            std::thread([this]{ this->write_file(); }).detach();
            return;
        }
        buffer_.push_back(t);
        buffer_.insert(buffer_.end(), values.begin(), values.end());
        rows_++;
    }

    void close()
    {
        if (!enabled_) return;
        enabled_ = false;
        write_file();
    }

private:
    void write_file()
    {
        if (rows_ == 0) return;
        FILE * f = std::fopen(path_.c_str(), "w");
        if (f == nullptr) {
            spdlog::error("TelemetryLogger: cannot write {}", path_);
            return;
        }
        for (size_t c = 0; c < columns_.size(); ++c) {
            std::fprintf(f, "%s%s", columns_[c].c_str(), c + 1 == columns_.size() ? "\n" : ",");
        }
        const size_t ncol = columns_.size();
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < ncol; ++c) {
                std::fprintf(f, "%.6g%s", buffer_[r * ncol + c], c + 1 == ncol ? "\n" : ",");
            }
        }
        std::fclose(f);
        spdlog::info("TelemetryLogger: wrote {} rows to {}", rows_, path_);

        buffer_.clear();
        buffer_.shrink_to_fit();
        rows_ = 0;
        if (active_ == this) active_ = nullptr;
    }

    static void install_signal_handler()
    {
        static bool installed = false;
        if (installed) return;
        installed = true;
        std::signal(SIGINT, [](int sig){
            if (active_) { active_->enabled_ = false; active_->write_file(); }
            std::signal(sig, SIG_DFL);
            std::raise(sig);
        });
    }

    static inline TelemetryLogger * active_ = nullptr;

    std::atomic<bool> enabled_{false};
    double max_seconds_ = 0.0;
    std::string path_;
    std::vector<std::string> columns_;
    std::vector<float> buffer_;
    std::atomic<size_t> rows_{0};
    std::chrono::steady_clock::time_point t0_;
};
