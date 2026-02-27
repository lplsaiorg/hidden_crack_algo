/**
 * run_test.cpp — multi-threaded efficiency test for YoloHiddenCrackDetector.
 *
 * Each thread owns one YoloHiddenCrackDetector instance.
 * All threads run WARMUP_RUNS warm-up iterations then TEST_RUNS timed
 * iterations on the same test image.  Per-thread latency stats and the
 * aggregate throughput (FPS) are printed at the end.
 *
 * Usage:
 *   ./run_test [model_xml] [image_path] [num_threads] [warmup] [runs]
 *
 * Defaults:
 *   model_xml   = hard-coded path below
 *   image_path  = hard-coded path below
 *   num_threads = 4
 *   warmup      = 10
 *   runs        = 100
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo.h"

// ---------------------------------------------------------------------------
// Defaults (override via argv)
// ---------------------------------------------------------------------------
static const std::string DEFAULT_MODEL =
    "openvino_int8_640/best.xml";
static const std::string DEFAULT_IMAGE =
    "test_imgs/001973.jpg";
static constexpr int   DEFAULT_THREADS = 2;
static constexpr int   DEFAULT_WARMUP  = 10;
static constexpr int   DEFAULT_RUNS    = 100;
static constexpr float CONF_THRESH     = 0.65f;
static constexpr float IOU_THRESH      = 0.8;

// ---------------------------------------------------------------------------
// Per-thread result
// ---------------------------------------------------------------------------
struct ThreadResult {
    int    thread_id;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p90_ms;
    double p99_ms;
    int    num_dets;   // detections on last run
    bool   ok{false};
    std::string error_msg;
};

// ---------------------------------------------------------------------------
// Worker function — one thread, one detector instance
// Image loading and preprocessing are done OUTSIDE the timed loop.
enum class Mode { INFER_ONLY, FAST_POST, FULL_POST };

// ---------------------------------------------------------------------------
static void worker(
    int               thread_id,
    const std::string& model_xml,
    const cv::Mat&    blob,
    int               orig_h,
    int               orig_w,
    int               warmup_runs,
    int               test_runs,
    Mode              mode,
    std::atomic<int>& ready_count,
    int               total_threads,
    ThreadResult&     result)
{
    result.thread_id = thread_id;
    try {
        YoloHiddenCrackDetector det;
        det.init(model_xml, "CPU");

        ++ready_count;
        while (ready_count.load() < total_threads)
            std::this_thread::yield();

        auto run_once = [&]() -> int {
            switch (mode) {
            case Mode::INFER_ONLY:
                det.detectFromBlob(blob, orig_h, orig_w, CONF_THRESH, IOU_THRESH, false);
                return -1;
            case Mode::FAST_POST:
                det.detectFromBlob(blob, orig_h, orig_w, CONF_THRESH, IOU_THRESH, false);
                return det.fast_postprocess(CONF_THRESH);
            case Mode::FULL_POST:
                return (int)det.detectFromBlob(blob, orig_h, orig_w,
                                               CONF_THRESH, IOU_THRESH, true).size();
            }
            return -1;
        };

        for (int i = 0; i < warmup_runs; ++i) run_once();

        std::vector<double> latencies;
        latencies.reserve(test_runs);
        int last_val = -1;
        for (int i = 0; i < test_runs; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            last_val = run_once();
            auto t1 = std::chrono::steady_clock::now();
            latencies.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::sort(latencies.begin(), latencies.end());
        double sum  = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / test_runs;
        double sq   = 0.0;
        for (double v : latencies) sq += (v - mean) * (v - mean);

        auto pct = [&](double p) {
            int idx = std::max(0, std::min(test_runs-1,
                      (int)std::ceil(p/100.0*test_runs)-1));
            return latencies[idx];
        };

        result.mean_ms  = mean;
        result.std_ms   = std::sqrt(sq / test_runs);
        result.min_ms   = latencies.front();
        result.max_ms   = latencies.back();
        result.p50_ms   = pct(50);
        result.p90_ms   = pct(90);
        result.p99_ms   = pct(99);
        result.num_dets = last_val;
        result.ok       = true;

    } catch (const std::exception& e) {
        result.ok        = false;
        result.error_msg = e.what();
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    std::string model_xml   = DEFAULT_MODEL;
    std::string image_path  = DEFAULT_IMAGE;
    int num_threads         = DEFAULT_THREADS;
    int warmup_runs         = DEFAULT_WARMUP;
    int test_runs           = DEFAULT_RUNS;

    if (argc > 1) model_xml    = argv[1];
    if (argc > 2) image_path   = argv[2];
    if (argc > 3) num_threads  = std::stoi(argv[3]);
    if (argc > 4) warmup_runs  = std::stoi(argv[4]);
    if (argc > 5) test_runs    = std::stoi(argv[5]);

    if (num_threads < 1) { std::fprintf(stderr, "num_threads must be >= 1\n"); return 1; }
    if (test_runs   < 1) { std::fprintf(stderr, "runs must be >= 1\n");        return 1; }

    // ---- load image and preprocess ONCE — excluded from all timing ----
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::fprintf(stderr, "[ERROR] Cannot read image: %s\n", image_path.c_str());
        return 1;
    }
    const int orig_h = image.rows;
    const int orig_w = image.cols;

    // Use a temporary detector only for preprocessing (no timing impact).
    YoloHiddenCrackDetector prep_det;
    prep_det.init(model_xml, "CPU");
    cv::Mat blob = prep_det.preprocess(image);   // NCHW float32, done once

    std::printf("=== YOLO Seg OpenVINO Efficiency Test ===\n");
    std::printf("Model      : %s\n", model_xml.c_str());
    std::printf("Image      : %s  (%d×%d)\n", image_path.c_str(), orig_w, orig_h);
    std::printf("Threads    : %d\n", num_threads);
    std::printf("Warmup runs: %d  |  Test runs: %d  (per thread)\n", warmup_runs, test_runs);
    std::printf("conf=%.2f  iou=%.2f\n", CONF_THRESH, IOU_THRESH);
    std::printf("(image loading and preprocessing excluded from timing)\n\n");
    std::fflush(stdout);

    // ---- helper: run one benchmark pass ----
    struct PassResult {
        std::vector<ThreadResult> rows;
        double wall_ms{0.0};
    };

    auto run_pass = [&](Mode mode) -> PassResult {
        PassResult pr;
        pr.rows.resize(num_threads);
        std::vector<std::thread> threads;
        std::atomic<int>         ready_count{0};

        auto wall_start = std::chrono::steady_clock::now();
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, t,
                std::cref(model_xml), std::cref(blob), orig_h, orig_w,
                warmup_runs, test_runs, mode,
                std::ref(ready_count), num_threads, std::ref(pr.rows[t]));
        }
        for (auto& th : threads) th.join();
        pr.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start).count();

        for (const auto& r : pr.rows)
            if (!r.ok)
                throw std::runtime_error("[Thread " + std::to_string(r.thread_id) +
                                         "] " + r.error_msg);
        return pr;
    };

    PassResult infer_pass, fast_pass, full_pass;
    try {
        infer_pass = run_pass(Mode::INFER_ONLY);
        fast_pass  = run_pass(Mode::FAST_POST);
        full_pass  = run_pass(Mode::FULL_POST);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    // ---- aggregate helper ----
    auto aggregate = [&](const PassResult& pr) {
        double sum = 0.0;
        for (const auto& r : pr.rows) sum += r.mean_ms;
        double mean     = sum / num_threads;
        long   total    = static_cast<long>(num_threads) * test_runs;
        return std::make_tuple(mean,
            1000.0 / mean,
            (1000.0 / mean) * num_threads,
            total / (pr.wall_ms / 1000.0));
    };

    auto [i_mean, i_fps1, i_tot, i_wall] = aggregate(infer_pass);
    auto [f_mean, f_fps1, f_tot, f_wall] = aggregate(fast_pass);
    auto [p_mean, p_fps1, p_tot, p_wall] = aggregate(full_pass);

    // ================================================================
    // Combined table  (three passes: INFER / FAST / FULL)
    // ================================================================
    const std::string sep(102, '-');

    std::printf("%-8s  %-5s  %8s %8s %8s %8s %8s %8s %8s  %6s  %8s %8s %8s\n",
                "Thread", "Mode",
                "mean_ms", "std_ms", "min_ms", "max_ms",
                "p50_ms", "p90_ms", "p99_ms", "result",
                "fps/thr", "fps_tot", "fps_wall");
    std::printf("%s\n", sep.c_str());

    auto print_rows = [&](const PassResult& pr, const char* tag,
                          double fps1, double fps_tot, double fps_wall_v) {
        for (int t = 0; t < num_threads; ++t) {
            const auto& r  = pr.rows[t];
            bool        last = (t == num_threads - 1);
            std::string res_str = (r.num_dets < 0) ? "-"
                                : std::to_string(r.num_dets);
            std::printf("%-8d  %-5s  %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  %6s  %8s %8s %8s\n",
                r.thread_id, tag,
                r.mean_ms, r.std_ms, r.min_ms, r.max_ms,
                r.p50_ms,  r.p90_ms, r.p99_ms,
                res_str.c_str(),
                last ? std::to_string((int)std::round(fps1)   ).c_str() : "",
                last ? std::to_string((int)std::round(fps_tot) ).c_str() : "",
                last ? std::to_string((int)std::round(fps_wall_v)).c_str() : "");
        }
    };

    print_rows(infer_pass, "INFER", i_fps1, i_tot, i_wall);
    std::printf("%s\n", sep.c_str());
    print_rows(fast_pass,  "FAST",  f_fps1, f_tot, f_wall);
    std::printf("%s\n", sep.c_str());
    print_rows(full_pass,  "FULL",  p_fps1, p_tot, p_wall);
    std::printf("%s\n", sep.c_str());

    std::printf("%-8s  %-5s  %8.2f %8s %8s %8s %8s %8s %8s  %8s  %8.1f %8.1f %8.1f\n",
                "AVG", "INFER", i_mean,"","","","","","","", i_fps1, i_tot, i_wall);
    std::printf("%-8s  %-5s  %8.2f %8s %8s %8s %8s %8s %8s  %8s  %8.1f %8.1f %8.1f\n",
                "AVG", "FAST",  f_mean,"","","","","","","", f_fps1, f_tot, f_wall);
    std::printf("%-8s  %-5s  %8.2f %8s %8s %8s %8s %8s %8s  %8s  %8.1f %8.1f %8.1f\n",
                "AVG", "FULL",  p_mean,"","","","","","","", p_fps1, p_tot, p_wall);
    std::printf("%-8s  %-5s  %8.2f  (vs INFER)\n",
                "COST", "FAST",  f_mean - i_mean);
    std::printf("%-8s  %-5s  %8.2f  (vs INFER)\n",
                "COST", "FULL",  p_mean - i_mean);

    return 0;
}
