/**
 * main.cpp — demonstrates every postprocess mode of YoloHiddenCrackDetector.
 *
 * Usage:
 *   ./openvino_yolo26_seg [model_xml] [image_path] [output_dir] [conf] [iou]
 *
 * Modes shown:
 *   Mode 1 — detect()          : one-call convenience (preprocess + infer + full decode)
 *   Mode 2 — fast_postprocess(): binary hit/miss gate after infer, no decode overhead
 *   Mode 3 — detectFromBlob()  : reuse a pre-built blob (preprocess once, infer many)
 *   Mode 4 — infer only        : detectFromBlob(..., do_postproc=false), raw speed test
 */

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo.h"

// ---- tiny timing helper ----
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---- print detection list ----
static void print_dets(const std::vector<Detection>& dets, float conf_thresh)
{
    std::printf("  → %zu detection(s)  [conf_thresh=%.2f]\n", dets.size(), conf_thresh);
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        std::printf("    [%zu] class=%d  conf=%.4f  bbox=(%.1f, %.1f, %.1f, %.1f)\n",
                    i, d.class_id, d.conf,
                    d.bbox.x, d.bbox.y,
                    d.bbox.x + d.bbox.width,
                    d.bbox.y + d.bbox.height);
    }
}

int main(int argc, char* argv[])
{
    std::string model_xml  = "openvino_int8/best.xml";
    std::string image_path = "test_imgs/001973.jpg";
    std::string out_dir    = "./result";
    float       conf       = 0.75f;
    float       iou        = 0.45f;

    if (argc > 1) model_xml  = argv[1];
    if (argc > 2) image_path = argv[2];
    if (argc > 3) out_dir    = argv[3];
    if (argc > 4) conf       = std::stof(argv[4]);
    if (argc > 5) iou        = std::stof(argv[5]);

    // ------------------------------------------------------------------
    // Load image
    // ------------------------------------------------------------------
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::fprintf(stderr, "[ERROR] Cannot read image: %s\n", image_path.c_str());
        return 1;
    }
    const int orig_h = image.rows, orig_w = image.cols;
    std::printf("Image : %s  (%d×%d)\n", image_path.c_str(), orig_w, orig_h);

    // ------------------------------------------------------------------
    // Init detector
    // ------------------------------------------------------------------
    YoloHiddenCrackDetector det;
    try {
        det.init(model_xml, "CPU");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] init: %s\n", e.what());
        return 1;
    }
    std::printf("Model : %s  [OK]\n\n", model_xml.c_str());

    // ==================================================================
    // Mode 1 — detect()
    //   Convenience path: preprocess + infer + NMS + mask decode in one call.
    //   Use this when simplicity matters more than latency breakdown.
    // ==================================================================
    std::printf("=== Mode 1: detect()  [preprocess + infer + full decode] ===\n");
    {
        auto t0 = Clock::now();
        auto dets = det.detect(image, conf, iou);
        std::printf("  elapsed: %.2f ms\n", ms_since(t0));
        print_dets(dets, conf);

        cv::Mat vis = det.visualize(image, dets);
        std::string out = out_dir + "/mode1_detect.jpg";
        cv::imwrite(out, vis);
        std::printf("  saved  → %s\n", out.c_str());
    }

    // ==================================================================
    // Mode 2 — fast_postprocess()
    //   Preprocess once, infer, then call fast_postprocess() for a cheap
    //   binary "hit / no-hit" answer.  No NMS, no mask, no coordinate work.
    //   Ideal as a gate: run the full decode only when hit == 1.
    // ==================================================================
    std::printf("\n=== Mode 2: fast_postprocess()  [infer + binary hit/miss] ===\n");
    {
        // Step A: preprocess once (reused across modes 2/3/4)
        cv::Mat blob = det.preprocess(image);

        // Step B: infer (no postprocess)
        auto t0 = Clock::now();
        det.detectFromBlob(blob, orig_h, orig_w, conf, iou, /*do_postproc=*/false);
        double infer_ms = ms_since(t0);

        // Step C: binary check — nearly free
        auto t1 = Clock::now();
        int  hit = det.fast_postprocess(conf);
        double fast_ms = ms_since(t1);

        std::printf("  infer: %.2f ms  |  fast_postprocess: %.3f ms  →  hit=%d\n",
                    infer_ms, fast_ms, hit);

        if (hit) {
            // Step D (optional): full decode only when something was found
            auto t2    = Clock::now();
            auto dets  = det.detectFromBlob(blob, orig_h, orig_w, conf, iou, true);
            std::printf("  full decode (on hit): %.2f ms\n", ms_since(t2));
            print_dets(dets, conf);

            cv::Mat vis = det.visualize(image, dets);
            std::string out = out_dir + "/mode2_fast_gate.jpg";
            cv::imwrite(out, vis);
            std::printf("  saved  → %s\n", out.c_str());
        } else {
            std::printf("  no detection — full decode skipped.\n");
        }
    }

    // ==================================================================
    // Mode 3 — detectFromBlob()
    //   Preprocess the image once, then call detectFromBlob() repeatedly
    //   (e.g. retrying with different thresholds, or benchmarking).
    //   Preprocessing cost is paid only once.
    // ==================================================================
    std::printf("\n=== Mode 3: detectFromBlob()  [reuse blob, vary thresholds] ===\n");
    {
        cv::Mat blob = det.preprocess(image);

        for (float c : {0.25f, 0.50f, 0.70f}) {
            auto t0  = Clock::now();
            auto dets = det.detectFromBlob(blob, orig_h, orig_w, c, iou);
            std::printf("  conf=%.2f  elapsed=%.2f ms", c, ms_since(t0));
            std::printf("  → %zu det(s)\n", dets.size());
        }

        // Save result at default threshold
        auto dets = det.detectFromBlob(blob, orig_h, orig_w, conf, iou);
        cv::Mat vis = det.visualize(image, dets);
        std::string out = out_dir + "/mode3_from_blob.jpg";
        cv::imwrite(out, vis);
        std::printf("  saved  → %s\n", out.c_str());
    }

    // ==================================================================
    // Mode 4 — infer only  (do_postproc = false)
    //   Runs preprocess + infer but skips all decode work.
    //   Use for measuring raw model throughput or when you only need
    //   to feed the model without caring about results yet.
    // ==================================================================
    std::printf("\n=== Mode 4: infer only  [do_postproc=false] ===\n");
    {
        cv::Mat blob = det.preprocess(image);

        auto t0 = Clock::now();
        auto dets = det.detectFromBlob(blob, orig_h, orig_w, conf, iou,
                                       /*do_postproc=*/false);
        std::printf("  elapsed: %.2f ms  |  returned %zu dets (always 0)\n",
                    ms_since(t0), dets.size());
        std::printf("  (use fast_postprocess() or re-call with do_postproc=true to decode)\n");
    }

    std::printf("\nDone.\n");
    return 0;
}
