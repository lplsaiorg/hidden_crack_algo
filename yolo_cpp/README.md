# YoloHiddenCrackDetector — C++ / OpenVINO

Real-time hidden-crack segmentation inference using an end2end-exported YOLO26-seg model running on Intel OpenVINO.

---

## Project layout

```
cpp/
├── CMakeLists.txt          # build system
├── include/
│   └── yolo.h              # YoloHiddenCrackDetector class + Detection struct
├── src/
│   ├── yolo.cpp            # detector implementation
│   ├── main.cpp            # single-image demo (all postprocess modes)
│   └── run_test.cpp        # multi-threaded efficiency benchmark
└── build/                  # cmake output (git-ignored)
```

---

## Dependencies

| Library | Tested version | Notes |
|---|---|---|
| CMake | ≥ 3.16 | |
| GCC / G++ | 13 (C++17) | needs `_GLIBCXX_USE_CXX11_ABI=1` |
| OpenCV | 4.6 | modules: core imgproc imgcodecs highgui |
| Intel OpenVINO Toolkit | 2025.3.0 | C++ Runtime only |

### OpenVINO toolkit path

The toolkit is expected at:

```
openvino_libs/
└── openvino_toolkit_ubuntu22_2025.3.0.19807.44526285f24_x86_64/
    └── runtime/
        ├── cmake/      ← CMake config files
        ├── include/    ← headers
        └── lib/intel64/← shared libraries (.so)
```

To use a different toolkit location, either:

- Edit `OV_TOOLKIT` in `CMakeLists.txt`, **or**
- Pass it on the cmake command line:
  ```bash
  cmake .. -DOV_TOOLKIT=/path/to/your/toolkit
  ```

---

## Build

```bash
# 1. create and enter the build directory
mkdir -p cpp/build && cd cpp/build

# 2. configure
cmake ..

# 3. compile both targets (parallel)
make -j$(nproc)
```

This produces two executables inside `cpp/build/`:

| Executable | Source | Purpose |
|---|---|---|
| `openvino_yolo26_seg` | `main.cpp` | single-image demo |
| `run_test` | `run_test.cpp` | multi-threaded benchmark |

---

## Running the demo (`openvino_yolo26_seg`)

```bash
./openvino_yolo26_seg [model_xml] [image_path] [output_dir] [conf] [iou]
```

| Argument | Default | Description |
|---|---|---|
| `model_xml` | hard-coded path | path to the exported `best.xml` |
| `image_path` | hard-coded path | input BGR image |
| `output_dir` | `/tmp` | directory for saved result images |
| `conf` | `0.75` | confidence threshold |
| `iou` | `0.45` | NMS IoU threshold |

The demo runs through **four postprocess modes** and saves a visualised result for each:

### Mode 1 — `detect()`
One-call convenience API. Includes preprocessing, inference, NMS, and mask generation.
```cpp
auto dets = det.detect(image, conf, iou);
```
Use when simplicity matters more than latency control.

### Mode 2 — `fast_postprocess()` gate
Cheapest possible check: scan the raw confidence column after inference and return `1` (hit) or `0` (miss) with early exit. Costs < 0.1 ms.
```cpp
det.detectFromBlob(blob, h, w, conf, iou, /*do_postproc=*/false);  // infer
int hit = det.fast_postprocess(conf);                               // ~0.01 ms
if (hit)
    auto dets = det.detectFromBlob(blob, h, w, conf, iou, true);   // full decode
```
Best pattern when most frames are expected to be negative.

### Mode 3 — `detectFromBlob()`
Preprocess once (`det.preprocess(image)`), then reuse the resulting blob to run inference with different thresholds or multiple times.
```cpp
cv::Mat blob = det.preprocess(image);
auto dets_low  = det.detectFromBlob(blob, h, w, 0.25f, iou);
auto dets_high = det.detectFromBlob(blob, h, w, 0.70f, iou);
```

### Mode 4 — infer only
Skip all decode work (`do_postproc = false`). Returns an empty vector. Use to measure raw model throughput.
```cpp
det.detectFromBlob(blob, h, w, conf, iou, /*do_postproc=*/false);
```

---

## Running the benchmark (`run_test`)

```bash
./run_test [model_xml] [image_path] [num_threads] [warmup_runs] [test_runs]
```

| Argument | Default | Description |
|---|---|---|
| `model_xml` | hard-coded | model XML path |
| `image_path` | hard-coded | test image |
| `num_threads` | `2` | number of parallel detector instances |
| `warmup_runs` | `10` | warm-up iterations per thread (excluded from stats) |
| `test_runs` | `100` | timed iterations per thread |

Image loading and preprocessing are **excluded** from all timing. The benchmark runs three passes automatically and prints a combined table:

```
Thread  Mode    mean_ms  std_ms  min_ms  max_ms  p50_ms  p90_ms  p99_ms  result  fps/thr  fps_tot  fps_wall
------------------------------------------------------------------------------------------------------
0       INFER    91.5    ...
1       INFER    91.4    ...                                                      11       22       19
------------------------------------------------------------------------------------------------------
0       FAST     95.1    ...
1       FAST     94.8    ...                                                      11       21       18
------------------------------------------------------------------------------------------------------
0       FULL    100.8    ...
1       FULL    100.9    ...                                                      10       20       17
------------------------------------------------------------------------------------------------------
AVG     INFER    91.4    ...                                                      10.9     21.9     19.0
AVG     FAST     95.1    ...                                                      10.5     21.0     18.3
AVG     FULL    100.9    ...                                                       9.9     19.8     17.3
COST    FAST      3.7    (vs INFER)
COST    FULL      9.4    (vs INFER)
```

| Mode | Measures |
|---|---|
| `INFER` | preprocess (pre-done) + forward pass only |
| `FAST` | forward pass + `fast_postprocess()` binary check |
| `FULL` | forward pass + NMS + mask generation |

`result` column: `-` = not applicable, `0` = no detection, `1` = detection found.

---

## Postprocess pipeline internals

```
preprocess()
  BGR → RGB → bilinear resize to (input_w × input_h) → /255 → NCHW float32

infer()
  OV InferRequest on model output[0] [1, 300, 38] and output[1] [1, 32, 320, 320]

postprocess()  (full)
  1. Confidence filter         — drop rows below conf_thresh
  2. NMS on bboxes only        — before mask generation (skips suppressed boxes)
  3. Per kept detection:
     a. cv::gemm               — [1,32] × [32,320×320] linear combination
     b. cv::exp vectorised     — sigmoid activation
     c. Crop proto mask to bbox region in proto coords (320×320)
     d. cv::resize bbox crop   — resize ONLY the bbox area (not the full image)
     e. Threshold at 0.5       — paste binary mask into full-image canvas

fast_postprocess()
  Scan conf column of output[0], return 1 on first hit, else 0.  No other work.
```

---

## Compile-time flags

| Flag | Value | Reason |
|---|---|---|
| `-O3 -march=native` | both targets | AVX2 autovectorisation for gemm/exp inner loops |
| `_GLIBCXX_USE_CXX11_ABI=1` | global | matches system OpenCV's `std::string` ABI |
| `BUILD_RPATH` | toolkit `lib/intel64` | find OpenVINO `.so` at runtime without `LD_LIBRARY_PATH` |
