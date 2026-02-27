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

---

## Path configuration

There are **three places** where paths must be set before building and running.

### 1. OpenVINO toolkit root — `CMakeLists.txt`

```cmake
# CMakeLists.txt  line 16
set(OV_TOOLKIT "openvino_libs/openvino_toolkit_ubuntu22_2025.3.0.19807.44526285f24_x86_64")
```

`OV_TOOLKIT` must point to the versioned toolkit directory that contains the `runtime/` subtree:

```
<OV_TOOLKIT>/
└── runtime/
    ├── cmake/       ← OpenVINO CMake config (OpenVINOConfig.cmake)
    ├── include/     ← C++ headers  (openvino/openvino.hpp …)
    └── lib/intel64/ ← shared libraries (libopenvino.so.2025.3.0 …)
```

**Option A — edit `CMakeLists.txt` directly** (recommended for a fixed install):

```cmake
set(OV_TOOLKIT "/absolute/path/to/your/toolkit")
```

**Option B — pass at cmake configure time** (no file edits needed):

```bash
cmake .. -DOV_TOOLKIT=/absolute/path/to/your/toolkit
```

**Option C — conda / pip OpenVINO** (if you installed via `pip install openvino`):

```bash
# find the cmake config shipped with the Python package
python -c "import openvino; print(openvino.__file__)"
# typically: /home/<user>/anaconda3/envs/<env>/lib/python3.x/site-packages/openvino/__init__.py

cmake .. -DOpenVINO_DIR=/home/<user>/anaconda3/envs/<env>/lib/python3.x/site-packages/openvino/cmake
```

> **Note:** the conda/pip package is built with the **old** `_GLIBCXX_USE_CXX11_ABI=0` ABI.
> If you see linker errors like `undefined reference to cv::imread(std::string const&, int)`
> you must switch to the standalone toolkit (Option A/B), which uses the new ABI matching system OpenCV.

---

### 2. Default model & image paths — `src/main.cpp` and `src/run_test.cpp`

Both executables have built-in fallback paths used when no command-line arguments are given.
Edit the constants near the top of each file:

**`src/main.cpp`** (lines 46–47):

```cpp
std::string model_xml  = "openvino_int8/best.xml";   // ← path to best.xml
std::string image_path = "test_imgs/001973.jpg";      // ← test image
```

**`src/run_test.cpp`** (lines 37–45):

```cpp
static const std::string DEFAULT_MODEL = "openvino_int8/best.xml";
static const std::string DEFAULT_IMAGE = "test_imgs/001973.jpg";

static constexpr int   DEFAULT_THREADS = 2;      // parallel detector instances
static constexpr int   DEFAULT_WARMUP  = 10;     // warm-up runs (excluded from stats)
static constexpr int   DEFAULT_RUNS    = 100;    // timed runs per thread
static constexpr float CONF_THRESH     = 0.65f;  // confidence threshold
static constexpr float IOU_THRESH      = 0.80f;  // NMS IoU threshold
```

All of these can also be overridden at **runtime** via positional command-line arguments
(see the *Running* sections below) without recompiling.

---

### 3. Quick path-check before building

```bash
# verify toolkit structure
ls <OV_TOOLKIT>/runtime/cmake/OpenVINOConfig.cmake   # must exist
ls <OV_TOOLKIT>/runtime/lib/intel64/libopenvino.so*  # must exist

# verify model export
ls openvino_int8/best.xml
ls openvino_int8/best.bin
```

---

## Build

```bash
# 1. create and enter the build directory
mkdir -p build && cd build

# 2. configure  (OV_TOOLKIT already set in CMakeLists.txt, or pass -D override)
cmake ..

# 2b. configure with an explicit toolkit path
cmake .. -DOV_TOOLKIT=/absolute/path/to/your/toolkit

# 3. compile both targets (parallel)
make -j$(nproc)
```

This produces two executables inside `build/`:

| Executable | Source | Purpose |
|---|---|---|
| `openvino_yolo26_seg` | `main.cpp` | single-image demo |
| `run_test` | `run_test.cpp` | multi-threaded benchmark |

---

## Running the demo (`openvino_yolo26_seg`)

```bash
# use built-in default paths (set in main.cpp)
./openvino_yolo26_seg

# override all arguments
./openvino_yolo26_seg <model_xml> <image_path> <output_dir> <conf> <iou>

# example
./openvino_yolo26_seg \
    openvino_int8/best.xml \
    test_imgs/001973.jpg \
    /tmp/results \
    0.75 \
    0.45
```

| Position | Argument | Default (in `main.cpp`) | Description |
|---|---|---|---|
| 1 | `model_xml` | `openvino_int8/best.xml` | path to the exported `best.xml` |
| 2 | `image_path` | `test_imgs/001973.jpg` | input BGR image |
| 3 | `output_dir` | `/tmp` | directory for saved result images |
| 4 | `conf` | `0.75` | confidence threshold |
| 5 | `iou` | `0.45` | NMS IoU threshold |

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
# use built-in defaults (set in run_test.cpp)
./run_test

# override all arguments
./run_test <model_xml> <image_path> <num_threads> <warmup_runs> <test_runs>

# example: 4 threads, 20 warm-up, 200 timed runs
./run_test \
    openvino_int8/best.xml \
    test_imgs/001973.jpg \
    4 20 200
```

| Position | Argument | Default (in `run_test.cpp`) | Description |
|---|---|---|---|
| 1 | `model_xml` | `openvino_int8/best.xml` | model XML path |
| 2 | `image_path` | `test_imgs/001973.jpg` | test image |
| 3 | `num_threads` | `2` | parallel detector instances (one model per thread) |
| 4 | `warmup_runs` | `10` | warm-up iterations per thread (not timed) |
| 5 | `test_runs` | `100` | timed iterations per thread |

> `CONF_THRESH` (0.65) and `IOU_THRESH` (0.80) are compile-time constants in
> `run_test.cpp` — edit and recompile to change them.

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
