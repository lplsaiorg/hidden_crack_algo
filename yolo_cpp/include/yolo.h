#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

/**
 * @brief Single instance-segmentation detection result.
 *
 * bbox coordinates are in the original (unscaled) image space.
 * mask is a binary (0/1) CV_8UC1 Mat with the same (h, w) as the original image.
 */
struct Detection {
    cv::Rect2f  bbox;       ///< x, y, w, h in original image coords
    float       conf{0.f};  ///< objectness / class confidence
    int         class_id{0};
    cv::Mat     mask;       ///< CV_8UC1, 0 or 1, same size as original image
};

/**
 * @brief YOLO-seg OpenVINO inference wrapper.
 *
 * Supports end2end-exported YOLO segmentation models (e.g. YOLO26-seg).
 * Output layout expected from the model:
 *   output[0]  [1, max_det, 4+1+nc+mask_dim]  — detections (x1y1x2y2, conf, class_id, coeffs)
 *   output[1]  [1, mask_dim, proto_h, proto_w] — prototype masks
 *
 * Preprocessing matches tools/infer_openvino.py exactly:
 *   BGR → RGB → bilinear resize to (input_w, input_h) → /255 → NCHW float32
 */
class YoloHiddenCrackDetector {
public:
    YoloHiddenCrackDetector() = default;
    ~YoloHiddenCrackDetector() = default;

    /**
     * @brief Load and compile the OpenVINO model.
     * @param model_xml  Path to the .xml file (the .bin must sit next to it).
     * @param device     OpenVINO device string, e.g. "CPU", "GPU".
     */
    void init(const std::string& model_xml, const std::string& device = "CPU");

    /**
     * @brief Run inference on one BGR image (includes preprocessing).
     * @param image        Input image in BGR format (any size).
     * @param conf_thresh  Minimum confidence threshold.
     * @param iou_thresh   IoU threshold for NMS.
     * @param do_postproc  If false, skip decoding/NMS and return empty.
     */
    std::vector<Detection> detect(const cv::Mat& image,
                                  float conf_thresh  = 0.25f,
                                  float iou_thresh   = 0.45f,
                                  bool  do_postproc  = true);

    /**
     * @brief Preprocess a BGR image into a flat float32 NCHW blob.
     *        Call this once outside the timing loop, then pass the result to
     *        detectFromBlob() to exclude preprocessing from latency measurement.
     * @param image  BGR input image (any size).
     * @return       CV_32F Mat, size [1, 3*input_h*input_w], ready for inference.
     *               Also stores orig_h/orig_w inside the returned blob as
     *               extra metadata via the 2-element header at blob.ptr<int>()
     *               — use makeBlob() instead to get the structured type below.
     */
    cv::Mat preprocess(const cv::Mat& image);

    /**
     * @brief Run inference on an already-preprocessed blob (skips preprocessing).
     * @param blob        Output of preprocess() — flat CV_32F NCHW buffer.
     * @param orig_h      Original image height (needed for bbox scaling).
     * @param orig_w      Original image width.
     * @param conf_thresh Minimum confidence threshold.
     * @param iou_thresh  IoU threshold for NMS.
     * @param do_postproc If false, skip decoding/NMS and return empty.
     */
    std::vector<Detection> detectFromBlob(const cv::Mat& blob,
                                          int   orig_h,
                                          int   orig_w,
                                          float conf_thresh  = 0.25f,
                                          float iou_thresh   = 0.45f,
                                          bool  do_postproc  = true);

    /**
     * @brief Fast binary check: returns 1 if any detection confidence >= conf_thresh,
     *        0 otherwise.  Skips NMS, mask generation, and all coordinate math —
     *        just scans the raw score column and early-exits on the first hit.
     *        Useful as a lightweight "is there anything here?" gate before running
     *        the full pipeline.
     * @param conf_thresh  Detection confidence threshold.
     * @return 1 if at least one box exceeds the threshold, 0 otherwise.
     */
    int fast_postprocess(float conf_thresh = 0.25f);

    /**
     * @brief Draw boxes and semi-transparent masks on a copy of the image.
     */
    cv::Mat visualize(const cv::Mat& image, const std::vector<Detection>& detections);

private:
    ov::Core            core_;
    ov::CompiledModel   compiled_model_;
    ov::InferRequest    infer_request_;

    // Model geometry (resolved at init time)
    int input_h_  {1280};
    int input_w_  {1280};
    int proto_h_  {320};
    int proto_w_  {320};
    int mask_dim_ {32};
    int max_det_  {300};

    /**
     * @brief Decode raw model outputs into Detection structs (with NMS).
     */
    std::vector<Detection> postprocess(
        const float* det_ptr,    ///< output[0] data, shape [max_det, stride]
        int          det_stride, ///< number of floats per detection row
        const float* proto_ptr,  ///< output[1] data, shape [mask_dim, proto_h, proto_w]
        int          orig_h,
        int          orig_w,
        float        conf_thresh,
        float        iou_thresh
    );

    /** @brief Greedy IoU-based NMS. Returns indices of kept detections. */
    static std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh);

    /** @brief Shared infer + optional decode path used by detect() and detectFromBlob(). */
    std::vector<Detection> infer_and_decode(cv::Mat& blob, int orig_h, int orig_w,
                                            float conf_thresh, float iou_thresh,
                                            bool  do_postproc);
};
