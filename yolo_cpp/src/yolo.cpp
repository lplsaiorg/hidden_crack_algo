#include "yolo.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
void YoloHiddenCrackDetector::init(const std::string& model_xml, const std::string& device)
{
    std::string bin_path = model_xml.substr(0, model_xml.rfind('.')) + ".bin";

    auto ov_model = core_.read_model(model_xml, bin_path);
    compiled_model_ = core_.compile_model(ov_model, device);
    infer_request_  = compiled_model_.create_infer_request();

    // ---- resolve input shape ----
    auto in_shape = compiled_model_.input(0).get_shape();
    // expected [1, 3, H, W]
    if (in_shape.size() == 4) {
        input_h_ = static_cast<int>(in_shape[2]);
        input_w_ = static_cast<int>(in_shape[3]);
    }

    // ---- resolve output[0] shape: [1, max_det, stride] ----
    auto out0_shape = compiled_model_.output(0).get_shape();
    if (out0_shape.size() == 3) {
        max_det_    = static_cast<int>(out0_shape[1]);
        int stride  = static_cast<int>(out0_shape[2]);
        // stride = 4 + 1 + nc + mask_dim; nc=1 → mask_dim = stride - 6
        mask_dim_ = stride - 6;
    }

    // ---- resolve output[1] shape: [1, mask_dim, proto_h, proto_w] ----
    auto out1_shape = compiled_model_.output(1).get_shape();
    if (out1_shape.size() == 4) {
        proto_h_ = static_cast<int>(out1_shape[2]);
        proto_w_ = static_cast<int>(out1_shape[3]);
    }
}

// ---------------------------------------------------------------------------
// preprocess  — identical to Python preprocess()
// ---------------------------------------------------------------------------
cv::Mat YoloHiddenCrackDetector::preprocess(const cv::Mat& image)
{
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);

    // HWC uint8 → float32 /255, then lay out as NCHW into a flat buffer
    // We keep the result as a 2-D Mat [3, H*W] so we can hand it straight to
    // the OV tensor, but we return the full [1,3,H,W] blob as CV_32F.
    cv::Mat blob;
    resized.convertTo(blob, CV_32F, 1.0 / 255.0);   // still HWC

    // HWC → NCHW
    std::vector<cv::Mat> chans(3);
    cv::split(blob, chans);
    cv::Mat nchw(1, 3 * input_h_ * input_w_, CV_32F);
    float* dst = nchw.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        const float* src = chans[c].ptr<float>();
        std::copy(src, src + input_h_ * input_w_, dst + c * input_h_ * input_w_);
    }
    return nchw;   // [1, 3*H*W] flat, reinterpreted as NCHW by OV
}

// ---------------------------------------------------------------------------
// RawDet  — lightweight bbox+meta struct used before mask generation
// ---------------------------------------------------------------------------
struct RawDet {
    float        x1, y1, x2, y2; // in model input (1280×1280) coords
    float        conf;
    int          class_id;
    const float* coeffs;          // pointer into OV output buffer
};

// ---------------------------------------------------------------------------
// nms  — greedy NMS operating on RawDet (no mask needed yet)
// ---------------------------------------------------------------------------
std::vector<int> YoloHiddenCrackDetector::nms(const std::vector<Detection>& dets, float iou_thresh)
{
    // kept for API compatibility — delegates to nms_raw internally
    std::vector<RawDet> raw;
    raw.reserve(dets.size());
    for (const auto& d : dets)
        raw.push_back({d.bbox.x, d.bbox.y,
                       d.bbox.x + d.bbox.width, d.bbox.y + d.bbox.height,
                       d.conf, d.class_id, nullptr});
    std::vector<int> order(raw.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return raw[a].conf > raw[b].conf; });
    std::vector<bool> sup(raw.size(), false);
    std::vector<int>  keep;
    for (int i = 0; i < (int)order.size(); ++i) {
        int idx = order[i];
        if (sup[idx]) continue;
        keep.push_back(idx);
        float ax1 = raw[idx].x1, ay1 = raw[idx].y1;
        float ax2 = raw[idx].x2, ay2 = raw[idx].y2;
        float aa  = std::max(0.f, ax2-ax1) * std::max(0.f, ay2-ay1);
        for (int j = i+1; j < (int)order.size(); ++j) {
            int jdx = order[j];
            if (sup[jdx]) continue;
            float ix1 = std::max(ax1, raw[jdx].x1), iy1 = std::max(ay1, raw[jdx].y1);
            float ix2 = std::min(ax2, raw[jdx].x2), iy2 = std::min(ay2, raw[jdx].y2);
            float inter = std::max(0.f,ix2-ix1) * std::max(0.f,iy2-iy1);
            if (inter <= 0.f) continue;
            float ab = std::max(0.f, raw[jdx].x2-raw[jdx].x1)
                     * std::max(0.f, raw[jdx].y2-raw[jdx].y1);
            if (inter / (aa + ab - inter) > iou_thresh)
                sup[jdx] = true;
        }
    }
    return keep;
}

// ---------------------------------------------------------------------------
// postprocess  — optimised:
//   1. Collect conf-filtered RawDets  (no mask work yet)
//   2. NMS on bboxes only             (skip mask for suppressed boxes)
//   3. Generate mask only for kept detections
//      a. cv::gemm  for the linear combination  [1,mask_dim]*[mask_dim,proto_area]
//      b. cv::exp   vectorised sigmoid
//      c. Crop proto mask to bbox region, resize ONLY that region to orig coords
//         → avoids resizing the full image (4096×3500) per detection
// ---------------------------------------------------------------------------
std::vector<Detection> YoloHiddenCrackDetector::postprocess(
    const float* det_ptr,
    int          det_stride,
    const float* proto_ptr,
    int          orig_h,
    int          orig_w,
    float        conf_thresh,
    float        iou_thresh)
{
    const float scale_x   = static_cast<float>(orig_w) / input_w_;
    const float scale_y   = static_cast<float>(orig_h) / input_h_;
    const float proto_sx  = static_cast<float>(proto_w_) / input_w_;  // 320/1280 = 0.25
    const float proto_sy  = static_cast<float>(proto_h_) / input_h_;
    const int   proto_area = proto_h_ * proto_w_;

    auto clampf = [](float v, float lo, float hi){
        return v < lo ? lo : (v > hi ? hi : v);
    };

    // ---- Phase 1: confidence filter ----------------------------------------
    std::vector<RawDet> raw;
    raw.reserve(32);
    for (int i = 0; i < max_det_; ++i) {
        const float* row = det_ptr + i * det_stride;
        float conf = row[4];
        if (conf < conf_thresh) continue;
        raw.push_back({row[0], row[1], row[2], row[3],
                       conf, static_cast<int>(row[5]), row + 6});
    }
    if (raw.empty()) return {};

    // ---- Phase 2: NMS on raw bboxes (no mask generated yet) ----------------
    std::vector<int> order(raw.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return raw[a].conf > raw[b].conf; });

    std::vector<bool> sup(raw.size(), false);
    std::vector<int>  kept_idx;
    for (int i = 0; i < (int)order.size(); ++i) {
        int idx = order[i];
        if (sup[idx]) continue;
        kept_idx.push_back(idx);
        float ax1 = raw[idx].x1, ay1 = raw[idx].y1;
        float ax2 = raw[idx].x2, ay2 = raw[idx].y2;
        float aa  = std::max(0.f, ax2-ax1) * std::max(0.f, ay2-ay1);
        for (int j = i+1; j < (int)order.size(); ++j) {
            int jdx = order[j];
            if (sup[jdx]) continue;
            float ix1 = std::max(ax1, raw[jdx].x1), iy1 = std::max(ay1, raw[jdx].y1);
            float ix2 = std::min(ax2, raw[jdx].x2), iy2 = std::min(ay2, raw[jdx].y2);
            float inter = std::max(0.f,ix2-ix1) * std::max(0.f,iy2-iy1);
            if (inter <= 0.f) continue;
            float ab = std::max(0.f, raw[jdx].x2-raw[jdx].x1)
                     * std::max(0.f, raw[jdx].y2-raw[jdx].y1);
            if (inter / (aa + ab - inter) > iou_thresh)
                sup[jdx] = true;
        }
    }

    // ---- Phase 3: mask generation for kept detections only -----------------
    // Wrap proto buffer as [mask_dim, proto_area] — zero-copy
    cv::Mat proto_2d(mask_dim_, proto_area, CV_32F,
                     const_cast<float*>(proto_ptr));

    std::vector<Detection> results;
    results.reserve(kept_idx.size());

    for (int idx : kept_idx) {
        const RawDet& rd = raw[idx];

        // 3a. gemm: [1, mask_dim] × [mask_dim, proto_area] → [1, proto_area]
        cv::Mat coeff_row(1, mask_dim_, CV_32F,
                          const_cast<float*>(rd.coeffs));  // zero-copy view
        cv::Mat mask_flat_row;
        cv::gemm(coeff_row, proto_2d, 1.0, cv::Mat(), 0.0, mask_flat_row);
        cv::Mat mask_proto = mask_flat_row.reshape(1, proto_h_); // [proto_h, proto_w]

        // 3b. vectorised sigmoid via cv::exp
        cv::Mat neg_exp;
        cv::exp(-mask_proto, neg_exp);
        cv::Mat mask_sig = 1.0f / (1.0f + neg_exp);   // element-wise

        // 3c. crop proto mask to the detection's bbox region
        int px1 = std::max(0,       (int)(rd.x1 * proto_sx));
        int py1 = std::max(0,       (int)(rd.y1 * proto_sy));
        int px2 = std::min(proto_w_, (int)(rd.x2 * proto_sx + 0.5f));
        int py2 = std::min(proto_h_, (int)(rd.y2 * proto_sy + 0.5f));
        if (px2 <= px1) px2 = std::min(px1 + 1, proto_w_);
        if (py2 <= py1) py2 = std::min(py1 + 1, proto_h_);
        cv::Mat cropped = mask_sig(cv::Range(py1, py2), cv::Range(px1, px2));

        // bbox in original image coords
        float bx1 = clampf(rd.x1 * scale_x, 0.f, orig_w - 1.f);
        float by1 = clampf(rd.y1 * scale_y, 0.f, orig_h - 1.f);
        float bx2 = clampf(rd.x2 * scale_x, 0.f, orig_w - 1.f);
        float by2 = clampf(rd.y2 * scale_y, 0.f, orig_h - 1.f);

        int ibx1 = (int)bx1, iby1 = (int)by1;
        int ibx2 = std::min((int)(bx2 + 0.5f) + 1, orig_w);
        int iby2 = std::min((int)(by2 + 0.5f) + 1, orig_h);
        int bw   = std::max(1, ibx2 - ibx1);
        int bh   = std::max(1, iby2 - iby1);

        // 3d. resize ONLY the bbox-sized crop (not the full image)
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(bw, bh), 0, 0, cv::INTER_LINEAR);

        // 3e. threshold + paste into a full-image binary mask
        cv::Mat binary_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        cv::Mat roi = binary_mask(cv::Range(iby1, iby2), cv::Range(ibx1, ibx2));
        cv::Mat thresh;
        cv::threshold(resized, thresh, 0.5, 1.0, cv::THRESH_BINARY);
        thresh.convertTo(thresh, CV_8UC1);
        thresh.copyTo(roi);

        Detection det;
        det.bbox     = cv::Rect2f(bx1, by1, bx2 - bx1, by2 - by1);
        det.conf     = rd.conf;
        det.class_id = rd.class_id;
        det.mask     = std::move(binary_mask);
        results.push_back(std::move(det));
    }

    return results;
}

// ---------------------------------------------------------------------------
// fast_postprocess  — binary hit/miss, no NMS, no masks, early exit
// ---------------------------------------------------------------------------
int YoloHiddenCrackDetector::fast_postprocess(float conf_thresh)
{
    ov::Tensor out0    = infer_request_.get_output_tensor(0); // [1, max_det, stride]
    const float* ptr   = out0.data<const float>();
    const int    stride = static_cast<int>(out0.get_shape()[2]);
    // confidence is at index 4 of each detection row
    for (int i = 0; i < max_det_; ++i) {
        if (ptr[i * stride + 4] >= conf_thresh)
            return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// infer_and_decode  (shared by detect and detectFromBlob)
// ---------------------------------------------------------------------------
std::vector<Detection> YoloHiddenCrackDetector::infer_and_decode(
    cv::Mat& blob, int orig_h, int orig_w,
    float conf_thresh, float iou_thresh, bool do_postproc)
{
    ov::Tensor input_tensor(
        ov::element::f32,
        {1, 3, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)},
        blob.ptr<float>()
    );
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();

    if (!do_postproc)
        return {};

    ov::Tensor out0 = infer_request_.get_output_tensor(0);
    ov::Tensor out1 = infer_request_.get_output_tensor(1);

    const float* det_ptr   = out0.data<const float>();
    const float* proto_ptr = out1.data<const float>();
    int  det_stride        = static_cast<int>(out0.get_shape()[2]);

    return postprocess(det_ptr, det_stride, proto_ptr, orig_h, orig_w, conf_thresh, iou_thresh);
}

// ---------------------------------------------------------------------------
// detect  — includes preprocessing
// ---------------------------------------------------------------------------
std::vector<Detection> YoloHiddenCrackDetector::detect(const cv::Mat& image,
                                                float conf_thresh,
                                                float iou_thresh,
                                                bool  do_postproc)
{
    if (image.empty())
        throw std::invalid_argument("detect(): input image is empty");

    cv::Mat blob = preprocess(image);
    return infer_and_decode(blob, image.rows, image.cols,
                            conf_thresh, iou_thresh, do_postproc);
}

// ---------------------------------------------------------------------------
// detectFromBlob  — skips preprocessing (blob produced by preprocess())
// ---------------------------------------------------------------------------
std::vector<Detection> YoloHiddenCrackDetector::detectFromBlob(const cv::Mat& blob,
                                                        int   orig_h,
                                                        int   orig_w,
                                                        float conf_thresh,
                                                        float iou_thresh,
                                                        bool  do_postproc)
{
    if (blob.empty())
        throw std::invalid_argument("detectFromBlob(): blob is empty");

    cv::Mat b = blob;   // non-const alias (OV tensor needs non-const ptr)
    return infer_and_decode(b, orig_h, orig_w, conf_thresh, iou_thresh, do_postproc);
}

// ---------------------------------------------------------------------------
// visualize
// ---------------------------------------------------------------------------
cv::Mat YoloHiddenCrackDetector::visualize(const cv::Mat& image,
                                   const std::vector<Detection>& detections)
{
    cv::Mat vis    = image.clone();
    cv::Mat overlay = vis.clone();
    const cv::Scalar color(0, 255, 0);  // green for crack

    for (const auto& det : detections) {
        // filled mask overlay
        overlay.setTo(color, det.mask);

        // bounding box
        cv::Rect box(
            static_cast<int>(std::round(det.bbox.x)),
            static_cast<int>(std::round(det.bbox.y)),
            static_cast<int>(std::round(det.bbox.width)),
            static_cast<int>(std::round(det.bbox.height))
        );
        cv::rectangle(vis, box, color, 3);

        // label
        std::string label = "crack " + std::to_string(det.conf).substr(0, 4);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
        int ty = std::max(box.y - 8, ts.height);
        cv::putText(vis, label, cv::Point(box.x, ty),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 2);
    }

    // blend: overlay (mask fill) × 0.35 + vis (boxes) × 0.65
    cv::addWeighted(overlay, 0.35, vis, 0.65, 0.0, vis);
    return vis;
}
