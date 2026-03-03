# Validation Report

**Date:** 2026-03-03  
**Model:** `best.pt`  
**Checkpoint:** `/root/autodl-tmp/ckpt/yinlie_seg_0302_wb/yolo26n3/weights/best.pt`  
**Split:** `val`  
**Image size:** 1280 &nbsp;|&nbsp; **Target class:** 0 &nbsp;|&nbsp; **Badcase min conf:** 0.1  

---

## Dataset Composition

| Dataset | Positive (NG) | Negative (OK) | Total |
|---------|:---:|:---:|:---:|
| sum | 2448 | 2448 | 4896 |
| real_data0302 | 38 | 500 | 538 |

---

## Performance Metrics

| Dataset | recall@precision=0.95 | precision@recall=0.95 | mAP50 | mAP50-95 | Operating Threshold |
|---------|:---:|:---:|:---:|:---:|:---:|
| sum | 0.983660 | 0.990217 | 0.901165 | 0.766169 | 0.1000 |
| real_data0302 | 0.500000 | 0.468354 | 0.569099 | 0.217101 | 0.5422 |

---

## Confusion Matrix at recall@precision=0.95 Threshold

### sum

Threshold = `0.1000` (max of computed operating point and `BADCASE_MIN_CONF=0.1`)  

|  | **Predicted Positive** | **Predicted Negative** |
|--|:---:|:---:|
| **Actual Positive** (2448 images) | TP = 2287 | FN = 161 |
| **Actual Negative** (2448 images) | FP = 13 | TN = 2435 |

- **Precision** at threshold = 0.9943
- **Recall** at threshold = 0.9342

### real_data0302

Threshold = `0.5422` (max of computed operating point and `BADCASE_MIN_CONF=0.1`)  

|  | **Predicted Positive** | **Predicted Negative** |
|--|:---:|:---:|
| **Actual Positive** (38 images) | TP = 19 | FN = 19 |
| **Actual Negative** (500 images) | FP = 1 | TN = 499 |

- **Precision** at threshold = 0.9500
- **Recall** at threshold = 0.5000
