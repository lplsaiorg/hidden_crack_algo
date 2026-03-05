# Validation Report

**Date:** 2026-03-05  
**Model:** `best.pt`  
**Checkpoint:** `/root/autodl-tmp/ckpt/yinlie_seg_0304/yolo26n_1000e/weights/best.pt`  
**Split:** `val`  
**Image size:** 1280 &nbsp;|&nbsp; **Target class:** 0 &nbsp;|&nbsp; **Badcase min conf:** 0.05  

---

## Dataset Composition

| Dataset | Positive (NG) | Negative (OK) | Total |
|---------|:---:|:---:|:---:|
| lipo0302 | 38 | 500 | 538 |
| laplace0304 | 114 | 964 | 1078 |

---

## Performance Metrics

| Dataset | recall@precision=0.95 | precision@recall=0.95 | mAP50 | mAP50-95 | Operating Threshold |
|---------|:---:|:---:|:---:|:---:|:---:|
| lipo0302 | 1.000000 | 0.974359 | 0.673481 | 0.256024 | 0.0500 |
| laplace0304 | 0.692982 | 0.851562 | 0.453828 | 0.172818 | 0.1419 |

---

## Confusion Matrix at recall@precision=0.95 Threshold

### lipo0302

Threshold = `0.0500` (max of computed operating point and `BADCASE_MIN_CONF=0.05`)  

|  | **Predicted Positive** | **Predicted Negative** |
|--|:---:|:---:|
| **Actual Positive** (38 images) | TP = 35 | FN = 3 |
| **Actual Negative** (500 images) | FP = 1 | TN = 499 |

- **Precision** at threshold = 0.9722
- **Recall** at threshold = 0.9211

### laplace0304

Threshold = `0.1419` (max of computed operating point and `BADCASE_MIN_CONF=0.05`)  

|  | **Predicted Positive** | **Predicted Negative** |
|--|:---:|:---:|
| **Actual Positive** (114 images) | TP = 79 | FN = 35 |
| **Actual Negative** (964 images) | FP = 4 | TN = 960 |

- **Precision** at threshold = 0.9518
- **Recall** at threshold = 0.6930
