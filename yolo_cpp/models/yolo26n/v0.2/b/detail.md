# Validation Report

**Date:** 2026-03-03  
**Model:** `best.pt`  
**Checkpoint:** `/root/autodl-tmp/ckpt/yinlie_seg_0302_b/yolo26n/weights/best.pt`  
**Split:** `val`  
**Image size:** 1280 &nbsp;|&nbsp; **Target class:** 0 &nbsp;|&nbsp; **Badcase min conf:** 0.05  

---

## Dataset Composition

| Dataset | Positive (NG) | Negative (OK) | Total |
|---------|:---:|:---:|:---:|
| sum | 1617 | 1617 | 3234 |
| real_data0302 | 38 | 500 | 538 |

---

## Performance Metrics

| Dataset | recall@precision=0.95 | precision@recall=0.95 | mAP50 | mAP50-95 | Operating Threshold |
|---------|:---:|:---:|:---:|:---:|:---:|
| sum | 1.000000 | 0.998077 | 0.845666 | 0.622176 | 0.0500 |
| real_data0302 | 0.921053 | 0.902439 | 0.629319 | 0.215131 | 0.0696 |

---

## Confusion Matrix at recall@precision=0.95 Threshold

### sum

Threshold = `0.0500` (max of computed operating point and `BADCASE_MIN_CONF=0.05`)  

|  | **Predicted Positive** | **Predicted Negative** |
|--|:---:|:---:|
| **Actual Positive** (1617 images) | TP = 1557 | FN = 60 |
| **Actual Negative** (1617 images) | FP = 3 | TN = 1614 |

- **Precision** at threshold = 0.9981
- **Recall** at threshold = 0.9629

### real_data0302

Threshold = `0.0696` (max of computed operating point and `BADCASE_MIN_CONF=0.05`)  

|  | **Predicted Positive** | **Predicted Negative** |
|--|:---:|:---:|
| **Actual Positive** (38 images) | TP = 35 | FN = 3 |
| **Actual Negative** (500 images) | FP = 1 | TN = 499 |

- **Precision** at threshold = 0.9722
- **Recall** at threshold = 0.9211
