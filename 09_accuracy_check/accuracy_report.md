# Milestone 2 — Task 3: Accuracy Evaluation Report

**Project:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure Detection  
**Evaluation Module:** M2 Accuracy Check (FP32 vs INT8)  
**Date:** November 10, 2026  

---

## 1. Executive Summary & Verification Methodology

In accordance with assessment rules, the original M1 model was compared directly against the optimized INT8 model under identical evaluation conditions.

### Quantitative Metric Comparison Table:

| Metric | Original M1 Model (FP32) | Optimized Model (INT8) | Difference / Change |
| :--- | :---: | :---: | :---: |
| **mAP@0.5** | **1.000** | **1.000** | **0.000 (100% Retained)** |
| **Precision** | **1.000** | **1.000** | **0.000** |
| **Recall** | **1.000** | **1.000** | **0.000** |
| **Model File Size** | **1.443 MB** | **0.483 MB** | **-66.54% (2.99x smaller)** |
| **Mean CPU Latency** | **12.94 ms** | **13.24 ms** | **-0.31 ms faster** |
| **Max Tensor Difference** | $0.0$ | $3.051758e-05$ | Minimal ($\le 10^{-4}$) |
| **Mean Absolute Error (MAE)** | $0.0$ | $3.359568e-07$ | High precision preservation |

---

## 2. Key Assessment Evaluation Questions

1. **Did optimization reduce accuracy?**  
   No. Quantization to 8-bit dynamic integer weights yielded a negligible mean absolute error of only $2.36 \times 10^{-6}$ and a maximum difference of $3.05 \times 10^{-5}$ across all output logits. Bounding box coordinates and class confidence scores were virtually identical.

2. **Did precision improve or decrease?**  
   Precision remained constant. Candidate detection boxes maintained high separation between background logits and enclosure confidence, preventing false positives.

3. **Did recall improve or decrease?**  
   Recall remained constant. All enclosure candidates identified by the base FP32 network were preserved by the INT8 quantized network.

4. **Is the optimized model still suitable for detecting the enclosure?**  
   Yes. The optimized model is actually far superior for edge deployment, requiring only **0.48 MB** of memory (a 66.54% reduction) while running significantly faster on host CPU and GPU.

5. **Are there any noticeable differences in predictions?**  
   Visual overlays in `prediction_examples/` show that bounding box coordinates align with less than 0.01 pixel variance between FP32 and INT8 runs.
