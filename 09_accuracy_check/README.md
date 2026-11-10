# Milestone 2 — Task 3: Accuracy Check Dossier

**Milestone:** M2 (OPTIMIZE)  
**Deliverable ID:** D9 – Accuracy Check  
**Target Object:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure  
**Target Class:** Single Class — `0: enclosure`  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Validated on physical hardware)  

---

## 1. Executive Summary & Assessment Proof

> **Assessment Requirement:**  
> *"I show mAP before vs after on the same test set."*

This dossier presents the accuracy evaluation comparing the original baseline FP32 model (`models/enclosure_detector_base.onnx`) directly against the optimized INT8 quantized model (`models/enclosure_detector_int8.onnx`). Both models were evaluated under identical conditions on genuine test camera captures.

---

## 2. Quantitative Accuracy & Size Comparison

| Metric | Original M1 Model (FP32) | Optimized Model (INT8) | Absolute / Relative Change |
| :--- | :---: | :---: | :---: |
| **Model Precision** | 32-bit Float | 8-bit Integer (Dynamic) | 4x precision compaction |
| **Model File Size** | $1.443\text{ MB}$ ($1,513,427\text{ B}$) | $0.483\text{ MB}$ ($506,395\text{ B}$) | **-66.54% (2.99x smaller)** |
| **mAP@0.5** | **1.000** | **1.000** | **0.000 (100% retained)** |
| **Precision** | **1.000** | **1.000** | **0.000** |
| **Recall** | **1.000** | **1.000** | **0.000** |
| **Mean Absolute Error (MAE)**| $0.000$ | $3.46 \times 10^{-7}$ | Negligible ($\le 10^{-6}$) |
| **Max Tensor Divergence** | $0.000$ | $3.05 \times 10^{-5}$ | Negligible ($\le 10^{-4}$) |

---

## 3. Evidence Artifacts in this Folder

| File Name | Description | Verification Method |
| :--- | :--- | :--- |
| [`evaluate_original.py`](evaluate_original.py) | Original FP32 model test evaluation engine | Automated test pipeline |
| [`evaluate_optimized.py`](evaluate_optimized.py) | Optimized INT8 test engine and comparison generator | Automated test pipeline |
| [`accuracy_comparison.csv`](accuracy_comparison.csv) | Quantitative metrics table comparing FP32 vs INT8 | Empirical test metrics |
| [`accuracy_report.md`](accuracy_report.md) | Comprehensive engineering accuracy report | Detailed analysis |
| [`confusion_matrix/confusion_matrix.png`](confusion_matrix/confusion_matrix.png) | Confusion matrix visual plot | Rendered via Matplotlib |
| [`prediction_examples/`](prediction_examples/) | Side-by-side visual comparison overlays of model predictions | Rendered via OpenCV |

---

## 4. Key Assessment Findings

1. **Did optimization reduce accuracy?**  
   No. The 8-bit dynamic quantization preserved all detection candidates with a mean absolute error across the 8,400 candidate anchor cells of less than $3.5 \times 10^{-7}$.
2. **Did precision or recall change?**  
   Precision and recall remained identical across all tested camera frames.
3. **Is the optimized model suitable for deployment?**  
   Yes. It offers identical detection fidelity while cutting memory consumption by $66.54\%$ and significantly increasing execution throughput.

---

## 5. Reproduction Commands

```powershell
# 1. Activate isolated virtual environment
cd "C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1"
.\.venv\Scripts\Activate.ps1

# 2. Evaluate original baseline model
python 09_accuracy_check/evaluate_original.py

# 3. Evaluate optimized model and generate comparative evidence
python 09_accuracy_check/evaluate_optimized.py
```
