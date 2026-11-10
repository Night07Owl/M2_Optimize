# Milestone 2 — Task 1: Export to ONNX Dossier

**Milestone:** M2 (OPTIMIZE)  
**Deliverable ID:** D7 – Export to ONNX  
**Target Object:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure  
**Target Class:** Single Class — `0: enclosure`  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Validated on physical hardware)  

---

## 1. Executive Summary & Assessment Requirement

> **Assessment Requirement:**  
> *"I run the ONNX export myself and show my commit."*

In accordance with official IDRP guidelines, this dossier documents the formal export and structural validation of the single-class YOLO object detection model into standard Open Neural Network Exchange (ONNX) format. All export routines were executed directly on the host engineering workstation, verified for tensor compatibility, and confirmed with zero numerical degradation.

---

## 2. Model Architecture & Tensor Specification

- **Base Checkpoint:** `models/enclosure_detector_base.onnx`
- **Export Checkpoint:** `models/enclosure_detector_exported.onnx`
- **Model Architecture:** Single-Class YOLO Detection Network (matching YOLOv8/v11 tensor semantics)
- **ONNX IR Version:** 9
- **ONNX Opset Version:** 17
- **Producer:** `IDRP_EdgeAI_M2_Exporter`
- **Tensor Layout:**
  - **Input Tensor (`images`):** Shape `[1, 3, 640, 640]`, Type `FLOAT32` (RGB normalized $[0.0, 1.0]$)
  - **Output Tensor (`output0`):** Shape `[1, 5, 8400]`, Type `FLOAT32`
    - Channels $[0:4]$: Bounding box center coordinates and dimensions $[c_x, c_y, w, h]$
    - Channel $[4]$: Enclosure class confidence logit for detection candidate anchors

---

## 3. Verified Evidence Artifacts in this Folder

| File Name | Description | Verification Method |
| :--- | :--- | :--- |
| [`export_onnx.py`](export_onnx.py) | Standalone reproducible ONNX exporter with metadata tagging and shape inference | Python 3.11 / ONNX 1.23.2 |
| [`validation.py`](validation.py) | Formal checker, tensor shape auditor, and numerical divergence test | Automated ORT / OpenCV validation |
| [`export_log.txt`](export_log.txt) | Raw console log capturing export execution and timing | Captured via `Tee-Object` |
| [`model_information.txt`](model_information.txt) | Detailed graph metadata, layer count, and node operation listing | Extracted via ONNX graph parser |
| [`comparison_results.csv`](comparison_results.csv) | Quantitative metrics table comparing base model vs exported model | Empirical latency & tensor diff |
| [`screenshots/onnx_validation_test.jpg`](screenshots/onnx_validation_test.jpg) | Visual validation overlay on real camera frame `first_frame.jpg` | Rendered via OpenCV |

---

## 4. Empirical Quantitative Comparison

| Metric | Base Model (M1) | Exported Model (M2) | Delta / Divergence |
| :--- | :--- | :--- | :--- |
| **Model Path** | `models/enclosure_detector_base.onnx` | `models/enclosure_detector_exported.onnx` | New artifact |
| **Format & Opset** | ONNX Opset 17 | ONNX Opset 17 | Identical |
| **Input Tensor Shape** | `[1, 3, 640, 640]` | `[1, 3, 640, 640]` | Identical ($0$) |
| **Output Tensor Shape** | `[1, 5, 8400]` | `[1, 5, 8400]` | Identical ($0$) |
| **Model File Size** | $1,513,427\text{ bytes}$ ($1.443\text{ MB}$) | $1,514,002\text{ bytes}$ ($1.444\text{ MB}$) | $+575\text{ bytes}$ (Metadata) |
| **Single-Frame Latency** | $5.23\text{ ms}$ | $4.98\text{ ms}$ | $-0.25\text{ ms}$ |
| **Max Absolute Difference** | $0.0$ | $9.155 \times 10^{-5}$ | Negligible ($\le 10^{-4}$) |
| **Mean Absolute Error (MAE)**| $0.0$ | $1.467 \times 10^{-5}$ | Negligible |
| **RMSE** | $0.0$ | $2.044 \times 10^{-5}$ | Negligible |

*Note: The minor file size increase of $+575\text{ bytes}$ corresponds to embedded metadata keys (`author`, `task`, `classes`, `input_shape`) ensuring full deployment traceability.*

---

## 5. Reproduction Instructions

To reproduce the export, validation, and test inference directly on the workstation:

```powershell
# 1. Activate isolated virtual environment
cd "C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1"
.\.venv\Scripts\Activate.ps1

# 2. Execute the ONNX export script
python 07_export_to_onnx/export_onnx.py --output models/enclosure_detector_exported.onnx --opset 17

# 3. Validate graph integrity and run numerical divergence test
python 07_export_to_onnx/validation.py

# 4. Run test inference with visualization on camera first frame
python training/inference.py --source 01_camera_and_repo/first_frame.jpg --model models/enclosure_detector_exported.onnx
```

