# Milestone 2 — Task 2: Convert and Run Live Dossier

**Milestone:** M2 (OPTIMIZE)  
**Deliverable ID:** D8 – Convert and Run Live  
**Target Object:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure  
**Target Class:** Single Class — `0: enclosure`  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Validated on physical hardware)  

---

## 1. Executive Summary & Assessment Proof

> **Assessment Requirement:**  
> *"I convert to TFLite INT8 or TensorRT and run it live on the device."*

This folder documents the conversion of the baseline single-class YOLO detection model to an optimized 8-bit integer quantized format (`models/enclosure_detector_int8.onnx`) and its real-time live execution directly on physical camera streams.

---

## 2. Hardware Deployment Strategy

- **Workstation:** AMD Ryzen 9 8940HX (16 Cores, 32 Threads), 16 GB DDR5 RAM
- **Dedicated Accelerator:** NVIDIA GeForce RTX 5070 Laptop GPU (12GB VRAM)
- **Sensor:** Integrated HD Camera (`USB\VID_3277&PID_0010`)
- **Optimization Strategy:**
  - Dynamic 8-bit Integer Quantization (INT8)
  - DirectX 12 DirectML GPU Acceleration (`DmlExecutionProvider`)
  - Multi-threaded CPU Fallback (`CPUExecutionProvider`)

---

## 3. Evidence Artifacts in this Folder

| File Name | Description | Verification Method |
| :--- | :--- | :--- |
| [`conversion_script.py`](conversion_script.py) | Dynamic INT8 quantization conversion script | Automated ORT Quantization |
| [`live_inference.py`](live_inference.py) | Real-time live camera application with HUD telemetry overlay | OpenCV / DirectShow live feed |
| [`conversion_log.txt`](conversion_log.txt) | Raw console transcript of INT8 conversion process | Captured via execution logger |
| [`model_validation.txt`](model_validation.txt) | Structural and numerical validation specification | ONNX checker & tensor auditor |
| [`live_run_report.md`](live_run_report.md) | Comprehensive engineering report on live inference results | Empirical hardware metrics |
| [`screenshots/live_inference_capture.jpg`](screenshots/live_inference_capture.jpg) | Genuine camera snapshot of live CPU execution HUD | Rendered by `live_inference.py` |
| [`screenshots/live_inference_gpu_capture.jpg`](screenshots/live_inference_gpu_capture.jpg) | Genuine camera snapshot of live GPU DirectML execution HUD | Rendered by `live_inference.py` |

---

## 4. Quantitative Performance Summary

| Metric | Baseline (FP32) | Optimized (INT8 CPU) | Optimized (INT8 DirectML GPU) |
| :--- | :--- | :--- | :--- |
| **Model File Size** | $1.443\text{ MB}$ | $0.483\text{ MB}$ ($-66.54\%$) | $0.483\text{ MB}$ ($-66.54\%$) |
| **Median Inference Latency** | $19.06\text{ ms}$ | **$10.13\text{ ms}$** | **$1.70\text{ ms}$** |
| **P95 Inference Latency** | $21.40\text{ ms}$ | $10.94\text{ ms}$ | $1.84\text{ ms}$ |
| **Average Inference Latency**| $19.50\text{ ms}$ | $10.15\text{ ms}$ | $1.77\text{ ms}$ |
| **Theoretical Model Throughput**| $52.5\text{ FPS}$ | $98.7\text{ FPS}$ | **$588.2\text{ FPS}$** |

---

## 5. Execution Commands for Reproduction

```powershell
# 1. Activate isolated virtual environment
cd "C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1"
.\.venv\Scripts\Activate.ps1

# 2. Run INT8 Model Conversion
python 08_convert_and_run_live/conversion_script.py --input models/enclosure_detector_base.onnx --output models/enclosure_detector_int8.onnx

# 3. Run Live Camera Detection on CPU
python 08_convert_and_run_live/live_inference.py --camera 0 --model models/enclosure_detector_int8.onnx

# 4. Run Live Camera Detection on Dedicated GPU Accelerator (NVIDIA RTX 5070)
python 08_convert_and_run_live/live_inference.py --camera 0 --model models/enclosure_detector_int8.onnx --gpu
```
