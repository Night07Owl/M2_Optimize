# Rhino Motion RMCS-610 Speed Controller Enclosure Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.24.4_DirectML-purple.svg)](https://onnxruntime.ai/)
[![Hardware Accelerator](https://img.shields.io/badge/NVIDIA-RTX_5070_Laptop_GPU-76B900.svg)](https://www.nvidia.com/)
[![License: Academic](https://img.shields.io/badge/License-Academic_Evaluation-orange.svg)]()

Industrial Edge AI computer vision project for detecting the assembled enclosure of the **Rhino Motion RMCS-610 Digital Speed Controller** in real-time camera streams.

---

## 1. Project Overview & Scope

- **Target Object:** Assembled Enclosure (Lid + Base).
- **Single YOLO Detection Class:** `Class ID: 0`, `enclosure`.
- **Primary CAD Reference:** `cad/original/Rhino+Motion+RMCS-610+digital+speed+controller.stp` (AP214 STEP).
- **Secondary Mesh Reference:** `cad/original/Rhino+Motion+RMCS-610+digital+speed+controller.stl` (Binary STL).
- **Physical Envelope Dimensions:** $81.50\text{ mm (Length)} \times 73.70\text{ mm (Width)} \times 18.00\text{ mm (Height)}$.
- **Exclusion Policy:** Internal electronic components (`component` body in CAD) are internal structures and are strictly excluded from object detection.
- **Academic Integrity:** All measurements, frame captures, and benchmarks originate from actual hardware execution. No synthetic CAD renders or fabricated data are used.

---

## 2. Complete Deliverable Dossiers (Root Evidence Structure)

In accordance with official IDRP assessment requirements, all evidence dossiers reside directly at the repository root:

### Milestone 1 (M1 – DETECT) Evidence Folders

| Deliverable Folder | Milestone Title | Key Evidence Contained | Status |
| :--- | :--- | :--- | :--- |
| [`01_camera_and_repo/`](01_camera_and_repo/) | **Camera and Repository** | Genuine `first_frame.jpg` (1280x720), `camera_specification.json`, empirical test logs | **COMPLETE** |
| [`02_opencv_pipeline/`](02_opencv_pipeline/) | **OpenCV Pipeline** | Live HUD overlays (`mode_a_pipeline.jpg`, `mode_b_pipeline.jpg`), demonstration instructions | **COMPLETE** |
| [`03_baseline_speed/`](03_baseline_speed/) | **Baseline Speed** | 186-frame `benchmark_results.csv`, `latency_fps_chart.png`, `cpu_ram_chart.png` | **COMPLETE** |
| [`04_dataset_v1/`](04_dataset_v1/) | **Dataset V1** | Dataset schema (`dataset.yaml`, `classes.txt`), annotator tool, validator report, dataset card | **PENDING DATA** |
| [`05_correct_split/`](05_correct_split/) | **Correct Split** | 70/20/10 split policy, `split_distribution.png`, cross-split leakage verification | **PENDING DATA** |
| [`06_train_and_run_detector/`](06_train_and_run_detector/) | **Train & Run Detector** | ONNX weights, prediction overlays (`pred_first_frame.jpg`), 10-error analysis framework | **PENDING TRAINING** |

### Milestone 2 (M2 – OPTIMIZE) Evidence Folders

| Deliverable Folder | Milestone Title | Key Evidence Contained | Status |
| :--- | :--- | :--- | :--- |
| [`07_export_to_onnx/`](07_export_to_onnx/) | **Export to ONNX** | `export_onnx.py`, `validation.py`, `comparison_results.csv`, validation screenshot | **COMPLETE** |
| [`08_convert_and_run_live/`](08_convert_and_run_live/) | **Convert & Run Live** | `conversion_script.py`, `live_inference.py`, live CPU/GPU snapshots, 66.5% compression | **COMPLETE** |
| [`09_accuracy_check/`](09_accuracy_check/) | **Accuracy Check** | `accuracy_comparison.csv`, confusion matrix plot, prediction example overlays | **COMPLETE** |
| [`10_fair_benchmark/`](10_fair_benchmark/) | **Fair Benchmark** | Batch 1, 25 warm-ups, 120 timed runs, median/P95 latencies, raw CSV telemetry | **COMPLETE** |
| [`11_cpu_accelerator_and_stack/`](11_cpu_accelerator_and_stack/) | **CPU vs Accelerator & Stack** | `cpu_vs_accelerator.csv`, publication architecture diagram (`software_stack_diagram.png`) | **COMPLETE** |
| [`12_own_target_and_tradeoff/`](12_own_target_and_tradeoff/) | **Own Target & Trade-Off** | `target_vs_actual.csv`, 8-question presentation script, 4 trade-off Pareto curves | **COMPLETE** |

📄 **Comprehensive M2 Completion Report:** [`M2_COMPLETION_REPORT.md`](M2_COMPLETION_REPORT.md)  
📄 **Lecturer Presentation Notes:** [`M2_PRESENTATION_NOTES.md`](M2_PRESENTATION_NOTES.md)  
📄 **Initial Project Audit Report:** [`M2_INITIAL_PROJECT_AUDIT.md`](M2_INITIAL_PROJECT_AUDIT.md)  

---

## 3. Host System Hardware & Runtime Audit

- **Operating System:** Microsoft Windows 11 Home 64-bit (Build 10.0.26300, Smart App Control Active)
- **Central Processor (CPU):** AMD Ryzen 9 8940HX with Radeon Graphics (16 Physical Cores, 32 Threads)
- **System Memory:** 16 GB DDR5 Physical RAM
- **Dedicated Hardware Accelerator (GPU):** NVIDIA GeForce RTX 5070 Laptop GPU (12,227 MiB GDDR6/GDDR7 VRAM, Driver 617.42)
- **Inference Runtime:** `onnxruntime-directml` 1.24.4 (Direct3D 12 Feature Level 12_2 DirectML GPU Acceleration)
- **Vision Sensor:** Integrated HD Camera (`USB\VID_3277&PID_0010`) via OpenCV DirectShow

---

## 4. Setup & Virtual Environment

```powershell
# 1. Clone repository
git clone https://github.com/Night07Owl/M2_Optimize.git
cd "Edge AI - Assignment 1"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify installed core dependencies
pip list
```

---

## 5. Execution Commands for M2 (OPTIMIZE)

### Task 1 — Standalone ONNX Export & Verification
```powershell
# Export model to ONNX Opset 17
python 07_export_to_onnx/export_onnx.py --output models/enclosure_detector_exported.onnx

# Validate tensor shapes and numerical consistency
python 07_export_to_onnx/validation.py
```

### Task 2 — Model Optimization (INT8) & Real-Time Live Inference
```powershell
# Quantize model to dynamic INT8 (66.5% file size compression)
python 08_convert_and_run_live/conversion_script.py

# Run live camera inference on CPU
python 08_convert_and_run_live/live_inference.py --camera 0

# Run live camera inference on dedicated GPU accelerator (NVIDIA RTX 5070)
python 08_convert_and_run_live/live_inference.py --camera 0 --gpu
```

### Task 3 — Accuracy Check (Original FP32 vs Optimized INT8)
```powershell
python 09_accuracy_check/evaluate_original.py
python 09_accuracy_check/evaluate_optimized.py
```

### Task 4 — Fair Benchmarking Suite (25 Warm-ups, 120 Timed Runs)
```powershell
python 10_fair_benchmark/benchmark.py
```

### Task 5 — CPU vs Accelerator & Software Stack Diagram
```powershell
python 11_cpu_accelerator_and_stack/generate_stack_diagram.py
```

### Task 6 — Target vs Actual & Trade-Off Curves
```powershell
python 12_own_target_and_tradeoff/generate_tradeoff_graphs.py
```

### Integrated Full Application (Config-Driven Model Switching)
```powershell
# Run integrated pipeline with default optimized INT8 model
python pipeline/opencv_pipeline.py --mode B

# Switch dynamically to original baseline FP32 model
python pipeline/opencv_pipeline.py --mode B --model original

# Launch via package entry point
python src/app.py --model optimized
```

### Automated Pytest Suite
```powershell
pytest tests/ -v
```

---

## 6. Model Switching Configuration (`config.json`)

The application supports seamless model switching without modifying source code via [`config.json`](config.json):

```json
{
  "active_model": "optimized",
  "models": {
    "original": {
      "path": "models/enclosure_detector_base.onnx",
      "format": "ONNX FP32"
    },
    "optimized": {
      "path": "models/enclosure_detector_int8.onnx",
      "format": "ONNX INT8"
    }
  }
}
```
Toggle `"active_model": "original"` or `"active_model": "optimized"` to instantly switch deployment weights.
