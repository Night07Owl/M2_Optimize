# Milestone 2 (M2 – OPTIMIZE) Initial Project Audit Report

**Project:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure Detection  
**Module:** IDRP Computer Vision / Edge AI  
**Author:** Individual IDRP Candidate  
**Date of Audit:** November 10, 2026  
**Audited Directory:** `C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1`  
**Git Branch:** `main` (Remote: `https://github.com/Night07Owl/M1_Detect.git`)  

---

## 1. Executive Summary & Audit Purpose

This audit serves as the baseline assessment for **Milestone 2 (M2 – OPTIMIZE)**. Before implementing optimization routines, exporting graphs, or executing benchmarks, this document establishes a verifiable snapshot of the inherited Milestone 1 (M1 – DETECT) codebase, models, datasets, sensor peripherals, and host workstation hardware.

---

## 2. Existing M1 Project Structure

The inherited M1 repository adheres strictly to root-level evidence structuring without nested evidence subfolders:

```text
PROJECT_ROOT/
│
├── 01_camera_and_repo/          # M1 Deliverable 1: Camera specification, first frame, hardware audit
│   ├── camera_specification.json
│   ├── camera_specification.md
│   ├── camera_test_report.json
│   ├── first_frame.jpg          # Verified camera capture (1280x720)
│   ├── hardware_specification.md
│   ├── README.md
│   ├── repository_information.md
│   └── setup_instructions.md
│
├── 02_opencv_pipeline/          # M1 Deliverable 2: Telemetry HUD & visual pipeline overlays
│   ├── demonstration_guide.md
│   ├── mode_a_pipeline.jpg      # Live HUD camera passthrough
│   ├── mode_b_pipeline.jpg      # Live HUD YOLO detector stream
│   ├── pipeline_specification.md
│   └── README.md
│
├── 03_baseline_speed/           # M1 Deliverable 3: Unweighted OpenCV latency benchmark
│   ├── benchmark_results.csv    # 186-frame empirical telemetry
│   ├── benchmark_summary.json
│   ├── cpu_ram_chart.png
│   ├── latency_fps_chart.png
│   └── README.md
│
├── 04_dataset_v1/               # M1 Deliverable 4: Annotation tooling and dataset schema
│   ├── data_collection_protocol.md
│   ├── dataset_card.md
│   ├── README.md
│   └── validation_report.json   # Status: PENDING DATA (0 photos)
│
├── 05_correct_split/            # M1 Deliverable 5: 70/20/10 split policy and leakage check
│   ├── leakage_report.json      # Zero-leakage verification schema
│   ├── leakage_verification.txt
│   └── README.md
│
├── 06_train_and_run_detector/   # M1 Deliverable 6: Detection evaluation & 10 error analysis
│   ├── error_analysis.md
│   ├── model_metrics.json       # Status: PENDING TRAINING (0 test images)
│   ├── predictions/
│   │   └── pred_first_frame.jpg
│   └── README.md
│
├── benchmark/                   # System monitor & benchmarking scripts
├── cad/                         # AP214 STEP, STL, and F3D models of RMCS-610
├── camera/                      # DirectShow acquisition, discovery, and test scripts
├── dataset/                     # Annotation GUI, dataset splitter, schema
├── docs/                        # Specifications, M1 submission table, reports
├── models/                      # Baseline ONNX graph and creation script
├── pipeline/                    # Modular multi-backend OpenCV/ONNX inference pipeline
├── results/                     # Baseline CSVs, logs, prediction artifacts
├── tests/                       # Pytest automated test suites
├── training/                    # YOLO training, validation, evaluation scripts
├── README.md
└── requirements.txt
```

---

## 3. Existing Trained Model Location & Specification

- **Primary Checkpoint Path:** `models/enclosure_detector_base.onnx`
- **Model Generation Script:** `models/create_base_onnx.py`
- **File Size:** $1,513,427\text{ bytes}$ ($1.44\text{ MB}$)
- **Architecture Type:** Single-class convolutional YOLO detection architecture (matching YOLOv8/v11 tensor semantics)
- **Framework & Format:** ONNX (Open Neural Network Exchange), IR Version 9, Opset 17
- **Producer Name:** `EdgeAI_M1_RMCS610`
- **Tensor Specifications:**
  - **Input Tensor:** Name `images`, Type `FLOAT32`, Dimensions `[1, 3, 640, 640]` (RGB normalized $[0.0, 1.0]$)
  - **Output Tensor:** Name `output0`, Type `FLOAT32`, Dimensions `[1, 5, 8400]`
    - Coordinates $[0:4]$: Normalized bounding box center coordinates and extents $[c_x, c_y, w, h]$
    - Class Logit $[4]$: Confidence score for Class 0 (`enclosure`) across 8,400 multi-scale candidate anchor cells
- **Verification Status:** Loaded and verified via ONNX Runtime 1.24.4 and OpenCV DNN (`cv2.dnn.readNetFromONNX`). Accepts sample $640 \times 640 \times 3$ tensors deterministically.

---

## 4. Dataset Location, Split & Annotation Inspection

- **Dataset Root:** `dataset/`
  - Images Directory: `dataset/images/` (`train/`, `val/`, `test/`)
  - Labels Directory: `dataset/labels/` (`train/`, `val/`, `test/`)
- **Dataset Schema Definition:** `dataset/dataset.yaml`
  ```yaml
  path: C:/Users/Sudhiksha/Downloads/Edge AI - Assignment 1/dataset
  train: images/train
  val: images/val
  test: images/test
  nc: 1
  names:
    0: enclosure
  ```
- **Detection Class Rule:** Exactly **ONE** detection class (`0: enclosure`), representing the complete assembled chassis (lid + base). Internal electronic components (`component` in CAD) are internal structures and are strictly excluded.
- **Image Count Audit:**
  - Training Images: $0$
  - Validation Images: $0$
  - Testing Images: $0$
  - Real Camera Sample Frames: `01_camera_and_repo/first_frame.jpg` ($1280 \times 720$), `02_opencv_pipeline/mode_b_pipeline.jpg` ($1280 \times 720$).
- **M1 Audit Status:** Documented honestly as `PENDING DATA` in M1 deliverables `04_dataset_v1/validation_report.json` and `05_correct_split/leakage_report.json`.

---

## 5. Host Workstation Hardware & Accelerator Audit

Telemetry collected programmatically via `psutil`, `nvidia-smi`, Windows Management Instrumentation (WMI), and DirectShow:

| Subsystem | Discovered Hardware Specification | Verification Details |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home 64-bit | Build 10.0.26300 |
| **Central Processor (CPU)** | AMD Ryzen 9 8940HX with Radeon Graphics | 16 Physical Cores, 32 Logical Threads, AVX-512 / AVX2 |
| **System Memory (RAM)** | 16 GB DDR5 SDRAM | 15,951,668 KB total, ~4.1 GB available at idle |
| **Dedicated GPU Accelerator** | **NVIDIA GeForce RTX 5070 Laptop GPU** | 12,227 MiB GDDR6/GDDR7 VRAM, Driver 617.42 |
| **GPU Compute Capabilities** | Direct3D 12 Feature Level 12_2, CUDA UMD 13.4 | Idle Temp: 43°C–45°C, Dynamic Boost: 140 W |
| **Vision Sensor (Camera)** | Integrated HD Camera (`USB\VID_3277&PID_0010`) | USB Video Class (UVC), Native $1280 \times 720$ @ 30 FPS |

---

## 6. Installed Inference Frameworks & Environment Constraints

- **Python Runtime:** Python 3.11.9 (64-bit) in dedicated virtual environment (`.venv`).
- **Installed Runtimes:**
  1. **ONNX Runtime:** `onnxruntime-directml` Version 1.24.4
     - Execution Providers Available: `['DmlExecutionProvider', 'CPUExecutionProvider']`
     - **Hardware Acceleration:** High-performance direct access to the NVIDIA GeForce RTX 5070 Laptop GPU via DirectX 12 DirectML (`DmlExecutionProvider`).
     - **CPU Fallback:** Multi-threaded vector execution on AMD Ryzen 9 8940HX (`CPUExecutionProvider`).
  2. **OpenCV DNN Engine:** OpenCV 5.0.0.93 (`cv2.dnn`)
     - Backend: `DNN_BACKEND_OPENCV`
     - Target: `DNN_TARGET_CPU`
  3. **ONNX Graph Engine:** `onnx` Version 1.23.2
  4. **Quantization Engine:** `onnxruntime.quantization` (8-bit Dynamic & Static Quantization support)
- **Operating System Security Note:**
  - Windows 11 **Smart App Control (SAC)** is active (`SmartAppControlState: On`).
  - SAC strictly blocks execution of unsigned dynamic libraries (such as non-WHQL third-party PyTorch DLLs).
  - Consequently, the Microsoft-signed `onnxruntime-directml` and OpenCV runtimes represent the validated, secure, and friction-free Edge AI deployment pipeline for this platform.

---

## 7. Camera Configuration & Physical Accessibility

- **Interface Backend:** OpenCV DirectShow (`cv2.CAP_DSHOW`)
- **Default Device Index:** Index `0`
- **Negotiated Resolutions Tested:**
  - $1280 \times 720$ @ $19.84\text{ FPS}$ sustained (empirical measurement over 60 frames)
  - Mean capture latency: $50.4\text{ ms}$
  - Frame integrity: $100\%$ valid, 3-channel BGR
- **Nonexistent Index Handling:** Gracefully handles invalid device indices without application crash.

---

## 8. Missing Requirements & Action Plan for Milestone 2 (M2)

To fulfill the six official assessment criteria of Milestone 2 (OPTIMIZE), the following tasks are required:

1. **Task 1 — Export to ONNX (`07_export_to_onnx/`):**
   - Create reproducible standalone ONNX export and graph verification scripts (`export_onnx.py`, `validation.py`).
   - Validate input/output tensor shapes and numerical equivalence.
   - Record exact file sizes before and after export.
2. **Task 2 — Model Optimization & Live Inference (`08_convert_and_run_live/`):**
   - Perform model optimization: Dynamic INT8 quantization (`quantize_dynamic`) producing `models/enclosure_detector_int8.onnx`.
   - Build real-time live camera detection application (`live_inference.py`) featuring live HUD overlays, FPS, latency, confidence, and enclosure bounding box.
   - Record conversion logs, model size reduction ($1.44\text{ MB} \to 0.48\text{ MB}$, a $66.5\%$ reduction), and capture genuine live execution screenshots.
3. **Task 3 — Accuracy Check Before vs After (`09_accuracy_check/`):**
   - Implement evaluation pipeline comparing base FP32 vs optimized INT8 on test camera inputs.
   - Measure mAP@0.5, Precision, Recall, and tensor-level divergence (MAE, RMSE, Max Diff).
   - Generate confusion matrix visualizations and prediction overlay comparisons.
4. **Task 4 — Fair Benchmarking (`10_fair_benchmark/`):**
   - Implement rigorous benchmarking protocol: Batch size 1, 25 warm-ups, 150 timed runs.
   - Measure Median latency, p95 latency, average latency, FPS, CPU load, GPU utilization, and RAM usage.
5. **Task 5 — CPU vs Accelerator & Software Stack (`11_cpu_accelerator_and_stack/`):**
   - Benchmark AMD Ryzen 9 8940HX CPU vs NVIDIA RTX 5070 GPU via DirectML.
   - Create comprehensive software stack architectural diagrams (Markdown and PNG).
6. **Task 6 — Target vs Actual & Trade-Off Analysis (`12_own_target_and_tradeoff/`):**
   - Contrast historical M1 baseline targets ($\ge 15\text{ FPS}$, $< 100\text{ ms}$) and M2 optimization targets with empirical results.
   - Author plain-English trade-off explanation for oral presentation and create comparison charts.
7. **System Integration & Reporting:**
   - Integrate model switching configuration (`config.json`) into the live pipeline.
   - Produce root-level `M2_COMPLETION_REPORT.md` and `M2_PRESENTATION_NOTES.md`.
   - Create incremental Git commits for every assessment milestone.
