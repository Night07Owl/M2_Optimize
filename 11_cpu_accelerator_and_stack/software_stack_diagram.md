# Milestone 2 — Task 5: Software & Hardware Architecture Stack

**Project:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure Detection  
**Module:** M2 (OPTIMIZE)  
**Deliverable ID:** D11 – CPU vs Accelerator & Software Stack  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Empirically verified on physical host)  

---

## 1. Complete Software & Hardware Stack Diagram

```text
                  +-------------------------------------------------------+
                  |                 1. PHYSICAL VISION SENSOR              |
                  |     Integrated HD Camera (USB\VID_3277&PID_0010)       |
                  |             Native HD: 1280x720 @ 30 FPS               |
                  +-------------------------------------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |         2. DEVICE DRIVER & FRAME CAPTURE BUS          |
                  |      Windows DirectShow (cv2.CAP_DSHOW) / MSMF API     |
                  |              Empirical Latency: ~47.9 ms              |
                  +-------------------------------------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |         3. OPENCV COMPUTER VISION PREPROCESSING        |
                  |         OpenCV 5.0.0.93 (CPython 3.11.9 64-bit)        |
                  |  Letterbox 640x640 | BGR to RGB | Normalization [0,1]  |
                  +-------------------------------------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |      4. OPTIMIZED DEPLOYMENT MODEL (ONNX INT8)        |
                  |       models/enclosure_detector_int8.onnx (506 KB)    |
                  |   Single-Class YOLO Graph (Class 0: 'enclosure')      |
                  +-------------------------------------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |            5. EDGE INFERENCE ENGINE (ORT)             |
                  |            ONNX Runtime DirectML 1.24.4               |
                  |    Graph Optimization: ORT_ENABLE_ALL | Batch Size: 1 |
                  +-------------------------------------------------------+
                                              |
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
    +-----------------------------------------+   +-----------------------------------------+
    |           6A. CPU EXECUTION             |   |       6B. HARDWARE ACCELERATOR          |
    |          CPUExecutionProvider           |   |          DmlExecutionProvider           |
    | AMD Ryzen 9 8940HX (16 Cores, 32 Thr)   |   |   NVIDIA GeForce RTX 5070 Laptop GPU    |
    |      Median Latency: 8.90 ms            |   |         Median Latency: 1.38 ms         |
    |        Throughput: 115.5 FPS            |   |          Throughput: 722.4 FPS          |
    |          CPU Load: ~50.9%               |   |     CPU Offload: CPU Load drops to 9.4% |
    +-----------------------------------------+   +-----------------------------------------+
                         |                                         |
                         +--------------------+--------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |        7. POSTPROCESSING & INVERSE RECONSTRUCTION     |
                  |       Confidence Rejection (tau >= 0.30) | NMS        |
                  |     Inverse Letterbox Coordinate Rescaling to 720p    |
                  +-------------------------------------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |              8. ENCLOSURE TARGET DETECTION            |
                  |  Enclosure Bounding Box: [x1, y1, x2, y2]             |
                  |  Detection Confidence Score | Single Class: enclosure |
                  +-------------------------------------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |        9. REAL-TIME HUD TELEMETRY & DISPLAY           |
                  |  Telemetry HUD: FPS, Latency (ms), Count, Backend     |
                  |  Keybindings: 's' snapshot, 'q' graceful exit         |
                  +-------------------------------------------------------+
```

---

## 2. Exhaustive Technical Subsystem Specification

| Architecture Layer | Discovered / Implemented Technology | Exact Version / Details |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home 64-bit | Build 10.0.26300 (WDDM Driver Model) |
| **Security Subsystem** | Windows Defender Smart App Control (SAC) | `SmartAppControlState: On` (Microsoft-signed execution) |
| **Programming Language** | Python | 3.11.9 (64-bit CPython) in dedicated `.venv` |
| **Camera Sensor Hardware** | Integrated HD Camera | `USB\VID_3277&PID_0010`, UVC Bridge |
| **Camera Capture Interface** | OpenCV VideoCapture DirectShow | `cv2.CAP_DSHOW` ($1280 \times 720$ resolution) |
| **Vision Framework** | OpenCV Python | Version 5.0.0.93 |
| **Neural Network Architecture** | Single-Class YOLO Object Detector | Class 0: `enclosure` (Assembled RMCS-610 Housing) |
| **Base Model Format** | ONNX (Open Neural Network Exchange) | IR Version 9, Opset 17 ($1.443\text{ MB}$) |
| **Optimized Model Format** | ONNX Dynamic Quantized INT8 | QInt8 precision, $506\text{ KB}$ ($66.54\%$ size reduction) |
| **Inference Runtime** | ONNX Runtime DirectML | Version 1.24.4 |
| **CPU Execution Hardware** | AMD Ryzen 9 8940HX with Radeon Graphics | 16 physical cores, 32 threads, AVX-512 / AVX2 |
| **Hardware Accelerator** | **NVIDIA GeForce RTX 5070 Laptop GPU** | 12,227 MiB GDDR6/GDDR7 VRAM, Driver 617.42 |
| **Accelerator Interface API** | Direct3D 12 Feature Level 12_2 | DirectML (`DmlExecutionProvider`) |
| **Visual Evidence Diagram** | High-resolution publication diagram | [`software_stack_diagram.png`](software_stack_diagram.png) |
