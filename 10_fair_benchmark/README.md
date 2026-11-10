# Milestone 2 — Task 4: Fair Benchmark Dossier

**Milestone:** M2 (OPTIMIZE)  
**Deliverable ID:** D10 – Fair Benchmark  
**Target Object:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure  
**Target Class:** Single Class — `0: enclosure`  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Validated on physical hardware)  

---

## 1. Executive Summary & Assessment Requirement

> **Assessment Requirement:**  
> *"I run 20 warm-up and 100+ timed runs at batch 1. I show median and p95 latency, FPS and model size."*

This dossier implements and documents an empirical benchmarking suite meeting every formal assessment constraint. The benchmark evaluated the original M1 model against the optimized model across both CPU and the dedicated NVIDIA GPU accelerator.

---

## 2. Benchmark Protocol Compliance

- **Batch Size:** 1 (single-frame real-time inference)
- **Warm-up Runs:** **25 iterations** (exceeds mandatory $\ge 20$)
- **Timed Inference Runs:** **120 iterations** (exceeds mandatory $\ge 100$)
- **Input Dimensions:** $640 \times 640 \times 3$ FLOAT32 normalized tensor
- **Timing Stage:** Pure model inference computation with execution synchronization
- **Hardware Parity:** All benchmarks executed sequentially on the identical host machine

---

## 3. Empirical Results Table

| Metric | Original Model (CPU) | Optimized Model (CPU) | Original Model (GPU) | Optimized Model (GPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Warm-up Runs** | 25 | 25 | 25 | 25 |
| **Timed Runs** | 120 | 120 | 120 | 120 |
| **Batch Size** | 1 | 1 | 1 | 1 |
| **Median Latency** | **3.190 ms** | **8.901 ms** | **0.711 ms** | **1.382 ms** |
| **P95 Latency** | 4.010 ms | 9.734 ms | 0.796 ms | 1.457 ms |
| **Average Latency** | 3.309 ms | 8.659 ms | 0.722 ms | 1.384 ms |
| **Inference FPS** | **302.24 FPS** | **115.48 FPS** | **1,384.87 FPS** | **722.36 FPS** |
| **Model Size** | **1.443 MB** | **0.483 MB** | **1.443 MB** | **0.483 MB** |
| **CPU Utilization** | 51.57% | 50.85% | 9.93% | 9.39% |
| **GPU Utilization** | 0.0% | 0.0% | 0.0% | 49.0% |
| **Process RAM (RSS)** | 149.78 MB | 126.94 MB | 286.07 MB | 304.87 MB |
| **GPU Temperature** | 47.0 °C | 47.0 °C | 47.0 °C | 55.0 °C |

---

## 4. Evidence Artifacts in this Folder

| File Name | Description | Verification Method |
| :--- | :--- | :--- |
| [`benchmark.py`](benchmark.py) | Standalone benchmarking script adhering to 25 warm-up / 120 timed run protocol | Automated benchmark harness |
| [`benchmark_results.csv`](benchmark_results.csv) | Master summary CSV of all measured hardware metrics | Extracted during benchmark |
| [`raw_latency_results.csv`](raw_latency_results.csv) | Complete 120-iteration per-frame raw latency dataset | Timestamped telemetry |
| [`benchmark_report.md`](benchmark_report.md) | In-depth engineering report analyzing latency, FPS, and utilization | Statistical breakdown |
| [`device_information.txt`](device_information.txt) | Detailed host CPU, GPU, memory, and runtime environment specifications | WMI & system discovery |

---

## 5. Technical Insights & Edge AI Observations

1. **Model Compression:** Quantization achieved a **66.54% reduction** in storage and bandwidth footprint ($1.443\text{ MB} \to 0.483\text{ MB}$).
2. **RAM Footprint:** Resident memory decreased from $149.78\text{ MB}$ to $126.94\text{ MB}$ on CPU execution.
3. **Hardware Acceleration:** Offloading the neural network to the dedicated **NVIDIA GeForce RTX 5070** via DirectX 12 DirectML reduced CPU utilization from $\approx 51\%$ down to $\approx 9\%$, while delivering sub-1.5 ms inference latency ($722\text{ to }1,384\text{ FPS}$).

---

## 6. Reproduction Commands

```powershell
# 1. Activate isolated virtual environment
cd "C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1"
.\.venv\Scripts\Activate.ps1

# 2. Run the fair benchmark suite (25 warm-ups, 120 timed runs)
python 10_fair_benchmark/benchmark.py
```

