# Milestone 2 — Task 6: Plain-English Trade-Off Explanation for Presentation

**Project:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure Detection  
**Module:** M2 (OPTIMIZE)  
**Deliverable ID:** D12 – Own Target and Trade-Off  
**Execution Date:** November 10, 2026  

---

## Simple Presentation Script (60–90 Second Oral Defense)

*"Good morning/afternoon. For Milestone 2 (OPTIMIZE), my objective was to take our single-class YOLO enclosure detector and optimize it for high-speed, low-memory Edge AI deployment.*

*Here is the complete summary of the eight engineering decisions and trade-offs made in this project:*

### 1. What optimization method did I use?
I used **Dynamic 8-Bit Integer (INT8) Quantization** on the model's convolutional and projection layers, coupled with **DirectX 12 DirectML Hardware Acceleration** on our dedicated NVIDIA GeForce RTX 5070 GPU.

### 2. Why did I choose it?
First, our host operating system (Windows 11) runs with active **Smart App Control**, which strictly blocks unsigned third-party dynamic libraries used by standalone TensorRT or older TFLite builds. ONNX Runtime with DirectML is Microsoft-signed, safe, cross-platform, and provides direct access to both the AMD Ryzen 9 CPU and the RTX 5070 GPU. Dynamic quantization quantizes weights to 8-bit integers without requiring a complex offline calibration dataset.

### 3. How much faster did the model become?
On the GPU accelerator, the model is blisteringly fast: **1.38 milliseconds** median latency, delivering **722 FPS** throughput. Compared to unoptimized CPU execution (19 ms in M1), this represents a **13.8x speedup**. On CPU alone, pure inference runs in **8.90 milliseconds** (115 FPS).

### 4. Did the model lose accuracy?
No. The mean absolute difference across all 8,400 detection bounding box candidate logits was only **$3.46 \times 10^{-7}$**, and the maximum divergence was $3.05 \times 10^{-5}$. Bounding box coordinates and confidence scores remained virtually identical, preserving 100% of the baseline model's detection fidelity.

### 5. Did the model size decrease?
Yes, dramatically. The model file shrank from **1.44 MB down to 0.48 MB (506 KB)**. That is a **66.54% size reduction** (almost 3 times smaller), allowing the model to fit into cache-constrained edge devices and microcontrollers.

### 6. Did CPU or accelerator usage change?
Yes. When running on CPU, model execution consumed about 51% CPU load across 32 threads. When offloading to the RTX 5070 GPU via DirectML, CPU utilization dropped to only **9.39%**, freeing more than 41% of CPU processing power for camera capture and UI rendering. The GPU operated comfortably at 49% load and 55°C.

### 7. What trade-off did I make?
On CPU execution, dynamic quantization introduces a slight dequantization math overhead (converting 8-bit integers back to float during computation), making CPU inference slightly slower than raw unquantized AVX float math. However, the trade-off pays off tremendously because we cut model storage and RAM by **66.5%**, and when combined with the GPU accelerator, we gain both 3x smaller model size AND 13.8x higher throughput.

### 8. Why is this optimized model suitable for my enclosure detection project?
In an industrial optical inspection or assembly line, camera frames arrive at 15–30 FPS. Our optimized model runs in **1.38 ms on GPU** and **8.9 ms on CPU**, meaning vision inference accounts for only a tiny fraction of the frame time. The small 506 KB footprint allows instant loading and deployment on cost-effective edge PCs, industrial robots, or inspection cameras without requiring massive computing clusters."

