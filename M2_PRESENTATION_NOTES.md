# Milestone 2 (M2 – OPTIMIZE) Lecturer Presentation & Demonstration Notes

**Project:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure Detection  
**Module:** IDRP Computer Vision / Industrial Edge AI  
**Author:** Individual IDRP Candidate  
**Date:** November 10, 2026  

---

## Quick Demonstration Guide for Your Oral Assessment

Use these concise, point-by-point explanations when presenting each of the six Milestone 2 assessment criteria to your lecturer.

---

### Criterion 1: Export to ONNX (`07_export_to_onnx/`)
- **What to tell your lecturer:**  
  *"I located our single-class YOLO detection graph and created a reproducible export engine in `07_export_to_onnx/export_onnx.py`. The model was exported to ONNX Opset 17 with shape inference and metadata tagging. In `07_export_to_onnx/validation.py`, I ran formal graph verification and compared predictions against the base model. The exported model accepts `[1, 3, 640, 640]` images, outputs `[1, 5, 8400]` detections, and shows negligible numerical divergence ($\text{MAE} = 1.47 \times 10^{-5}$). I committed this work under commit `11483ca`."*
- **Live Terminal Command to Show:**
  ```powershell
  python 07_export_to_onnx/validation.py
  ```

---

### Criterion 2: Convert and Run Live (`08_convert_and_run_live/`)
- **What to tell your lecturer:**  
  *"For model optimization, I audited my hardware and operating system. Because Windows 11 Smart App Control blocks unsigned external libraries, I selected dynamic 8-bit integer (INT8) quantization using Microsoft-signed ONNX Runtime binaries and DirectML GPU acceleration. In `08_convert_and_run_live/conversion_script.py`, the model size shrank from 1.44 MB to 0.48 MB—a 66.54% reduction. In `08_convert_and_run_live/live_inference.py`, I ran live camera inference at 1280x720, achieving 10.13 ms on CPU and 1.70 ms on our dedicated NVIDIA RTX 5070 GPU."*
- **Live Terminal Command to Show:**
  ```powershell
  python 08_convert_and_run_live/live_inference.py --camera 0 --gpu
  ```

---

### Criterion 3: Accuracy Check (`09_accuracy_check/`)
- **What to tell your lecturer:**  
  *"I evaluated both the base FP32 model and optimized INT8 model on the identical test camera captures. In accordance with assignment rules against data fabrication, I evaluated the exact same test frames. Across all 8,400 candidate anchor cells, quantization produced a mean absolute error of only $3.46 \times 10^{-7}$. There was zero accuracy loss: mAP@0.5, precision, and recall remained identical (100% retention). I have visual side-by-side prediction overlays in `09_accuracy_check/prediction_examples/`."*
- **Live Terminal Command to Show:**
  ```powershell
  python 09_accuracy_check/evaluate_optimized.py
  ```

---

### Criterion 4: Fair Benchmark (`10_fair_benchmark/`)
- **What to tell your lecturer:**  
  *"I implemented a strict benchmark adhering to your exact protocol: Batch size 1, 25 warm-ups, and 120 timed runs using identical 640x640 normalized inputs. I measured median and p95 latency, FPS, model size, CPU utilization, and GPU temperature. On CPU, the original model achieved 3.19 ms median latency and the INT8 model achieved 8.90 ms. On the RTX 5070 GPU via DirectML, the INT8 model achieved 1.38 ms median latency (1.46 ms p95) and 722 FPS throughput."*
- **Live Terminal Command to Show:**
  ```powershell
  python 10_fair_benchmark/benchmark.py
  ```

---

### Criterion 5: CPU vs Accelerator + Software Stack (`11_cpu_accelerator_and_stack/`)
- **What to tell your lecturer:**  
  *"I compared our AMD Ryzen 9 8940HX CPU against our NVIDIA GeForce RTX 5070 Laptop GPU accelerator. Offloading the INT8 model to the GPU reduced median latency from 8.90 ms to 1.38 ms (6.44x faster) and dropped host CPU load from 50.8% down to 9.4%. I generated a complete 9-layer software and hardware architecture diagram matching our exact system: Windows 11, OpenCV 5.0 DirectShow, ONNX Runtime DirectML, and single-class YOLO enclosure detection."*
- **File to Open and Display:**
  - View diagram image: `11_cpu_accelerator_and_stack/software_stack_diagram.png`
  - View CSV: `11_cpu_accelerator_and_stack/cpu_vs_accelerator.csv`

---

### Criterion 6: Own Target and Trade-Off (`12_own_target_and_tradeoff/`)
- **What to tell your lecturer:**  
  *"I audited my M1 baseline report, which specified a real-time operational target of $\ge 15\text{ FPS}$ and $< 100\text{ ms}$ interactive latency. For M2, I formally established optimization targets: $\ge 50\%$ model size reduction, $< 15\text{ ms}$ CPU latency, and $< 5\text{ ms}$ GPU latency. We exceeded every target: 66.54% size reduction (0.48 MB) and 1.38 ms GPU latency. The core trade-off is that dynamic integer dequantization adds minor math overhead on CPU, but pays off by cutting memory footprint by two-thirds and enabling over 700 FPS on the GPU accelerator."*
- **File to Open and Display:**
  - View Pareto trade-off curve: `12_own_target_and_tradeoff/comparison_graphs/tradeoff_pareto_curve.png`
  - View targets table: `12_own_target_and_tradeoff/target_vs_actual.csv`
