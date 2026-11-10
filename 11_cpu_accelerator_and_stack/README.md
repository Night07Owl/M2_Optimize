# Milestone 2 — Task 5: CPU vs Accelerator & Software Stack Dossier

**Milestone:** M2 (OPTIMIZE)  
**Deliverable ID:** D11 – CPU vs Accelerator & Software Stack  
**Target Object:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure  
**Target Class:** Single Class — `0: enclosure`  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Validated on physical hardware)  

---

## 1. Executive Summary & Assessment Requirement

> **Assessment Requirement:**  
> *"I show the CPU vs accelerator table and draw the software stack."*

This folder documents the empirical comparison between CPU inference (AMD Ryzen 9 8940HX) and hardware-accelerated inference (NVIDIA GeForce RTX 5070 Laptop GPU via DirectX 12 DirectML) and presents the end-to-end software and hardware pipeline architecture.

---

## 2. Evidence Artifacts in this Folder

| File Name | Description | Verification Method |
| :--- | :--- | :--- |
| [`cpu_vs_accelerator.csv`](cpu_vs_accelerator.csv) | Empirical comparison metrics table (Latency, FPS, Memory, Utilization) | Benchmark suite output |
| [`comparison_report.md`](comparison_report.md) | Comprehensive technical report analyzing CPU vs GPU tradeoffs | Empirical analysis |
| [`software_stack_diagram.png`](software_stack_diagram.png) | High-resolution publication-quality software and hardware stack diagram | Matplotlib rendering |
| [`software_stack_diagram.md`](software_stack_diagram.md) | Markdown & ASCII architectural diagram detailing all components | Structural specification |
| [`hardware_information.txt`](hardware_information.txt) | Complete host CPU, GPU, memory, and runtime environment specifications | System discovery |
| [`generate_stack_diagram.py`](generate_stack_diagram.py) | Python generator for diagram rendering and CSV synchronization | Automated generator |

---

## 3. Reproduction Commands

```powershell
# 1. Activate isolated virtual environment
cd "C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1"
.\.venv\Scripts\Activate.ps1

# 2. Re-generate architecture diagrams and comparison metrics
python 11_cpu_accelerator_and_stack/generate_stack_diagram.py
```

