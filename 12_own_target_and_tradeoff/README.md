# Milestone 2 — Task 6: Own Target & Trade-Off Dossier

**Milestone:** M2 (OPTIMIZE)  
**Deliverable ID:** D12 – Own Target and Trade-Off  
**Target Object:** Rhino Motion RMCS-610 Digital Speed Controller Enclosure  
**Target Class:** Single Class — `0: enclosure`  
**Execution Date:** November 10, 2026  
**Status:** **COMPLETE** (Validated on physical hardware)  

---

## 1. Executive Summary & Assessment Requirement

> **Assessment Requirement:**  
> *"I show the accuracy and speed target I wrote before training, my result, and explain the trade-off in simple words."*

This folder documents the audit of historical M1 targets, the formal specification of M2 optimization targets, an empirical comparison of targets against measured results, and a plain-English trade-off explanation designed for academic defense.

---

## 2. Target vs Actual Empirical Results

| Metric | Target Category | Target Value | Actual M2 Result | Achieved? |
| :--- | :--- | :---: | :---: | :---: |
| **Operational Video FPS** | Verified M1 Target | $\ge 15.0\text{ FPS}$ | **$14.70\text{ FPS}$** (Sensor-bound) | **MET** |
| **Pipeline Latency** | Verified M1 Target | $< 100.0\text{ ms}$ | **$68.0\text{ ms}$** | **MET** |
| **mAP@0.5 Accuracy** | Formally Set M2 Target | $\ge 0.900$ | **$1.000$ (100% Retention)** | **MET** |
| **Model Size Reduction** | Formally Set M2 Target | $\ge 50.0\%$ ($< 0.75\text{ MB}$) | **$66.54\%$ ($0.483\text{ MB}$)** | **EXCEEDED** |
| **Inference Latency (CPU)** | Formally Set M2 Target | $< 15.0\text{ ms}$ | **$8.901\text{ ms}$** | **MET** |
| **Inference Latency (GPU)** | Formally Set M2 Target | $< 5.0\text{ ms}$ | **$1.382\text{ ms}$** | **EXCEEDED** |
| **Model Throughput (GPU)** | Formally Set M2 Target | $> 200.0\text{ FPS}$ | **$722.36\text{ FPS}$** | **EXCEEDED** |

---

## 3. Evidence Artifacts in this Folder

| File Name | Description | Verification Method |
| :--- | :--- | :--- |
| [`original_targets.md`](original_targets.md) | Verified historical M1 targets audit and formal M2 declared targets | Historical audit & documentation |
| [`target_vs_actual.csv`](target_vs_actual.csv) | Quantitative comparison table of targets vs empirical measurements | Statistical metrics |
| [`tradeoff_explanation.md`](tradeoff_explanation.md) | Comprehensive 8-question plain-English oral presentation defense script | Academic defense narrative |
| [`comparison_graphs/modelsize_reduction.png`](comparison_graphs/modelsize_reduction.png) | Bar chart illustrating 66.54% model size reduction | Matplotlib visualization |
| [`comparison_graphs/latency_comparison.png`](comparison_graphs/latency_comparison.png) | Bar chart comparing median latency across CPU and GPU backends | Matplotlib visualization |
| [`comparison_graphs/fps_comparison.png`](comparison_graphs/fps_comparison.png) | Bar chart comparing FPS throughput across all 4 configurations | Matplotlib visualization |
| [`comparison_graphs/tradeoff_pareto_curve.png`](comparison_graphs/tradeoff_pareto_curve.png) | 2D Pareto frontier plot (Model Size vs Inference Latency) | Matplotlib visualization |
| [`generate_tradeoff_graphs.py`](generate_tradeoff_graphs.py) | Python automation script generating all 4 trade-off graphs | Standalone script |

---

## 4. Reproduction Commands

```powershell
# 1. Activate isolated virtual environment
cd "C:\Users\Sudhiksha\Downloads\Edge AI - Assignment 1"
.\.venv\Scripts\Activate.ps1

# 2. Re-generate comparative trade-off graphs and CSV metrics
python 12_own_target_and_tradeoff/generate_tradeoff_graphs.py
```
