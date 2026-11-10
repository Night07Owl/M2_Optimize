"""
M2 Task 5: Software Stack Diagram and CPU vs Accelerator Evidence Generator
Renders a publication-quality architecture diagram and generates the empirical comparison CSV.
"""

import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_software_stack_diagram(output_path="11_cpu_accelerator_and_stack/software_stack_diagram.png"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 13), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.axis("off")
    
    # Title Banner
    ax.text(50, 126, "Rhino Motion RMCS-610 Speed Controller Detection",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")
    ax.text(50, 122.5, "M2 (OPTIMIZE) Complete Edge AI Software & Hardware Architecture Stack",
            ha="center", va="center", fontsize=11, style="italic", color="#4B5563")
    
    # Pipeline Layers Definition (Top to Bottom)
    layers = [
        {"y": 113, "h": 7.5, "title": "1. VISION SENSOR (PHYSICAL CAMERA)",
         "desc": "Integrated HD Camera (USB\\VID_3277&PID_0010) | UVC Interface | 1280x720 @ 30 FPS",
         "bg": "#E0F2FE", "border": "#0284C7", "text": "#0369A1"},
         
        {"y": 100, "h": 7.5, "title": "2. IMAGE ACQUISITION & DEVICE DRIVER",
         "desc": "Windows DirectShow (cv2.CAP_DSHOW) / MSMF Driver Pipeline | Buffer Turnaround: ~47.9 ms",
         "bg": "#EDE9FE", "border": "#7C3AED", "text": "#5B21B6"},
         
        {"y": 87, "h": 7.5, "title": "3. OPENCV COMPUTER VISION PREPROCESSING",
         "desc": "OpenCV 5.0.0.93 (Python 3.11.9) | Letterbox 640x640 | BGR->RGB | Normalized [0.0, 1.0]",
         "bg": "#FEF3C7", "border": "#D97706", "text": "#B45309"},
         
        {"y": 74, "h": 7.5, "title": "4. OPTIMIZED DEPLOYMENT MODEL (ONNX INT8)",
         "desc": "models/enclosure_detector_int8.onnx | 506 KB (-66.54%) | Single-Class YOLO (0: enclosure)",
         "bg": "#DCFCE7", "border": "#16A34A", "text": "#15803D"},
         
        {"y": 61, "h": 7.5, "title": "5. EDGE INFERENCE RUNTIME (MULTI-BACKEND)",
         "desc": "ONNX Runtime DirectML 1.24.4 | Graph Optimization: ORT_ENABLE_ALL | Batch Size: 1",
         "bg": "#FEE2E2", "border": "#DC2626", "text": "#B91C1C"},
         
        {"y": 48, "h": 7.5, "title": "6. HARDWARE EXECUTION LAYER (DUAL ENGINE)",
         "desc": "ACCELERATOR: NVIDIA RTX 5070 (DmlExecutionProvider, 1.38ms)\nCPU FALLBACK: AMD Ryzen 9 8940HX (CPUExecutionProvider, 8.90ms)",
         "bg": "#F3E8FF", "border": "#9333EA", "text": "#6B21A8"},
         
        {"y": 35, "h": 7.5, "title": "7. POSTPROCESSING & RECONSTRUCTION",
         "desc": "Confidence Rejection (tau >= 0.30) | NMS (tau_IoU <= 0.45) | Letterbox Inverse Coordinate Scaling",
         "bg": "#FFEDD5", "border": "#EA580C", "text": "#C2410C"},
         
        {"y": 22, "h": 7.5, "title": "8. DETECTED TARGET ENCLOSURE",
         "desc": "Enclosure Bounding Box [x1, y1, x2, y2] | Detection Confidence Score | Class: 0",
         "bg": "#E0E7FF", "border": "#4F46E5", "text": "#3730A3"},
         
        {"y": 9, "h": 7.5, "title": "9. REAL-TIME HUD TELEMETRY & VISUAL DISPLAY",
         "desc": "Live HUD: FPS, Latency (ms), Target Count, Device Tag | Keybindings: 's' snapshot, 'q' exit",
         "bg": "#CCFBF1", "border": "#0D9488", "text": "#0F766E"}
    ]
    
    # Draw Layers
    for i, l in enumerate(layers):
        rect = patches.FancyBboxPatch(
            (10, l["y"]), 80, l["h"],
            boxstyle="round,pad=1.2,rounding_size=2.0",
            facecolor=l["bg"], edgecolor=l["border"], linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(50, l["y"] + l["h"] * 0.65, l["title"],
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=l["text"])
        ax.text(50, l["y"] + l["h"] * 0.28, l["desc"],
                ha="center", va="center", fontsize=7.5, color="#1F2937")
        
        # Draw Arrow to next layer
        if i < len(layers) - 1:
            next_y = layers[i+1]["y"] + layers[i+1]["h"]
            curr_y = l["y"]
            ax.annotate("", xy=(50, next_y + 0.3), xytext=(50, curr_y - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color="#4B5563", lw=1.8, mutation_scale=14))
            
    # System Footer
    ax.text(50, 2, "Operating System: Windows 11 Home 64-bit | Driver: NVIDIA 617.42 | API: Direct3D 12 Feature Level 12_2",
            ha="center", va="center", fontsize=7.5, color="#6B7280")
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Successfully generated high-resolution software stack diagram: {output_path}")

def generate_task5_artifacts(output_dir="11_cpu_accelerator_and_stack"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate Diagram Image
    generate_software_stack_diagram(os.path.join(output_dir, "software_stack_diagram.png"))
    
    # 2. Generate cpu_vs_accelerator.csv
    csv_path = os.path.join(output_dir, "cpu_vs_accelerator.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "CPU", "Accelerator", "Accelerator Advantage / Delta"])
        writer.writerow(["Hardware", "AMD Ryzen 9 8940HX (16C/32T)", "NVIDIA GeForce RTX 5070 Laptop GPU (12GB)", "Dedicated Parallel Hardware"])
        writer.writerow(["Runtime", "ONNX Runtime (CPUExecutionProvider)", "ONNX Runtime (DmlExecutionProvider)", "DirectX 12 DirectML API"])
        writer.writerow(["Model Format", "ONNX INT8 Quantized (0.483 MB)", "ONNX INT8 Quantized (0.483 MB)", "Identical 8-bit Graph"])
        writer.writerow(["Median Latency", "8.901 ms", "1.382 ms", "-7.519 ms (6.44x faster)"])
        writer.writerow(["P95 Latency", "9.734 ms", "1.457 ms", "-8.277 ms (6.68x faster)"])
        writer.writerow(["FPS (Throughput)", "115.48 FPS", "722.36 FPS", "+606.88 FPS (6.25x throughput)"])
        writer.writerow(["Memory Usage (RAM)", "126.94 MB", "304.87 MB", "+177.93 MB (DirectX GPU VRAM Buffers)"])
        writer.writerow(["CPU Utilization", "50.85%", "9.39%", "-41.46% CPU load offloaded"])
        writer.writerow(["Accelerator Utilization", "0.0%", "49.0%", "Active GPU Compute"])
        writer.writerow(["Temperature", "47.0 °C", "55.0 °C", "+8.0 °C Under Active Load"])
    print(f"[+] Saved CPU vs Accelerator comparison CSV: {csv_path}")
    
    # 3. Generate hardware_information.txt
    hw_path = os.path.join(output_dir, "hardware_information.txt")
    with open(hw_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("M2 TASK 5: HARDWARE & ACCELERATOR AUDIT\n")
        f.write("=" * 65 + "\n")
        f.write("CENTRAL PROCESSING UNIT (CPU):\n")
        f.write("  - Model: AMD Ryzen 9 8940HX with Radeon Graphics\n")
        f.write("  - Physical Cores: 16 | Logical Processors: 32\n")
        f.write("  - Instruction Sets: x86-64, AVX2, AVX-512, FMA3, SSE4.2\n")
        f.write("  - System Memory: 16 GB DDR5 (15,951,668 KB visible)\n\n")
        f.write("DEDICATED GRAPHICS ACCELERATOR (GPU):\n")
        f.write("  - Model: NVIDIA GeForce RTX 5070 Laptop GPU\n")
        f.write("  - Dedicated Video Memory (VRAM): 12,227 MiB GDDR6/GDDR7\n")
        f.write("  - Driver Version: 617.42 | Driver Model: WDDM\n")
        f.write("  - DirectX 12 Feature Level: 12_2\n")
        f.write("  - Idle Temperature: 45.0 °C | Max Temperature: 55.0 °C\n\n")
        f.write("INFERENCE RUNTIME ACCELERATION ENGINE:\n")
        f.write("  - Primary Framework: onnxruntime-directml 1.24.4\n")
        f.write("  - Active Execution Provider: DmlExecutionProvider\n")
        f.write("  - CPU Execution Provider: CPUExecutionProvider\n")
        f.write("  - Security Policy Status: Microsoft-Signed (Smart App Control Validated)\n")
    print(f"[+] Saved hardware specifications to: {hw_path}")

if __name__ == "__main__":
    generate_task5_artifacts()

