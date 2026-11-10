"""
Main Entrypoint for RMCS-610 Speed Controller Detection System (M2)
Supports config-driven model selection, live camera inference, and benchmark routines.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

from pipeline.opencv_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Rhino Motion RMCS-610 Edge AI Vision System")
    parser.add_argument("--mode", type=str, default="B", choices=["A", "B"], help="Mode A (Passthrough) or B (Detection)")
    parser.add_argument("--model", type=str, default="optimized", choices=["original", "optimized"],
                        help="Model to run: 'original' (FP32) or 'optimized' (INT8)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--cpu-only", action="store_true", help="Run strictly on CPU without GPU acceleration")
    parser.add_argument("--headless", action="store_true", help="Headless execution mode")
    parser.add_argument("--frames", type=int, default=None, help="Frame count limit")
    args = parser.parse_args()
    
    model_alias = "models/enclosure_detector_int8.onnx" if args.model == "optimized" else "models/enclosure_detector_base.onnx"
    
    run_pipeline(
        mode=args.mode,
        model_path=model_alias,
        cam_index=args.cam,
        use_gpu=not args.cpu_only,
        headless=args.headless,
        max_frames=args.frames
    )

if __name__ == "__main__":
    main()

