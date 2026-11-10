"""
Real-Time OpenCV Computer Vision Application
Implements Mode A (Camera Passthrough) and Mode B (YOLO Enclosure Detection)
Displays live video feed with HUD telemetry, detection overlays, and model switching support.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import argparse
import json
import cv2

from pipeline.performance_monitor import PerformanceMonitor
from pipeline.visualization import draw_hud, draw_detections
from pipeline.detector import EnclosureDetector

def run_pipeline(mode="B", model_path=None, config_path="config.json", cam_index=0,
                 width=1280, height=720, use_gpu=True, headless=False,
                 max_frames=None, save_screenshot=None):
    """
    Execute real-time computer vision pipeline.
    mode: 'A' (Passthrough) or 'B' (YOLO Detection)
    """
    mode = mode.upper()
    print("=" * 65)
    print(f"LAUNCHING OPENCV PIPELINE - MODE {mode}")
    print(f"Target Object: Rhino Motion RMCS-610 Enclosure (Class ID: 0)")
    
    # Resolve model alias if provided
    if model_path:
        if model_path.lower() == "original":
            model_path = "models/enclosure_detector_base.onnx"
        elif model_path.lower() == "optimized":
            model_path = "models/enclosure_detector_int8.onnx"
            
    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[!] Note: Could not parse {config_path}: {e}")
            
    if model_path is None and mode == "B":
        active_key = cfg.get("active_model", "optimized")
        model_path = cfg.get("models", {}).get(active_key, {}).get("path", "models/enclosure_detector_int8.onnx")
        
    print(f"Model Path: {model_path if mode == 'B' else 'None (Passthrough)'}")
    print("=" * 65)
    
    # Initialize detector if in Mode B
    detector = None
    if mode == "B":
        if not model_path or not os.path.exists(model_path):
            print(f"[-] Error: Model file '{model_path}' not found.")
            print("    Please run: python 08_convert_and_run_live/conversion_script.py")
            return False
        detector = EnclosureDetector(model_path=model_path, use_gpu=use_gpu)
        
    # Initialize Camera
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    
    if not cap.isOpened():
        print(f"[-] Error: Unable to connect to camera device index {cam_index}.")
        print("    Please check that your webcam is connected and enabled.")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Camera connected: {actual_w}x{actual_h}")
    
    perf = PerformanceMonitor(window_size=30)
    window_name = f"Edge AI RMCS-610 Detector - MODE {mode}"
    
    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1280, actual_w), min(720, actual_h))
        except Exception as e:
            print(f"[!] GUI window creation failed ({e}). Defaulting to headless.")
            headless = True
            
    frame_idx = 0
    saved_screenshot_done = False
    
    try:
        while True:
            # 1. Frame Capture
            t_cap_start = time.perf_counter()
            ret, frame = cap.read()
            t_cap_end = time.perf_counter()
            
            if not ret or frame is None:
                print("[-] Video stream disconnected or frame grab failed.")
                break
                
            cap_duration_ms = (t_cap_end - t_cap_start) * 1000.0
            perf.record_capture_latency(cap_duration_ms)
            
            # 2. Detect Enclosure
            detections = []
            inf_duration_ms = 0.0
            
            t_proc_start = time.perf_counter()
            if mode == "B" and detector is not None and detector.is_loaded():
                detections = detector.detect(frame)
            t_proc_end = time.perf_counter()
            
            inf_duration_ms = (t_proc_end - t_proc_start) * 1000.0
            perf.record_inference_latency(inf_duration_ms)
            
            total_latency_ms = cap_duration_ms + inf_duration_ms
            perf.record_total_latency(total_latency_ms)
            perf.tick_frame()
            
            # 3. Draw Overlays
            display_frame = frame.copy()
            if detections:
                display_frame = draw_detections(display_frame, detections)
                
            display_frame = draw_hud(
                display_frame,
                mode=mode,
                fps=perf.get_fps(),
                latency_ms=total_latency_ms,
                enclosure_count=len(detections),
                capture_ms=cap_duration_ms
            )
            
            # 4. Save evidence snapshot if requested
            if save_screenshot and not saved_screenshot_done and frame_idx >= 5:
                os.makedirs(os.path.dirname(os.path.abspath(save_screenshot)), exist_ok=True)
                cv2.imwrite(save_screenshot, display_frame)
                print(f"[+] Saved pipeline evidence screenshot to: {save_screenshot}")
                saved_screenshot_done = True
                
            # 5. Interactive controls
            if not headless:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[+] Clean exit requested by user.")
                    break
                elif key == ord('s'):
                    snap_path = f"snapshot_mode_{mode.lower()}_{int(time.time())}.jpg"
                    cv2.imwrite(snap_path, display_frame)
                    print(f"[+] Snapshot saved: {snap_path}")
                    
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                print(f"[+] Reached frame budget target: {max_frames} frames.")
                break
                
    except KeyboardInterrupt:
        print("\n[+] Cleanly terminating on keyboard interrupt.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
            
    summary = perf.get_summary()
    print("\n--- Pipeline Execution Summary ---")
    print(json.dumps(summary, indent=2))
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time OpenCV Enclosure Detection Application")
    parser.add_argument("--mode", type=str, default="B", choices=["A", "B", "a", "b"],
                        help="Pipeline Mode: A (Passthrough) or B (YOLO Detection)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO model or alias: 'original' / 'optimized'")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable DirectML GPU acceleration")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU inference")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--frames", type=int, default=None, help="Maximum frames to run")
    parser.add_argument("--save-screenshot", type=str, default=None, help="Save an annotated screenshot")
    
    args = parser.parse_args()
    use_gpu = not args.cpu_only
    
    run_pipeline(
        mode=args.mode,
        model_path=args.model,
        config_path=args.config,
        cam_index=args.cam,
        width=args.width,
        height=args.height,
        use_gpu=use_gpu,
        headless=args.headless,
        max_frames=args.frames,
        save_screenshot=args.save_screenshot
    )
