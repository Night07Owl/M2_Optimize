"""
M2 Task 2: Real-Time Live Camera Inference Engine with Optimized INT8 Model
Runs real-time camera stream, preprocesses frames, performs INT8 inference,
detects the assembled RMCS-610 enclosure, and renders a live telemetry HUD.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

class LiveEnclosureDetector:
    def __init__(self, model_path="models/enclosure_detector_int8.onnx",
                 conf_thres=0.30, iou_thres=0.45, use_gpu=False):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = (640, 640)
        self.class_name = "enclosure"
        
        # Configure execution provider
        providers = ['CPUExecutionProvider']
        if use_gpu and 'DmlExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'DmlExecutionProvider')
            
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"[+] Initializing ONNX Runtime Session for: {model_path}")
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        print(f"[+] Model active on provider: {self.active_provider}")

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Preprocess
        t_pre0 = time.perf_counter()
        boxed_img, ratio, (pad_w, pad_h) = self._letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(boxed_img, 1.0 / 255.0, self.input_size, swapRB=True, crop=False)
        t_pre = (time.perf_counter() - t_pre0) * 1000.0
        
        # 2. INT8 Inference
        t_inf0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})[0]
        t_inf = (time.perf_counter() - t_inf0) * 1000.0
        
        # 3. Postprocess & NMS
        t_post0 = time.perf_counter()
        if outputs.ndim == 3 and outputs.shape[1] < outputs.shape[2]:
            outputs = np.transpose(outputs[0], (1, 0))  # (8400, 5)
        elif outputs.ndim == 3:
            outputs = outputs[0]
            
        boxes = []
        confidences = []
        
        for row in outputs:
            score = float(row[4])
            if score >= self.conf_thres:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = (cx - w / 2 - pad_w) / ratio
                y1 = (cy - h / 2 - pad_h) / ratio
                x2 = (cx + w / 2 - pad_w) / ratio
                y2 = (cy + h / 2 - pad_h) / ratio
                
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(score)
                
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                detections.append({
                    "box": [bx, by, bx + bw, by + bh],
                    "class_name": self.class_name,
                    "confidence": confidences[idx]
                })
        t_post = (time.perf_counter() - t_post0) * 1000.0
        
        return detections, t_pre, t_inf, t_post

def draw_hud(frame, detections, fps, t_inf, provider, model_name="INT8 Quantized"):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label banner
        label = f"enclosure: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    # Semi-transparent HUD Header Bar
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status Telemetry
    cv2.putText(frame, f"RMCS-610 ENCLOSURE DETECTOR | M2 LIVE", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Model: {model_name} | Backend: {provider}", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:5.1f} | Latency: {t_inf:5.2f} ms | Enclosures: {len(detections)}", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0) if len(detections) > 0 else (100, 255, 100), 2)
    
    # Controls hint
    cv2.putText(frame, "Press 's' snapshot | 'q' quit", (w - 230, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame

def run_live(cam_index=0, model_path="models/enclosure_detector_int8.onnx",
             use_gpu=False, max_frames=None, save_screenshot_path=None, show_window=True):
    print("=" * 70)
    print("M2 TASK 2: RUNNING LIVE CAMERA INFERENCE")
    print(f"Camera Device Index: {cam_index}")
    print(f"Model Checkpoint: {model_path}")
    print(f"Hardware Acceleration (GPU): {use_gpu}")
    print("=" * 70)
    
    detector = LiveEnclosureDetector(model_path=model_path, use_gpu=use_gpu)
    
    # Open Camera via DirectShow
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[-] ERROR: Failed to open camera device index {cam_index}.")
        print("    Troubleshooting steps:")
        print("    1. Verify camera privacy settings in Windows 11 Settings.")
        print("    2. Ensure no other application (e.g., Zoom, Teams) has exclusive lock.")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Camera opened successfully: {actual_w}x{actual_h}")
    
    frame_count = 0
    t_start = time.perf_counter()
    fps_ema = 0.0
    latencies = []
    
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[-] Failed to grab frame from camera stream.")
                break
                
            frame_count += 1
            
            # Run detection
            detections, t_pre, t_inf, t_post = detector.process_frame(frame)
            latencies.append(t_inf)
            
            # Calculate instantaneous FPS
            dt = time.perf_counter() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst_fps if frame_count == 1 else (0.9 * fps_ema + 0.1 * inst_fps)
            
            # Render HUD
            rendered = draw_hud(frame, detections, fps_ema, t_inf, detector.active_provider)
            
            # Save screenshot if requested
            if save_screenshot_path and frame_count == (max_frames // 2 if max_frames else 30):
                os.makedirs(os.path.dirname(os.path.abspath(save_screenshot_path)), exist_ok=True)
                cv2.imwrite(save_screenshot_path, rendered)
                print(f"[+] Saved live camera frame snapshot to: {save_screenshot_path}")
                
            if show_window:
                cv2.imshow("RMCS-610 Live Enclosure Detector", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[+] User exit requested via keyboard.")
                    break
                elif key == ord('s'):
                    snap_path = f"08_convert_and_run_live/screenshots/manual_snap_{int(time.time())}.jpg"
                    cv2.imwrite(snap_path, rendered)
                    print(f"[+] Manual snapshot saved: {snap_path}")
                    
            if max_frames and frame_count >= max_frames:
                print(f"[+] Reached frame budget target: {max_frames} frames.")
                break
                
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
            
    t_total = time.perf_counter() - t_start
    avg_fps = frame_count / t_total if t_total > 0 else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    
    print("=" * 70)
    print("LIVE INFERENCE STREAM SUMMARY:")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Elapsed Time: {t_total:.2f} seconds")
    print(f"Sustained Pipeline FPS: {avg_fps:.2f} FPS")
    print(f"Median Inference Latency: {med_lat:.2f} ms")
    print(f"P95 Inference Latency: {p95_lat:.2f} ms")
    print(f"Average Inference Latency: {avg_lat:.2f} ms")
    print("=" * 70)
    return {
        "frames": frame_count,
        "elapsed_sec": t_total,
        "avg_fps": avg_fps,
        "median_latency_ms": med_lat,
        "p95_latency_ms": p95_lat,
        "provider": detector.active_provider
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live RMCS-610 INT8 Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=str, default="models/enclosure_detector_int8.onnx")
    parser.add_argument("--gpu", action="store_true", help="Use DirectML GPU accelerator")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI")
    parser.add_argument("--snapshot", type=str, default="08_convert_and_run_live/screenshots/live_inference_capture.jpg")
    args = parser.parse_args()
    
    run_live(
        cam_index=args.camera,
        model_path=args.model,
        use_gpu=args.gpu,
        max_frames=args.max_frames,
        save_screenshot_path=args.snapshot,
        show_window=not args.headless
    )

