"""
YOLO Enclosure Detector Module
Multi-backend inference engine supporting ONNX Runtime (DirectML / CPU) and OpenCV DNN.
Target Class: 0 ('enclosure') - Rhino Motion RMCS-610 digital speed controller enclosure.
Config-driven model switching supporting both Baseline (FP32) and Optimized (INT8) checkpoints.
"""

import os
import json
import cv2
import numpy as np

class EnclosureDetector:
    def __init__(self, model_path=None, conf_thres=None, iou_thres=None, input_size=None, use_gpu=True):
        self.classes = {0: "enclosure"}
        self.backend = None
        self.session = None
        self.net = None
        self.active_provider = None
        self.model_profile = "Unknown"
        self.use_gpu = use_gpu
        
        # Load from config.json if parameters are omitted or defaults needed
        config = self._load_config()
        cfg_runtime = config.get("runtime", {})
        
        self.conf_thres = conf_thres if conf_thres is not None else cfg_runtime.get("confidence_threshold", 0.30)
        self.iou_thres = iou_thres if iou_thres is not None else cfg_runtime.get("iou_threshold", 0.45)
        self.input_size = tuple(input_size) if input_size is not None else tuple(cfg_runtime.get("input_resolution", [640, 640]))
        
        # Resolve model path
        if model_path is None:
            active_key = config.get("active_model", "optimized")
            model_info = config.get("models", {}).get(active_key, {})
            model_path = model_info.get("path", "models/enclosure_detector_int8.onnx")
            self.model_profile = model_info.get("format", active_key)
        elif "int8" in model_path.lower():
            self.model_profile = "ONNX INT8 (Optimized)"
        elif "base" in model_path.lower() or "export" in model_path.lower():
            self.model_profile = "ONNX FP32 (Base)"
            
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"[!] Warning: Specified model path does not exist: {model_path}")

    def _load_config(self):
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_model(self, model_path):
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Check available execution providers
                available = ort.get_available_providers()
                providers = ['CPUExecutionProvider']
                if self.use_gpu and 'DmlExecutionProvider' in available:
                    providers.insert(0, 'DmlExecutionProvider')
                elif self.use_gpu and 'CUDAExecutionProvider' in available:
                    providers.insert(0, 'CUDAExecutionProvider')
                    
                self.session = ort.InferenceSession(model_path, opts, providers=providers)
                self.backend = "onnxruntime"
                self.active_provider = self.session.get_providers()[0]
                self.input_name = self.session.get_inputs()[0].name
                print(f"[+] Loaded ONNX model via onnxruntime ({self.active_provider}): {model_path} [{self.model_profile}]")
                return
            except Exception as e:
                print(f"[!] ONNX Runtime init failed ({e}), falling back to cv2.dnn...")
                
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "opencv_dnn"
                self.active_provider = "OpenCV_DNN_CPU"
                print(f"[+] Loaded ONNX model via cv2.dnn: {model_path}")
                return
            except Exception as e:
                print(f"[-] Failed loading ONNX with cv2.dnn: {e}")
                
        self.backend = None
        print(f"[!] Warning: No active detector backend loaded for: {model_path}")

    def is_loaded(self):
        return self.backend is not None

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dw / 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def detect(self, frame):
        """
        Runs object detection on the input frame.
        Returns: list of detections:
        [{'box': [x1, y1, x2, y2], 'class_id': 0, 'class_name': 'enclosure', 'confidence': 0.92}]
        """
        if not self.is_loaded() or frame is None:
            return []
            
        orig_h, orig_w = frame.shape[:2]
        input_img, ratio, (pad_w, pad_h) = self._letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(input_img, 1.0 / 255.0, self.input_size, swapRB=True, crop=False)
        
        if self.backend == "onnxruntime":
            outputs = self.session.run(None, {self.input_name: blob})[0]
        elif self.backend == "opencv_dnn":
            self.net.setInput(blob)
            outputs = self.net.forward()
        else:
            return []
            
        # YOLOv8/v11 output shape: (1, 5, 8400)
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
                    "class_id": 0,
                    "class_name": "enclosure",
                    "confidence": round(confidences[idx], 4)
                })
        return detections
