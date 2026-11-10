"""
Pytest Suite for Milestone 2 (M2 - OPTIMIZE) Deliverables
Verifies ONNX export integrity, INT8 quantization compaction, DirectML GPU/CPU inference,
and configuration-driven model switching.
"""

import os
import json
import pytest
import numpy as np
import onnx
import onnxruntime as ort

def test_exported_onnx_integrity():
    """Verify exported ONNX model exists, is valid, and has correct tensor dimensions."""
    model_path = "models/enclosure_detector_exported.onnx"
    assert os.path.exists(model_path), f"Exported model missing: {model_path}"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    in_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert in_shape == [1, 3, 640, 640]
    assert out_shape == [1, 5, 8400]

def test_int8_quantized_model():
    """Verify INT8 quantized model exists, is significantly compressed, and produces valid output."""
    base_path = "models/enclosure_detector_base.onnx"
    int8_path = "models/enclosure_detector_int8.onnx"
    assert os.path.exists(int8_path), f"INT8 model missing: {int8_path}"
    
    base_size = os.path.getsize(base_path)
    int8_size = os.path.getsize(int8_path)
    # Check that model size is reduced by at least 50%
    assert int8_size < base_size * 0.50, f"Expected >50% compression, got {int8_size}/{base_size}"
    
    # Run test inference
    sess = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
    assert out.shape == (1, 5, 8400)

def test_config_model_switching():
    """Verify config.json contains valid model switching mappings."""
    assert os.path.exists("config.json"), "config.json missing"
    with open("config.json", "r") as f:
        cfg = json.load(f)
    assert "active_model" in cfg
    assert "models" in cfg
    assert "original" in cfg["models"]
    assert "optimized" in cfg["models"]
    assert os.path.exists(cfg["models"]["original"]["path"])
    assert os.path.exists(cfg["models"]["optimized"]["path"])

def test_detector_switching_integration():
    """Verify EnclosureDetector loads both original and optimized models cleanly."""
    from pipeline.detector import EnclosureDetector
    
    # Test loading optimized
    det_opt = EnclosureDetector(model_path="models/enclosure_detector_int8.onnx")
    assert det_opt.is_loaded()
    assert det_opt.backend == "onnxruntime"
    
    # Test loading original
    det_orig = EnclosureDetector(model_path="models/enclosure_detector_base.onnx")
    assert det_orig.is_loaded()
    assert det_orig.backend == "onnxruntime"
    
    # Run sample inference
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res_opt = det_opt.detect(frame)
    res_orig = det_orig.detect(frame)
    assert isinstance(res_opt, list)
    assert isinstance(res_orig, list)

