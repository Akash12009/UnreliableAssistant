"""Create INT8 ONNX variants (CPU path, ADR-002). Skips existing files unless --force.
Acceptance (ADR-002): nDCG@4 within 0.01 of FP32 -> `python -m eval.ir_bench --quant fp32` vs default. Measured: 0.9381 vs 0.9370."""
import sys
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
force = "--force" in sys.argv
for name in ("bge-small-en-v1.5", "ms-marco-MiniLM-L-6-v2"):
    src, dst = ROOT / "models" / name / "model.onnx", ROOT / "models" / name / "model_int8.onnx"
    if src.exists() and (force or not dst.exists()):
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        print("quantized", dst.name, "for", name)
    else:
        print("skip", name)
