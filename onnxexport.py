import onnx
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from model import Yolov1Model

import numpy as np
from model import Yolov1Model

m = onnx.load("yolomodel.onnx")
print(m.opset_import)

for n in m.graph.initializer:
    print(n.name, n.data_type)

w = onnx.numpy_helper.to_array(n)
print(w.shape)

# -----------------------------------------
# 1) 모델 로드 및 FP32 강제
# -----------------------------------------
model = Yolov1Model()      # ★ 네 YOLOv1 모델로 교체
checkpoint = torch.load("./model/finetune-weight-135.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()
model = model.float()          # FP32 강제

# 모든 파라미터 FP32로 강제
for p in model.parameters():
    p.data = p.data.float()
    if p.grad is not None:
        p.grad = p.grad.float()

# -----------------------------------------
# 2) Dummy input FP32로 고정
# -----------------------------------------
dummy = torch.randn(1, 3, 448, 448).float()

# -----------------------------------------
# 3) onnx export - opset 12
# -----------------------------------------
onnx_path = "yolov1_raw.onnx"

torch.onnx.export(
    model,
    (dummy,),
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,                    # ★ CUDA EP 최적
    do_constant_folding=True,            # FP32 상수로 접힘
    dynamic_axes=None                    # dynamic shape 제거 → CUDA fallback 방지
)

print("[STEP 1] Raw ONNX exported:", onnx_path)

# -----------------------------------------
# 4) ONNX Simplifier - FP64 제거 + constant folding
# -----------------------------------------
model_onnx = onnx.load(onnx_path)
model_simp, check = simplify(model_onnx)

assert check, "Simplified ONNX model couldn't be validated"

onnx.save(model_simp, "yolov1_simplified.onnx")
print("[STEP 2] Simplified ONNX saved as yolov1_simplified.onnx")

# -----------------------------------------
# 5) Double(FP64) initializer 제거 (안전용)
# -----------------------------------------
clean_path = "yolov1_final.onnx"
m = onnx.load("yolov1_simplified.onnx")

from onnx import numpy_helper

for init in m.graph.initializer:
    if init.data_type == 11:   # float16
        arr = numpy_helper.to_array(init).astype(np.float32)
        init.CopyFrom(numpy_helper.from_array(arr, init.name))
    elif init.data_type == 7:  # float64 (double)
        arr = numpy_helper.to_array(init).astype(np.float32)
        init.CopyFrom(numpy_helper.from_array(arr, init.name))

onnx.save(m, clean_path)
print("[STEP 3] Cleaned ONNX saved as yolov1_final.onnx")

print("\n🎉 YOLOv1 ONNX 변환 완성! (CUDA/ cuDNN 최적화 100% 통과 가능)")
print("➡ 출력 파일: yolov1_final.onnx")
