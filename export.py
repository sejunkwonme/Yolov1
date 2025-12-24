import os
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

from model import Yolov1Model

cwd = os.getcwd()
PTH_PATH = os.path.join(cwd, 'model', 'finetune-weight-third145.pth')
ONNX_PATH_BACKBONE = os.path.join(cwd, 'thirdmodel-backbone.onnx')
ONNX_PATH_HEAD     = os.path.join(cwd, 'thirdmodel-head.onnx')

ONNX_SIM_BACKBONE  = os.path.join(cwd, 'thirdmodel-backbone.sim.onnx')
ONNX_SIM_HEAD      = os.path.join(cwd, 'thirdmodel-head.sim.onnx')

device = "cpu"

model = Yolov1Model(S=7, B=2, C=20).to(device=device)
missing, unexpected = model.load_state_dict(torch.load(PTH_PATH, map_location=device), strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
model.eval()

dummy1 = torch.randn(1, 3, 448, 448, device=device)

backbone = nn.Sequential(model.yolomodel[0]).eval()

torch.onnx.export(
    backbone,
    (dummy1,),
    ONNX_PATH_BACKBONE,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    keep_initializers_as_inputs=False,
)

dummy2 = torch.randn(1, 1024, 14, 14, device=device)

head = nn.Sequential(model.yolomodel[1], model.yolomodel[2]).eval()

torch.onnx.export(
    head,
    (dummy2,),
    ONNX_PATH_HEAD,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
)

print("Export done.")

def simplify_onnx(in_path: str, out_path: str, input_shapes: dict):
    model_onnx = onnx.load(in_path)

    sim_model, ok = simplify(
        model_onnx,
        input_shapes=input_shapes,
        dynamic_input_shape=False,
        check_n=3,
    )

    if not ok:
        raise RuntimeError(f"onnxsim simplify failed: {in_path}")

    onnx.save(sim_model, out_path)
    print(f"Simplified saved: {out_path}")

simplify_onnx(
    ONNX_PATH_BACKBONE,
    ONNX_SIM_BACKBONE,
    input_shapes={"input": [1, 3, 448, 448]},
)

simplify_onnx(
    ONNX_PATH_HEAD,
    ONNX_SIM_HEAD,
    input_shapes={"input": [1, 1024, 14, 14]},
)
