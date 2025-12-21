import torch
import os

from model import Yolov1Model

cwd = os.getcwd()
PTH_PATH = os.path.join(cwd, 'model', 'finetune-weight-third145.pth')
ONNX_PATH = os.path.join(cwd, 'thirdmodel.onnx')

# 입력 크기 (필요하면 수정)
B, C, H, W = 1, 3, 448, 448

device = "cpu"

# 1) 모델 로드 (torch.save(model)로 저장된 경우에만 동작)
model = Yolov1Model(S=7, B=2, C=20).to(device=device)
model.load_state_dict(torch.load(PTH_PATH))
model.eval()

# 2) 더미 입력
dummy = torch.randn(B, C, H, W, device=device)

# 3) ONNX export
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

print("Exported:", ONNX_PATH)
