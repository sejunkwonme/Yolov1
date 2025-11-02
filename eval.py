import os
import torch
from torch.utils.data import (DataLoader, random_split, Subset)
from model import Yolov1Model
from dataset import VOCDataset
from loss import Yolov1DetectionLoss, Yolov1ClassificationLoss
from utils import (NMS, mAP, IoU, plotImage)
from tqdm import tqdm
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2

cwd = os.getcwd() # 현재 워킹디렉토리 경로 저장
# 학습에 쓰일 하이퍼파라미터
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
MOMENTUM = 0
NUM_EPOCHS = 50
NUM_WORKERS = 20
PIN_MEMORY = True
IMG_DIR = os.path.join(cwd, "data", "VOC", "images")
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")

test_transform = v2.Compose([
    v2.Resize((448, 448),),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def main():
    model = Yolov1Model(S = 7, B = 2, C = 20)
    checkpoint = torch.load(os.path.join(cwd, 'model', 'finetune-weight-135.pth'), weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model.to(DEVICE, non_blocking=True)
    example_input = torch.randn(1, 3, 448, 448).to(DEVICE)
    example_tuple = (example_input,)
    onnx_program = torch.onnx.export(model, example_tuple, dynamo=True)
    onnx_program.save("yolomodel.onnx")
    return
    test_csv_path = os.path.join(cwd, "data", "VOC", "2007test.csv")
    dataset = VOCDataset(test_csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,transform=test_transform,)

    generator = torch.Generator().manual_seed(100)

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model.eval()
    for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for images, labels in test_loader:
                images = images.to(DEVICE)
                preds = model(images)
                plotImage(images.to("cpu"), preds)
                all_preds.append(preds)
                all_targets.append(labels)

            # mAP 계산
            #mAP_result = mAP(all_preds, all_targets)
            #print(f"Epoch {epoch}: mAP={mAP:.4f}")


if __name__ == "__main__":
    main()