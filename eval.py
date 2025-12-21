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
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd() # 현재 워킹디렉토리 경로 저장
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1
PIN_MEMORY = True
IMG_DIR = os.path.join(cwd, "data", "VOC", "images")
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")

test_transform = v2.Compose([
    v2.Resize((448, 448),),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def voc07_11pt_ap(prec, rec):
    prec = np.asarray(prec, dtype=np.float64)
    rec  = np.asarray(rec,  dtype=np.float64)

    if prec.size == 0 or rec.size == 0:
        return 0.0

    r_levels = np.linspace(0.0, 1.0, 11)
    p_interp = np.zeros_like(r_levels)

    for i, r in enumerate(r_levels):
        mask = rec >= r
        p_interp[i] = prec[mask].max() if mask.any() else 0.0

    return float(p_interp.mean())

def main():
    model = Yolov1Model(S = 7, B = 2, C = 20)
    checkpoint = torch.load(os.path.join(cwd, 'model', 'finetune-weight-third145.pth'), weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE, non_blocking=True)
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

    all_boxes = []
    GT_num_list = [0 for _ in range(20)]
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            pred = model(image).detach().cpu()
            result = NMS(pred, label)
            #print(result)
            #plotImage(image, result, label)

            all_boxes.append(result)
            for clsidx in range(20): # 모든 클래스별 Ground Truth 의 개수 세기 (모든 평가 이미지에 대해 누적)
                GT_num_list[clsidx] += int(label[:, clsidx:clsidx+1, :, :].sum().item())


    table = []
    for clsidx in range(20):
        APtable_in_a_class = []
        for image in all_boxes:
            for box in image:
                if box[0] == clsidx: # 0번째, 1번째, ... 19번재 클래스까지 각가 반복하면서 각 클래스에 해당하는 deteciton들을 따로 모은다
                    APtable_in_a_class.append(box.copy())

        table.append(APtable_in_a_class.copy())

    AllPrec, AllRecall = mAP(table, GT_num_list)
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()
    sum_ap = 0

    for cls in range(20):
        ax = axes[cls]

        rec = AllRecall[cls]
        prec = AllPrec[cls]
        ap11 = voc07_11pt_ap(prec, rec)
        sum_ap += ap11
        ax.plot(rec, prec, marker='o')
        ax.set_title(f"Class {classes[cls]} | AP(11pt)={ap11:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        # plot 안에도 텍스트로 표시(원치 않으면 이 두 줄 삭제)
        ax.text(0.02, 0.02, f"AP11={ap11:.3f}", transform=ax.transAxes,
                va='bottom', ha='left')

    plt.tight_layout()
    plt.show()

    print(f"mAP : {sum_ap/20})")

if __name__ == "__main__":
    main()