import os
import torch
from torch.utils.data import (DataLoader, random_split, Subset)
from model import Yolov1Model
from dataset import VOCDataset
from loss import Yolov1DetectionLoss, Yolov1ClassificationLoss
from utils import (NMS, mAP, IoU, plotImage)
from tqdm import tqdm
from torchvision import datasets, transforms
from pathlib import Path
import torch.nn.init as init
import torch.nn as nn

cwd = os.getcwd() # 현재 워킹디렉토리 경로 저장
# 학습에 쓰일 하이퍼파라미터
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS = 135
NUM_WORKERS = 20
PIN_MEMORY = True
IMG_DIR = os.path.join(cwd, "data", "VOC", "images")
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")
warmup_epochs = 10

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    elif epoch < 75:
        return 1.0          # 1e-2 유지
    elif epoch < 105:
        return 0.1          # 1e-3 (1/10)
    else:
        return 0.01         # 1e-4 (1/100)

def init_weights_normal(m):
    import torch.nn.init as init
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def main():
    # 1️⃣ 모델 선언 (모듈 그대로)
    model = Yolov1Model(S=7, B=2, C=20, mode="finetune").to(DEVICE)

    # 2️⃣ checkpoint 로드
    checkpoint = torch.load(os.path.join(cwd, 'model', 'convert-weight-42.pth'), weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    """
    print(list(torch.load(os.path.join(cwd, 'model', 'pretrain-weight-42.pth')).keys())[:10])
    backbone_state_dict = {}
    for k, v in dic.items():
        if k.startswith("pretrainmodel.0.backbone20layers"):
            new_k = k.replace("pretrainmodel.0.", "finetunemodel.0.")
            backbone_state_dict[new_k] = v
    print(list(backbone_state_dict.keys())[:10])
    torch.save(backbone_state_dict, os.path.join(cwd, 'model', 'convert-weight-42.pth'))
    """

    model.finetunemodel[1].apply(init_weights_normal)
    model.finetunemodel[2].apply(init_weights_normal)
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    loss_finetune = Yolov1DetectionLoss().to(DEVICE)
    train_csv_path = os.path.join(cwd, "data", "VOC", "allexamples.csv")
    test_csv_path = os.path.join(cwd, "data", "VOC", "2007test.csv")
    train_set = VOCDataset(train_csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,)
    test_set = VOCDataset(test_csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_finetune, lr_lambda=lr_lambda)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(NUM_EPOCHS):
        # 프로그레스 바 객체 설정, dataloader객체를 담는다
        with tqdm(train_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader:
            train_loss = 0.0

            # 학습 단계
            model.train() # 모델을 학습 모드로 설정
            for images, labels in tqdmloader:
                tqdmloader.set_description(f"Train_Epoch {epoch + 1:04d}")
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                preds = model(images)
                loss = loss_finetune(preds, labels)

                # backward
                optimizer_finetune.zero_grad()
                loss.backward()
                # gradient clipping (max_norm=1.0)
                if epoch < warmup_epochs:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer_finetune.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with tqdm(test_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_val:
            with torch.no_grad():
                for images, labels in tqdmloader_val:
                    tqdmloader_val.set_description(f"Validation_Epoch {epoch + 1:04d}")
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    out = model(images)
                    loss = loss_finetune(out, labels)
                    val_loss += loss.item()


            val_loss /= len(test_loader)

        scheduler.step()

        with open(os.path.join(cwd, "finetune-log.txt"), "a") as f:
            f.write(f"Epoch {epoch + 1:04d} "
                    f"train_loss: {train_loss:.4f} "
                    f"val_loss: {val_loss:.4f} "
                    f"LR: {optimizer_finetune.param_groups[0]['lr']:.6}\n"
                    )
        if epoch + 1 == 135:
            torch.save(model.state_dict(), os.path.join(cwd, "model", f"finetune-weight-{epoch + 1}.pth"))

if __name__ == "__main__":
    main()