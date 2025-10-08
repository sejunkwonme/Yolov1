import os
import torch
from torch.utils.data import (DataLoader, random_split)
from model import Yolov1Model
from dataset import VOCDataset
from loss import Yolov1Loss
from utils import (NMS, mAP, IoU)
from tqdm import tqdm
from colorama import init

init()

cwd = os.getcwd() # 현재 워킹디렉토리 경로 저장

# 학습에 쓰일 하이퍼파라미터
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
MOMENTUM = 0
NUM_EPOCHS = 1000
NUM_WORKERS = 20
PIN_MEMORY = True
IMG_DIR = os.path.join(cwd, "data", "VOC", "images")
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")

def main():
    model = Yolov1Model(S = 7, B = 2, C = 20, mode='finetune').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = Yolov1Loss()
    csv_path = os.path.join(cwd, "data", "VOC", "100examples.csv")
    dataset = VOCDataset(csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
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

    total_steps = NUM_EPOCHS * len(train_loader)
    progress = tqdm(total=total_steps, ncols=70, ascii=" =")

    for epoch in range(NUM_EPOCHS):
        mean_loss = []

        # 학습 단계
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # forward
            preds = model(images)
            loss = loss_fn(preds, labels)
            mean_loss.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tqdm 진행 업데이트
            progress.update(1)
            progress.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {loss.item():.4f}")

        with open("train_log.txt", "a") as f:
            print(f"Epoch {epoch + 1}: Mean loss = {sum(mean_loss) / len(mean_loss)}", file=f)

        # 검증 단계
        """
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for images, labels in val_loader:
                preds = model(images)
                all_preds.append(preds)
                all_targets.append(labels)
        """
        # mAP 계산
        #mAP_result = mAP(all_preds, all_targets)
        #print(f"Epoch {epoch}: mAP={mAP:.4f}")

    progress.close()

if __name__ == "__main__":
    main()