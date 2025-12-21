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
import math
import torchvision.transforms.v2 as v2

cwd = os.getcwd()
LEARNING_RATE = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 96
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
NUM_EPOCHS = 100
NUM_WORKERS = 12
PIN_MEMORY = True
IMG_DIR = os.path.join(cwd, "data", "VOC", "images")
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")
WARMUP = 10

@torch.no_grad()
def initialize_HE_conv4(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

@torch.no_grad()
def initialize_detection_block(model):
    linear_count = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if linear_count == 0:
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu'
                )
            else:
                nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

            linear_count += 1

train_transform = v2.Compose([
    v2.ColorJitter(brightness=0.3, saturation=0.3),
    v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), fill=(127, 127, 127)),
    v2.Resize((448, 448),),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

test_transform = v2.Compose([
    v2.Resize((448, 448),),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def warmup_anneal(epoch, warmup_epochs=10, start_factor=0.1, first_factor = 1.0, second_factor = 0.1, final_factor=0.01):
    if epoch < warmup_epochs:
        progress = epoch / warmup_epochs
        factor = 10 ** (math.log10(start_factor) + progress * math.log10(first_factor / start_factor))
        return factor
    if epoch < 36:
        return first_factor
    elif epoch < 80:
        return second_factor
    else:
        return final_factor

def main():
    model = Yolov1Model(S=7, B=2, C=20, mode="finetune").to(DEVICE)
    checkpoint = torch.load(os.path.join(cwd, 'model', 'pretrain-weight-secondattempt-36.pth'), weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print("Missing keys:")
    for k in missing:
        print("  ", k)

    print("Unexpected keys:")
    for k in unexpected:
        print("  ", k)

    initialize_HE_conv4(model.yolomodel[1])
    initialize_detection_block(model.yolomodel[2])

    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    loss_finetune = Yolov1DetectionLoss().to(DEVICE)
    train_csv_path = os.path.join(cwd, "data", "VOC", "allexamples.csv")
    test_csv_path = os.path.join(cwd, "data", "VOC", "2007test.csv")
    train_set = VOCDataset(train_csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,transform=train_transform,)
    test_set = VOCDataset(test_csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,transform=test_transform,)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer_finetune,
        lr_lambda=[
            lambda e: warmup_anneal(e, warmup_epochs=WARMUP, start_factor = 0.1, first_factor = 1.0, second_factor = 0.1, final_factor=0.01),
        ]
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
        prefetch_factor=8,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
        prefetch_factor=8,
        persistent_workers=True,
    )

    for epoch in range(NUM_EPOCHS):
        with tqdm(train_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader:
            train_loss = 0.0

            # 학습 단계
            model.train() # 모델을 학습 모드로 설정
            for images, labels in tqdmloader:
                tqdmloader.set_description(f"Train_Epoch {epoch + 1:04d}")
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # forward
                preds = model(images)
                loss = loss_finetune(preds, labels)

                # backward
                optimizer_finetune.zero_grad()
                loss.backward()
                optimizer_finetune.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with tqdm(test_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_val:
            with torch.no_grad():
                for images, labels in tqdmloader_val:
                    tqdmloader_val.set_description(f"Validation_Epoch {epoch + 1:04d}")
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    out = model(images)
                    loss = loss_finetune(out, labels)
                    val_loss += loss.item()


            val_loss /= len(test_loader)

        with open(os.path.join(cwd, "finetune-final-log.txt"), "a") as f:
            f.write(f"Epoch {epoch + 1:04d} "
                    f"train_loss: {train_loss:.4f} "
                    f"val_loss: {val_loss:.4f} "
                    f"LR: {optimizer_finetune.param_groups[0]['lr']:.6} "
                    f"Momentem: {optimizer_finetune.param_groups[0]['momentum']:.6}\n"
                    )

        scheduler.step()

        if epoch + 1 >= 49:
            torch.save(model.state_dict(), os.path.join(cwd, "model", f"finetune-final-batchnorm-weight-{epoch + 1}.pth"))


if __name__ == "__main__":
    main()