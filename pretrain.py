import os
import torch
from torch.utils.data import (DataLoader, random_split, Subset)
from model import Yolov1Model
from loss import Yolov1ClassificationLoss
from utils import (NMS, mAP, IoU, plotImage)
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.optim as optim
import logging
import torch.nn as nn

cwd = os.getcwd() # 현재 워킹디렉토리 경로 저장

# 학습에 쓰일 하이퍼파라미터
# ImageNet - 1K Dataset Pretrain
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS = 90
NUM_WORKERS = 20
PIN_MEMORY = True
IMAGENET_DIR = os.path.join(cwd, "ImageNet")
IMAGENET_VAL_DIR = os.path.join(cwd, "ImageNet_val")

def initialize_weights_vgg16(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

# top-k accuracy 계산 함수 추가
def topk_accuracy(output, target, topk=(1, 5)):
    """output: (N, C), target: (N,)"""
    maxk = max(topk)
    batch_size = target.size(0)

    # top-k index 추출
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, N)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res  # [top1, top5]

def main():
    model_pretrain = Yolov1Model(mode = "pretrain").to(DEVICE)
    initialize_weights_vgg16(model_pretrain)
    optimizer_pretrain = torch.optim.SGD(model_pretrain.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    loss_pretrain = Yolov1ClassificationLoss().to(DEVICE)

    # ReduceLROnPlateau 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pretrain,
        mode='min',  # 'min'이면 loss 감소를 목표로 함
        factor=0.1,  # LR을 1/10로 줄임
        patience=1,  # 1 epoch 동안 개선이 없으면 감소
        threshold=1e-4,  # 개선으로 간주할 최소 변화량
        cooldown=0,  # 감소 후 대기 epoch 수
        min_lr=1e-6,  # LR 하한선
    )

    # 이미지 전처리 (Data Augmentation + Normalization)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 이미지 크롭 후 224x224
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        transforms.ToTensor(),  # Tensor 변환
        transforms.Normalize(  # ImageNet 평균/표준편차로 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset_train = datasets.ImageFolder(root=IMAGENET_DIR, transform=train_transforms)
    dataset_val = datasets.ImageFolder(root=IMAGENET_VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=dataset_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    best_val_loss = float('inf')
    wait = 0
    patience = 3

    for epoch in range(NUM_EPOCHS):
        model_pretrain.train()

        # 프로그레스 바 객체 설정, dataloader객체를 담는다
        with tqdm(train_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_train:
            # 학습 단계
            train_loss = 0.0
            model_pretrain.train() # 모델을 학습 모드로 설정
            for images, labels in tqdmloader_train:
                tqdmloader_train.set_description(f"Train_Epoch {epoch + 1:04d}")
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                out = model_pretrain(images)
                loss = loss_pretrain(out, labels)

                # backward
                optimizer_pretrain.zero_grad()
                loss.backward()
                optimizer_pretrain.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

        model_pretrain.eval()
        val_loss = 0.0
        top1_acc_total, top5_acc_total = 0.0, 0.0

        with tqdm(test_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_val:
            with torch.no_grad():
                for images, labels in tqdmloader_val:
                    tqdmloader_val.set_description(f"Validation_Epoch {epoch + 1:04d}")
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    out = model_pretrain(images)
                    loss = loss_pretrain(out, labels)
                    val_loss += loss.item()

                    # Top-1 / Top-5 accuracy 계산
                    top1, top5 = topk_accuracy(out, labels, topk=(1, 5))
                    top1_acc_total += top1
                    top5_acc_total += top5

            val_loss /= len(test_loader)
            top1_acc = (top1_acc_total / len(test_loader)) * 100
            top5_acc = (top5_acc_total / len(test_loader)) * 100

        with open(os.path.join(cwd, "log.txt"), "a") as f:
            f.write(f"Epoch {epoch + 1:04d} "
                    f"train_loss: {train_loss:.4f} "
                    f"val_loss: {val_loss:.4f} "
                    f"Top1: {top1_acc:.2f}% "
                    f"Top5: {top5_acc:.2f}% "
                    f"LR: {optimizer_pretrain.param_groups[0]['lr']:.6}\n"
                    )


        torch.save(model_pretrain.state_dict(), os.path.join(cwd, "model", f"pretrain-weight-{epoch + 1}.pth"))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1

        if wait >= patience or optimizer_pretrain.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()