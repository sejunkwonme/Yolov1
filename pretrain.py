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
import torchvision.transforms.v2 as v2

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

@torch.no_grad()
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

# top-k accuracy 계산 함수 추가
def topk_accuracy(output, target, topk=(1, 5)):
    """output: (N, C), target: (N,)"""
    # (256,1000), (256,)
    maxk = max(topk)
    batch_size = target.size(0)

    # top-k index 추출
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, N) (5, 256)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res  # [top1, top5]

def main():
    model = Yolov1Model(mode = "pretrain").to(DEVICE)
    initialize_weights(model)
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.mean().item():.5f}, std={param.std().item():.5f}")
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    criterion = Yolov1ClassificationLoss().to(DEVICE)

    # ReduceLROnPlateau 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 'min'이면 loss 감소를 목표로 함
        factor=0.1,  # LR을 1/10로 줄임
        patience=1,  # 1 epoch 동안 개선이 없으면 감소
        threshold=1e-4,  # 개선으로 간주할 최소 변화량
        cooldown=0,  # 감소 후 대기 epoch 수
        min_lr=1e-6,  # LR 하한선
    )

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    val_transforms = v2.Compose([
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
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
        prefetch_factor=6,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        dataset=dataset_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
        prefetch_factor=6,
        persistent_workers=True,
    )

    best_val_loss = float('inf')
    wait = 0
    patience = 3

    for epoch in range(NUM_EPOCHS):
        model.train()

        # 프로그레스 바 객체 설정, dataloader객체를 담는다
        with tqdm(train_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_train:
            # 학습 단계
            train_loss = 0.0
            model.train() # 모델을 학습 모드로 설정
            for images, labels in tqdmloader_train:
                tqdmloader_train.set_description(f"Train_Epoch {epoch + 1:04d}")
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # forward
                out = model(images)
                loss = criterion(out, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        top1_acc_total, top5_acc_total = 0.0, 0.0

        with tqdm(test_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_val:
            with torch.no_grad():
                for images, labels in tqdmloader_val:
                    tqdmloader_val.set_description(f"Validation_Epoch {epoch + 1:04d}")
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    out = model(images)
                    loss = criterion(out, labels)
                    val_loss += loss.item()

                    # Top-1 / Top-5 accuracy 계산
                    top1, top5 = topk_accuracy(out, labels, topk=(1, 5))
                    top1_acc_total += top1
                    top5_acc_total += top5

            val_loss /= len(test_loader)
            top1_acc = (top1_acc_total / len(test_loader)) * 100
            top5_acc = (top5_acc_total / len(test_loader)) * 100

        with open(os.path.join(cwd, "log-secondattempt.txt"), "a") as f:
            f.write(f"Epoch {epoch + 1:04d} "
                    f"train_loss: {train_loss:.4f} "
                    f"val_loss: {val_loss:.4f} "
                    f"Top1: {top1_acc:.2f}% "
                    f"Top5: {top5_acc:.2f}% "
                    f"LR: {optimizer.param_groups[0]['lr']:.6}\n"
                    )


        torch.save(model.state_dict(), os.path.join(cwd, "model", f"pretrain-weight-secondattempt-{epoch + 1}.pth"))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1

        if wait >= patience or optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()