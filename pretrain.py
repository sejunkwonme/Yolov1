import os
import torch
from torch.utils.data import (DataLoader, random_split, Subset)
from model import Yolov1Model
from loss import Yolov1ClassificationLoss
from utils import (NMS, mAP, IoU, plotImage)
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.optim as optim

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
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")

def main():
    model_pretrain = Yolov1Model(mode = "pretrain").to(DEVICE)
    optimizer_pretrain = torch.optim.SGD(model_pretrain.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    loss_pretrain = Yolov1ClassificationLoss().to(DEVICE)

    # ReduceLROnPlateau 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pretrain,
        mode='min',  # 'min'이면 loss 감소를 목표로 함
        factor=0.1,  # LR을 1/10로 줄임
        patience=3,  # 3 epoch 동안 개선이 없으면 감소
        threshold=1e-4,  # 개선으로 간주할 최소 변화량
        cooldown=0,  # 감소 후 대기 epoch 수
        min_lr=1e-6,  # LR 하한선
        verbose=True  # 감소 시 콘솔에 출력
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

    dataset_train = datasets.ImageFolder(root=IMAGENET_DIR, transform=train_transforms)
    dataset_val = datasets.ImageFolder(root=IMAGENET_VAL_DIR, transform=train_transforms)
    #generator = torch.Generator().manual_seed(200)
    #train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

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


    for epoch in range(NUM_EPOCHS):
        model_pretrain.train()
        train_loss = 0.0
        # 프로그레스 바 객체 설정, dataloader객체를 담는다
        with tqdm(train_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_train:
            # 학습 단계
            model_pretrain.train() # 모델을 학습 모드로 설정
            for images, labels in tqdmloader_train:
                tqdmloader_train.set_description(f"Epoch {epoch + 1:04d}")
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

            tqdmloader_train.set_postfix(loss=train_loss)

        with tqdm(test_loader, unit="batch", ascii=" =", ncols=100) as tqdmloader_val:
            model_pretrain.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in tqdmloader_val:
                    out = model_pretrain(images)
                    loss = loss_pretrain(out, labels)
                    val_loss += loss.item()
            val_loss /= len(test_loader)
            tqdmloader_val.set_postfix(loss=val_loss)


        if (epoch + 1) % 10 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model_pretrain.state_dict(),
            'optimizer_state_dict': loss_pretrain.state_dict(),
            'loss': loss,
            }, os.path.join(cwd, "model", f"model-finetune-{epoch + 1}.pth"))


if __name__ == "__main__":
    main()