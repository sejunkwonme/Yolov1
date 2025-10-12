import os
import torch
from torch.utils.data import (DataLoader, random_split, Subset)
from model import Yolov1Model
from dataset import VOCDataset
from loss import Yolov1DetectionLoss, Yolov1ClassificationLoss
from utils import (NMS, mAP, IoU, plotImage)
from tqdm import tqdm
from torchvision import datasets, transforms

cwd = os.getcwd() # 현재 워킹디렉토리 경로 저장

# 학습에 쓰일 하이퍼파라미터
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 120
WEIGHT_DECAY = 1e-4
MOMENTUM = 0
NUM_EPOCHS = 10
NUM_WORKERS = 20
PIN_MEMORY = True
IMAGENET_DIR = os.path.join(cwd, 'ImageNet')
IMG_DIR = os.path.join(cwd, "data", "VOC", "images")
LABEL_DIR = os.path.join(cwd, "data", "VOC", "labels")

def main():
    #model = Yolov1Model(S = 7, B = 2, C = 20, mode="finetune").to(DEVICE)
    model = Yolov1Model(mode = "pretrain").to(DEVICE)
    #optimizer_finetune = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #loss_finetune = Yolov1DetectionLoss()
    loss_pretrain = Yolov1ClassificationLoss()
    csv_path = os.path.join(cwd, "data", "VOC", "500examples.csv")
    dataset = VOCDataset(csv_path, img_dir=IMG_DIR, label_dir=LABEL_DIR,)
    # 1) 이미지 전처리 (Data Augmentation + Normalization)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 이미지 크롭 후 224x224
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        transforms.ToTensor(),  # Tensor 변환
        transforms.Normalize(  # ImageNet 평균/표준편차로 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    ImageNet_dataset = datasets.ImageFolder(root=IMAGENET_DIR, transform=train_transforms)
    # 2) 사용할 이미지 개수 제한
    #num_samples = 500000  # 원하는 개수
    #subset_indices = list(range(num_samples))
    #subset_dataset = Subset(ImageNet_dataset, subset_indices)
    ImageNet_loader = DataLoader(ImageNet_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)

    total_size = len(dataset)
    train_size = int(0.9 * total_size) # train:test 0.9:0.1
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(200)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

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
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(NUM_EPOCHS):
        mean_loss = []
        preds_list = []

        # 프로그레스 바 객체 설정, dataloader객체를 담는다
        with tqdm(ImageNet_loader, unit="batch", ascii=" =", ncols=70) as tqdmloader:

            # 학습 단계
            model.train() # 모델을 학습 모드로 설정
            for images, labels in tqdmloader:
                tqdmloader.set_description(f"Epoch {epoch + 1:04d}")
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                preds = model(images)
                preds_list.append(preds.detach().clone()) # 추론결과 리스트에 보관
                loss = loss_pretrain(preds, labels)
                mean_loss.append(loss.item())

                # backward
                optimizer_pretrain.zero_grad()
                loss.backward()
                optimizer_pretrain.step()

                tqdmloader.set_postfix(loss = loss.item())


        #if (epoch + 1) % 5 == 0:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_pretrain.state_dict(),
        'loss': loss,
        }, os.path.join(cwd, "model", f"model-pretrain-{epoch + 1}.pth"))




    """
    # 검증 단계
    model.load_state_dict(torch.load(os.path.join(cwd, "model", f"model-199.pth"), weights_only=True))
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds = model(images)
            print(preds)
            print(preds.shape)
            plotImage(images.to("cpu"), preds)
            all_preds.append(preds)
            all_targets.append(labels)

        # mAP 계산
        #mAP_result = mAP(all_preds, all_targets)
        #print(f"Epoch {epoch}: mAP={mAP:.4f}")
        """

if __name__ == "__main__":
    main()