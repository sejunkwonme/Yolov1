import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

transform = Compose([
    transforms.ToTensor(),
    transforms.Normalize(  # ImageNet 평균/표준편차로 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
])

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20,):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # .txt파일에 접근
        boxes = []
        with open(label_path) as f: # 각 이미지의 비율로 변환한 x, y, w, h를 불러온다
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height]) # 좌표 변환전이다 x,y,w,h 모두 이미지 전체를 기준으로 0~1 사이의 값이다

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        orig_w, orig_h = image.size  # 원본 크기

        # --- YOLO 비율 좌표를 픽셀 단위로 변환 ---
        # (x, y, w, h)는 [0, 1] 기준 비율이므로, 픽셀로 환산
        boxes = torch.tensor(boxes) # (오브젝트 개수, (class_label, x, y, width, height)) 2차원 텐서
        boxes[:, 1] = boxes[:, 1] * orig_w  # x_center
        boxes[:, 2] = boxes[:, 2] * orig_h  # y_center
        boxes[:, 3] = boxes[:, 3] * orig_w  # width
        boxes[:, 4] = boxes[:, 4] * orig_h  # height

        # --- 이미지 리사이즈 ---
        new_w, new_h = 448, 448
        image = image.resize((new_w, new_h))

        # --- 바운딩박스 비율 보정 ---
        x_scale = new_w / orig_w
        y_scale = new_h / orig_h
        boxes[:, 1] *= x_scale
        boxes[:, 2] *= y_scale
        boxes[:, 3] *= x_scale
        boxes[:, 4] *= y_scale

        #if self.transform: # 이미지를 448x448 로 스케일링한다 이때는 비율이 무시됨
        image = self.transform(image)

        # Convert To Cells
        label_matrix = torch.zeros((self.C + 5 * self.B, self.S, self.S))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # 픽셀 좌표를 다시 448기준 비율로 변환 (S셀 기준 매핑용)
            x /= new_w
            y /= new_h
            width /= new_w
            height /= new_h

            i, j = int(self.S * y), int(self.S * x) # 박스 중앙점의 비율값에 셀의 개수를 곱하여 몇번재 셀에 있는지 알아낸다
            x_cell, y_cell = self.S * x - j, self.S * y - i # 박스 중앙점이 셀 내부에서 상대적으로 어디에 있는지 구한다

            if label_matrix[20, i, j] == 0: # 이미 셀자리에서 오브젝트를 처리하면 다음엔 안한다 한 셀당 하나의 오브젝트만 다루기때문
                # 이 루프에 들어온거자체가 오브젝트가 존재하는것이므로 존재확률 1로 설정
                label_matrix[20, i, j] = 1

                # Box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width, height])

                label_matrix[21:25, i, j] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[class_label, i, j] = 1

        return image, label_matrix # 이때 image 와 label_matrix는 Dataloader에서 배치가 생성될 때 배치차원이 추가됨