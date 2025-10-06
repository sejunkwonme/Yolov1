import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
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
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes) # (오브젝트 개수, (class_label, x, y, width, height))

        if self.transform: # 이미지를 448x448 로 스케일링한다 이때는 비율이 무시됨
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.C + 5 * self.B, self.S, self.S))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

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