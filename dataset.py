import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxes
import torchvision.transforms.v2.functional as TF
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S, self.B, self.C = S, B, C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # --- 이미지 & 라벨 불러오기 ---
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = map(float, label.strip().split())
                boxes.append([class_label, x, y, w, h])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        boxes = torch.tensor(boxes)

        # --- BoundingBoxes 객체 생성 ---
        bbox = BoundingBoxes(
            data=boxes[:, 1:5],
            format="XYXY",
            canvas_size=(orig_h, orig_w)  # (height, width)
        )
        target = {"boxes": bbox, "labels": boxes[:, 0].to(torch.int64)}

        """
        img_tensor = TF.to_tensor(image)
        img_with_boxes = draw_bounding_boxes(
            (img_tensor * 255).to(torch.uint8),
            boxes=target["boxes"],
            colors="red",
            labels=[str(l.item()) for l in target["labels"]],
            width=2
        )
        plt.figure(figsize=(6, 6))
        plt.title("Before Transform")
        plt.imshow(TF.to_pil_image(img_with_boxes))
        plt.axis("off")
        plt.show()
        """

        # --- transform 적용 (이미지 & bbox 동시 변환) ---
        image, target = self.transform(image, target)

        """
        count = 0
        if count <= 10:
            img_tensor = (image * 255).to(torch.uint8)
            img_with_boxes = draw_bounding_boxes(
                img_tensor,
                boxes=target["boxes"],
                colors="lime",
                labels=[str(l.item()) for l in target["labels"]],
                width=2
            )
            plt.figure(figsize=(6, 6))
            plt.title("After Transform")
            plt.imshow(TF.to_pil_image(img_with_boxes))
            plt.axis("off")
            plt.show()
            count += 1
        """
        if isinstance(image, torch.Tensor):
            _, new_h, new_w = image.shape
        else:
            new_w, new_h = image.size
        boxes_cxcywh = box_convert(target["boxes"].to("cpu"), in_fmt="xyxy", out_fmt="cxcywh")

        # --- YOLO 라벨 매트릭스 변환 ---
        label_matrix = torch.zeros((self.C + 5 * self.B, self.S, self.S))
        for box, cls in zip(boxes_cxcywh, target["labels"]):
            x, y, w, h = box.tolist()
            x = x / new_w
            y = y / new_h
            w = w / new_w
            h = h / new_h
            class_label = int(cls)

            i = min(self.S - 1, int(self.S * y))
            j = min(self.S - 1, int(self.S * x))
            x_cell, y_cell = self.S * x - j, self.S * y - i

            if label_matrix[20, i, j] == 0:
                label_matrix[20, i, j] = 1
                label_matrix[21:25, i, j] = torch.tensor([x_cell, y_cell, w, h], dtype=label_matrix.dtype)
                label_matrix[class_label, i, j] = 1

        return image, label_matrix
