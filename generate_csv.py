# 이미지파일이름, 이미지 이름.txt 짝의 csv를 생성한다
# 생성하는 이유는 데이터셋은 전체 데이터를 담고 있는데 이러면 끝나는데 너무 오래 걸리므로
# 100개, 1000개 등의 일부 데이터셋으로 불러오기 위해 csv 변환파일을 만들어줄 필요가 있음

import os
import csv
from os import getcwd

cwd = getcwd()

read_train = open(f"{cwd}/data/VOC/train.txt", "r").readlines()

with open(f"{cwd}/data/VOC/100examples.csv", mode="w", newline="") as train_file:
    for line in read_train[:100]: # 여기서 전체 데이서셋에서 사용할 데이터 개수만큼 지정 가능하다
        image_file = os.path.basename(line.strip())
        text_file = image_file.replace(".jpg", ".txt") # 파일이름.txt로 변환 여기엔 클래스와 바운딩박스의 위치가 담겨있음
        data = [image_file, text_file]
        writer = csv.writer(train_file)
        writer.writerow(data)

"""
read_train = open("test.txt", "r").readlines()

with open(f"{cwd}/data/VOC/test.csv", mode="w", newline="") as train_file:
    for line in read_train:
        image_file = line.split("/")[-1].replace("\n", "")
        text_file = image_file.replace(".jpg", ".txt")
        data = [image_file, text_file]
        writer = csv.writer(train_file)
        writer.writerow(data)
"""