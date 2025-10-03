import xml.etree.ElementTree as ET
import os
from os import getcwd

voc=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# 클래스 20개
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

"""
VOC데이터셋의 이미지의 annotation 데이터에는
바운딩박스 데이터가 절대 좌표로 담겨 있음
Yolov1에서는 이미지 내에서의 상대 비율을 사용하기 때문에
절대 좌표를 상대 비율로 변환해 줘야 한다
"""
def cvt_coordnate(imgsize, vocbox):
    dw = 1./(imgsize[0])
    dh = 1./(imgsize[1])
    x = (vocbox[0] + vocbox[1])/2.0 - 1 # 바운딩박스의 중앙점 x 를 구한다 voc는 이미지 전체기준 좌표이다
    y = (vocbox[2] + vocbox[3])/2.0 - 1 # 바운딩박스의 중앙점 y 를 구한다
    w = vocbox[1] - vocbox[0] # 바운딩박스의 절대적 width
    h = vocbox[3] - vocbox[2] # 바운디박스의 절대적 height
    x = x*dw # 이미지의 width 로 나누어 0 ~ 1 사이로 스케일링한다 이미지 내의 상대 위치, 크기로 변환
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h # 학습에 쓰일 수 있도록 절대좌표 -> 이미지에 크기에 대한 상대 비율로 변경한 후 튜플로 변환한다

"""
VOC데이터셋의 annotation 데이터를 파싱해서
label을 따로 텍스트 파일에 저장한다
"""
def cvt_annotation(year, image_id):
    annotation = open(f"archive/VOC_dataset/VOCdevkit/VOC{year}/Annotations/{image_id}.xml")
    out_file = open(f"archive/VOC_dataset/VOCdevkit/VOC{year}/labels/{image_id}.txt", 'w') # 이미지 1개당 레이블 파일 txt파일 1개 생성
    tree=ET.parse(annotation)
    root = tree.getroot()
    size = root.find('size')
    imgw = int(size.find('width').text)
    imgh = int(size.find('height').text)

    for obj in root.iter('object'): # 한 이미지 내의 오브젝트는 2개 이상일 수 있음
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        annotation_bndbox = obj.find('bndbox')

        # voc 데이터셋의 bounding box는 절대좌표로 되어 있음 (0 ~ 이미지의 width, height 사이의 값)
        vocbox = (float(annotation_bndbox.find('xmin').text), float(annotation_bndbox.find('xmax').text), float(annotation_bndbox.find('ymin').text), float(annotation_bndbox.find('ymax').text))
        cvted_bbox = cvt_coordnate((imgw,imgh), vocbox)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in cvted_bbox]) + '\n') # ex) aeroplane x y w h 로 나오고 이미지 1개에 여러 줄의 바운딩박스 레이들이 나올 수 있음

wd = getcwd()

for year, image_set in voc:
    if not os.path.exists(f"archive/VOC_dataset/VOCdevkit/VOC{year}/labels"): # 기존 VOC 데이터셋 폴더에 labels 폴더 새로 만들기
        os.makedirs(f"archive/VOC_dataset/VOCdevkit/VOC{year}/labels")
    image_ids = open(f"archive/VOC_dataset/VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt").read().strip().split()
    list_file = open(f"{year}_{image_set}.txt", 'w') # 연도_이미지셋으로 새로 txt파일 생성한다 여기에는 이미지 파일의 절대경로가 들어간다
    for image_id in image_ids:
        list_file.write(f"{wd}/archive/VOC_dataset/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg\n") # 이미지파일의 경로를 하나씩 쓴다
        cvt_annotation(year, image_id)
    list_file.close()

# 합칠 파일 목록
train_files = ["2007_train.txt", "2007_val.txt", "2012_train.txt", "2012_val.txt"]
train_all_files = ["2007_train.txt", "2007_val.txt", "2007_test.txt", "2012_train.txt", "2012_val.txt"]

# train.txt 생성
with open("train.txt", "w", encoding="utf-8") as outfile:
    for fname in train_files:
        with open(fname, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            #outfile.write("\n")  # 파일 사이에 개행 추가 (선택적)

# train.all.txt 생성
with open("train.all.txt", "w", encoding="utf-8") as outfile:
    for fname in train_all_files:
        with open(fname, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            #outfile.write("\n")
