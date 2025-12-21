import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from collections import Counter
from jaxtyping import Float
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from operator import itemgetter

# 셀에 상대적인 바운딩박스의 중심좌표를 이미지 전체에 상대적인 좌표로 변환해준다
def cvtCellCoord2ImgCoord(input: Float[torch.Tensor, "Batch bbox_params S S"], S = 7):
    device = input.device
    i = torch.arange(S, device=device).view(1, 1, S, 1)
    j = torch.arange(S, device=device).view(1, 1, 1, S)
    x_img = (j + input[:, 0:1, :, :]) / S
    y_img = (i + input[:, 1:2, :, :]) / S
    w = input[:, 2:3, :, :]
    h = input[:, 3:4, :, :]
    return torch.cat([x_img, y_img, w, h], dim = 1) # (Batch, 4, S, S)

# 박스의 중심좌표와  너비를 받아박스의 좌상단 좌표와 우하단 좌표로 변환
def cvtCenter2Corner(input: Float[torch.Tensor,"Batch bbox_params S S"]):
    xmin = input[:, 0:1, :, :] - abs(input[:, 2:3, :, :]) / 2
    ymin = input[:, 1:2, :, :] - abs(input[:, 3:4, :, :]) / 2
    xmax = input[:, 0:1, :, :] + abs(input[:, 2:3, :, :]) / 2
    ymax = input[:, 1:2, :, :] + abs(input[:, 3:4, :, :]) / 2
    return torch.cat([xmin, ymin, xmax, ymax], dim = 1) # (Batch, 4, S, S)

# 예측 박스와 레이블 박스 간의 IoU 를 구한다 (배치, 셀의 i행 j열에 대해 한번에 계산 가능하다)
def IoU(boxes_preds: Float[torch.Tensor, "Batch bbox_params S S"], boxes_labels: Float[torch.Tensor, "Batch bbox_params S S"], S = 7, mode = "mid"):
    if mode == "mid":
        img_coord_preds = cvtCellCoord2ImgCoord(boxes_preds, S)
        img_coord_labels = cvtCellCoord2ImgCoord(boxes_labels, S)
        box1_corners = cvtCenter2Corner(img_coord_preds)
        box2_corners = cvtCenter2Corner(img_coord_labels)
        xmin = torch.max(box1_corners[:, 0:1, :, :], box2_corners[:, 0:1, :, :])
        ymin = torch.max(box1_corners[:, 1:2, :, :], box2_corners[:, 1:2, :, :])
        xmax = torch.min(box1_corners[:, 2:3, :, :], box2_corners[:, 2:3, :, :])
        ymax = torch.min(box1_corners[:, 3:4, :, :], box2_corners[:, 3:4, :, :])
    elif mode == "corner":
        xmin = torch.max(boxes_preds[:, 0:1, :, :], boxes_labels[:, 0:1, :, :])
        ymin = torch.max(boxes_preds[:, 1:2, :, :], boxes_labels[:, 1:2, :, :])
        xmax = torch.min(boxes_preds[:, 2:3, :, :], boxes_labels[:, 2:3, :, :])
        ymax = torch.min(boxes_preds[:, 3:4, :, :], boxes_labels[:, 3:4, :, :])
    intersection_area = (xmax - xmin).clamp(min = 0) * (ymax - ymin).clamp(min = 0) # 음수이면 0으로 클램프한다
    if mode == "mid":
        box1_area = abs((box1_corners[:,2:3,:,:] - box1_corners[:,0:1,:,:]) * (box1_corners[:,3:4,:,:] - box1_corners[:,1:2,:,:]))
        box2_area = abs((box2_corners[:,2:3,:,:] - box2_corners[:,0:1,:,:]) * (box2_corners[:,3:4,:,:] - box2_corners[:,1:2,:,:]))

    elif mode == "corner":
        box1_area = abs((boxes_preds[:, 2:3, :, :] - boxes_preds[:, 0:1, :, :]) * (
                    boxes_preds[:, 3:4, :, :] - boxes_preds[:, 1:2, :, :]))
        box2_area = abs((boxes_labels[:, 2:3, :, :] - boxes_labels[:, 0:1, :, :]) * (
                    boxes_labels[:, 3:4, :, :] - boxes_labels[:, 1:2, :, :]))
    return intersection_area / (box1_area + box2_area - intersection_area + 1e-9) # (Batch, 1, S, S)

# 모델에서 추론한 텐서를 가져와서 non-maximum suppression 을 수행한다 이미지 한장씩 수행, 텐서 입력하기 전에 Batch차원 없애야 제대로 작동한다
def NMS(predictions: Float[torch.Tensor, "features"], trues, iou_threshold = 0.6, threshold = 0.2, S: int=7, B: int=2, C: int=20):
    predictions = predictions.view(-1, C + B * 5, S, S) # (1, 30, 7, 7)
    predictions[:,21:25,:,:] = cvtCenter2Corner(cvtCellCoord2ImgCoord(predictions[:,21:25,:,:]))
    predictions[:,26:30,:,:] = cvtCenter2Corner(cvtCellCoord2ImgCoord(predictions[:,26:30,:,:]))
    label = trues.clone()
    label[:,21:25,:,:] = cvtCenter2Corner(cvtCellCoord2ImgCoord(trues[:, 21:25, :, :]))

    predictions = predictions.view(C + B * 5, S, S) # (30, 7, 7)
    box1_scores = predictions[20:21,:,:] * predictions[0:20, :, :] #(20, 7, 7)
    box2_scores = predictions[25:26,:,:] * predictions[0:20, :, :] #(20, 7, 7)
    box1_scores_masked = torch.where(box1_scores[0:20, :, :] < threshold,
                                     torch.tensor(0., device=box1_scores.device),
                                     box1_scores[0:20,:,:])
    box2_scores_masked = torch.where(box2_scores[0:20, :, :] < threshold,
                                     torch.tensor(0., device=box2_scores.device),
                                     box2_scores[0:20, :, :])

    box1_scores_coord = torch.cat([box1_scores_masked, predictions[21:25,:,:]], dim = 0) # (24, 7, 7)
    box2_scores_coord = torch.cat([box2_scores_masked, predictions[26:30,:,:]], dim = 0) # (24, 7, 7)
    all_scores = torch.cat([box1_scores_coord, box2_scores_coord], dim = 1) # (24, 14, 7)

    flatten = all_scores.view(all_scores.size(0), -1).clone() # (24, 98)
    flatten[0:20,:], indices = torch.sort(flatten[0:20,:], dim = 1, descending = True)
    for i in range(20): # 20개의 클래스에 대해 순차적 진행 (NMS는 매 클래스마다 다르게 처리되므로 벡터화 불가능
        for boxi in range(S*S*2):
            if flatten[i,boxi] <= 0:
                continue
            start = boxi + 1
            for boxj in range(start,S*S*2):
                IoUofBoxes = IoU(flatten[20:24,indices[i,boxi]].view(1,4,1,1), flatten[20:24,indices[i,boxj]].view(1,4,1,1), mode = "corner")
                if IoUofBoxes[:,0:1,:,:].item() > iou_threshold:
                    flatten[i,boxj] = 0

    #ret_tensor = torch.zeros(C + 5 * B , S , S, device=predictions.device)
    gt_used = torch.zeros((20, 1, S, S), dtype=torch.bool, device=label.device)
    drawing_boxes = []
    for boxi in range(S*S*2): # [0,98)
        maxscore, classnum = torch.max(flatten[0:20,boxi:boxi+1], dim = 0)
        classnum = classnum.item()
        if maxscore > 0:
            box = []
            oriidx = indices[classnum, boxi]
            #if oriidx > 48: # 이부분이 문제...
                #oriidx = oriidx - (S * S)

            box.append(classnum) # classindex
            box.append(maxscore.item()) # score
            box.append(flatten[20:24,oriidx].detach().clone().tolist()) # x y w h

            #y = oriidx // S
            #x = oriidx % S

            maxiou = 0
            maxx, maxy = -1, -1
            for a in range(S*S):
                y = a // S
                x = a % S
                if gt_used[classnum:classnum+1, 0, y:y+1, x:x+1].item() == True:
                    continue
                if label[:, classnum:classnum + 1, y:y + 1, x:x + 1] != 1:
                    continue
                iou = IoU(flatten[20:24, oriidx].view(1, 4, 1, 1), label[:, 21:25, y : y+1, x:x + 1], mode="corner").item()
                if maxiou < iou:
                    maxiou = iou
                    maxx = x
                    maxy = y

            if maxiou > 0.5:
                box.append("TP")
                box.append(maxiou)
                gt_used[classnum:classnum + 1, 0, maxy:maxy + 1, maxx:maxx + 1] = True

            else:
                box.append("FP")
                box.append(maxiou)

            #ret_tensor[21:25, y:y + 1, x:x + 1] = flatten[20:24, oriidx].reshape(1, 4, 1, 1)
            #ret_tensor[classnum, y, x] = 1
            #ret_tensor[20, y, x] = maxscore
            drawing_boxes.append(box)
    return drawing_boxes #[[classnum, maxscore, [xmin, ymin, xmax, ymax], "TP" or "FP"], ... ]

def mAP(table, GT_num): # 예측결과 텐서 배치의 리스트 레이블 텐서 배치의 리스트 받아 mAP를 계산한다. 박스 변환은 내부에서 처리한다
    allprecision = []
    allrecall = []
    for idx, table_class in enumerate(table):
        precision = []
        recall = []
        TP = 0
        FP = 0

        GTNUM = GT_num[idx]
        score_sorted = sorted(table_class, key=itemgetter(1), reverse=True) # score 내림차순으로 정렬
        for row in score_sorted:
            if row[3] == "TP":
                TP += 1
            elif row[3] == "FP":
                FP += 1
            PRECISION = TP / (TP + FP)
            RECALL = TP / GTNUM
            precision.append(PRECISION)
            recall.append(RECALL)

        allprecision.append(precision)
        allrecall.append(recall)

    return allprecision, allrecall

def plotImage(image, pred, label):
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    label[:,21:25,:,:] = cvtCenter2Corner(cvtCellCoord2ImgCoord(label[:,21:25,:,:]))
    labelbox = []
    for a in range(49):
        y = a // 7
        x = a % 7

        if label[:, 20:21, y: y + 1, x:x + 1].item() == 1: # 원본 박스 존재 확인
            box = []
            cls_idx = torch.argmax(label[:, 0:20, y: y + 1, x:x + 1], dim = 1).item()
            box.append(cls_idx)
            coords = label[:, 21:25, y:y+1, x:x+1].detach().cpu().squeeze().tolist()
            box.append(coords)
            labelbox.append(box)

    # image: (C,H,W) or (1,C,H,W) on GPU 가능
    if image.ndim == 4:
        image = image[0]

    img = image.detach().cpu()
    img = img.permute(1, 2, 0)  # HWC
    if img.max() > 1.5:         # 0~255인 경우 대비
        img = img / 255.0

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img.clamp(0, 1))
    ax.axis("off")

    for obj in pred:
        cls = int(obj[0])
        score = float(obj[1])
        xmin, ymin, xmax, ymax = obj[2]
        xmin = xmin * 448
        xmax = xmax * 448
        ymin = ymin * 448
        ymax = ymax * 448

        tag = obj[3] if len(obj) > 3 else ""

        w = xmax - xmin
        h = ymax - ymin

        rect = patches.Rectangle((xmin, ymin), w, h,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        name = classes[cls] if 0 <= cls < len(classes) else str(cls)
        ax.text(xmin, ymin, f"{name} {score:.2f} {tag}",
                color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.5, pad=2, edgecolor="none"))

    for box in labelbox:
        cls = int(box[0])
        xmin, ymin, xmax, ymax = box[1]
        xmin = xmin * 448
        xmax = xmax * 448
        ymin = ymin * 448
        ymax = ymax * 448

        w = xmax - xmin
        h = ymax - ymin

        rect = patches.Rectangle((xmin, ymin), w, h,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        name = classes[cls] if 0 <= cls < len(classes) else str(cls)
        ax.text(xmin, ymin, f"{name}",
                color="white", fontsize=10,
                bbox=dict(facecolor="green", alpha=0.5, pad=2, edgecolor="none"))

    plt.show()