import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from collections import Counter
from jaxtyping import Float

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

# 박스의 중심좌표와 너비를 받아 박스의 좌상단 좌표와 우하단 좌표로 변환
def cvtCenter2Corner(input: Float[torch.Tensor,"Batch bbox_params S S"]):
    xmin = input[:, 0:1, :, :] - input[:, 2:3, :, :] / 2
    ymin = input[:, 1:2, :, :] - input[:, 3:4, :, :] / 2
    xmax = input[:, 0:1, :, :] + input[:, 2:3, :, :] / 2
    ymax = input[:, 1:2, :, :] + input[:, 3:4, :, :] / 2
    return torch.cat([xmin, ymin, xmax, ymax], dim = 1) # (Batch, 4, S, S)

# 예측 박스와 레이블 박스 간의 IoU 를 구한다 (배치, 셀의 i행 j열에 대해 한번에 계산 가능하다)
def IoU(boxes_preds: Float[torch.Tensor, "Batch bbox_params S S"], boxes_labels: Float[torch.Tensor, "Batch bbox_params S S"], S = 7):
    img_coord_preds = cvtCellCoord2ImgCoord(boxes_preds, S)
    img_coord_labels = cvtCellCoord2ImgCoord(boxes_labels, S)
    box1_corners = cvtCenter2Corner(img_coord_preds)
    box2_corners = cvtCenter2Corner(img_coord_labels)
    xmin = torch.max(box1_corners[:, 0:1, :, :], box2_corners[:, 0:1, :, :])
    ymin = torch.max(box1_corners[:, 1:2, :, :], box2_corners[:, 1:2, :, :])
    xmax = torch.min(box1_corners[:, 2:3, :, :], box2_corners[:, 2:3, :, :])
    ymax = torch.min(box1_corners[:, 3:4, :, :], box2_corners[:, 3:4, :, :])
    intersection_area = (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0) # 음수이면 0으로 클램프한다
    box1_area = abs((box1_corners[:,2:3,:,:] - box1_corners[:,0:1,:,:]) * (box1_corners[:,3:4,:,:] - box1_corners[:,1:2,:,:]))
    box2_area = abs((box2_corners[:,2:3,:,:] - box2_corners[:,0:1,:,:]) * (box2_corners[:,3:4,:,:] - box2_corners[:,1:2,:,:]))
    return intersection_area / (box1_area + box2_area - intersection_area + 1e-6) # (Batch, 1, S, S)

# 모델에서 추론한 텐서를 가져와서 non-maximum suppression 을 수행한다 이미지 한장씩 수행, 텐서 입력하기 전에 Batch차원 없애야 제대로 작동한다
def NMS(predictions: Float[torch.Tensor, "features"], iou_threshold = 0.5, threshold = 0.4, S: int=7, B: int=2, C: int=20):
    predictions = predictions.reshape(C + B * 5, S, S) # (30, 7, 7)
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

    flatten = all_scores.view(all_scores.size(0), -1) # (24, 98)
    flatten_sorted, indices = torch.sort(flatten, dim = 1, descending = True)
    for i in range(20): # 20개의 클래스에 대해 순차적 진행 (NMS는 매 클래스마다 다르게 처리되므로 벡터화 불가능
        for boxi in range(S*S*2):
            if flatten_sorted[i,boxi] == 0:
                continue
            start = boxi + 1
            for boxj in range(start,S*S*2):
                IoUofBoxes = IoU(flatten_sorted[20:24,boxi:boxi+1].view(1,4,1,1), flatten_sorted[20:24,boxj:boxj+1].view(1,4,1,1))
                if IoUofBoxes[:,0:1,:,:].item() > iou_threshold:
                    flatten_sorted[i,boxj] = 0

    ret_tensor = torch.zeros(C + 5 * B , S , S)
    for boxi in range(S*S*2): # [0,98)
        maxscore, classnum = torch.max(flatten_sorted[0:20,boxi:boxi+1], dim = 0)
        if maxscore > 0 and ret_tensor[classnum,boxi // S, boxi % S] == 0:
            ret_tensor[21:25, (boxi // S):(boxi // S) + 1, (boxi % S):(boxi % S) + 1] = flatten_sorted[20:24, boxi:boxi+1].view(4,1,1)
            ret_tensor[classnum,boxi // S, boxi % S] = 1
            ret_tensor[20, boxi // S, boxi % S] = maxscore
    return ret_tensor # (C + 5 * B, S, S)


def mAP(pred_tensor_list, true_tensor_list, iou_threshold=0.5, S = 7, classnum = 20): # 예측결과 텐서 배치의 리스트 레이블 텐서 배치의 리스트 받아 mAP를 계산한다. 박스 변환은 내부에서 처리한다

    pred_nms_list = [] # NMS를 거친 3차원 텐서들의 리스트
    for batch in pred_tensor_list:
        for i in range(batch.size(0)):
            pred = batch[i] # (C + 5 * B, S, S)
            pred_nms = NMS(pred, iou_threshold = iou_threshold)
            pred_nms_list.append(pred_nms)

    pred_nms_batch = torch.stack(pred_nms_list, dim = 0) # 배치차원을 만들면서 cat 해야하기때문에
    true_tensor_batch = torch.stack(true_tensor_list, dim = 0)


    for i in range(20): # 클래스별로, 그리고 confidence threshold별로 TP, FP, FN 구해야 한다
        thresholds = np.arange(0.9, 0.1, -0.1, dtype=float)
        for threshold in thresholds: # threshold마다 반복
            mask = (pred_nms_batch[:,i,:,:] == 1) # i번재 클래스 위치가 1인것만 추출한다 (B, S, S)
            indices = torch.nonzero(mask) # (N, 3) [batch_idx, 행, 열]
            socre_vector = pred_nms_batch[indices[:, 0], i, indices[:, 1], indices[:, 2]] # (N,) 예측값들의 confidence 1차원 벡터
            

            result = torch.where(mask, pred_nms_batch[:,20:21, :,:], )
            pr_matrix = pred_nms_batch[:,i:i+1,:,:].view(pred_nms_batch.size(0)*S*S,-1)
            true_tensor_batch[:,i:i+1,:,:] pred_nms_batch[:,i:i+1,:,:]