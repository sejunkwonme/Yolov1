import torch
import torch.nn as nn
from utils import IoU
from jaxtyping import Float

class Yolov1Loss(nn.Module):
    def __init__(self, S: int=7, B: int=2, C: int=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions: Float[torch.Tensor, "Batch features"],
                target: Float[torch.Tensor, "Batch features"]) -> float:
        predictions = predictions.reshape(-1, self.C + self.B * 5, self.S, self.S)

        # target boundingbox 는 21:25 한개만 사용, 2개의 예측에 대해IoU 계산
        iou_b1 = IoU(predictions[:, 21:25, :, :], target[:, 21:25, :, :])
        iou_b2 = IoU(predictions[:, 26:30, :, :], target[:, 21:25, :, :])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # IoU가 높은것을 선택해서 학습에 이용한다
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[:, 20:21, :, :]  # in paper this is Iobj_i 0 하고 1중 하나이다
        # (매우 중요 브로드캐스팅 된다. 20:21 은 2번째 차원 값의 개수가 1이므로

        # Box Localization loss
        # 레이블에서, 셀에 오브젝트가 없으면 0을 곱해 없는것도 학습 해야한다
        # 오브젝트가 있다면, exists_box 는 1이고
        # bestbox는 예측 박스 2개중에 iou가 큰 것의 인덱스이다 이 때 위의 torch.max 에 의해 0 또는 1이 될 수 있으므로
        # 2개 중에 iou가 큰 박스만 적용 가능하다 첫번째 것이 best이면 bestbox 는 0으로 나오므로
        # 21:25 의 박스좌표계를 살리고 두번째 것이 best이면 반대로 작용
        box_predictions = exists_box * (
            (
                bestbox * predictions[:, 26:30, :, :]
                + (1 - bestbox) * predictions[:, 21:25, :, :]
            )
        )
        box_targets = exists_box * target[:, 21:25, :, :]

        box_predictions_root = torch.sign(box_predictions[:, 2:4, :, :]) * torch.sqrt(
            torch.abs(box_predictions[:, 2:4, :, :] + 1e-6)
        )
        box_targets_root = torch.sqrt(box_targets[:,2:4,:,:])

        box_loss = self.mse(box_predictions_root,box_targets_root)

        # Object loss
        pred_box = (bestbox * predictions[:, 25:26, :, :] + (1 - bestbox) * predictions[:, 20:21, :, :])
        object_loss = self.mse((exists_box * pred_box),(exists_box * target[:, 20:21, :, :]))

        # No Object loss
        no_object_loss = self.mse((1 - exists_box) * predictions[:, 20:21, :, :], (1 - exists_box) * target[:, 20:21, :, :])
        no_object_loss += self.mse((1 - exists_box) * predictions[:, 25:26, :, :],(1 - exists_box) * target[:, 20:21, :, :])

        # Class loss
        class_loss = self.mse(exists_box * predictions[:,:20,:,:],exists_box * target[:,:20,:,:])

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss