import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, in_tensor): # nn.Module 의 forward 를 오버라이드하여 정의해야 한다.
        return self.block(in_tensor)

class DetectionBlock(nn.Module):
    def __init__(self, S, B, C):
        super().__init__()
        self.detectionlayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S * S * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, in_tensor):
        return self.detectionlayers(in_tensor)

class ClassificationBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.classificationlayers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1000),
        )

    def forward(self,in_tensor):
        return self.classificationlayers(in_tensor)

class Yolov1Backbone20(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() # nn.Module 부모클래스의 기능을 온전히 활용하기위해 부모클래스의 생성자를 초기화 한다
        self.backbone20layers = nn.Sequential(
            CNNBlock(3, 64, kernel_size=7, stride=2, padding=3, ),  # 1
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            CNNBlock(64, 192, kernel_size=3, stride=1, padding=1, ),  # 2
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            CNNBlock(192, 128, kernel_size=1, stride=1, padding=0, ),  # 3
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1, ),  # 4
            CNNBlock(256, 256, kernel_size=1, stride=1, padding=0, ),  # 5
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, ),  # 6
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, ),  # 7
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, ),  # 8
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, ),  # 9
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, ),  # 10
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, ),  # 11
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, ),  # 12
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, ),  # 13
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, ),  # 14
            CNNBlock(512, 512, kernel_size=1, stride=1, padding=0, ),  # 15
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, ),  # 16
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, ),  # 17
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, ),  # 18
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, ),  # 19
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, ),  # 20
        )

    def forward(self, in_tensor):
        return self.backbone20layers(in_tensor)

class Yolov1Backbone4(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone4layers = nn.Sequential(
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1, ), # 21
            CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1, ), # 22
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1, ), # 23
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1, ), # 24
        )

    def forward(self, in_tensor):
        return self.backbone4layers(in_tensor)

class Yolov1Model(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20, **kwargs):
        super().__init__()
        self.mode = kwargs.get("mode")

        if self.mode == "pretrain":
            self.pretrainmodel = nn.Sequential(
                Yolov1Backbone20(**kwargs),
                ClassificationBlock()
            )
        elif self.mode == "finetune":
            self.finetunemodel = nn.Sequential(
                Yolov1Backbone20(**kwargs),
                Yolov1Backbone4(**kwargs),
                DetectionBlock(S, B, C),
            )

    def forward(self, x):
        if self.mode == "pretrain":
            return self.pretrainmodel(x)
        elif self.mode == "finetune":
            return self.finetunemodel(x)