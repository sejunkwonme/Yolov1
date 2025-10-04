import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, in_tensor): # nn.Module 의 forward 를 오버라이드하여 정의해야 한다.
        return self.leakyrelu(self.conv(in_tensor))

class DetectionBlock(nn.Module):
    def __init__(self, S, B, C):
        super().__init__()
        FCLayers = []

        FCLayers += [nn.Flatten()];
        FCLayers += [nn.Linear(S * S * 1024, 4096)]
        FCLayers += [nn.LeakyReLU(0.1)]
        FCLayers += [nn.Dropout(0.5)]
        FCLayers += [nn.Linear(4096, S * S * (B * 5 + C))]

        self.detectionhead = nn.Sequential(*FCLayers)

    def forward(self, in_tensor):
        return self.detectionhead(in_tensor)

class ClassificationBlock(nn.Module):
    def __init__(self, S, B, C):
        super().__init__()
        FCLayers = []

class Yolov1Backbone20(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() # nn.Module 부모클래스의 기능을 온전히 활용하기위해 부모클래스의 생성자를 초기화 한다
        cnnlayers = []

        cnnlayers += [CNNBlock(3, 64, kernel_size=7, stride=2, padding=3, )] # 1
        cnnlayers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        cnnlayers += [CNNBlock(64, 192, kernel_size=3, stride=1, padding=1, )] # 2
        cnnlayers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        cnnlayers += [CNNBlock(192, 128, kernel_size=1, stride=1, padding=0, )] # 3
        cnnlayers += [CNNBlock(128, 256, kernel_size=3, stride=1, padding=1, )] # 4
        cnnlayers += [CNNBlock(256, 256, kernel_size=1, stride=1, padding=0, )] # 5
        cnnlayers += [CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, )] # 6
        cnnlayers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        cnnlayers += [CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, )] # 7
        cnnlayers += [CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, )] # 8
        cnnlayers += [CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, )] # 9
        cnnlayers += [CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, )] # 10
        cnnlayers += [CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, )] # 11
        cnnlayers += [CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, )] # 12
        cnnlayers += [CNNBlock(512, 256, kernel_size=1, stride=1, padding=0, )] # 13
        cnnlayers += [CNNBlock(256, 512, kernel_size=3, stride=1, padding=1, )] # 14
        cnnlayers += [CNNBlock(512, 512, kernel_size=1, stride=1, padding=0, )] # 15
        cnnlayers += [CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, )] # 16
        cnnlayers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        cnnlayers += [CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, )] # 17
        cnnlayers += [CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, )] # 18
        cnnlayers += [CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0, )] # 19
        cnnlayers += [CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1, )] # 20


        self.backbone20 = nn.Sequential(*cnnlayers)

    def forward(self, x):
        return self.backbone20(x)

class Yolov1Backbone4(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cnnlayers = []

        cnnlayers += [CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1, )]  # 21
        cnnlayers += [CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1, )]  # 22

        cnnlayers += [CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1, )]  # 23
        cnnlayers += [CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1, )]  # 24
        self.backbone_4 = nn.Sequential(*cnnlayers)

    def forward(self, x):
        return self.backbone4(x)

class Yolov1Model(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20, **kwargs):
        super().__init__()
        self.mode = kwargs.get("mode", "finetune")

        if self.mode == "pretrain":
            self.backbone20 = Yolov1Backbone20(**kwargs)
            self.classificationhead = ClassificationBlock()
        if self.mode == "finetune":
            self.backbone20 = Yolov1Backbone20(**kwargs)
            self.backbone4 = Yolov1Backbone4(**kwargs)
            self.detectionhead = DetectionBlock()

    def forward(self, x):
        x = self.backbone20(x)
        if self.mode == "pretrain":
            return self.classificationhead(x)
        elif self.mode == "finetune":
            x = self.backbone4(x)
            return self.detectionhead(x)