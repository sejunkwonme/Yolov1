# Yolov1 - Personal Project

**Machine Vision Engineer** | 대한민국, 서울 | sejunkwon@outlook.com |
***

## 1. 레포지토리 설명
**Introduction**
* Pytorch를 이용해 모델 구성과 loss 그리고 평가 metric을 Joseph Redmon 의 Paper를 보고 직접 구현한 구현체가 있는 레포지토리
* 세부 구현은 모델의 출력에 Sigmoid를 씌운 것을 제외하고 동일
* ImageNet 데이터와 CNN레이어 + classification layer를 통해 pretrain을 진행하고 이후에 detection lyaer를 붙여 finetuing 가능
* 대부분의 loss와 Iou등의 utility 구현이 torch.tensor의 연산을 이용해 vectorized 되어 있어 학습속도가 매우 빠름

**Prerequisites**
* Python과 Pytorch, Numpy, tqdm 이 설치된 쉘 환경
* 빠른 학습을 위한 Nvidia GPU와 Cuda Toolkit 설치
* 학습을 위해서는 ImageNet-1k 데이터셋과 VOC 2007, 2012 데이터셋이 필요
* 사용한 데이터셋의 다운로드 링크\
VOC - <https://www.kaggle.com/datasets/bardiaardakanian/voc0712>\
ImageNet - <https://academictorrents.com/collection/imagenet-2012>

**테스트 환경 - Personal Desktop**
* i7 265K 20Core 20Threads
* 96 GB Main Memory
* RTX 3090 Founders Edition VRAM 24GB
* Pretrain 훈련시간 약 1.5일 Finetuing 약 14시간
***

## 2. 구현 과정

* Data 준비
  * 데이터 처리 코드는 Joseph Redmon 의 Darknet 레포지토리의 VOC_Label.py 를 참고해 데이터셋을 전처리
  * ImageNet 데이터는 Pytorch에 내장된 datasets.ImageFolder메소드를 이용해 데이터셋 객체를 만들어 pretrain에 사용
  * NCHW 텐서 shape인 (Batch_size, num_Channels, row, col) 형식의 4차원 Tensor를 입력으로 받고 출력하도록 데이터 처리 함수들을 작성
  * 이미지와 바운딩박스를 448x448로 리사이징할 때 동시에 적용하기 위해 torchvision v2 의 transform과 boundingbboxes를 사용
* Layer 생성
  * conv2d와 leakyRelu 모듈을 이용하여 CNN 레이어를 쌓음
  * ImageNet 데이터를 이용하여 Paper와 동일하게 Pretrain 을 수행, Early Stopping 시 Top5 Accuray 에서 82% 가량에서 학습이 멈춰 이 가중치를 사용
* 학습 전략
  * detection head 와 backbone을 분리구현 및 조건문을 통해 pretrain 과 finetuing 시에 원활하게 교체 가능
***

## 3. 깨달은 점

* 구현
  * 손실함수 계산을 MSE(평균) 이 아니고 SSE(합) 으로 계산하기 때문에 loss 의 반환값을 배치 사이즈로 나눠줘야 제대로 작동함을 깨달았음. 이렇게 하지 않으면 기울기 폭발(nan) 이 일어남
  * torch.tensor의 broadcasting을 통해 반복문을 통한 복잡한 계산을 단 몇 줄의 코드로 수행할 수 있음을 깨달음. 한 텐서 차원에서 요소 하나의 값을 다른 텐서의 한 차원 전체에 대해 연산할 수 있음
* 학습
  * 데이터로더 객체에서 대이터를 로드할 때 단순히 .to("cuda") 를 사용하면 배치가 로드할 때와 학습할 때 순차적으로 작업이 blocking 된다는 사실을 깨달았음. train 과 validation 사이에 유휴시간이 너무 김
  * Blocking을 해결하기 위해 배치를 로드할때 .cuda(non_blocking = True) 옵션을 통해 비동기적으로 처리할 수 있음을 깨달았고 데이터로더 생성시에 persistant_workers = True를 통해 일정량의 배치를 미리 만들어 두어 유휴시간을 줄일수 있었음
  * BatchNorm 을 사용하면 Paper 의 학습 전략대로 학습이 되지 않음, 아마 Dropout Layer 때문에 BatchNorm 을 사용하면 가중치 학습이 잘 안 됨.
  * Paper 에서는 Sigmoid 사용에 대한 언급이 없지만 Sigmoid를 사용하지 않으면 IoU 와 좌표 스케일링 변환 등이 음수 좌표 때문에 제대로 이루어지지 않아 기울기 폭발이 일어남. 그래서 모든 출력값에 Sigmoid를 씌워 주었음.
* 내보내기
  * ONNX 형식으로 export할 때 계산 그래프 내에 동적인 조건(if문)이 있으면 오류가 나고, 각 클래스의 이름들을 적절하게 통일되게 설정해 줘야 에러가 나지 않음.
***

## 4. Report

* Image Example

![screenshot1](/assets/Figure_1.png)
![screenshot2](/assets/Figure_2.png)

* Average Precision per Classes

![screenshot3](/assets/mAP_grpah.png)

* 직접 구현한 모델의 mAP : 0.473
* 논문상 mAP : 0.63