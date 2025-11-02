# Yolov1

**Computer Vision Engineer** | 대한민국, 서울 | sejunkwon@outlook.com |

***

## 1. 레포지토리 설명
**Introduction**
* Pytorch를 이용해 모델 구성과 loss 그리고 평가 metric을 Joseph Redmon 의 Paper를 보고 직접 구현한 구현체가 있는 레포지토리 입니다.
* 세부 구현은 모델의 출력에 Sigmoid를 씌운 것을 제외하고 동일합니다.
* ImageNet 데이터와 CNN레이어 + classification layer를 통해 pretrain을 진행하고 이후에 detection lyaer를 붙여 finetuing 할 수 있습니다.
* 대부분의 loss와 metric 구현이 vectorized 되어 있어 학습속도가 매우 빠릅니다.
* 데이터 처리 코드는 Joseph Redmon 의 Darknet 레포지토리의 VOC_Label.py 를 참고해 데이터셋을 전처리하고 경로를 지정했습니다.
* ImageNet 데이터는 Pytorch에 내장된 datasets.ImageFolder메소드를 이용해 데이터셋 객체를 만들어 pretrain에 사용했습니다.

**Prerequisites**
* Python과 Pytorch, Numpy, tqdm 이 설치된 쉘 환경
* 빠른 학습을 위한 Nvidia GPU와 Cuda Toolkit 설치
* 학습을 위해서는 ImageNet-1k 데이터셋과 VOC 2007, 2012 데이터셋이 필요합니다.
* 제가 사용한 데이터셋의 다운로드 링크는 다음과 같습니다. (Joseph Redmon은 더이상 데이터셋과 데이터셋에 대한 링크를 제공하지 않습니다.)\
VOC - <https://www.kaggle.com/datasets/bardiaardakanian/voc0712>\
ImageNet - <https://academictorrents.com/collection/imagenet-2012>

**Computational Power Specification**
* i7 265K 20Core 20Threads
* 96 GB Main Memory
* RTX 3090 Founders Edition

***

## 2. 구현 과정

**Process**
* Computer Vision 을 학습할 때 표준적으로 쓰이는 텐서 shape인 (Batch_size, num_Channels, row, col) 형식의 4차원 Tensor를 입력으로 받고 출력하도록
  함수들을 작성했습니다. 이를 통해 직관적으로 이미지와 피쳐 맵이 어떻게 변화하는지 파악할 수 있습니다.
* ImageNet 데이터를 이용하여 Paper와 동일하게 Pretrain 을 수행하였습니다. Early Stopping 시 Top5 Accuray 에서 82% 가량에서 학습이 멈춰 이 가중치를 사용했습니다.
* BatchNorm 을 사용하면 Paper 의 학습 전략대로 학습이 되지 않습니다. 아마 Dropout Layer 때문에 BatchNorm 을 사용하면 가중치 학습이 잘 안되는 것 같습니다.
* Paper 에서는 Sigmoid 사용에 대한 언급이 없지만 Sigmoid를 사용하지 않으면 IoU 와 좌표 스케일링 변환 등이 음수 좌표 때문에 제대로 이루어지지 않아 기울기 폭발이 일어납니다 그래서 모든 출력값에 Sigmoid를 씌워 주었습니다.
***

## 3. 깨달은 점

* 손실함수 계산을 MSE(평균) 이 아니고 SSE(합) 으로 계산하기 때문에 loss 의 반환값을 배치 사이즈로 나눠줘야 제대로 작동함을 깨달았습니다. 이렇게 하지 않으면 기울기 폭발(nan) 이 일어납니다.
* torch.tensor의 broadcasting을 통해 반복문을 통한 복잡한 계산을 단 몇 줄의 코드로 수행할 수 있음을 깨달았습니다. 한 텐서 차원에서 요소 하나의 값을 다른 텐서의 한 차원 전체에 대해 연산할 수 있다는 점이 인상깊었습니다.
* 데이터로더 객체에서 대이터를 로드할 때 단순히 .to("cuda") 를 사용하면 배치가 로드할 때와 학습할 때 순차적으로 작업이 blocking 된다는 사실을 깨달았습니다. train 과 validation 사이에 유휴시간이 너무 길었습니다.
* Blocking을 해결하기 위해 배치를 로드할때 .cuda(non_blocking = True) 옵션을 통해 비동기적으로 처리할 수 있음을 깨달았고 데이터로더 생성시에 persistant_workers = True를 통해 일정량의 배치를 미리 만들어 두어 유휴시간을 줄일수 있었습니다.
***
