# Yolov1

**Computer Vision Engineer** | 대한민국, 서울 | sejunkwon@outlook.com |  

***

## 1. 레포지토리 설명  
* Pytorch를 이용해 모델 구성과 loss 그리고 평가 metric을 Joseph Redmon 의 Paper를 보고 직접 구현 구현체가 있는 레포지토리 입니다.
* 밑의 명령어를 통해 바로 학습을 시작할 수 있습니다.
'''
python train.py
'''
* Managed social media accounts, content calendars, and performance metrics to ensure consistency across platforms.
* Spearheaded marketing efforts for product launches, achieving a 20% increase in sales within the first quarter.

**Digital Marketing Specialist at Growth Solutions**
New York, NY | May 2016 – July 2019 | 3 years 2 months

* Developed content for email marketing campaigns that achieved a 35% open rate.
* Managed Google Ads campaigns, optimizing keywords and budgets for maximum ROI.
* Analyzed website traffic using Google Analytics and prepared reports to adjust marketing strategies.

***

## 프로젝트


 
**Pytorch를 이용한 Yolov1 구현 - (PersonalProject)**

* ai 모델 플랫폼에서 제공되는 모델을 사용하지 않고 직접 Paper를 보고 모델의 구조, 손실함수, metric 등을 구현
* 그리드 셀 내부와 이미지 전체에 대한 좌표 스케일링이 필요하므로 모든 배치와 i, j 번째 셀에 대해 좌표변환을 수행할 수 있는 함수 구현
* IoU(Intersection over Union), NMS(Non-Maximum-suppression) 그리고 mAP(mean-Average-Precisoin) 등의 메트릭, 후처리 기능을 배치처리로 수행할 수 있도록 구현
* 위의 매트릭 후처리는 for문 없이 torch.Tensor의 연산으로 병렬처리 됨 (NMS는 배치처리가 불가하므로 이미지 한장씩 처리)



***

## 기술스택

**프로그래밍 언어:** C/C++, Python  

**라이브러리:** OpenCV, OpenCL, Cuda, Gstreamer  

**프레임워크:** CMake, Qt, Pytorch  

***
