# p4-opt-10-lwbaby
  <img src=https://user-images.githubusercontent.com/56903243/119317896-99c62680-bcb3-11eb-8495-a3372fabd656.jpg width = 30%>
부스트캠프 AI tech 프로젝트4 모델 최적화 'Lightweight Baby!'팀입니다.

# Task Overview
**What**: 분리수거 자동화를 위해 이미지 분류 모델 경량화, 모든 클래스에 대해 정확도 향상.  
**Why**: 저사양 디바이스 혹은 클라우드 서버와의 통신을 이용할 때 실시간 분리수거를 수행할 수 있는 빠른 처리 속도가 요구됨.  
**How**:  
 경량화의 경우, 우선 Hyper parameter를 선정하고 (1) NAS를 통한 모델 구조 최적화, (2) Knowledge Distillation을 이용한 스크래치 학습, (3) Pruning 및 Tensor Decomposition을 이용한 모델 압축 (4) Quantization 및 Compiling을 통한 수행 속도 향상의 파이프라인을 거친다.  
 분류 정확도의 경우, Data Preprocessing, Data Augmentation, Hyper parameter Optimization 등의 방법을 이용한다.

## 평가 방법
경량화는 연산량 감소를, 정확도는 F1 score 향상으로 판단한다.  
따라서 검증 및 평가는 MACs 값이 작을수록, F1 score가 클수록 좋은 점수를 기준으로 수행한다.  
![image](https://user-images.githubusercontent.com/66929142/122687740-f2f18d80-d252-11eb-964d-71dcf7f57c8e.png)

# Getting Started
 저희는 프로젝트를 통하여 새로운 모델을 발견하기 보다는 기존에 존재하는 작은 모델에 대해서 성능을 높이는 것에 초점을 두었습니다.

 그래서 주어진 베이스라인 코드를 수정하여 사용할 부분만 압축하여 새로운 베이스라인 코드를 작성하였습니다.
 
 베이스라인은 ```torchvision```의 모델을 불러와 학습시키며, 이 과정에서 다양한 시도를 통해 성능을 높이기 위해 노력하였습니다.

## train.py
 ```torchvision```에서 모델을 불러오고 학습 설정 파일을 읽어와 그대로 학습을 수행합니다.
 
 다음과 같이 선언하여 학습을 수행합니다.
 ```bash
 $ python train.py --train_name test_train --train_config train_config/example.yaml
 ```
 
 wandb의 sweep을 이용한 hyper-parameter optimization  
 ![image](https://user-images.githubusercontent.com/66929142/122791498-3cde8000-d2f4-11eb-9097-4cba45030bd9.png)  
 ![image](https://user-images.githubusercontent.com/66929142/122791516-40720700-d2f4-11eb-8614-3a8c86473a19.png)  

## model.py
 이 파일에는 모델을 학습시키는 코드와 모델의 사전학습된 가중치를 업데이트하는 코드가 구현되어 있습니다.
 
 ```torchvision```에 구현되어 있는 모델들은 아웃풋 개수만 달라져도 사전학습된 가중치를 받아오지 못 합니다.
 
 그래서 사전학습된 백본의 가중치를 업데이트할 수 있도록 코드를 구현하였습니다.
 
 또한 이 과정에서 백본을 고정하여 백본의 학습이 되지 않도록 설정할 수 있습니다.
 
 Tucker Decomposition for Conv Layer  
 분해할 conv layer가 model의 0번째 레이어인 경우의 예시입니다.   
 ![image](https://user-images.githubusercontent.com/66929142/122791848-90e96480-d2f4-11eb-991e-26111b0f343a.png)

## dataloader.py
 ```DataLoader```를 만드는 코드가 구현되어 있습니다.
 
 저희는 모델을 학습시키는 과정에서 기존의 베이스라인 코드에서 
 
 random augmentation이 모델의 성능을 끌어올리는데 그다지 효과가 없음을 발견하였습니다.
 
 그래서 random_crop만 사용하여 ```Dataset```을 구성하였습니다.
 
## utils.py
 학습을 수행하는데 기존에 베이스라인 코드에서 저희가 사용할 기능들만 가져왔습니다.
 
 MACs를 계산하는 calc_macs 함수와 학습의 설정파일을 읽을 read_yaml 함수가 구현되어 있습니다.
 
## loss.py
 모델의 학습을 수행할 때 사용한 loss가 구현되어 있습니다.
 
 f1 loss를 Cross Entropy loss와 함께 사용해 약간의 성능 향상을 볼 수 있었습니다.
 
 f1 loss는 [링크](https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354)의 코드를 사용하였습니다.
 
 또한 성능 향상 효과는 볼 수 없었지만 KL divergence를 이용한 distillation loss도 구현되어 있습니다.
 
 KL_distillation_loss는 다음과 같이 선언합니다.
 ```python3
 import torch.nn as nn
 
 student_loss = nn.CrossEntropy()
 criterion = KL_distillation_loss(teacher_model, student_loss, alpha=0.9, tau=3)
 ```
 학습에서는 다음과 같이 loss값을 계산합니다.
 ```python3
for data, labels in dataloader:
    ...
    
    if isinstance(criterion, KL_distillation_loss):
        loss = criterion(outputs, labels, data)
    else:
        loss = criterion(outputs, labels)
    ...
    
 ```
 이 때 각각 ```running_student_loss```멤버 변수와 ```running_distillation_loss```멤버 변수를 이용해 축적된 각각의 loss 값을 확인할 수 있습니다.
 
 또한, ```reset_running_loss``` 메소드를 이용해 위의 변수들의 값을 초기화할 수 있습니다.
 
# Model Overview
**Backbone**:  
- ShuffleNet V2의 채널을 반으로 줄인 모델  
- torchvision의 pretrained 모델 사용  

**구조 변경**:  
stage4 및 Conv-BN-ReLU 레이어 제거  
![image](https://user-images.githubusercontent.com/66929142/122687812-5e3b5f80-d253-11eb-967c-ceb62cb879ab.png)

# Configs
## Data Configs
- Image size: 64 x 64  
backbone의 사전학습에 사용된 image size는 224 x 224이며, 모델 경량화를 위해 작은 이미지를 사용한다.
## Training Configs
-	Criterion: CrossEntropyLoss (Pytorch)
-	Optimizer: AdamW (Pytorch)
-	LR Scheduler: CosineAnnealingWarmRestarts (Pytorch)

# Techniques
## Data Preprocessing & Augmentation
- Zero Padding
- CutOut, CutMix
## Training
- F1 Loss, Distillation Loss
