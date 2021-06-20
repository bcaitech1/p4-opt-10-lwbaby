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
backbone의 pretraining에 사용된 image size는 224 x 224이며, 모델 경량화를 위해 작은 이미지를 사용한다.
## Training Configs
-	Criterion: CrossEntropyLoss (Pytorch)
-	Optimizer: AdamW (Pytorch)
-	LR Scheduler: CosineAnnealingWarmRestarts (Pytorch)
