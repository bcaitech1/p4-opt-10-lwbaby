# new_baseline
 저희는 프로젝트를 통하여 새로운 모델을 발견하기 보다는 기존에 존재하는 작은 모델에 대해서 성능을 높이는 것에 초점을 두었습니다.

 그래서 주어진 베이스라인 코드를 수정하여 사용할 부분만 압축하여 새로운 베이스라인 코드를 작성하였습니다.
 
 베이스라인은 ```torchvision```의 모델을 불러와 학습시키며, 이 과정에서 다양한 시도를 통해 성능을 높이기 위해 노력하였습니다.

# train.py
 ```torchvision```에서 모델을 불러오고 학습 설정 파일을 읽어와 그대로 학습을 수행합니다.
 
 다음과 같이 선언하여 학습을 수행합니다.
 ```bash
 $ python train.py --train_name test_train --train_config train_config/example.yaml
 ```
 
# model.py
 이 파일에는 모델을 학습시키는 코드와 모델의 사전학습된 가중치를 업데이트하는 코드가 구현되어 있습니다.
 
 ```torchvision```에 구현되어 있는 모델들은 아웃풋 개수만 달라져도 사전학습된 가중치를 받아오지 못 합니다.
 
 그래서 사전학습된 백본의 가중치를 업데이트할 수 있도록 코드를 구현하였습니다.
 
 또한 이 과정에서 백본을 고정하여 백본의 학습이 고정되지 않도록 설정할 수 있습니다.
 
# dataloader.py
 ```DataLoader```를 만드는 코드가 구현되어 있습니다.
 
 저희는 모델을 학습시키는 과정에서 기존의 베이스라인 코드에서 
 
 random augmentation이 모델의 성능을 끌어올리는데 그다지 효과가 없음을 발견하였습니다.
 
 그래서 random_crop만 사용하여 ```Dataset```을 구성하였습니다.
 
# utils.py
 학습을 수행하는데 기존에 베이스라인 코드에서 저희가 사용할 기능들만 가져왔습니다.
 
 MACs를 계산하는 calc_macs 함수와 학습의 설정파일을 읽을 read_yaml 함수가 구현되어 있습니다.
 
# loss.py
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
