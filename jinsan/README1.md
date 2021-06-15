**분리수거** 로봇을 위한 인공지능을 만들었습니다. 외부와 연결되지 않고 로봇의 내부 시스템으로만 분리수거를 시행하려면 경량화된 인공지능이 필요합니다.  
구체적인 업무는 재활용품 이미지를 입력받아 9가지 품목 중 하나로 분류하는 것입니다.  
**class imbalance** 문제는 실제 환경과 비슷하도록 데이터를 구성하였기 때문에 발생하였으며, data augmentation과 logit adjustment loss를 통해 해결하려고 하였습니다.  
불균형 문제를 해결하였는지 알아보기 위한 지표로 F1 스코어를 평가에 포함시켰습니다.
**경량화**는 이번 프로젝트의 주된 목적이며, quantization과 tensor decomposition을 적용하였습니다.  
경량화를 나타내는 지표로는 MACs 연산량을 사용하였습니다.

### Augmentation: transforms1(정의), policies(호출)
adjust_bright
- 의도: 이미지 데이터 중 알아보기 힘들 정도로 어두운 이미지가 존재하여, 이를 보정하려고 했습니다.
- 역할: 각 RGB 이미지의 bright 평균값을 128로 만드는 함수입니다.
- 결과: simple augmentation > adjust brightness > random augmentation 순으로 성능이 좋았습니다.
- 분석: 밝기를 통일하는 것은 새로운 정보를 주는 것이 아니고 오히려 clip 등의 함수에서 정보를 손실시키며,
컴퓨터는 어둡다는 것 자체는 상관이 없습니다. 컴퓨터가 아닌 사람을 위해 시각화할 때 필요할 수 있는 작업입니다.
또한 밝기를 통일하여 일반화를 방해하므로 성능을 하락시킨다고 결론지었습니다.
- 기타: random augmentation에서 필요하다고 생각하는 identity, rotate, brightness만 포함하여 training해봤지만, 성능이 좋지 않았습니다.

SquareCrop
- 의도: 이미지를 정사각형으로 만들어야 하는데, 제로 패딩은 이미지의 가장자리에 의미없는 정보를 추가하여 혼란을 줄 수 있다고 생각하였습니다.
이전에 너비와 높이의 비율은 보존해야 성능이 좋았기 때문에, 비율은 유지하면서 기존의 이미지만 사용하고자 했습니다.
- 역할: 이미지의 중앙에서, 너비와 높이 중 작은 길이를 갖는 정사각형으로 잘라냅니다.
- 결과: zero padding >> no padding > square crop 순으로 성능이 좋았습니다. (성능이 매우 좋지 않았습니다.)
- 분석: crop 자체가 정보를 손실시키며, 이미지의 너비와 높이 비율이 최대 2.5배로 매우 커서 정보 손실이 많습니다.

SquareReflectPad_Tensor
- 의도: 정보의 손실 없이 이미지의 비율을 유지하면서 기존의 이미지로 정사각형 이미지를 만들려고 했습니다.
- 역할: 이미지의 너비와 높이 중 더 긴 방향으로 이미지를 반사형태로 반전시켜 붙여넣습니다.
- 결과: zero padding에 비해 accuracy는 높았지만, f1 score는 낮았습니다.
- 분석: zero padding은 중요 부위를 가리지 않은 마스킹과 같으며, 이에 비해 일반화가 덜 되어 성능이 하락했다고 결론지었습니다.

### Augmentation: dataloader, trainer
```
# dataloader.py
from cutmix.cutmix import CutMix

train_dataset = ImageFolder(root=train_path, transform=transform_train)
train_dataset = CutMix(train_dataset, num_class=9, beta=1.0, prob=0.8, num_mix=1)

# train.py
from cutmix.utils import CutMixCrossEntropyLoss

criterion = CutMixCrossEntropyLoss()
```
- 의도: 일반화를 위해 augmentation 방법인 Cutmix를 적용하였습니다.
- 역할: 80% 확률로 한 이미지의 부분을 덧붙입니다.
- 결과: 성능이 다소 상승하였습니다.
- 분석: reflect padding이나 밝기 조정 등에 비해 다른 이미지의 정보가 추가되어 일반화가 잘 되었다고 결론지었습니다.
- 기타: 다른 augmentation 방법인 Cutout 역시 일반화에 도움을 주어 성능을 상승시켰습니다.

### Model Compression
Conv layer Tucker decomposition
- 의도: conv layer의 입출력 채널을 쪼개어 계산량을 줄이고
- 결과: 

depthwise conv layer Tucker decomposition
- 의도: 사용한 모델 shufflenet_v2_x0_5은 대부분의 layer가 decomposition 결과와 유사한 depthwise conv 구조를 가지며,
depthwise conv layer에는 tucker decomposition을 그대로 적용할 수 없어서 적용할 수 있는 형태로 만들었습니다.
- 역할: Pytorch에서 depthwise conv 구조는 입력 채널수가 1인 conv처럼 weight가 구성됩니다.
따라서 weight를 입력 채널수에 맞도록 index 1에 대해 복제하여 tucker decomposition을 수행합니다.
분해된 세 개의 conv layer들을 depthwise 형태로 만듭니다.
- 결과: 성능이 매우 좋지 않았습니다.
- 분석: depthwise conv는 이미 파라미터 수가 매우 작은데, 여기서 더 줄이려다 보니 rank 수를 작게 설정할 수밖에 없었고
따라서 성능이 많이 낮아졌다고 판단했습니다.
