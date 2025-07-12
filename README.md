# 스마트 배치 사이즈 선택기 (Smart Batch Size Selector)

훈련 정확도와 시간 효율을 기반으로, 모델 학습에 가장 적절한 배치 사이즈를 자동으로 선택해주는 경량 유틸리티입니다.

## 개요

이 도구는 √N 주변의 2의 거듭제곱 배치 크기들에 대해 훈련 효율을 측정한 후,  
정확도 대비 시간 효율이 가장 좋은 배치 사이즈를 선택합니다.  
다음과 같은 기능이 포함되어 있습니다:

- `selectBatchSize`: 어떤 모델과 데이터셋에도 사용할 수 있는 범용 함수
- `getBatchSizeMNIST`: PyTorch와 MNIST를 이용한 데모 함수

## 주요 특징

- 실험 기반 선택: 고정 규칙이나 추측 없이 선택
- √N 중심의 2의 거듭제곱 후보 비교 (기본: widen=2)
- 모델/훈련 루프 함수는 자유롭게 정의 가능
- 코드 통합이 쉽고 가벼움

## 함수 설명

### `selectBatchSize(N, model_fn, train_fn, widen=2)`

사용자 정의 모델과 훈련 로직을 통해 가장 효율적인 배치 사이즈를 선택합니다.

**매개변수**:
- `N` *(int)*: 전체 학습 데이터 수
- `model_fn()` *(Callable)*: 새 모델을 반환하는 함수
- `train_fn(model, batch_size)` *(Callable)*: 모델을 학습하고 `(정확도, 시간)` 반환
- `widen` *(int)*: √N 기준으로 좌우 비교할 2ⁿ 후보 개수

**반환값**: `int` - 가장 효율적인 배치 사이즈

---

### `getBatchSizeMNIST(N, widen=2)`

MNIST 데이터셋과 기본 모델을 사용한 내장 실험을 실행합니다.

**반환값**: `int` - 지정한 N에 대해 가장 적합한 배치 사이즈

---

## 사용 예시

```python
from batch_selector import selectBatchSize, getBatchSizeMNIST

# 간편 데모: MNIST
best = getBatchSizeMNIST(N=3000)
print("Best batch size (MNIST):", best)

# 사용자 정의 모델 적용
def model_fn():
    return MyModel()

def train_fn(model, batch_size):
    # 모델 학습 및 평가
    return accuracy, time

best = selectBatchSize(N=10000, model_fn=model_fn, train_fn=train_fn)
print("Best batch size (custom):", best)
```

---

## 실행 환경

- Python 3.7 이상
- PyTorch (MNIST 데모용)
- NumPy (선택)
- CPU 또는 GPU에서 실행 가능

---

## 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다.
