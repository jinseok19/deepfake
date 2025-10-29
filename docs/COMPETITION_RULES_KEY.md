# ⚠️ 대회 핵심 규칙 요약

## 🚨 반드시 지켜야 할 규칙

### 1. **모델 앙상블 금지** ❌
- **여러 개의 모델을 조합하여 예측하는 것은 금지**
- 단일 모델만 제출 가능

#### ❌ 금지되는 것:
```python
# 여러 모델 로드하여 앙상블
model1 = load_model("model1")
model2 = load_model("model2")
pred = (model1(x) + model2(x)) / 2  # ❌ 금지!

# 베이스라인 + 파인튜닝 모델 조합
baseline_pred = baseline_model(x)
finetuned_pred = finetuned_model(x)
final = (baseline_pred + finetuned_pred) / 2  # ❌ 금지!

# 여러 체크포인트 조합
checkpoint1 = load("epoch_5")
checkpoint2 = load("epoch_10")
ensemble = (checkpoint1(x) + checkpoint2(x)) / 2  # ❌ 금지!
```

#### ✅ 허용되는 것:
```python
# 단일 모델 내에서 TTA (Test Time Augmentation)
model = load_model("single_model")
pred_orig = model(image)
pred_flip = model(flip(image))
final = (pred_orig + pred_flip) / 2  # ✅ 허용!

# 단일 모델의 여러 프레임 집계
predictions = []
for frame in video_frames:
    predictions.append(model(frame))
final = mean(predictions)  # ✅ 허용!

# 단일 모델 파인튜닝
model = load_pretrained("vit-base")
model = finetune(model, new_data)
pred = model(x)  # ✅ 허용!
```

---

### 2. 제출 규칙
- **하루 3회** 제출 가능
- 재제출 시 **30분 간격** 필수
- 추론 시간 **최대 3시간**

### 3. 파일 규칙
- 파일명: 반드시 **`task.ipynb`**
- 경로: **상대 경로** 사용 (`./model/...`)
- 불필요 파일 제거 (`.git`, 개인정보 등)

### 4. 환경 제약
- CPU: 8 core
- RAM: 48GB
- GPU: L4 또는 T4
- 스토리지: 수GB (여유 적음)

---

## 💡 전략 가이드

### 단일 모델 내에서 최대 성능 내기

#### 1. **TTA (Test Time Augmentation)** ✅
- 좌우 반전, 회전, 스케일 등
- 같은 모델로 여러 번 추론 → 평균
- **단일 모델 내 기법이므로 허용!**

#### 2. **파인튜닝** ✅
- 공개 데이터셋으로 학습
- FaceForensics++, Celeb-DF 등
- 최종 모델 1개만 제출

#### 3. **프레임 집계 최적화** ✅
- Mean, Max, Median 조합
- 가중치 조정
- **단일 모델의 출력을 집계하는 것이므로 허용!**

#### 4. **전처리 개선** ✅
- 얼굴 탐지 개선
- 프레임 샘플링 최적화
- 데이터 증강

---

## 🎯 추천 실험 순서

1. **TTA 적용** (EXP-003-Fixed 진행 중)
   - 좌우 반전 앙상블
   - 단일 모델, 규칙 준수 ✅

2. **파인튜닝** (다음 단계)
   - FaceForensics++ 200개로 시작
   - 단일 파인튜닝 모델 제출 ✅

3. **전처리 최적화**
   - 프레임 수 증가
   - 동적 샘플링

4. **아키텍처 변경** (최후)
   - ViT 변형 시도
   - 여전히 단일 모델 ✅

---

## ⚠️ 주의사항

### 잘못된 전략 예시
```python
# ❌ 이렇게 하면 안 됨!
baseline = load("baseline_model")
finetuned = load("finetuned_model")
vit = load("vit_model")
resnet = load("resnet_model")

# 여러 모델 앙상블 - 규칙 위반!
final_pred = (baseline(x) + finetuned(x) + vit(x) + resnet(x)) / 4
```

### 올바른 전략
```python
# ✅ 이렇게 해야 함!
# 1. 최고 성능 단일 모델 선택
best_model = load("finetuned_on_ff++")

# 2. TTA 적용 (단일 모델 내)
pred_orig = best_model(image)
pred_flip = best_model(flip(image))
pred_rotate = best_model(rotate(image, 5))

# 3. TTA 결과 평균
final_pred = (pred_orig + pred_flip + pred_rotate) / 3
```

---

## 📌 요약

| 기법 | 허용 여부 | 비고 |
|------|----------|------|
| 모델 앙상블 | ❌ 금지 | 여러 모델 조합 |
| TTA | ✅ 허용 | 단일 모델 내 기법 |
| 파인튜닝 | ✅ 허용 | 최종 1개 모델만 |
| 프레임 집계 | ✅ 허용 | Mean, Max 등 |
| 전처리 개선 | ✅ 허용 | 얼굴 탐지, 샘플링 등 |

**핵심: 단일 모델 내에서 모든 것을 해결해야 함!** 🎯

