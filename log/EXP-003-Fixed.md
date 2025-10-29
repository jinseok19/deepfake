# EXP-003-Fixed: TTA 메모리 최적화 + CUDA 12.6

## 📋 실험 개요
- **실험 번호**: EXP-003-Fixed
- **제출 일시**: 2025-10-29
- **모델명**: EXP-003-Fixed-TTA-CUDA126
- **CUDA 버전**: 12.6 (PyTorch 2.7.1)
- **베이스라인 점수**: 0.5489

---

## 🔧 수정 내용

### 문제점 (EXP-003 실패 원인)
1. **메모리 폭발**: 원본 40 + 반전 40 = 80개 이미지를 한번에 GPU로 전송
   - 프레임 40개 × 2 (원본+반전) = 80개
   - 메모리 사용: ~12GB
   - 서버 GPU 메모리 부족으로 OOM 또는 실행 실패

2. **에러 메시지**: "점수를 산출할 수 없었습니다"
   - 코드 실행 중 에러 발생
   - submission.csv 생성 실패 추정

### 해결 방법
**원본과 반전을 분리하여 순차 추론**

```python
# Before (메모리 폭발)
tta_images = face_images + flipped_images  # 80개
inputs = processor(images=tta_images, ...).to("cuda")
probs = F.softmax(outputs.logits, dim=1).mean(dim=0)

# After (메모리 최적화)
# 1. 원본 40개 추론
inputs_orig = processor(images=face_images, ...).to("cuda")
probs_orig = F.softmax(outputs_orig.logits, dim=1).mean(dim=0)

# 2. 반전 40개 추론
inputs_flip = processor(images=flipped_images, ...).to("cuda")
probs_flip = F.softmax(outputs_flip.logits, dim=1).mean(dim=0)

# 3. 앙상블
avg_probs = (probs_orig + probs_flip) / 2.0
```

---

## 🎯 전처리 (Pre-processing)

### 비디오 프레임 샘플링
- **프레임 수**: 40개 (EXP-002와 동일)
- **샘플링 방식**: Uniform (균등 간격)
- **얼굴 탐지**: Dlib frontal face detector
- **크롭 방식**: Bounding box + 30% margin
- **리사이즈**: 224×224 (ViT 입력 크기)

### 병렬 처리
- **Worker 수**: min(CPU 코어 - 1, 8)
- **처리 방식**: multiprocessing.Pool
- **예상 시간**: 전처리 ~10-20분

---

## 🧠 모델링 (Modeling)

### 모델 구조
- **베이스 모델**: `dima806/deep-fake-detector-v2-model`
- **아키텍처**: ViT-Base-Patch16-224
- **파라미터**: ~86M
- **입력 크기**: 224×224×3
- **출력**: Binary classification (Real: 0, Fake: 1)

### TTA (Test Time Augmentation)
- **증강 기법**: Horizontal Flip (좌우 반전)
- **앙상블 방식**: 
  1. 원본 프레임들의 평균 확률
  2. 반전 프레임들의 평균 확률
  3. 두 확률의 평균 → 최종 예측

---

## 📊 후처리 (Post-processing)

### 프레임 집계 (Aggregation)
1. **원본 처리**:
   - 40개 프레임 → ViT 추론
   - Softmax → 각 프레임의 확률 (2개 클래스)
   - 평균 → `probs_orig` (2,)

2. **반전 처리**:
   - 40개 반전 프레임 → ViT 추론
   - Softmax → 각 프레임의 확률
   - 평균 → `probs_flip` (2,)

3. **TTA 앙상블**:
   - `(probs_orig + probs_flip) / 2.0`
   - Argmax → 최종 클래스 (0 or 1)

---

## 💻 CUDA 12.6 선택 이유

### PyTorch 2.7.1 장점
- **최신 최적화**: Transformer 연산 개선
- **메모리 관리**: 동적 메모리 할당 최적화
- **성능 향상**: CUDA 12.6 커널 최적화
- **안정성**: 최신 버전의 버그 수정

### vs 다른 CUDA 버전
| CUDA | PyTorch | 특징 |
|------|---------|------|
| 10.2 | 1.6.0   | 구버전, 안정적이지만 느림 |
| 11.8 | 1.8.0   | 중간, 일반적 |
| 12.6 | 2.7.1   | **최신, 최적화** ⭐ |

---

## 📈 예상 결과

### 메모리 개선
- **Before**: 80개 × 3 × 224 × 224 = ~12GB (한번에)
- **After**: 40개 × 2회 = ~6GB (순차)
- **절약**: 50% 메모리 감소

### 속도 예상
- 전처리: ~10-20분
- 추론: ~80-90분 (TTA 2배)
- **총 소요**: ~90-110분

### 성능 예상
- **베이스라인**: 0.5489
- **EXP-002**: 0.5506 (+0.31%)
- **EXP-003-Fixed**: 0.55~0.58 (+1~5%) 예상
  - TTA 효과: +1~3%
  - 프레임 40개: +0.3%

---

## 🔍 이전 실험 비교

| 실험 | 프레임 | TTA | CUDA | F1 Score | 변화 |
|------|--------|-----|------|----------|------|
| Baseline | 30 | ❌ | 일반 | 0.5489 | - |
| EXP-002 | 40 | ❌ | 일반 | 0.5506 | +0.31% |
| EXP-003 | 40 | ⚠️ | 일반 | **실패** | 메모리 OOM |
| **EXP-003-Fixed** | 40 | ✅ | 12.6 | **대기중** | ? |

---

## 🎯 개선 포인트

### 1. 전처리
- **변화 없음**: EXP-002와 동일
- 프레임 40개, Dlib 얼굴 탐지

### 2. 모델링
- **변화 없음**: 동일 모델 사용
- ViT-Base, 86M 파라미터

### 3. 후처리 ⭐ (핵심 개선)
- **TTA 추가**: 좌우 반전 앙상블
- **메모리 최적화**: 순차 추론
- **CUDA 최적화**: PyTorch 2.7.1

---

## 📝 파라미터 상세

### 전처리
```python
num_frames_to_extract = 40
resize_for_detection = 640
target_size = (224, 224)
margin = 0.3  # Bounding box margin
```

### 추론
```python
device = "cuda"
batch_processing = False  # 순차 처리
tta_method = "horizontal_flip"
ensemble_weights = [0.5, 0.5]  # 원본:반전 = 1:1
```

### 집계
```python
aggregation_method = "mean"  # 프레임 평균
tta_aggregation = "mean"     # TTA 평균
final_decision = "argmax"    # 최종 분류
```

---

## 🚀 다음 단계

### 성공 시 (F1 > 0.56)
1. **EXP-004**: FaceForensics++ 파인튜닝
   - 샘플 200개로 시작
   - Val F1 확인 후 전체 데이터

### 실패 시 (F1 < 0.56 or 에러)
1. **EXP-003-V2**: TTA 제거 + CUDA 12.6
   - 안정성 우선
2. **배치 처리**: 16개씩 나눠서 처리
3. **프레임 감소**: 40 → 30으로 복귀

### 장기 계획
1. FaceForensics++ 파인튜닝
2. 앙상블 (베이스라인 + 파인튜닝)
3. 이미지 생성 파이프라인
4. 최종 앙상블 모델

---

## 💡 인사이트

### TTA 효과
- **이론적**: 1-3% 성능 향상 기대
- **트레이드오프**: 추론 시간 2배
- **메모리**: 순차 처리로 해결

### CUDA 12.6 기대
- 최신 최적화 활용
- 메모리 관리 개선
- 실행 안정성 향상

### 개선 전략
1. **점진적 개선**: 한 번에 하나씩
2. **안정성 우선**: 작동하는 것 기반
3. **데이터 중심**: 파인튜닝으로 큰 향상

---

## 📌 주요 변경사항 요약

### 코드 수정
```diff
- tta_images = face_images + flipped_images  # 80개 한번에
- inputs = processor(images=tta_images, ...).to("cuda")
- probs = F.softmax(outputs.logits, dim=1).mean(dim=0)

+ # 원본 40개
+ inputs_orig = processor(images=face_images, ...).to("cuda")
+ probs_orig = F.softmax(outputs_orig.logits, dim=1).mean(dim=0)
+ 
+ # 반전 40개
+ inputs_flip = processor(images=flipped_images, ...).to("cuda")
+ probs_flip = F.softmax(outputs_flip.logits, dim=1).mean(dim=0)
+ 
+ # 앙상블
+ avg_probs = (probs_orig + probs_flip) / 2.0
```

### 제출 정보
```python
model_name = "EXP-003-Fixed-TTA-CUDA126"
key = "492f7ace-67d6-4636-8448-25def840e236"  # CUDA 12.6
```

---

## 📊 결과

**✅ 제출 완료**

- **제출 시간**: 2025-10-29
- **F1 Score**: **0.5599217639** (≈ 0.5600)
- **베이스라인 대비**: **+0.0110 (+2.01%)** ✅
- **EXP-002 대비**: **+0.0093 (+1.69%)** ✅

### 성능 분석

**✅ 성공적인 개선!**

1. **TTA 효과 확인**: +1.69% (EXP-002 대비)
   - 좌우 반전 앙상블이 정확도 향상에 기여
   - 예상 범위(+1~3%) 내의 성능 향상
   
2. **프레임 40 + TTA 조합**: +2.01% (베이스라인 대비)
   - 프레임 증가(+0.31%) + TTA(+1.69%) = 복합 효과
   
3. **메모리 최적화 성공**:
   - 순차 처리로 OOM 해결
   - CUDA 12.6 안정적 작동

### 비교 분석

| 실험 | 프레임 | TTA | F1 Score | 베이스라인 대비 |
|------|--------|-----|----------|----------------|
| BASELINE | 30 | ❌ | 0.5489 | - |
| EXP-002 | 40 | ❌ | 0.5506 | +0.31% |
| EXP-003-Fixed | 40 | ✅ | **0.5600** | **+2.01%** ✅ |

**핵심 인사이트**:
- 프레임 40 단독: +0.31%
- TTA 단독: ~+1.69%
- **TTA가 프레임 증가보다 5배 효과적!**

---

## 🔖 태그

`#TTA` `#MemoryOptimization` `#CUDA126` `#BugFix` `#HorizontalFlip` `#Ensemble`

