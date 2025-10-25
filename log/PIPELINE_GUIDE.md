# 🔄 딥페이크 탐지 파이프라인 가이드

> 전처리 / 모델링 / 후처리 완벽 이해하기

---

## 📊 전체 파이프라인 개요

```
입력 데이터 (이미지/동영상)
         ↓
┌─────────────────────────────────────┐
│ 1️⃣ 전처리 (Pre-processing)          │
│  - 데이터 준비 및 변환                │
│  - CPU 작업                          │
├─────────────────────────────────────┤
│ • 동영상 → 프레임 샘플링              │
│ • 얼굴 검출 (dlib)                   │
│ • 얼굴 크롭 & 리사이즈                │
│ • 데이터 증강 (선택)                 │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 2️⃣ 모델링 (Modeling)                │
│  - AI 모델 추론                      │
│  - GPU 작업                          │
├─────────────────────────────────────┤
│ • ViT 모델 로드                      │
│ • 배치 처리                          │
│ • Softmax 확률 출력                  │
│   → [Real 확률, Fake 확률]           │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3️⃣ 후처리 (Post-processing)         │
│  - 예측 결과 집계 및 최종화            │
│  - CPU 작업                          │
├─────────────────────────────────────┤
│ • 프레임 필터링                       │
│ • 확률 집계 (Mean, Max 등)           │
│ • Argmax 최종 예측                   │
│   → 0 (Real) or 1 (Fake)            │
└─────────────────────────────────────┘
         ↓
    최종 결과
```

---

## 1️⃣ 전처리 (Pre-processing)

### 정의
**원본 데이터를 모델이 이해할 수 있는 형태로 변환하는 단계**

### 주요 작업
1. **프레임 샘플링** (동영상 → 이미지들)
2. **얼굴 검출** (dlib, MTCNN 등)
3. **크롭 & 리사이즈** (224x224)
4. **데이터 증강** (좌우 반전, 회전 등)
5. **정규화** (0~1 범위로 변환)

### 관련 파라미터
```python
# 전처리 파라미터
NUM_FRAMES = 30              # 동영상 샘플 프레임 수
margin = 1.3                 # 얼굴 박스 여유 비율
TARGET_SIZE = (224, 224)     # 최종 입력 크기
resize_for_detection = 640   # 얼굴 검출용 해상도

# 증강 설정
horizontal_flip = False      # 좌우 반전 여부
rotation_degree = 0          # 회전 각도
```

### 코드 예시
```python
# 전처리 핵심 함수
def detect_and_crop_face(image, target_size=(224, 224)):
    """
    전처리: 얼굴 검출 → 크롭 → 리사이즈
    """
    # 1. 얼굴 검출
    faces = face_detector(image)
    
    # 2. 크롭 (얼굴 있으면 얼굴, 없으면 중앙)
    if faces:
        face_img = crop_face(image, faces[0], margin=1.3)
    else:
        face_img = center_crop(image)
    
    # 3. 리사이즈
    face_img = face_img.resize(target_size)
    
    return face_img

def process_video(video_path):
    """
    전처리: 동영상 → 프레임 샘플링
    """
    # 프레임 샘플링
    frame_indices = np.linspace(0, total_frames-1, NUM_FRAMES)
    
    for idx in frame_indices:
        frame = extract_frame(video_path, idx)
        face = detect_and_crop_face(frame)
        face_images.append(face)
    
    return face_images
```

### 개선 아이디어
- ✅ **프레임 수 증가**: 30 → 40 (정보량 ↑)
- ✅ **중앙 크롭 추가**: 얼굴 미검출 시 정보 활용
- ⏸️ **TTA 적용**: 좌우 반전, 회전 등
- ⏸️ **동적 샘플링**: 장면 전환 감지

---

## 2️⃣ 모델링 (Modeling)

### 정의
**전처리된 데이터를 AI 모델에 입력하여 예측 확률을 얻는 단계**

### 주요 작업
1. **모델 로드** (ViT, ResNet 등)
2. **배치 처리** (여러 이미지 동시 처리)
3. **순전파** (Forward Pass)
4. **Softmax** (확률 변환)

### 관련 파라미터
```python
# 모델링 파라미터
model_name = "ViT-base"      # 사용 모델
BATCH_SIZE = 16              # GPU 배치 크기
device = "cuda"              # GPU 사용
num_workers = 8              # CPU 워커 수

# 모델 설정 (변경 시)
# model.config.xxx = yyy
```

### 코드 예시
```python
# 모델링 핵심 함수
def predict_batch(model, processor, images, device="cuda"):
    """
    모델링: 배치 추론
    """
    # 1. 전처리 (정규화 등)
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    # 2. 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, 2)
    
    # 3. Softmax (확률 변환)
    probs = F.softmax(logits, dim=1)  # shape: (batch_size, 2)
    # probs[i] = [Real 확률, Fake 확률]
    
    return probs.cpu().numpy()

# 배치 처리
all_probs = []
for i in range(0, len(images), BATCH_SIZE):
    batch = images[i:i + BATCH_SIZE]
    batch_probs = predict_batch(model, processor, batch)
    all_probs.extend(batch_probs)
```

### 개선 아이디어
- ⏸️ **모델 앙상블**: 여러 번 추론 후 평균
- ⏸️ **다른 아키텍처**: ResNet, EfficientNet
- ✅ **배치 크기 조정**: 메모리 vs 속도

---

## 3️⃣ 후처리 (Post-processing)

### 정의
**모델의 예측 확률들을 집계하여 최종 예측을 만드는 단계**

### 주요 작업
1. **프레임 필터링** (신뢰도 낮은 프레임 제거)
2. **확률 집계** (Mean, Max, Median 등)
3. **최종 예측** (Argmax)

### 관련 파라미터
```python
# 후처리 파라미터
MEAN_WEIGHT = 0.6            # 평균 가중치
MAX_WEIGHT = 0.4             # 최대 가중치
use_face_filter = True       # 프레임 필터링 여부
threshold = 0.5              # 확률 임계값 (사용 시)
```

### 코드 예시
```python
# 후처리 핵심 함수
def aggregate_video_predictions(probs_list, face_detected_flags):
    """
    후처리: 여러 프레임 확률 → 1개 최종 예측
    
    Input:
        probs_list: [(0.3, 0.7), (0.4, 0.6), ..., (0.2, 0.8)]
        face_detected_flags: [True, True, False, ...]
    
    Output:
        final_prediction: 0 or 1
    """
    probs_array = np.array(probs_list)  # shape: (30, 2)
    
    # 1. 프레임 필터링 (얼굴 검출 성공만)
    detected_indices = np.where(face_detected_flags)[0]
    if detected_indices:
        probs_array = probs_array[detected_indices]
    
    # 2. 확률 집계
    mean_probs = probs_array.mean(axis=0)  # [Real평균, Fake평균]
    max_probs = probs_array.max(axis=0)    # [Real최대, Fake최대]
    
    # Mean 60% + Max 40% 조합
    combined_probs = MEAN_WEIGHT * mean_probs + MAX_WEIGHT * max_probs
    
    # 3. 최종 예측
    final_prediction = np.argmax(combined_probs)  # 0 or 1
    
    return final_prediction
```

### 집계 방식 비교

#### 방식 1: Mean only (베이스라인)
```python
avg_probs = probs.mean(axis=0)
prediction = np.argmax(avg_probs)

# 장점: 안정적, 노이즈에 강함
# 단점: 강한 신호 놓칠 수 있음
```

#### 방식 2: Mean + Max (EXP-001)
```python
combined = 0.6 * mean + 0.4 * max
prediction = np.argmax(combined)

# 장점: 강한 신호 포착, 균형잡힘
# 단점: Max가 노이즈일 수 있음
```

#### 방식 3: Median
```python
median_probs = np.median(probs, axis=0)
prediction = np.argmax(median_probs)

# 장점: 아웃라이어에 강함
# 단점: 정보 손실 가능
```

### 개선 아이디어
- ✅ **Mean+Max 조합**: 강한 신호 포착
- ✅ **프레임 필터링**: 신뢰도 높은 프레임만
- ⏸️ **가중치 최적화**: Grid Search
- ⏸️ **Median 추가**: Mean+Max+Median 조합

---

## 🎯 단계별 개선 전략

### 전처리 개선이 효과적인 경우
- **정보 손실이 있을 때** (얼굴 미검출 등)
- **데이터가 부족할 때** (증강 필요)
- **샘플링이 부적절할 때** (프레임 수, 간격)

**예시**:
- 얼굴 미검출 → 중앙 크롭 (+3~4%)
- 프레임 30 → 40 (+2~3% 예상)

### 모델링 개선이 효과적인 경우
- **모델 성능이 부족할 때**
- **앙상블이 필요할 때**
- **계산 자원이 충분할 때**

**예시**:
- ViT → ViT+ResNet 앙상블 (+5~10% 예상)
- 단일 추론 → 5회 추론 평균 (+2~3% 예상)

### 후처리 개선이 효과적인 경우
- **집계 방식이 단순할 때**
- **노이즈가 많을 때**
- **빠른 실험이 필요할 때** (모델 재학습 불필요)

**예시**:
- Mean only → Mean+Max (+2~3%)
- 전체 프레임 → 필터링 (+1~2%)

---

## 📊 실험 설계 가이드

### 좋은 실험 설계
```
EXP-002: 전처리 개선 실험

목표: 프레임 수 증가로 정보량 향상
가설: 더 많은 프레임 = 더 정확한 예측

변경:
  1️⃣ 전처리: NUM_FRAMES 30→40
  2️⃣ 모델링: 변경 없음
  3️⃣ 후처리: 변경 없음

예상 효과: +3~4%
```

### 나쁜 실험 설계
```
EXP-XXX: 여러 개 바꿔보기

변경:
  - NUM_FRAMES 증가
  - MEAN_WEIGHT 변경
  - margin 변경
  - 모델도 바꿔보기

→ 뭐가 효과 있었는지 알 수 없음!
```

---

## 💡 핵심 정리

| 단계 | 작업 | 개선 방법 | 효과 |
|------|------|-----------|------|
| **전처리** | 데이터 준비 | 프레임 수, 증강, 크롭 | 정보량 ↑ |
| **모델링** | AI 추론 | 모델 변경, 앙상블 | 성능 ↑ |
| **후처리** | 결과 집계 | 집계 방식, 필터링 | 정확도 ↑ |

### 개선 우선순위
1. **전처리** - 빠르고 효과적 (재학습 불필요)
2. **후처리** - 빠르고 쉬움 (재학습 불필요)
3. **모델링** - 느리고 어려움 (재학습 필요할 수 있음)

---

**마지막 업데이트**: 2024.10.24  
**버전**: v1.0

