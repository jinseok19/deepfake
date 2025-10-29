# 🗺️ 실험 로드맵

> 딥페이크 탐지 대회 - 전략적 실험 순서

---

## 📋 실험 순서

### ✅ 완료된 실험

| EXP | 날짜 | 내용 | 결과 | 상태 |
|-----|------|------|------|------|
| 000 | - | 베이스라인 | F1 0.5489 | ✅ 완료 |
| 001 | 10.24 | 중앙크롭+Mean+Max+필터 | F1 0.5354 (-2.46%) | ❌ 실패 |
| 002 | 10.29 | 프레임 수 증가 (30→40) | **대기중...** | ⏳ 진행중 |

---

## 🎯 예정된 실험 (우선순위 순)

### 1️⃣ EXP-003: TTA (Test Time Augmentation)

**목표**: 추론 시 데이터 증강으로 예측 안정성 향상

**방법**:
```python
# 좌우 반전 앙상블
predictions = []
predictions.append(model(image))           # 원본
predictions.append(model(flip(image)))     # 좌우 반전
final = mean(predictions)
```

**작업 내용**:
- `submit/task.ipynb` 수정
- TTA 함수 추가
- 배치 처리 유지

**예상 효과**: +1~3%
**예상 시간**: 코드 수정 1시간 + 추론 140분
**리스크**: 낮음 (검증된 기법)

**전제 조건**:
- EXP-002 결과 확인 후

---

### 2️⃣ EXP-004: FaceForensics++ 파인튜닝

**목표**: 공개 데이터셋으로 ViT 모델 성능 향상

**방법**:
```python
# 1. FaceForensics++ c23 다운로드
# 2. 얼굴 크롭 전처리 (margin=1.3)
# 3. 베이스라인 모델 파인튜닝
- Base: deep-fake-detector-v2-model
- Epochs: 3~5
- LR: 1e-5
- Batch: 16
```

**작업 내용**:
1. **데이터 준비** (2~3일)
   - FaceForensics++ 신청/다운로드
   - 얼굴 검출 및 크롭
   - Train/Val 분할
   
2. **파인튜닝** (1일)
   - 학습 스크립트 작성
   - GPU 학습 (4~8시간)
   - 검증 및 체크포인트 선택
   
3. **제출 테스트** (반나절)
   - 추론 시간 확인
   - 로컬 검증
   - 제출

**예상 효과**: +5~10%
**예상 시간**: 3~5일
**리스크**: 중간 (오버피팅 주의)

**데이터셋 정보**:
- 이름: FaceForensics++ c23
- 신청: https://github.com/ondyari/FaceForensics
- 용량: 5~10GB (압축 버전)
- 포함: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- 라이선스: 학술 연구 가능 ✅

**주의사항**:
- 모델 크기 유지 (~450MB)
- 추론 시간 3시간 이내
- 스토리지 제한 고려

**전제 조건**:
- EXP-003 완료 및 결과 분석

---

### 3️⃣ EXP-005+: 이미지 생성 파이프라인

**목표**: 최신 딥페이크 기법으로 자체 데이터셋 생성

**방법**:
```python
# ComfyUI 워크플로
1. T2I (Text to Image)
   - 다양한 얼굴 생성
   - 프롬프트: "portrait, single face, clear, ..."
   
2. I2V (Image to Video)  
   - 이미지를 비디오로 변환
   - 5초 내외 영상

3. Face Swap
   - 실제 얼굴 → 생성 얼굴
   - 다양한 각도/조명

4. 상용 플랫폼 병행
   - Kling AI
   - 기타 최신 플랫폼
```

**작업 내용**:
1. **환경 구축** (1~2일)
   - ComfyUI 설치
   - 모델/체크포인트 다운로드
   - 워크플로 템플릿 설정
   
2. **데이터 생성** (3~5일)
   - 프롬프트 템플릿 작성
   - 대량 생성 (수백~수천 개)
   - 품질 필터링
   
3. **전처리** (1~2일)
   - 얼굴 검출 및 크롭
   - 해상도/압축 정규화
   - Real 데이터 수집 (라이선스 확인)
   
4. **파인튜닝** (1일)
   - 자체 데이터로 학습
   - 검증
   - 제출

**예상 효과**: +10~15% (최신 기법 커버)
**예상 시간**: 1~2주
**리스크**: 높음 (시간 소요, 라이선스 관리)

**장점**:
- ✅ 최신 생성 기법 커버
- ✅ 대회 환경과 유사한 분포
- ✅ 세미나에서 권장한 방법

**단점**:
- ⚠️ 시간 많이 걸림
- ⚠️ 리소스 필요 (GPU, 스토리지)
- ⚠️ 라이선스 관리 복잡

**전제 조건**:
- EXP-004 완료
- 대회 마감까지 2주 이상 남음

---

## 📊 전체 타임라인

```
Week 1 (현재 - 11/5):
├─ EXP-002 결과 확인 ✅
├─ EXP-003 TTA 실험 🎯
└─ FaceForensics++ 신청/다운로드 시작

Week 2 (11/6 - 11/12):
├─ FaceForensics++ 전처리
├─ EXP-004 파인튜닝 시작
└─ ComfyUI 환경 구축 (병행)

Week 3 (11/13 - 11/19):
├─ EXP-005 이미지 생성 (선택)
├─ 최종 모델 선택
├─ 앙상블 실험 (시간 있으면)
└─ 최종 제출 및 검증

마감: 11/20 17:00
```

---

## 🎯 성능 목표

| 단계 | F1 Score | 누적 향상 |
|------|----------|-----------|
| 베이스라인 | 0.5489 | - |
| EXP-002 (40프레임) | 0.57~0.59 | +4~8% |
| EXP-003 (TTA) | 0.59~0.61 | +8~11% |
| EXP-004 (파인튜닝) | 0.63~0.68 | +15~24% |
| EXP-005 (직접 생성) | 0.68~0.75 | +24~37% |
| **최종 목표** | **0.70+** | **+28%+** |

---

## 💡 각 단계별 의사결정 포인트

### EXP-002 결과 후
```
IF F1 > 0.57:
  → EXP-003 진행 (TTA 추가)
ELSE:
  → 베이스라인 재분석, 프레임 수 조정
```

### EXP-003 결과 후
```
IF F1 > 0.60:
  → EXP-004 진행 (파인튜닝)
ELSE:
  → TTA 조합 실험 (회전, 밝기 등)
```

### EXP-004 결과 후
```
IF F1 > 0.65 AND 남은시간 > 2주:
  → EXP-005 진행 (직접 생성)
ELSE:
  → 앙상블/하이퍼파라미터 튜닝
```

---

## 🚨 비상 계획 (Plan B)

### 파인튜닝 실패 시
```
Plan B-1: 다른 공개 모델 앙상블
- EfficientNet-B7
- Xception
- 가중 평균

Plan B-2: 하이퍼파라미터 Grid Search
- NUM_FRAMES: 35, 40, 45
- margin: 1.2, 1.3, 1.4
- TTA 조합
```

### 시간 부족 시
```
우선순위:
1. TTA (1시간, +1~3%)
2. 간단한 앙상블 (1일, +2~3%)
3. 하이퍼파라미터 튜닝 (1일, +1~2%)
```

---

## 📚 참고 자료

### 데이터셋
- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics
- DFDC: https://ai.facebook.com/datasets/dfdc/

### ComfyUI
- 공식: https://github.com/comfyanonymous/ComfyUI
- 워크플로: https://comfyworkflows.com/
- 튜토리얼: https://www.youtube.com/results?search_query=comfyui+tutorial

### 대회 자료
- 대회 페이지: https://aifactory.space/task/9197
- 데이터셋 세미나: https://www.youtube.com/watch?v=xHvVRlA5JbQ
- Q&A: https://aifactory.space/task/9197/qna

---

**작성일**: 2025.10.29  
**마지막 업데이트**: EXP-002 제출 후  
**대회 마감**: 2025.11.20 17:00

