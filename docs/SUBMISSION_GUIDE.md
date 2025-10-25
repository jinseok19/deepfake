# 딥페이크 탐지 모델 제출 가이드

## 📋 개선 사항 요약

### ✅ 주요 개선 포인트

1. **얼굴 미검출 대응 강화**
   - 기존: 얼굴 미검출 시 무조건 Real(0) 레이블
   - 개선: 중앙 크롭 사용하여 모델에 입력 (더 정확한 예측)

2. **동영상 집계 방식 개선**
   - 기존: 평균 확률만 사용
   - 개선: Mean(60%) + Max(40%) 가중 조합
   - 효과: 강한 Fake 신호 포착 능력 향상

3. **배치 처리 최적화**
   - 기존: 프레임별 순차 GPU 처리
   - 개선: 16개씩 배치 처리
   - 효과: GPU 활용률 향상, 추론 시간 단축

4. **얼굴 검출 프레임 필터링**
   - 동영상에서 얼굴 검출 성공 프레임만 집계에 사용
   - 중앙 크롭 프레임의 영향 최소화

---

## 🚀 제출 방법

### 1단계: 파일 준비

```bash
# 제출할 파일 구조
submit/
├── task.ipynb              # 제출용 노트북 (task_improved.ipynb를 복사)
└── model/
    └── deep-fake-detector-v2-model/
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── training_args.bin
```

### 2단계: task.ipynb 수정

1. `task_improved.ipynb` 파일을 `task.ipynb`로 복사
```bash
copy task_improved.ipynb submit\task.ipynb
```

2. Cell 19 (제출 셀) 수정:
```python
import aifactory.score as aif
import time

t = time.time()

aif.submit(
    model_name="deepfake_detector_improved_v1",  # 본인이 원하는 모델명
    key="YOUR_KEY_HERE"  # ← 본인의 Competition Key로 교체!
)

print(f"\n소요 시간: {time.time() - t:.2f}초")
```

### 3단계: Competition Key 가져오기

1. AI Factory 웹사이트 로그인
2. 우상단 아이콘 → **마이페이지**
3. **활동히스토리** → "딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회" 찾기
4. **키복사** 버튼 클릭

**CUDA 버전별 Competition:**
- **CUDA 11.8** (추천): 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회
- CUDA 12.6: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회 CUDA 12.6
- CUDA 10.2: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회 CUDA 10.2

### 4단계: 모델 폴더 복사

```bash
# baseline 폴더의 모델을 submit 폴더로 복사
xcopy baseline\model submit\model /E /I /Y
```

### 5단계: 제출 실행

1. Jupyter Notebook으로 `submit/task.ipynb` 열기
2. 셀을 **순서대로 전부 실행**
3. 마지막 제출 셀 실행 → 자동 업로드

---

## ⚠️ 주의사항

### 필수 체크리스트

- [ ] 파일명이 정확히 `task.ipynb`인지 확인
- [ ] Competition Key가 올바른지 확인
- [ ] CUDA 버전과 Key가 일치하는지 확인
- [ ] 모델 경로가 `./model/deep-fake-detector-v2-model`인지 확인
- [ ] `.git` 폴더 등 불필요한 파일 제거

### 제출 제한

- 하루 최대 **3회** 제출
- 재제출: 정상 처리 후 **30분** 대기
- 추론 시간: 최대 **3시간** (타임아웃 주의)

### 환경 사양

- CPU: 8 core
- RAM: 48GB
- GPU: L4 (또는 T4)
- 스토리지: 수GB (여유 적음)

### 예상 소요 시간

- 베이스라인: 약 70분
- 개선 버전: 약 50-60분 (배치 처리 최적화)

---

## 📊 예상 성능 향상

| 항목 | 베이스라인 | 개선 버전 |
|------|----------|----------|
| 얼굴 미검출 처리 | 레이블 0 고정 | 중앙 크롭 예측 |
| 동영상 집계 | Mean only | Mean(60%) + Max(40%) |
| 배치 크기 | 1 | 16 |
| 예상 추론 시간 | 70분 | 50-60분 |
| 예상 F1 개선 | - | +2~5% |

---

## 🐛 트러블슈팅

### "점수를 산출할 수 없었습니다"

**원인:** OOM(메모리 부족) 또는 디스크 초과

**해결:**
```python
# task.ipynb에서 배치 크기 줄이기
BATCH_SIZE = 8  # 기본 16에서 8로 변경
```

### "Environment timeout"

**원인:** 3시간 초과

**해결:**
- NUM_FRAMES 줄이기 (30 → 20)
- 불필요한 파일 제거
- 모델 최적화

### ImportError: No module named 'dlib'

**원인:** 라이브러리 설치 실패

**해결:**
```python
# Cell 8에 추가
!pip install -U dlib --no-cache-dir
```

---

## 📞 문의

- 운영사무국: cs@aifactory.page
- Q&A 게시판: https://aifactory.space/task/9197/qna

---

## 🎯 다음 단계 개선 아이디어

1. **모델 앙상블**: ViT + EfficientNet
2. **TTA (Test Time Augmentation)**: 좌우 반전, 약간의 회전
3. **얼굴 검출기 다양화**: MTCNN, RetinaFace 추가
4. **학습 데이터 구축**: FaceForensics++, Celeb-DF 활용
5. **Temporal Consistency**: 연속 프레임 간 일관성 체크

---

**행운을 빕니다! 🍀**

