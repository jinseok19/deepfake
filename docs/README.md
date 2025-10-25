# 딥페이크 탐지 모델 경진대회

## 📁 프로젝트 구조

```
deepfake/
├── baseline/                          # 원본 베이스라인
│   ├── task.ipynb                     # 원본 노트북
│   ├── task(less_cpu_intensive_ver).ipynb
│   └── model/
│       └── deep-fake-detector-v2-model/
├── samples/                           # 샘플 데이터
│   └── fake/
│       ├── image/ (7개)
│       └── video/ (5개)
├── task_improved.ipynb                # 개선된 노트북 ⭐
├── improved_inference.py              # 개선 로직 (Python 스크립트)
├── prepare_submission.bat             # 제출 자동 준비 스크립트 ⭐
├── SUBMISSION_GUIDE.md                # 제출 가이드 ⭐
└── README.md                          # 이 파일
```

---

## 🎯 대회 정보

- **대회명**: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회
- **주최**: 행정안전부, 한국지능정보사회진흥원
- **주관**: 국립과학수사연구원
- **기간**: 2024.10.23 ~ 2024.11.20
- **평가지표**: Macro F1-score (Fake=양성)
- **제출방식**: 모델 제출 (추론 자동화)

---

## ✨ 개선 사항

### 1. 얼굴 미검출 대응 강화
- **기존**: 얼굴 미검출 시 레이블 0 (Real) 고정
- **개선**: 중앙 크롭하여 모델에 입력 → 더 정확한 예측

### 2. 동영상 집계 방식 개선
- **기존**: 평균 확률만 사용
- **개선**: Mean(60%) + Max(40%) 가중 조합
- **효과**: 강한 Fake 신호 포착 능력 향상

### 3. 배치 처리 최적화
- **기존**: 프레임별 순차 GPU 처리
- **개선**: 16개씩 배치 처리
- **효과**: 추론 시간 70분 → 50-60분

### 4. 얼굴 검출 프레임 필터링
- 동영상에서 얼굴 검출 성공 프레임만 집계 사용
- 중앙 크롭 프레임의 영향 최소화

---

## 🚀 빠른 시작

### 방법 1: 자동 준비 스크립트 사용 (추천)

```bash
# 제출 폴더 자동 생성
prepare_submission.bat

# submit/ 폴더가 생성되면:
# 1. submit/task.ipynb 열기
# 2. Cell 19에서 YOUR_KEY_HERE를 본인의 Key로 교체
# 3. 모든 셀 실행
```

### 방법 2: 수동 준비

```bash
# 1. 제출 폴더 생성
mkdir submit
mkdir submit\model

# 2. 노트북 복사
copy task_improved.ipynb submit\task.ipynb

# 3. 모델 복사
xcopy baseline\model submit\model /E /I /Y

# 4. submit/task.ipynb에서 Key 수정 후 실행
```

---

## 📊 성능 비교

| 항목 | 베이스라인 | 개선 버전 | 개선율 |
|------|-----------|----------|--------|
| 얼굴 미검출 처리 | 레이블 0 고정 | 중앙 크롭 예측 | ✅ |
| 동영상 집계 | Mean only | Mean + Max | ✅ |
| 배치 크기 | 1 | 16 | ✅ |
| 추론 시간 | 70분 | 50-60분 | **-20~30%** |
| 예상 F1 개선 | - | +2~5% | **+2~5%** |

---

## 📝 제출 전 체크리스트

- [ ] `submit/task.ipynb` 파일명 확인
- [ ] Cell 19에 Competition Key 입력
- [ ] CUDA 버전 확인 (기본: 11.8)
- [ ] 모델 경로 확인: `./model/deep-fake-detector-v2-model`
- [ ] `.git` 폴더 등 불필요한 파일 제거

---

## ⚙️ 하이퍼파라미터 튜닝

`task_improved.ipynb` Cell 15에서 조정 가능:

```python
# 상수
NUM_FRAMES = 30        # 동영상 프레임 수 (20~40)
BATCH_SIZE = 16        # GPU 배치 크기 (8~32)

# 집계 가중치
MEAN_WEIGHT = 0.6      # 평균 확률 가중치 (0.5~0.7)
MAX_WEIGHT = 0.4       # 최대 확률 가중치 (0.3~0.5)
```

**추천 조합:**
- **안정성 중시**: NUM_FRAMES=40, MEAN_WEIGHT=0.7
- **속도 중시**: NUM_FRAMES=20, BATCH_SIZE=32
- **균형**: NUM_FRAMES=30, MEAN_WEIGHT=0.6 (기본값)

---

## 🐛 트러블슈팅

### OOM (Out of Memory)
```python
BATCH_SIZE = 8  # 16 → 8로 감소
```

### 타임아웃 (3시간 초과)
```python
NUM_FRAMES = 20  # 30 → 20으로 감소
```

### 디스크 공간 부족
```bash
# submit/ 폴더에서 불필요한 파일 제거
del submit\*.pyc
rmdir /s /q submit\.git
```

---

## 📚 참고 자료

- [제출 가이드](SUBMISSION_GUIDE.md) - 상세 제출 방법
- [대회 페이지](https://aifactory.space/task/9197)
- [베이스라인 코드](https://aifactory.space/task/9197/baseline)
- [Q&A 게시판](https://aifactory.space/task/9197/qna)

---

## 🔮 향후 개선 아이디어

1. **모델 앙상블**
   - ViT + EfficientNet-B7
   - Xception (deepfake 특화)

2. **TTA (Test Time Augmentation)**
   - 좌우 반전
   - 약간의 회전 (±5도)

3. **얼굴 검출기 다양화**
   - MTCNN
   - RetinaFace
   - dlib (현재) + 추가 검출기

4. **학습 데이터 구축**
   - FaceForensics++ (c23, c40)
   - Celeb-DF v2
   - DFDC (Facebook)

5. **Temporal Consistency**
   - 연속 프레임 간 일관성 체크
   - LSTM/Transformer 기반 시퀀스 모델

---

## 📞 문의

- 운영사무국: cs@aifactory.page
- Q&A 게시판: https://aifactory.space/task/9197/qna

---

**제작**: 2024.10.24  
**버전**: v1.0 (Improved Baseline)  
**라이센스**: 대회 규정에 따름

🍀 **행운을 빕니다!**

