# 🎯 딥페이크 탐지 모델 경진대회

**행정안전부 / 국립과학수사연구원 주관**

이미지 및 동영상 속 딥페이크를 탐지하는 AI 모델 개발 프로젝트

---

## 📁 프로젝트 구조

```
deepfake/
├── 📓 task_improved.ipynb        ⭐ 개선된 제출용 노트북 (메인)
├── 📁 baseline/                   원본 베이스라인 코드
├── 📁 samples/                    샘플 데이터 (Fake 12개)
├── 📁 venv/                       Python 가상환경
│
├── 📁 log/                       📊 실험 로그 (NEW!)
│   ├── LOG_SUMMARY.md             전체 실험 요약
│   ├── EXP-001.md                 실험 1 상세
│   ├── TEMPLATE.md                실험 템플릿
│   └── README.md                  로그 관리 가이드
│
├── 📁 docs/                       📚 문서
│   ├── README.md                  상세 프로젝트 문서
│   ├── SUBMISSION_GUIDE.md        제출 가이드
│   ├── ENV_SETUP_GUIDE.md         환경 설정 가이드
│   └── EXPERIMENT_LOG.md          실험 통합 로그
│
├── 📁 scripts/                    🛠️ 유틸리티 스크립트
│   ├── setup_env.bat              가상환경 자동 설정
│   └── prepare_submission.bat     제출 자동 준비
│
├── 📁 dev/                        개발용 파일
│   └── improved_inference.py      추론 로직 (Python)
└── requirements.txt               라이브러리 목록
```

---

## 🚀 빠른 시작 (3단계)

### 1️⃣ 환경 설정
```bash
# 가상환경 자동 설치
scripts\setup_env.bat

# 가상환경 활성화
venv\Scripts\activate
```

### 2️⃣ 제출 준비
```bash
# 제출 폴더 자동 생성
scripts\prepare_submission.bat
```

### 3️⃣ 제출 실행
1. `submit/task.ipynb` 열기
2. Competition Key 입력
3. 모든 셀 실행

---

## ⚠️ 실험 결과

| 항목 | 베이스라인 | EXP-001 시도 | 결과 |
|------|-----------|-------------|------|
| **얼굴 미검출 처리** | 레이블 0 고정 | 중앙 크롭 예측 | ❌ 성능 하락 |
| **동영상 집계** | Mean only | Mean(60%) + Max(40%) | ❌ 성능 하락 |
| **배치 처리** | 1개씩 | 16개씩 | ✅ 시간 -30% |
| **프레임 필터링** | 전체 사용 | 얼굴 검출 성공만 | ❌ 성능 하락 |

**실제 성능:** 
- 베이스라인: F1 **0.5489**
- EXP-001: F1 0.5354 (**-2.46%** 하락) ❌
- 다음 계획: 베이스라인으로 복구 필요

---

## 📊 대회 정보

- **대회명**: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회
- **기간**: 2024.10.23 ~ 2024.11.20
- **평가**: Macro F1-score (Fake=양성)
- **제출**: 모델 제출 (추론 자동화)
- **제한**: 하루 3회, 30분 간격
- **상금**: 총 9,200만원

---

## 📝 주요 문서

| 문서 | 설명 |
|------|------|
| [📊 LOG_SUMMARY.md](log/LOG_SUMMARY.md) | ⭐ 전체 실험 요약 (NEW!) |
| [📘 SUBMISSION_GUIDE.md](docs/SUBMISSION_GUIDE.md) | 제출 방법 및 체크리스트 |
| [⚙️ ENV_SETUP_GUIDE.md](docs/ENV_SETUP_GUIDE.md) | 환경 설정 및 트러블슈팅 |
| [📖 README.md](docs/README.md) | 프로젝트 상세 문서 |
| [🧪 EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | 실험 통합 로그 |

---

## ⚠️ 중요 사항

### dlib 설치 이슈 (얼굴 검출 라이브러리)

**증상:** `pip install dlib` 실패 (CMake 필요)

**해결 방법 3가지:**

#### 방법 1: CMake 설치 후 재시도 (권장)
```bash
# 1. CMake 다운로드 및 설치
#    https://cmake.org/download/

# 2. Visual Studio Build Tools 설치
#    https://visualstudio.microsoft.com/downloads/
#    "Desktop development with C++" 선택

# 3. 가상환경 재활성화 후 재설치
venv\Scripts\activate
pip install dlib
```

#### 방법 2: 미리 빌드된 wheel 사용
```bash
# Python 3.10용 dlib wheel 다운로드 후 설치
pip install dlib-19.24.0-cp310-cp310-win_amd64.whl
```

#### 방법 3: 로컬 테스트 스킵 (제출만 할 경우)
- 제출 환경에는 dlib이 이미 설치되어 있음
- 로컬 테스트 없이 바로 제출 가능

---

## 🔧 하이퍼파라미터 튜닝

`task_improved.ipynb` Cell 15에서 수정:

```python
# 프레임 수 (동영상)
NUM_FRAMES = 30  # 20~40 (작을수록 빠름)

# 배치 크기
BATCH_SIZE = 16  # 8~32 (클수록 빠르지만 메모리 증가)

# 집계 가중치
MEAN_WEIGHT = 0.6  # 평균 비중 (0.5~0.7)
MAX_WEIGHT = 0.4   # 최대 비중 (0.3~0.5)
```

**추천 조합:**
- **안정성**: NUM_FRAMES=40, MEAN_WEIGHT=0.7
- **속도**: NUM_FRAMES=20, BATCH_SIZE=32
- **균형**: 기본값 (NUM_FRAMES=30, BATCH_SIZE=16)

---

## 🐛 트러블슈팅

### OOM (메모리 부족)
```python
BATCH_SIZE = 8  # 16 → 8로 감소
```

### 타임아웃 (3시간 초과)
```python
NUM_FRAMES = 20  # 30 → 20으로 감소
```

### dlib 설치 실패
→ [ENV_SETUP_GUIDE.md](docs/ENV_SETUP_GUIDE.md) 참고

---

## 📞 문의

- **운영 문의**: cs@aifactory.page
- **Q&A 게시판**: https://aifactory.space/task/9197/qna
- **대회 페이지**: https://aifactory.space/task/9197

---

## 🎓 참고 자료

- [PyTorch 설치](https://pytorch.org/get-started/locally/)
- [Transformers 문서](https://huggingface.co/docs/transformers)
- [ViT 모델 설명](https://huggingface.co/google/vit-base-patch16-224)

---

**제작**: 2024.10.24  
**버전**: v1.0 (Improved Baseline)  
**라이센스**: 대회 규정 준수

🍀 **행운을 빕니다!**

