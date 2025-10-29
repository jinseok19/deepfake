# 프로젝트 구조 가이드

## 📁 전체 디렉토리 구조

```
deepfake/
├── baseline/              # 베이스라인 코드
│   ├── model/            # 베이스라인 모델 파일
│   └── task.ipynb        # 원본 베이스라인 노트북
│
├── dev/                  # 개발용 스크립트
│   └── improved_inference.py  # 로컬 테스트용 추론 코드
│
├── docs/                 # 📚 모든 문서
│   ├── CompetitionDescription.md  # 대회 상세 설명
│   ├── DATASET_SEMINAR.md        # 데이터셋 가이드
│   ├── DLIB_INSTALL.md           # dlib 설치 가이드
│   ├── ENV_SETUP_GUIDE.md        # 환경 설정 가이드
│   ├── EXPERIMENT_LOG.md         # 실험 로그 (구버전)
│   ├── PROJECT_STRUCTURE.md      # 이 파일
│   ├── QUICK_START.md            # 빠른 시작 가이드
│   ├── README.md                 # 문서 인덱스
│   ├── STRATEGY.md               # 전략 문서
│   └── SUBMISSION_GUIDE.md       # 제출 가이드
│
├── log/                  # 📊 실험 로그 (최신)
│   ├── EXP-001.md        # 실험 001 상세 로그
│   ├── EXP-002.md        # 실험 002 상세 로그
│   ├── LOG_SUMMARY.md    # 전체 실험 요약
│   ├── PIPELINE_GUIDE.md # 파이프라인 가이드
│   ├── README.md         # 로그 관리 가이드
│   └── TEMPLATE.md       # 실험 로그 템플릿
│
├── samples/              # 샘플 데이터
│   └── fake/
│       ├── image/        # 샘플 이미지 (7개)
│       └── video/        # 샘플 비디오 (5개)
│
├── scripts/              # 🔧 유틸리티 스크립트
│   ├── prepare_submission.bat  # 제출 준비 스크립트
│   └── setup_env.bat          # 환경 설정 스크립트
│
├── submit/               # 🚀 제출용 폴더
│   ├── model/           # 제출용 모델 파일
│   └── task.ipynb       # 제출용 노트북 (최신)
│
├── venv/                # Python 가상환경 (Git 제외)
│
├── README.md            # 📖 프로젝트 메인 README
└── requirements.txt     # Python 의존성
```

---

## 📂 각 폴더의 역할

### 🎯 `baseline/`
- **목적**: 대회 제공 베이스라인 코드 보관
- **주요 파일**:
  - `task.ipynb`: 원본 베이스라인 노트북
  - `model/`: 베이스라인 모델 (ViT-base)
- **수정 금지**: 참조용으로만 사용

### 🔬 `dev/`
- **목적**: 로컬 개발 및 테스트
- **주요 파일**:
  - `improved_inference.py`: 로컬 테스트용 추론 스크립트
- **용도**: 빠른 로컬 테스트, 디버깅

### 📚 `docs/`
- **목적**: 모든 문서 중앙 관리
- **카테고리**:
  - **설치/설정**: `ENV_SETUP_GUIDE.md`, `DLIB_INSTALL.md`, `QUICK_START.md`
  - **대회 정보**: `CompetitionDescription.md`, `DATASET_SEMINAR.md`
  - **가이드**: `SUBMISSION_GUIDE.md`, `STRATEGY.md`
  - **아카이브**: `EXPERIMENT_LOG.md` (구버전, log/ 폴더가 최신)

### 📊 `log/`
- **목적**: 실험 기록 및 분석 (최신!)
- **주요 파일**:
  - `LOG_SUMMARY.md`: 전체 실험 요약 및 추이
  - `EXP-XXX.md`: 개별 실험 상세 로그
  - `TEMPLATE.md`: 새 실험 로그 작성 시 사용
  - `PIPELINE_GUIDE.md`: 전처리/모델링/후처리 가이드
- **구조화**: 전처리-모델링-후처리 중심

### 🎬 `samples/`
- **목적**: 로컬 테스트용 샘플 데이터
- **구성**:
  - `fake/image/`: 가짜 이미지 7개
  - `fake/video/`: 가짜 비디오 5개
- **용도**: dlib 테스트, 로컬 추론 검증

### 🔧 `scripts/`
- **목적**: 자동화 스크립트
- **파일**:
  - `setup_env.bat`: 가상환경 생성 및 라이브러리 설치
  - `prepare_submission.bat`: 제출 폴더 준비

### 🚀 `submit/`
- **목적**: 실제 제출용 파일만 보관
- **구성**:
  - `task.ipynb`: 제출용 노트북 (최신 실험 코드)
  - `model/`: 제출용 모델 파일
- **주의**: 제출 전 반드시 확인!

---

## 🔄 워크플로우

### 1. 새 실험 준비
```bash
# 1. baseline 기반으로 submit/task.ipynb 수정
# 2. log/TEMPLATE.md 복사하여 log/EXP-XXX.md 생성
```

### 2. 로컬 테스트 (선택)
```bash
# dev/improved_inference.py에서 빠른 테스트
python dev/improved_inference.py
```

### 3. 제출
```bash
# submit/task.ipynb에서 aif.submit() 실행
```

### 4. 결과 기록
```bash
# log/EXP-XXX.md 업데이트 (결과, 분석)
# log/LOG_SUMMARY.md 업데이트 (요약)
```

---

## 📋 파일 네이밍 규칙

### 실험 로그
- `EXP-001.md`, `EXP-002.md`, ... (3자리 숫자)
- 항상 `log/` 폴더에 위치

### 문서
- PascalCase + .md (예: `CompetitionDescription.md`)
- 모두 `docs/` 폴더에 위치

### 스크립트
- snake_case 또는 kebab-case
- 확장자에 맞게 분류 (.py, .bat, .sh)

---

## 🗑️ 제외된 파일/폴더 (.gitignore)

```
venv/                  # 가상환경
__pycache__/          # Python 캐시
*.pyc                 # Python 컴파일 파일
.ipynb_checkpoints/   # Jupyter 체크포인트
.DS_Store             # macOS
*.zip                 # 압축 파일
data/                 # 데이터셋 (용량 큼)
```

---

## 📝 문서 우선순위

### 🔥 자주 참조
1. `README.md` (루트) - 프로젝트 개요
2. `log/LOG_SUMMARY.md` - 실험 현황
3. `log/EXP-XXX.md` - 개별 실험 상세

### 📚 설정 시
1. `docs/QUICK_START.md` - 빠른 시작
2. `docs/ENV_SETUP_GUIDE.md` - 환경 설정
3. `docs/DLIB_INSTALL.md` - dlib 설치

### 🎯 실험 계획 시
1. `log/PIPELINE_GUIDE.md` - 파이프라인 이해
2. `log/TEMPLATE.md` - 실험 로그 템플릿
3. `docs/STRATEGY.md` - 전략 문서

### 📤 제출 시
1. `docs/SUBMISSION_GUIDE.md` - 제출 가이드
2. `submit/task.ipynb` - 제출 파일 확인

---

## 💡 꿀팁

### 빠른 네비게이션
```bash
# 프로젝트 루트
cd C:\Users\jinse\Documents\GitHub\deepfake

# 제출 폴더
cd submit

# 로그 확인
cd log

# 문서 확인
cd docs
```

### 실험 시작 체크리스트
- [ ] baseline/task.ipynb 확인
- [ ] log/TEMPLATE.md 복사 → log/EXP-XXX.md
- [ ] submit/task.ipynb 수정
- [ ] 변경사항 log/EXP-XXX.md에 기록
- [ ] aif.submit() 실행
- [ ] 결과 log/EXP-XXX.md + LOG_SUMMARY.md 업데이트

---

**작성일**: 2025.10.29  
**버전**: 1.0  
**마지막 업데이트**: EXP-002 제출 후

