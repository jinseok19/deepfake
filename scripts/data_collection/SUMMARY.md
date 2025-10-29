# 데이터 수집 자동화 - 완성 요약

## ✅ 완료된 작업

### 1. 핵심 스크립트 (7개)

| 파일 | 기능 | 출력 |
|-----|------|------|
| `download_real_datasets.py` | FFHQ + CelebA-HQ 다운로드 | 18,000개 |
| `crawl_youtube.py` | YouTube 얼굴 프레임 추출 | 7,000개 |
| `crawl_reddit.py` | Reddit AI 이미지 크롤링 | 3,000개 |
| `crawl_galleries.py` | 온라인 갤러리 크롤링 | 2,000개 |
| `comfyui_automation.py` | FLUX.1 배치 생성 | 3,000개 |
| `merge_metadata.py` | 메타데이터 통합 | CSV + JSON |
| `run_all.py` | 마스터 실행 스크립트 | 전체 자동화 |

### 2. 보조 파일

- `requirements.txt` - 의존성 목록
- `README.md` - 상세 사용 가이드
- `AUTOMATION_GUIDE.md` - 자동화 전략
- `.env.example` - 환경변수 템플릿
- `quickstart.sh` - Linux/Mac 빠른 시작
- `quickstart.bat` - Windows 빠른 시작

## 📊 자동화 현황

### 목표: 50,000개

```
자동화 완료: 33,000개 (66%)
├── Real: 25,000개
│   ├── FFHQ: 10,000 ✅
│   ├── CelebA-HQ: 8,000 ✅
│   └── YouTube: 7,000 ✅
│
└── Fake: 8,000개
    └── Generation: 8,000 ✅
        ├── FLUX.1: 3,000
        ├── Reddit: 3,000
        └── Galleries: 2,000

수동 작업 필요: 17,000개 (34%)
└── Fake: 17,000개
    ├── Face Swap: 7,500 (ReActor)
    ├── Reenactment: 5,000 (LivePortrait)
    └── Lip Sync: 2,500 (Wav2Lip)
```

## 🚀 사용 방법

### 빠른 실행 (권장)

```bash
cd scripts/data_collection

# Windows
quickstart.bat

# Linux/Mac
chmod +x quickstart.sh
./quickstart.sh
```

### 수동 실행

```bash
# 1. 환경 설정
pip install -r requirements.txt
cp .env.example .env
# .env 파일 편집 (API 키 설정)

# 2. 전체 자동 실행
python run_all.py --skip-comfyui

# 3. 개별 실행
python download_real_datasets.py
python crawl_youtube.py
python crawl_reddit.py
python crawl_galleries.py

# 4. ComfyUI (서버 실행 필요)
python comfyui_automation.py

# 5. 메타데이터 통합
python merge_metadata.py
```

## 📋 API 설정 필요

### 1. Kaggle (FFHQ 다운로드)
- https://www.kaggle.com/settings
- "Create New API Token"
- `~/.kaggle/kaggle.json`에 저장

### 2. Reddit (크롤링)
- https://www.reddit.com/prefs/apps
- "Create App" (script 타입)
- `.env`에 ID/Secret 설정

### 3. ComfyUI (FLUX.1 생성)
- ComfyUI 설치 및 실행
- FLUX.1 모델 다운로드
- http://127.0.0.1:8188 접속 확인

## 📂 출력 구조

```
dataset/
├── fake/
│   └── generation/
│       ├── flux1/
│       ├── reddit/
│       └── galleries/
├── real/
│   ├── ffhq/
│   ├── celebahq/
│   └── youtube/
└── metadata/
    ├── combined_dataset.csv
    ├── dataset_statistics.json
    └── collection_log.json
```

## ⏱️ 예상 소요 시간

| 작업 | 시간 | 자동화 |
|-----|------|-------|
| Real 다운로드 | 3-5시간 | ✅ 완전 자동 |
| YouTube 크롤링 | 3-5시간 | ✅ 완전 자동 |
| Reddit 크롤링 | 1-2시간 | ✅ 완전 자동 |
| 갤러리 크롤링 | 1-2시간 | ✅ 완전 자동 |
| FLUX.1 생성 | 17-20시간 | ✅ 완전 자동 |
| **소계** | **25-34시간** | **병렬 시 2-3일** |
| Face Swap | 2-3일 | ⚠️ 반자동 |
| Reenactment | 3-4일 | ⚠️ 반자동 |
| Lip Sync | 2-3일 | ⚠️ 반자동 |
| **전체** | **2-3주** | **자동 66%** |

## 💡 다음 단계

### 즉시 실행 가능
1. `quickstart.sh` 또는 `quickstart.bat` 실행
2. API 키 설정 (Kaggle, Reddit)
3. 자동 수집 시작 (33,000개)
4. 진행 상황 모니터링

### 추가 작업 (수동)
1. **Face Swap (7,500개)**
   - ComfyUI ReActor 노드 설정
   - 소스 얼굴 DB 준비 (FFHQ에서 100명)
   - 배치 스크립트 실행

2. **Face Reenactment (5,000개)**
   - LivePortrait 설치
   - 드라이빙 영상 준비 (표정 30개)
   - 배치 처리

3. **Lip Sync (2,500개)**
   - Wav2Lip 설치
   - TTS 음성 생성
   - 배치 처리

### 학습 및 제출
1. 데이터 전처리
2. 모델 파인튜닝
3. EXP-006 제출
4. 목표: F1 0.68-0.72

## 🎯 성능 예측

```
현재 (베이스라인): F1 0.5600
목표 (50K 학습):   F1 0.68-0.72 (+21-29%)

단계별 예상:
- 33K 데이터:  F1 ~0.60-0.62 (+7-11%)
- 50K 데이터:  F1 ~0.68-0.72 (+21-29%)
```

## 📝 참고 문서

- `README.md` - 상세 사용법
- `AUTOMATION_GUIDE.md` - 자동화 전략
- `../strategy/DATA_GENERATION_PLAN.md` - 전체 계획
- `../strategy/EXECUTION_SCHEDULE.md` - 3주 일정

## 🆘 문제 해결

### Kaggle API 오류
```bash
pip install --upgrade kaggle
chmod 600 ~/.kaggle/kaggle.json
```

### Reddit API 오류
```bash
pip install --upgrade praw
echo $REDDIT_CLIENT_ID  # 환경변수 확인
```

### yt-dlp 오류
```bash
pip install --upgrade yt-dlp
ffmpeg -version  # ffmpeg 설치 확인
```

### ComfyUI 연결 오류
```bash
curl http://127.0.0.1:8188/system_stats
# 서버 재시작: cd ComfyUI && python main.py
```

---

## ✨ 하이라이트

### 자동화 성과
- ✅ **7개 핵심 스크립트** 완성
- ✅ **66% 자동화** (33K/50K)
- ✅ **병렬 실행** 가능
- ✅ **메타데이터 자동 통합**
- ✅ **원클릭 실행** (quickstart)

### 주요 기능
- 🔄 **자동 재시도** 로직
- 📊 **실시간 진행 상황** (tqdm)
- 💾 **메타데이터 자동 저장**
- 🔍 **데이터 검증**
- 📈 **통계 자동 생성**

### 확장성
- 🔌 **모듈화 설계**
- ⚙️ **설정 파일** (.env)
- 📝 **상세 로깅**
- 🛠️ **에러 핸들링**

---

**작성일**: 2025-10-30  
**자동화 완료**: ✅  
**다음**: API 키 설정 후 `quickstart` 실행!

