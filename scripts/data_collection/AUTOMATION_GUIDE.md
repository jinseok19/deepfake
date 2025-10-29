# 데이터 수집 자동화 가이드

## 🎯 목표

**50,000개 샘플** 자동 수집 (Fake 25K + Real 25K)

현재 자동화된 부분: **33,000개** (66%)

## 📋 자동화된 작업

### ✅ 완료된 스크립트

| 스크립트 | 대상 | 개수 | 예상 시간 |
|---------|------|------|----------|
| `download_real_datasets.py` | FFHQ + CelebA-HQ | 18,000 | 3-5시간 |
| `crawl_youtube.py` | YouTube 프레임 | 7,000 | 3-5시간 |
| `crawl_reddit.py` | Reddit AI 이미지 | 3,000 | 1-2시간 |
| `crawl_galleries.py` | 온라인 갤러리 | 2,000 | 1-2시간 |
| `comfyui_automation.py` | FLUX.1 생성 | 3,000 | 17-20시간 |
| `merge_metadata.py` | 메타데이터 통합 | - | 5-10분 |

### 📝 TODO (수동 작업 필요)

1. **Face Swap (7,500개)**
   - ReActor 워크플로우 수동 설정 필요
   - 소스 얼굴 DB 준비
   - 반자동 가능 (스크립트는 준비됨)

2. **Face Reenactment (5,000개)**
   - LivePortrait 설치 및 실행
   - 드라이빙 영상 준비
   - 배치 스크립트 작성 가능

3. **Lip Sync (2,500개)**
   - Wav2Lip 설치
   - TTS 음성 생성
   - 배치 처리 가능

## 🚀 빠른 실행

### Windows
```batch
cd scripts\data_collection
quickstart.bat
```

### Linux/Mac
```bash
cd scripts/data_collection
chmod +x quickstart.sh
./quickstart.sh
```

### 수동 실행
```bash
# 1. 의존성 설치
pip install -r scripts/data_collection/requirements.txt

# 2. 환경변수 설정
cp scripts/data_collection/.env.example scripts/data_collection/.env
# .env 파일 편집 (Reddit API 키 등)

# 3. 전체 실행 (ComfyUI 제외)
cd scripts/data_collection
python run_all.py --skip-comfyui

# 4. 메타데이터 확인
python merge_metadata.py
```

## 📊 진행 상황 확인

```bash
# 통계 확인
cat dataset/metadata/dataset_statistics.json

# CSV로 확인
python -c "import pandas as pd; df = pd.read_csv('dataset/metadata/combined_dataset.csv'); print(df.describe())"
```

## 🔧 API 설정

### 1. Kaggle (FFHQ)

```bash
# 1. https://www.kaggle.com/settings 접속
# 2. "Create New API Token" 클릭
# 3. kaggle.json 다운로드

# Windows
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\

# Linux/Mac
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Reddit (크롤링)

```bash
# 1. https://www.reddit.com/prefs/apps 접속
# 2. "Create App" → Type: script
# 3. client_id, client_secret 복사

# .env 파일에 추가
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
```

## 📂 출력 구조

```
dataset/
├── fake/
│   └── generation/
│       ├── flux1/          # 3,000개
│       ├── reddit/         # 3,000개
│       └── galleries/      # 2,000개
│
├── real/
│   ├── ffhq/              # 10,000개
│   ├── celebahq/          # 8,000개
│   └── youtube/           # 7,000개
│
└── metadata/
    ├── combined_dataset.csv       # 전체 통합
    ├── dataset_statistics.json    # 통계
    └── collection_log.json        # 수집 로그
```

## 💡 다음 단계

### 자동 수집 완료 후 (33,000개)

1. **데이터 검증**
   ```bash
   python scripts/data_collection/validate_dataset.py
   ```

2. **첫 파인튜닝 테스트**
   - 33K 샘플로 빠른 학습
   - Val F1 확인 (~0.60-0.62 예상)

3. **수동 작업 진행**
   - Face Swap (ReActor)
   - Face Reenactment (LivePortrait)
   - Lip Sync (Wav2Lip)

### 50,000개 완료 후

1. **최종 학습**
   - 전체 데이터셋
   - 3-5 epochs
   - 목표 F1: 0.68-0.72

2. **제출**
   - EXP-006
   - 기대 성능: +16~34% (vs 베이스라인)

## 🆘 문제 해결

### "Kaggle API not found"
```bash
pip install --upgrade kaggle
# API 키 재설정
```

### "Reddit API error"
```bash
# praw 재설치
pip install --upgrade praw

# API 키 확인
python -c "import os; print(os.getenv('REDDIT_CLIENT_ID'))"
```

### "yt-dlp download failed"
```bash
# 업데이트
pip install --upgrade yt-dlp

# ffmpeg 설치 확인
ffmpeg -version
```

### "ComfyUI connection error"
```bash
# 서버 실행 확인
curl http://127.0.0.1:8188/system_stats

# 또는 브라우저에서
# http://127.0.0.1:8188
```

## 📈 예상 타임라인

| 단계 | 작업 | 소요 시간 | 누적 |
|-----|------|----------|------|
| Day 1 | Real 데이터 다운로드 | 3-5시간 | 18,000 |
| Day 2 | YouTube + Reddit 크롤링 | 4-7시간 | 28,000 |
| Day 3 | 갤러리 크롤링 | 1-2시간 | 30,000 |
| Day 4-5 | FLUX.1 생성 (백그라운드) | 17-20시간 | 33,000 ✅ |
| Day 6-10 | Face Swap (수동) | 2-3일 | 40,000 |
| Day 11-14 | Reenactment + Lip Sync | 3-4일 | 50,000 🎯 |

**자동화로 약 5일 → 수동 포함 2주 완성**

---

**작성일**: 2025-10-30  
**자동화 진행률**: 66% (33K/50K)  
**남은 작업**: Face Swap, Reenactment, Lip Sync

