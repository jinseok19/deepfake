# 데이터 수집 자동화

딥페이크 탐지 대회용 데이터셋 자동 수집 스크립트

## 빠른 시작

### 1. 환경 설정

```bash
cd scripts/data_collection

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일 편집 (Reddit API 키 등)
```

### 2. API 키 설정

#### Kaggle (필수)
✅ 이미 설정됨: `kaggle.json`

추가 설정 (Linux/Mac):
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Windows:
```cmd
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\
```

#### Reddit (선택)
1. https://www.reddit.com/prefs/apps
2. "Create App" (script 타입)
3. `.env`에 ID/Secret 설정

### 3. 실행

```bash
# 전체 자동 실행 (권장)
python run_all.py --skip-comfyui

# 개별 실행
python download_real_datasets.py      # FFHQ + CelebA-HQ
python crawl_youtube.py                # YouTube 프레임 추출
python crawl_reddit.py                 # Reddit AI 이미지
python crawl_galleries.py              # 온라인 갤러리
python comfyui_automation.py           # FLUX.1 생성 (ComfyUI 실행 필요)
python merge_metadata.py               # 메타데이터 통합
```

## 데이터 구조

```
dataset/
├── fake/
│   └── generation/
│       ├── flux1/      # 3,000개
│       ├── reddit/     # 3,000개
│       └── galleries/  # 2,000개
├── real/
│   ├── ffhq/           # 10,000개
│   ├── celebahq/       # 8,000개
│   └── youtube/        # 7,000개
└── metadata/
    ├── real_manifest.csv
    ├── fake_manifest.csv
    └── combined_dataset.csv
```

## 예상 시간

- Real 다운로드: 3-5시간
- YouTube 크롤링: 3-5시간  
- Reddit 크롤링: 1-2시간
- FLUX.1 생성: 17-20시간
- **총: 25-34시간** (병렬 시 2-3일)

## 문제 해결

### Kaggle API 오류
```bash
pip install --upgrade kaggle
chmod 600 ~/.kaggle/kaggle.json
```

### ffmpeg 없음 (YouTube)
Ubuntu: `sudo apt install ffmpeg`
Mac: `brew install ffmpeg`
Windows: https://ffmpeg.org/download.html

### ComfyUI 연결 오류
```bash
curl http://127.0.0.1:8188/system_stats
# ComfyUI 재시작 필요
```

## 다음 단계

1. ✅ 환경 설정 완료
2. 🔄 자동 수집 실행 (33,000개)
3. ⏳ 수동 작업 (Face Swap, Reenactment, Lip Sync)
4. 🎯 모델 파인튜닝
5. 📊 제출 (EXP-006)

