# 데이터 생성 세부 계획

## 📊 전체 구성 (50,000개)

```
═══════════════════════════════════════════
Fake 데이터: 25,000개
═══════════════════════════════════════════

1. Generation (리더보드):     10,000개 (40%)
2. Face Swap:                  7,500개 (30%)
3. Face Reenactment:           5,000개 (20%)
4. Lip Sync:                   2,500개 (10%)

═══════════════════════════════════════════
Real 데이터: 25,000개
═══════════════════════════════════════════

1. FFHQ:                      10,000개
2. CelebA-HQ:                  8,000개
3. YouTube 일반 영상:          7,000개
```

---

## 1️⃣ Generation (생성형 이미지) - 10,000개

### 리더보드 상위 모델 기반

#### A. FLUX.1 (직접 생성) - 3,000개 ⭐⭐⭐

```python
도구: ComfyUI + FLUX.1

설치:
1. ComfyUI 설치
2. ComfyUI Manager 설치
3. FLUX.1 모델 다운로드
   - flux1-schnell.safetensors (4GB)
   - flux1-dev.safetensors (12GB)

워크플로우:
[Load Checkpoint] → [CLIP Text Encode] → [KSampler] → [VAE Decode] → [Save Image]

프롬프트 템플릿 (100개 변형):
- "professional headshot of [age] [gender] [ethnicity], realistic, 8k"
- "portrait photography, [lighting] lighting, detailed face"
- "candid photo of a person, high quality, DSLR"
- "close-up face, photorealistic, sharp details"

자동화 스크립트:
prompts = load_prompts("templates.txt")
for prompt in prompts:
    for seed in range(30):  # 30 variations per prompt
        generate_image(
            prompt=prompt,
            seed=seed,
            steps=20,
            cfg=7.5,
            sampler="euler"
        )

생성 속도: RTX 3090 기준 ~3 images/min
예상 시간: 17시간 → 3-4일 (여유있게)

출력 구조:
generated_data/
└── generation/
    └── flux1/
        ├── img_0001.png
        ├── img_0002.png
        └── metadata.csv (prompt, seed, cfg)
```

#### B. Leonardo.Ai (무료 체험) - 1,000개 ⭐⭐

```python
방법:
1. 무료 계정 생성 (일일 150 토큰)
2. Lucid Origin Ultra 모델 선택
3. 배치 생성

제약사항:
- 일일 제한: 150 토큰
- 1 이미지 = 1 토큰
→ 하루 150개 → 7일 필요

팁:
- 여러 계정 사용 (윤리적 범위 내)
- API 활용 (유료 고려)

출력 구조:
generated_data/
└── generation/
    └── leonardo/
        ├── batch_001/
        └── metadata.csv
```

#### C. Recraft V3 (무료 체험) - 1,000개 ⭐

```python
방법:
1. https://www.recraft.ai/ 가입
2. 무료 크레딧 활용
3. API 또는 웹 인터페이스

예상 시간: 7-10일
```

#### D. Midjourney (수집) - 2,000개 ⭐⭐

```python
수집 소스:
- Reddit r/midjourney
- Midjourney 공식 갤러리 (showcase)
- Discord 공개 채널

크롤링 스크립트:
import praw  # Reddit API

reddit = praw.Reddit(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET",
    user_agent="deepfake_research"
)

subreddit = reddit.subreddit("midjourney")
for submission in subreddit.hot(limit=2000):
    if submission.url.endswith(('.jpg', '.png')):
        download_image(submission.url, metadata={
            'title': submission.title,
            'score': submission.score,
            'created': submission.created_utc
        })
```

#### E. Stable Diffusion XL (보조) - 2,000개 ⭐⭐

```python
ComfyUI + SDXL
- 로컬 생성
- 빠른 생성 (FLUX보다 빠름)
- 보조 데이터

예상 시간: 2-3일
```

#### F. 기타 리더보드 모델 (수집) - 1,000개

```python
소스:
- Seedream 4.0: Reddit, AI 커뮤니티
- Imagen 4: Google AI Test Kitchen
- Gemini 2.5: 공개 샘플
- Kolors 2.1: KlingAI 갤러리

크롤링 대상:
- CivitAI 갤러리
- Artstation (AI Art 태그)
- Lexica.art (SDXL, FLUX)
```

---

## 2️⃣ Face Swap (얼굴 교체) - 7,500개

### 최신 도구 활용

#### A. ReActor (ComfyUI) - 3,000개 ⭐⭐⭐

```python
도구: ComfyUI + ReActor Node

설치:
1. ComfyUI Manager → "ReActor" 검색
2. InsightFace 모델 다운로드
   - inswapper_128.onnx

워크플로우:
[Load Image (Source)] → [ReActor Face Swap] → [Load Image (Target)] → [Save]

소스 얼굴 DB (100명):
- FFHQ에서 선별
- 다양한 인종/연령/성별
sources/
├── asian_male_young/
├── caucasian_female_middle/
└── ... (100개 폴더)

타겟 영상/이미지:
- YouTube 크롤링 (7,000개)
- 공개 영상 데이터셋

자동화:
for source in source_faces:
    for target in target_images:
        swap_face(source, target)
        
생성 조합:
- 100 sources × 30 targets = 3,000개

생성 속도: ~1 image/sec
예상 시간: 1시간 (실제 작업 포함 2-3일)

출력 구조:
generated_data/
└── face_swap/
    └── reactor/
        ├── asian_male_young_001.png
        ├── metadata.csv (source_id, target_id)
        └── ...
```

#### B. Roop (오픈소스) - 2,000개 ⭐⭐

```bash
설치:
git clone https://github.com/s0md3v/roop
cd roop
pip install -r requirements.txt

실행:
python run.py \
  --source source_face.jpg \
  --target target_video.mp4 \
  --output output.mp4

배치 처리:
bash scripts/batch_roop.sh

예상 시간: 2-3일
```

#### C. 온라인 수집 (DeepFaceLive 등) - 2,500개

```python
수집 소스:
- Reddit r/SFWdeepfakes
- YouTube (공개 페이스 스왑)
- Twitter AI 계정

주의사항:
- NSFW 필터링
- 저작권 확인
- 고품질만 선별
```

---

## 3️⃣ Face Reenactment (표정 제어) - 5,000개

### 최신 도구 (2024)

#### A. LivePortrait (2024) - 3,000개 ⭐⭐⭐

```python
도구: LivePortrait (KwaiVGI)

설치:
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait
pip install -r requirements.txt

# 모델 다운로드
bash scripts/download_models.sh

실행:
python inference.py \
  --source source_face.jpg \
  --driving driving_video.mp4 \
  --output output.mp4

소스: FFHQ 이미지 (100명)
드라이빙: 표정 영상 (30개)
- 웃음, 화남, 놀람, 슬픔 등

조합: 100 × 30 = 3,000개

생성 속도: ~30 frames/sec (GPU)
예상 시간: 3-5일

출력 구조:
generated_data/
└── reenactment/
    └── liveportrait/
        ├── person001_smile.mp4
        ├── person001_angry.mp4
        └── metadata.csv
```

#### B. V-Express (2024) - 1,000개 ⭐

```python
도구: V-Express (조건부 표정 제어)

설치 및 사용:
git clone https://github.com/tencent-ailab/V-Express
# 설치 및 실행 가이드 참조

예상 시간: 2-3일
```

#### C. First Order Motion Model - 1,000개

```python
도구: FOMM (검증된 기법)

설치:
git clone https://github.com/AliaksandrSiarohin/first-order-model
cd first-order-model
pip install -r requirements.txt

# VoxCeleb pretrained 모델 다운로드

실행:
python demo.py \
  --config config/vox-256.yaml \
  --checkpoint vox-cpk.pth.tar \
  --source_image source.png \
  --driving_video driving.mp4

예상 시간: 2-3일
```

---

## 4️⃣ Lip Sync (립싱크) - 2,500개

#### A. Wav2Lip (SOTA) - 1,500개 ⭐⭐

```python
도구: Wav2Lip

설치:
git clone https://github.com/Rudrabha/Wav2Lip
cd Wav2Lip
pip install -r requirements.txt

# Pretrained 모델 다운로드
wget 'https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1' -O 'checkpoints/wav2lip.pth'

실행:
python inference.py \
  --checkpoint_path checkpoints/wav2lip.pth \
  --face input_video.mp4 \
  --audio input_audio.wav

음성 소스:
1. TTS 생성 (다양한 언어/성별)
   - Google TTS
   - Azure TTS
   - ElevenLabs

2. 공개 음성 데이터셋
   - LibriSpeech
   - Common Voice

영상 소스: 정면 얼굴 영상 (500개)
음성: TTS 생성 (다양한 문장 30개)

조합: 50 × 30 = 1,500개

생성 속도: ~1 video/min
예상 시간: 1-2일

출력:
generated_data/
└── lip_sync/
    └── wav2lip/
        ├── video_001_audio_01.mp4
        └── metadata.csv
```

#### B. SadTalker - 1,000개 ⭐

```python
도구: SadTalker (이미지 + 음성)

설치:
git clone https://github.com/OpenTalker/SadTalker
cd SadTalker
pip install -r requirements.txt

실행:
python inference.py \
  --driven_audio audio.wav \
  --source_image image.jpg \
  --result_dir results/

특징: 이미지만으로 말하는 영상 생성
예상 시간: 2-3일
```

---

## 5️⃣ Real 데이터 - 25,000개

#### A. FFHQ (Flickr-Faces-HQ) - 10,000개

```bash
다운로드:
wget https://drive.google.com/uc?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL -O ffhq-dataset-v2.json

스크립트:
python download_ffhq.py --num_images 10000

특징:
- 고해상도 (1024×1024)
- 다양한 인종/연령
- 정면 얼굴 중심
```

#### B. CelebA-HQ - 8,000개

```bash
다운로드:
wget https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv -O celeba-hq.zip

특징:
- 연예인 얼굴
- 고품질
- 다양한 각도
```

#### C. YouTube 크롤링 - 7,000개

```python
도구: yt-dlp + 프레임 추출

스크립트:
import yt_dlp
import cv2

# 영상 다운로드
ydl_opts = {
    'format': 'best',
    'outtmpl': 'downloads/%(id)s.%(ext)s',
}

# 프레임 추출
cap = cv2.VideoCapture(video_path)
# 얼굴 검출 + 저장

검색 키워드:
- "interview"
- "vlog"
- "portrait"
- "talking head"

필터:
- Creative Commons 라이선스
- 고화질 (720p+)
- 얼굴 명확히 보임
```

---

## 📁 최종 폴더 구조

```
dataset/
├── fake/
│   ├── generation/
│   │   ├── flux1/           (3,000)
│   │   ├── leonardo/        (1,000)
│   │   ├── recraft/         (1,000)
│   │   ├── midjourney/      (2,000)
│   │   ├── sdxl/            (2,000)
│   │   └── others/          (1,000)
│   │
│   ├── face_swap/
│   │   ├── reactor/         (3,000)
│   │   ├── roop/            (2,000)
│   │   └── collected/       (2,500)
│   │
│   ├── reenactment/
│   │   ├── liveportrait/    (3,000)
│   │   ├── v_express/       (1,000)
│   │   └── fomm/            (1,000)
│   │
│   └── lip_sync/
│       ├── wav2lip/         (1,500)
│       └── sadtalker/       (1,000)
│
├── real/
│   ├── ffhq/                (10,000)
│   ├── celebahq/            (8,000)
│   └── youtube/             (7,000)
│
└── metadata/
    ├── fake_manifest.csv
    ├── real_manifest.csv
    └── combined_dataset.csv
```

---

## 📊 메타데이터 구조

```csv
filename,label,category,method,source_model,quality_score,created_date
flux1_0001.png,1,generation,text-to-image,FLUX.1,0.95,2025-10-30
reactor_0001.png,1,face_swap,face-swap,ReActor,0.92,2025-10-30
ffhq_0001.png,0,real,original,FFHQ,1.0,2025-10-30
```

---

## ⏱️ 전체 예상 시간

```
Generation:     5-7일
Face Swap:      3-5일
Reenactment:    5-7일
Lip Sync:       3-4일
Real 수집:      2-3일
─────────────────────
합계:           18-26일 (병렬 작업 시 2-3주)
```

---

**작성일**: 2025-10-30  
**목표**: 50,000개 샘플  
**시작 우선순위**: FLUX.1 → ReActor → LivePortrait

