# 도구 설치 가이드

## 🛠️ 필수 도구 설치

### 시스템 요구사항

```
GPU: NVIDIA RTX 3060 이상 (VRAM 8GB+)
     권장: RTX 3090, 4090 (24GB)
RAM: 32GB 이상
저장공간: 500GB+ (SSD 권장)
OS: Windows 10/11 or Ubuntu 20.04+
Python: 3.10 or 3.11
```

---

## 1️⃣ ComfyUI 설치 (핵심!)

### Windows 설치

```bash
# 1. Python 3.10/3.11 설치 확인
python --version

# 2. Git 설치 (https://git-scm.com/)

# 3. ComfyUI 클론
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 4. 가상환경 생성
python -m venv venv
venv\Scripts\activate

# 5. 의존성 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 6. 실행
python main.py

# 브라우저에서 http://127.0.0.1:8188 접속
```

### Linux/Ubuntu 설치

```bash
# 1. Python 및 Git 설치
sudo apt update
sudo apt install python3.10 python3.10-venv git

# 2. ComfyUI 클론
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 3. 가상환경
python3 -m venv venv
source venv/bin/activate

# 4. 의존성
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 5. 실행
python main.py
```

---

## 2️⃣ ComfyUI Manager 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# ComfyUI 재시작
cd ../..
python main.py
```

**사용법**:
- ComfyUI 우하단 "Manager" 버튼
- "Install Custom Nodes" 클릭
- 원하는 노드 검색 및 설치

---

## 3️⃣ FLUX.1 설치 (최우선!)

### 모델 다운로드

```bash
cd ComfyUI/models/checkpoints

# FLUX.1 schnell (빠른 버전, 4GB)
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors

# FLUX.1 dev (고품질, 12GB)
wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
```

또는 Hugging Face에서 수동 다운로드:
- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://huggingface.co/black-forest-labs/FLUX.1-dev

### FLUX 노드 설치

ComfyUI Manager에서:
1. "ComfyUI-FLUX" 검색
2. 설치
3. ComfyUI 재시작

### 테스트 워크플로우

```
워크플로우 예제 다운로드:
https://comfyanonymous.github.io/ComfyUI_examples/flux/

기본 구조:
[Load Checkpoint: FLUX.1] 
  → [CLIP Text Encode: 프롬프트]
  → [KSampler]
  → [VAE Decode]
  → [Save Image]
```

---

## 4️⃣ ReActor (Face Swap) 설치

### ComfyUI Manager로 설치

```
1. Manager → Install Custom Nodes
2. "ReActor" 검색
3. "ReActor Node for ComfyUI" 설치
4. ComfyUI 재시작
```

### InsightFace 모델 다운로드

```bash
cd ComfyUI/models/insightface

# inswapper 모델 (필수)
wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
```

### 테스트

```
워크플로우:
[Load Image: 소스 얼굴]
  → [ReActor Face Swap]
  → [Load Image: 타겟]
  → [Save Image]

설정:
- Face Detection: retinaface_resnet50
- Face Model: inswapper_128.onnx
```

---

## 5️⃣ LivePortrait 설치 (Face Reenactment)

```bash
# 1. 클론
git clone https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait

# 2. 가상환경
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성
pip install -r requirements.txt

# 4. 모델 다운로드
bash scripts/download_models.sh

# 5. 테스트
python inference.py \
  --source examples/source/s0.jpg \
  --driving examples/driving/d0.mp4 \
  --output results/output.mp4
```

### GPU 메모리 부족 시

```python
# inference.py 수정
# batch_size 줄이기
config.batch_size = 1  # 기본: 4
```

---

## 6️⃣ Wav2Lip 설치 (Lip Sync)

```bash
# 1. 클론
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# 2. 가상환경
python -m venv venv
source venv/bin/activate

# 3. 의존성
pip install -r requirements.txt

# 4. ffmpeg 설치
# Windows: https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg

# 5. Pretrained 모델 다운로드
wget 'https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1' -O 'checkpoints/wav2lip.pth'

# 고품질 모델 (선택)
wget 'https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?download=1' -O 'checkpoints/wav2lip_gan.pth'

# 6. 테스트
python inference.py \
  --checkpoint_path checkpoints/wav2lip.pth \
  --face sample_data/video.mp4 \
  --audio sample_data/audio.wav \
  --outfile results/result_voice.mp4
```

---

## 7️⃣ Roop 설치 (Face Swap)

```bash
# 1. 클론
git clone https://github.com/s0md3v/roop.git
cd roop

# 2. 가상환경
python -m venv venv
source venv/bin/activate

# 3. 의존성
pip install -r requirements.txt

# 4. 실행
python run.py

# GUI가 열림
# Source: 소스 얼굴 선택
# Target: 타겟 영상 선택
# Start 클릭
```

### CLI 모드

```bash
python run.py \
  --source source_face.jpg \
  --target target_video.mp4 \
  --output output.mp4 \
  --execution-provider cuda  # GPU 사용
```

---

## 8️⃣ 데이터 수집 도구

### Reddit API (praw)

```bash
pip install praw

# Reddit App 생성 (https://www.reddit.com/prefs/apps)
# client_id, client_secret 발급
```

```python
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="deepfake_research_v1.0"
)

# 사용 예제
subreddit = reddit.subreddit("StableDiffusion")
for submission in subreddit.hot(limit=100):
    if submission.url.endswith(('.jpg', '.png')):
        print(submission.url, submission.title)
```

### gallery-dl (범용 크롤러)

```bash
pip install gallery-dl

# 사용
gallery-dl https://civitai.com/images?modelId=123456
gallery-dl https://www.reddit.com/r/midjourney/
```

### yt-dlp (YouTube)

```bash
pip install yt-dlp

# 영상 다운로드
yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID" \
  -f "best[height<=1080]" \
  --output "downloads/%(id)s.%(ext)s"

# Creative Commons만
yt-dlp "ytsearch100:interview CC" \
  --match-filter "license=Creative Commons"
```

---

## 9️⃣ 기타 유용한 도구

### dlib (얼굴 검출)

```bash
# Windows (미리 컴파일된 버전)
pip install https://github.com/jloh02/dlib/releases/download/v19.24.1/dlib-19.24.1-cp310-cp310-win_amd64.whl

# Linux
sudo apt install cmake
pip install dlib
```

### OpenCV

```bash
pip install opencv-python opencv-contrib-python
```

### Pillow (이미지 처리)

```bash
pip install Pillow
```

### tqdm (진행 바)

```bash
pip install tqdm
```

---

## 🔧 트러블슈팅

### CUDA Out of Memory

```python
# 1. 배치 크기 줄이기
batch_size = 1

# 2. 낮은 해상도 사용
resolution = 512  # 대신 1024

# 3. 모델 언로드
torch.cuda.empty_cache()

# 4. gradient checkpointing
model.enable_gradient_checkpointing()
```

### ImportError: DLL load failed

```bash
# Visual C++ Redistributable 설치
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### ffmpeg not found

```bash
# Windows: 환경변수 PATH에 ffmpeg 추가
# Linux: sudo apt install ffmpeg
```

### ComfyUI 느림

```
1. --lowvram 플래그 사용
   python main.py --lowvram

2. --cpu 모드 (최후)
   python main.py --cpu

3. GPU 드라이버 업데이트
```

---

## 📊 성능 벤치마크

### RTX 3090 (24GB)

```
FLUX.1 schnell:  ~3 images/min (512×512)
FLUX.1 dev:      ~1 image/min (1024×1024)
ReActor:         ~10 swaps/sec
LivePortrait:    ~30 frames/sec
Wav2Lip:         ~1 video/min (30sec clip)
```

### RTX 3060 (12GB)

```
FLUX.1 schnell:  ~2 images/min
ReActor:         ~5 swaps/sec
(FLUX.1 dev는 메모리 부족 가능)
```

---

## ✅ 설치 체크리스트

### 필수 (Week 1)
- [ ] ComfyUI + Manager
- [ ] FLUX.1 (schnell + dev)
- [ ] ReActor + InsightFace

### 중요 (Week 2)
- [ ] LivePortrait
- [ ] Roop
- [ ] praw (Reddit API)

### 선택 (Week 3)
- [ ] Wav2Lip
- [ ] SadTalker
- [ ] V-Express

---

## 🆘 도움말 링크

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- FLUX.1: https://huggingface.co/black-forest-labs
- ReActor: https://github.com/Gourieff/comfyui-reactor-node
- LivePortrait: https://github.com/KwaiVGI/LivePortrait
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip

---

**작성일**: 2025-10-30  
**업데이트**: 설치 중 이슈 발생 시

