# FaceForensics++ 파인튜닝 완전 가이드 (A to Z)

> 처음부터 끝까지 단계별 상세 가이드

---

## 📋 목차

1. [개요](#1-개요)
2. [데이터셋 신청 및 다운로드](#2-데이터셋-신청-및-다운로드)
3. [환경 설정](#3-환경-설정)
4. [데이터 전처리](#4-데이터-전처리)
5. [학습 코드 작성](#5-학습-코드-작성)
6. [파인튜닝 실행](#6-파인튜닝-실행)
7. [검증 및 평가](#7-검증-및-평가)
8. [모델 제출](#8-모델-제출)
9. [문제 해결](#9-문제-해결)

---

## 1. 개요

### 목표
베이스라인 ViT 모델을 FaceForensics++ 데이터로 파인튜닝하여 성능 향상

### 예상 효과
- F1 Score: +5~10%
- 베이스라인 0.5489 → 0.60~0.65 목표

### 소요 시간
- 데이터 준비: 1~2일
- 파인튜닝: 4~8시간 (GPU)
- 검증 및 제출: 반나절
- **총: 3~5일**

---

## 2. 데이터셋 신청 및 다운로드

### Step 2.1: 데이터셋 신청

#### 방법 A: 공식 GitHub (추천)

```bash
# 1. FaceForensics++ GitHub 방문
https://github.com/ondyari/FaceForensics

# 2. 신청 양식 작성
https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md

필수 정보:
- 이름
- 이메일
- 소속 기관 (학생/연구원)
- 사용 목적: "Academic research for deepfake detection competition"
```

#### 승인 시간
- 보통 1~3일 (영업일 기준)
- 승인 후 다운로드 링크 이메일 수신

---

### Step 2.2: 데이터셋 다운로드

#### 추천: c23 버전 (압축, 경량)

```bash
# 다운로드 스크립트 (승인 후 제공)
python download-FaceForensics.py \
    --dataset FaceForensics++ \
    --compression c23 \
    --type videos \
    --num_videos 1000

# 예상 용량: 5~10GB
# 예상 시간: 1~3시간 (인터넷 속도에 따라)
```

#### 데이터셋 구조

```
FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/          # Real 비디오 (1,000개)
│
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   └── c23/
│   │       └── videos/          # Deepfakes (1,000개)
│   ├── Face2Face/
│   │   └── c23/
│   │       └── videos/          # Face2Face (1,000개)
│   ├── FaceSwap/
│   │   └── c23/
│   │       └── videos/          # FaceSwap (1,000개)
│   └── NeuralTextures/
│       └── c23/
│           └── videos/          # NeuralTextures (1,000개)
```

#### 다운로드 위치

```bash
# 프로젝트 외부에 저장 (용량 때문)
C:\Datasets\FaceForensics++\

# 또는
D:\Data\FaceForensics++\
```

---

## 3. 환경 설정

### Step 3.1: GPU 확인

```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch GPU 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 3.2: 필요 라이브러리 설치

```bash
# 가상환경 활성화
cd C:\Users\jinse\Documents\GitHub\deepfake
venv\Scripts\activate

# 추가 라이브러리
pip install -U transformers==4.30
pip install -U datasets
pip install -U scikit-learn
pip install -U tqdm
pip install -U tensorboard  # 학습 모니터링용
```

---

## 4. 데이터 전처리

### Step 4.1: 전처리 스크립트 생성

```python
# scripts/preprocess_ff++.py
import os
import cv2
import dlib
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# 설정
FF_ROOT = "C:/Datasets/FaceForensics++"  # 다운로드 위치
OUTPUT_ROOT = "./data/ff++_processed"
NUM_FRAMES = 10  # 비디오당 프레임 수
TARGET_SIZE = (224, 224)
MARGIN = 1.3

# 베이스라인과 동일한 얼굴 검출 함수
def get_boundingbox(face, width, height, margin=MARGIN):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * margin)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb

def extract_face(image, detector):
    """이미지에서 얼굴 추출"""
    if isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = np.array(image)
    
    height, width = img_np.shape[:2]
    
    # 검출용 리사이즈
    if width > 640:
        scale = 640 / float(width)
        resized_h = int(height * scale)
        resized = cv2.resize(img_np, (640, resized_h))
    else:
        scale = 1.0
        resized = img_np
    
    # 얼굴 검출
    faces = detector(resized, 1)
    if not faces:
        return None
    
    # 가장 큰 얼굴
    face = max(faces, key=lambda r: r.width() * r.height())
    
    # 스케일 조정
    face_rect = dlib.rectangle(
        left=int(face.left() / scale),
        top=int(face.top() / scale),
        right=int(face.right() / scale),
        bottom=int(face.bottom() / scale)
    )
    
    # 크롭
    x, y, size = get_boundingbox(face_rect, width, height)
    cropped = img_np[y:y+size, x:x+size]
    
    # 리사이즈
    face_img = Image.fromarray(cropped).resize(TARGET_SIZE, Image.BICUBIC)
    return face_img

def process_video(video_path, label, detector, output_dir):
    """비디오에서 프레임 추출 및 얼굴 크롭"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return 0
    
    # 균등 샘플링
    frame_indices = np.linspace(0, total_frames-1, NUM_FRAMES, dtype=int)
    
    saved_count = 0
    video_name = video_path.stem
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 추출
        face_img = extract_face(frame_rgb, detector)
        
        if face_img is None:
            continue
        
        # 저장
        save_path = output_dir / f"{video_name}_frame{i:03d}.jpg"
        face_img.save(save_path, quality=95)
        saved_count += 1
    
    cap.release()
    return saved_count

def main():
    print("FaceForensics++ 전처리 시작")
    
    # 출력 디렉토리 생성
    output_train = Path(OUTPUT_ROOT) / "train"
    output_val = Path(OUTPUT_ROOT) / "val"
    
    for split in [output_train, output_val]:
        (split / "real").mkdir(parents=True, exist_ok=True)
        (split / "fake").mkdir(parents=True, exist_ok=True)
    
    # dlib 검출기
    detector = dlib.get_frontal_face_detector()
    
    # 데이터셋 경로
    real_path = Path(FF_ROOT) / "original_sequences/youtube/c23/videos"
    fake_paths = [
        Path(FF_ROOT) / "manipulated_sequences/Deepfakes/c23/videos",
        Path(FF_ROOT) / "manipulated_sequences/Face2Face/c23/videos",
        Path(FF_ROOT) / "manipulated_sequences/FaceSwap/c23/videos",
        Path(FF_ROOT) / "manipulated_sequences/NeuralTextures/c23/videos",
    ]
    
    # Real 비디오 처리
    print("\nReal 비디오 처리 중...")
    real_videos = list(real_path.glob("*.mp4"))
    
    # Train/Val 분할 (80/20)
    split_idx = int(len(real_videos) * 0.8)
    train_real = real_videos[:split_idx]
    val_real = real_videos[split_idx:]
    
    for videos, output_dir in [(train_real, output_train / "real"), 
                                (val_real, output_val / "real")]:
        for video in tqdm(videos, desc=f"Real ({output_dir.parent.name})"):
            process_video(video, 0, detector, output_dir)
    
    # Fake 비디오 처리
    print("\nFake 비디오 처리 중...")
    all_fake_videos = []
    for fake_path in fake_paths:
        all_fake_videos.extend(list(fake_path.glob("*.mp4")))
    
    # Train/Val 분할
    split_idx = int(len(all_fake_videos) * 0.8)
    train_fake = all_fake_videos[:split_idx]
    val_fake = all_fake_videos[split_idx:]
    
    for videos, output_dir in [(train_fake, output_train / "fake"), 
                                (val_fake, output_val / "fake")]:
        for video in tqdm(videos, desc=f"Fake ({output_dir.parent.name})"):
            process_video(video, 1, detector, output_dir)
    
    # 통계
    print("\n전처리 완료!")
    print(f"Train Real: {len(list((output_train/'real').glob('*.jpg')))}")
    print(f"Train Fake: {len(list((output_train/'fake').glob('*.jpg')))}")
    print(f"Val Real: {len(list((output_val/'real').glob('*.jpg')))}")
    print(f"Val Fake: {len(list((output_val/'fake').glob('*.jpg')))}")
    print(f"\n저장 위치: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
```

### Step 4.2: 전처리 실행

```bash
# 전처리 시작 (시간 소요: 2~4시간)
python scripts/preprocess_ff++.py

# 예상 결과:
# data/ff++_processed/
# ├── train/
# │   ├── real/     (~8,000 이미지)
# │   └── fake/     (~32,000 이미지)
# └── val/
#     ├── real/     (~2,000 이미지)
#     └── fake/     (~8,000 이미지)
```

---

## 5. 학습 코드 작성

### Step 5.1: 학습 스크립트

```python
# scripts/finetune_vit.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json

# 설정
BASELINE_MODEL = "./baseline/model/deep-fake-detector-v2-model"
DATA_ROOT = "./data/ff++_processed"
OUTPUT_DIR = "./models/vit_finetuned"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터셋 클래스
class DeepfakeDataset(Dataset):
    def __init__(self, root, split='train', processor=None):
        self.root = Path(root) / split
        self.processor = processor
        
        # 파일 목록 수집
        self.samples = []
        
        # Real (label=0)
        real_dir = self.root / "real"
        for img_path in real_dir.glob("*.jpg"):
            self.samples.append((str(img_path), 0))
        
        # Fake (label=1)
        fake_dir = self.root / "fake"
        for img_path in fake_dir.glob("*.jpg"):
            self.samples.append((str(img_path), 1))
        
        print(f"{split} dataset: {len(self.samples)} samples")
        print(f"  Real: {len(list(real_dir.glob('*.jpg')))}")
        print(f"  Fake: {len(list(fake_dir.glob('*.jpg')))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 전처리
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        else:
            pixel_values = torch.zeros((3, 224, 224))
        
        return pixel_values, label

# 학습 함수
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(pixel_values=images)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1

# 검증 함수
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1

# 메인 학습 루프
def main():
    print("="*60)
    print("FaceForensics++ 파인튜닝 시작")
    print("="*60)
    
    # 출력 디렉토리
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 모델 및 프로세서 로드
    print("\n베이스라인 모델 로드...")
    model = ViTForImageClassification.from_pretrained(BASELINE_MODEL)
    processor = ViTImageProcessor.from_pretrained(BASELINE_MODEL)
    model = model.to(DEVICE)
    print(f"Device: {DEVICE}")
    
    # 데이터셋
    print("\n데이터셋 준비...")
    train_dataset = DeepfakeDataset(DATA_ROOT, 'train', processor)
    val_dataset = DeepfakeDataset(DATA_ROOT, 'val', processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # 옵티마이저 및 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    best_f1 = 0
    history = []
    
    print("\n학습 시작!")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*60)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # 결과 출력
        print(f"\nTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # 히스토리 저장
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        # 최고 모델 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = Path(OUTPUT_DIR) / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"✓ Best model saved! (F1: {best_f1:.4f})")
    
    # 최종 모델 저장
    final_path = Path(OUTPUT_DIR) / "final_model"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    # 히스토리 저장
    with open(Path(OUTPUT_DIR) / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("학습 완료!")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"모델 저장: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
```

---

## 6. 파인튜닝 실행

### Step 6.1: 학습 시작

```bash
# 전처리 완료 확인
ls data/ff++_processed/train/

# 학습 시작 (4~8시간 소요)
python scripts/finetune_vit.py

# 예상 출력:
# ============================================================
# FaceForensics++ 파인튜닝 시작
# ============================================================
# 
# 베이스라인 모델 로드...
# Device: cuda
# 
# 데이터셋 준비...
# train dataset: 40000 samples
#   Real: 8000
#   Fake: 32000
# val dataset: 10000 samples
#   Real: 2000
#   Fake: 8000
# 
# 학습 시작!
# ============================================================
# 
# Epoch 1/5
# ------------------------------------------------------------
# Training: 100%|████████████████████| 2500/2500 [12:34<00:00,  3.31it/s]
# Validating: 100%|██████████████████|  625/625 [01:23<00:00,  7.49it/s]
# 
# Train Loss: 0.2345 | Acc: 0.9123 | F1: 0.9056
# Val   Loss: 0.3456 | Acc: 0.8876 | F1: 0.8834
# ✓ Best model saved! (F1: 0.8834)
```

### Step 6.2: 학습 모니터링

```python
# 학습 중 다른 터미널에서 확인
import json
with open('./models/vit_finetuned/history.json') as f:
    history = json.load(f)
    
for h in history:
    print(f"Epoch {h['epoch']}: Val F1 = {h['val_f1']:.4f}")
```

---

## 7. 검증 및 평가

### Step 7.1: 로컬 검증

```python
# scripts/evaluate_model.py
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 파인튜닝된 모델 로드
model_path = "./models/vit_finetuned/best_model"
model = ViTForImageClassification.from_pretrained(model_path).to("cuda")
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()

# 샘플 이미지로 테스트
test_image = Image.open("samples/fake/image/sample_image_1.png")

inputs = processor(images=test_image, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

print(f"Prediction: {pred} ({'Fake' if pred == 1 else 'Real'})")
```

---

## 8. 모델 제출

### Step 8.1: 제출 준비

```bash
# 파인튜닝 모델을 submit 폴더로 복사
mkdir -p submit/model_finetuned
cp -r models/vit_finetuned/best_model/* submit/model_finetuned/

# 또는 baseline 모델 대체
rm -rf submit/model/deep-fake-detector-v2-model/*
cp -r models/vit_finetuned/best_model/* submit/model/deep-fake-detector-v2-model/
```

### Step 8.2: task.ipynb 수정

```python
# submit/task.ipynb에서 모델 경로 확인
model_path = "./model/deep-fake-detector-v2-model"  # 파인튜닝 모델 포함

# 나머지는 동일 (EXP-003 TTA 코드)
```

### Step 8.3: 제출

```python
# submit/task.ipynb Cell 19
import aifactory.score as aif

aif.submit(
    model_name="EXP-004_finetuned",
    key="your_key_here"
)
```

---

## 9. 문제 해결

### Q1: GPU 메모리 부족 (OOM)

```python
# finetune_vit.py에서 배치 크기 감소
BATCH_SIZE = 16  # → 8 또는 4
```

### Q2: 데이터 불균형 (Real < Fake)

```python
# 가중치 손실 함수 사용
from torch.nn import CrossEntropyLoss

# Real:Fake = 1:4 비율
weight = torch.tensor([4.0, 1.0]).to(DEVICE)
criterion = CrossEntropyLoss(weight=weight)
```

### Q3: 오버피팅

```python
# Early stopping 추가
patience = 3
best_val_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    # ... 학습 ...
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### Q4: 추론 시간 초과

```python
# 모델 양자화 (선택)
import torch.quantization

model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)
```

---

## 📊 예상 결과

### 파인튜닝 전 (베이스라인)
```
Val F1: ~0.92 (FaceForensics++ val set)
Competition F1: 0.5489
```

### 파인튜닝 후
```
Val F1: ~0.88-0.92 (FaceForensics++ val set)
Competition F1: 0.60~0.65 (예상)
개선: +5~10%
```

---

## 📝 체크리스트

- [ ] FaceForensics++ 신청 (1~3일)
- [ ] 데이터 다운로드 (1~3시간)
- [ ] 전처리 실행 (2~4시간)
- [ ] 학습 코드 작성 (1시간)
- [ ] 파인튜닝 실행 (4~8시간)
- [ ] 로컬 검증 (30분)
- [ ] 모델 제출 준비 (30분)
- [ ] 대회 제출 (2~3시간)

---

## 🚀 다음 단계

파인튜닝 성공 후:
1. TTA + 파인튜닝 조합
2. 앙상블 (베이스라인 + 파인튜닝)
3. 하이퍼파라미터 튜닝
4. 직접 생성 데이터 추가

---

**작성일**: 2025.10.29  
**버전**: 1.0  
**난이도**: 중급  
**예상 소요**: 3~5일

