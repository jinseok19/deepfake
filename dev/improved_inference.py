"""
딥페이크 탐지 모델 - Improved Inference Script

개선사항:
1. 얼굴 미검출 시 중앙 크롭 사용 (레이블 0 부여 대신)
2. 동영상 집계: mean + max 확률 조합
3. 배치 처리로 GPU 효율 향상
4. 에러 핸들링 강화
"""

import os
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import cv2
import dlib
from pathlib import Path
import numpy as np
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing

### 경로 설정
model_path = "./baseline/model/deep-fake-detector-v2-model"
test_dataset_path = Path("./samples/fake")  # 로컬 테스트용
output_csv_path = Path("submission.csv")

# 상수
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".avi", ".mp4"}
TARGET_SIZE = (224, 224)
NUM_FRAMES = 30
BATCH_SIZE = 16

# 집계 가중치
MEAN_WEIGHT = 0.6
MAX_WEIGHT = 0.4


def get_boundingbox(face, width, height, margin=1.3):
    """얼굴 바운딩 박스 추출"""
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * margin)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def center_crop_image(image: Image.Image, target_size=(224, 224)):
    """중앙 크롭"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.BICUBIC)


def detect_and_crop_face(image: Image.Image, target_size=(224, 224), resize_for_detection=640):
    """얼굴 검출 및 크롭 (개선: 미검출 시 중앙 크롭)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_np = np.array(image)
    original_h, original_w, _ = original_np.shape
    
    # 얼굴 검출용 리사이즈
    if original_w > resize_for_detection:
        scale = resize_for_detection / float(original_w)
        resized_h = int(original_h * scale)
        resized_np = cv2.resize(original_np, (resize_for_detection, resized_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        resized_np = original_np
    
    # dlib 얼굴 검출
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(resized_np, 1)
    
    # 얼굴 검출 실패 시 중앙 크롭
    if not faces:
        return center_crop_image(image, target_size), False
    
    # 가장 큰 얼굴 선택
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # 원본 이미지 좌표로 변환
    scaled_face_rect = dlib.rectangle(
        left=int(face.left() / scale),
        top=int(face.top() / scale),
        right=int(face.right() / scale),
        bottom=int(face.bottom() / scale)
    )
    
    x, y, size = get_boundingbox(scaled_face_rect, original_w, original_h)
    cropped_np = original_np[y:y + size, x:x + size]
    face_img = Image.fromarray(cropped_np).resize(target_size, Image.BICUBIC)
    
    return face_img, True


def process_single_file(file_path):
    """파일 하나 전처리"""
    face_images = []
    face_detected_flags = []
    ext = file_path.suffix.lower()
    
    try:
        if ext in IMAGE_EXTS:
            image = Image.open(file_path)
            face_img, detected = detect_and_crop_face(image, TARGET_SIZE)
            face_images.append(face_img)
            face_detected_flags.append(detected)
            
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face_img, detected = detect_and_crop_face(image, TARGET_SIZE)
                    face_images.append(face_img)
                    face_detected_flags.append(detected)
            
            cap.release()
            
    except Exception as e:
        return file_path.name, [], [], str(e)
    
    return file_path.name, face_images, face_detected_flags, None


def predict_batch(model, processor, images, device="cuda"):
    """배치 단위 GPU 추론"""
    if not images:
        return []
    
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
    
    return probs.cpu().numpy()


def aggregate_video_predictions(probs_list, face_detected_flags):
    """동영상 프레임 확률 집계 (개선)"""
    if not probs_list:
        return 0
    
    probs_array = np.array(probs_list)
    
    # 얼굴 검출 성공 프레임만 사용
    detected_indices = [i for i, flag in enumerate(face_detected_flags) if flag]
    if detected_indices:
        probs_array = probs_array[detected_indices]
    
    # Mean과 Max 확률 조합
    mean_probs = probs_array.mean(axis=0)
    max_probs = probs_array.max(axis=0)
    
    combined_probs = MEAN_WEIGHT * mean_probs + MAX_WEIGHT * max_probs
    
    return np.argmax(combined_probs)


def main():
    print("=" * 50)
    print("딥페이크 탐지 모델 - Improved Version")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 모델 로드
    print("\n[1/4] 모델 로딩...")
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    processor = ViTImageProcessor.from_pretrained(model_path)
    model.eval()
    print("✓ 모델 로드 완료")
    
    # 파일 리스트 (이미지 + 동영상 모두)
    files = []
    for root, dirs, filenames in os.walk(test_dataset_path):
        for filename in filenames:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
                files.append(file_path)
    
    files = sorted(files)
    print(f"\n[2/4] 테스트 데이터: {len(files)}개 파일")
    
    # CPU 워커 설정
    num_workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
    print(f"\n[3/4] 전처리 시작 (workers: {num_workers})")
    
    results = {}
    
    # 병렬 전처리 + 순차 추론
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(files), desc="Processing") as pbar:
            for filename, face_images, face_flags, error in pool.imap_unordered(process_single_file, files):
                if error:
                    print(f"\n⚠ Error: {filename} - {error}")
                    results[filename] = 0
                    pbar.update(1)
                    continue
                
                if not face_images:
                    results[filename] = 0
                    pbar.update(1)
                    continue
                
                # 배치 단위로 GPU 추론
                all_probs = []
                for i in range(0, len(face_images), BATCH_SIZE):
                    batch = face_images[i:i + BATCH_SIZE]
                    batch_probs = predict_batch(model, processor, batch, device)
                    all_probs.extend(batch_probs)
                
                # 이미지: 단일 예측, 동영상: 집계
                if len(all_probs) == 1:
                    predicted_class = np.argmax(all_probs[0])
                else:
                    predicted_class = aggregate_video_predictions(all_probs, face_flags)
                
                results[filename] = predicted_class
                pbar.update(1)
    
    # CSV 저장
    print("\n[4/4] 결과 저장 중...")
    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for file_path in files:
            filename = file_path.name
            label = results.get(filename, 0)
            writer.writerow([filename, label])
    
    print(f"\n✓ 추론 완료! 결과 저장: {output_csv_path}")
    
    # 결과 요약
    print("\n=== 결과 요약 ===")
    labels = list(results.values())
    print(f"Real (0): {labels.count(0)}개")
    print(f"Fake (1): {labels.count(1)}개")
    print("=" * 50)


if __name__ == "__main__":
    main()

