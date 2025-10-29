"""
dlib 얼굴 검출 성능 테스트
samples 폴더의 이미지로 얼굴 검출 결과 시각화
"""

import dlib
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def get_boundingbox(face, width, height, margin=1.3):
    """베이스라인과 동일한 바운딩박스 계산"""
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * margin)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb

def detect_faces_on_image(image_path):
    """이미지에서 얼굴 검출"""
    # 이미지 로드
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_np = np.array(img)
    height, width = img_np.shape[:2]
    
    # dlib 검출기
    detector = dlib.get_frontal_face_detector()
    
    # 검출 (베이스라인과 동일하게 640으로 리사이즈)
    if width > 640:
        scale = 640 / float(width)
        resized_h = int(height * scale)
        resized_np = cv2.resize(img_np, (640, resized_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        resized_np = img_np
    
    # 얼굴 검출
    faces = detector(resized_np, 1)
    
    # 원본 크기로 스케일 조정
    scaled_faces = []
    for face in faces:
        scaled_face = dlib.rectangle(
            left=int(face.left() / scale),
            top=int(face.top() / scale),
            right=int(face.right() / scale),
            bottom=int(face.bottom() / scale)
        )
        scaled_faces.append(scaled_face)
    
    return img_np, scaled_faces

def visualize_detection(image_path, save_path=None):
    """원본 이미지와 검출 결과를 나란히 표시"""
    img_np, faces = detect_faces_on_image(image_path)
    height, width = img_np.shape[:2]
    
    # 검출 결과 이미지 복사
    detected_img = img_np.copy()
    
    # 얼굴 사각형 그리기
    for i, face in enumerate(faces):
        # margin 적용한 바운딩박스
        x, y, size = get_boundingbox(face, width, height, margin=1.3)
        
        # 빨간색 사각형 (margin 적용)
        cv2.rectangle(detected_img, (x, y), (x + size, y + size), (255, 0, 0), 3)
        
        # 초록색 사각형 (원본 검출 영역)
        cv2.rectangle(detected_img, 
                     (face.left(), face.top()), 
                     (face.right(), face.bottom()), 
                     (0, 255, 0), 2)
        
        # 얼굴 번호 표시
        cv2.putText(detected_img, f"Face {i+1}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (255, 0, 0), 2)
    
    # 결과 표시
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 원본 이미지
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original Image\n{Path(image_path).name}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 검출 결과
    axes[1].imshow(detected_img)
    title = f'Detected Faces: {len(faces)}\n'
    if faces:
        title += 'Red: Margin 1.3 | Green: Original Detection'
    else:
        title += 'No Face Detected!'
    axes[1].set_title(title, fontsize=12, fontweight='bold', 
                     color='green' if faces else 'red')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return len(faces)

def test_all_samples():
    """모든 샘플 이미지 테스트"""
    samples_dir = Path("samples/fake/image")
    output_dir = Path("dev/face_detection_results")
    output_dir.mkdir(exist_ok=True)
    
    image_files = sorted(samples_dir.glob("*.png")) + sorted(samples_dir.glob("*.jpg"))
    
    print(f"Testing {len(image_files)} images...")
    print("="*60)
    
    results = []
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        output_path = output_dir / f"detect_{img_path.stem}.png"
        num_faces = visualize_detection(img_path, output_path)
        
        results.append({
            'file': img_path.name,
            'faces': num_faces
        })
        
        status = "[OK] SUCCESS" if num_faces > 0 else "[FAIL] NO FACE"
        print(f"  {status} - Detected {num_faces} face(s)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_images = len(results)
    detected = sum(1 for r in results if r['faces'] > 0)
    failed = total_images - detected
    
    print(f"Total Images: {total_images}")
    print(f"Successfully Detected: {detected} ({detected/total_images*100:.1f}%)")
    print(f"Failed to Detect: {failed} ({failed/total_images*100:.1f}%)")
    print(f"\nResults saved to: {output_dir}/")
    
    # 상세 결과
    print("\nDetailed Results:")
    print("-"*60)
    for r in results:
        status = "[OK]  " if r['faces'] > 0 else "[FAIL]"
        print(f"{status} {r['file']:30s} - {r['faces']} face(s)")
    
    return results

if __name__ == "__main__":
    print("dlib Face Detection Performance Test")
    print("="*60)
    results = test_all_samples()

