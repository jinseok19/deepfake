"""
비디오 파일의 dlib 얼굴 검출 성능 테스트
30개 프레임 샘플링하여 얼굴 검출률 확인
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

def detect_faces_in_video(video_path, num_frames=30):
    """비디오에서 프레임 샘플링하여 얼굴 검출"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        cap.release()
        return None, []
    
    # 30개 프레임 균등 샘플링 (베이스라인과 동일)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    detector = dlib.get_frontal_face_detector()
    
    results = []
    sampled_frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        
        # 검출용 리사이즈 (베이스라인과 동일)
        if width > 640:
            scale = 640 / float(width)
            resized_h = int(height * scale)
            resized_frame = cv2.resize(frame_rgb, (640, resized_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            resized_frame = frame_rgb
        
        # 얼굴 검출
        faces = detector(resized_frame, 1)
        
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
        
        results.append({
            'frame_idx': idx,
            'frame': frame_rgb,
            'faces': scaled_faces,
            'detected': len(scaled_faces) > 0
        })
        
        sampled_frames.append(frame_rgb)
    
    cap.release()
    
    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'sampled': len(results),
        'detected': sum(1 for r in results if r['detected']),
        'detection_rate': sum(1 for r in results if r['detected']) / len(results) * 100 if results else 0
    }
    
    return video_info, results

def visualize_video_detection(video_path, output_dir):
    """비디오 검출 결과 시각화"""
    video_name = Path(video_path).stem
    
    print(f"\nProcessing: {Path(video_path).name}")
    
    video_info, results = detect_faces_in_video(video_path)
    
    if video_info is None:
        print("  [FAIL] Could not read video")
        return None
    
    # 대표 프레임 6개 선택 (첫, 중간들, 끝)
    sample_indices = [0, 5, 10, 15, 20, 29] if len(results) >= 30 else range(min(6, len(results)))
    
    # 2x3 그리드로 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        if idx >= len(results):
            axes[i].axis('off')
            continue
        
        result = results[idx]
        frame = result['frame'].copy()
        height, width = frame.shape[:2]
        
        # 얼굴 사각형 그리기
        for face in result['faces']:
            # margin 적용
            x, y, size = get_boundingbox(face, width, height, margin=1.3)
            cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 0, 0), 3)
            
            # 원본 검출 영역
            cv2.rectangle(frame, 
                         (face.left(), face.top()), 
                         (face.right(), face.bottom()), 
                         (0, 255, 0), 2)
        
        axes[i].imshow(frame)
        status = "DETECTED" if result['detected'] else "NO FACE"
        color = 'green' if result['detected'] else 'red'
        axes[i].set_title(f"Frame {result['frame_idx']} - {status}", 
                         fontsize=10, fontweight='bold', color=color)
        axes[i].axis('off')
    
    # 전체 정보 표시
    info_text = f"{video_name}\n"
    info_text += f"Total Frames: {video_info['total_frames']}\n"
    info_text += f"Sampled: {video_info['sampled']}\n"
    info_text += f"Detected: {video_info['detected']}/{video_info['sampled']}\n"
    info_text += f"Detection Rate: {video_info['detection_rate']:.1f}%"
    
    fig.suptitle(info_text, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 저장
    output_path = output_dir / f"video_detect_{video_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Detection Rate: {video_info['detection_rate']:.1f}% ({video_info['detected']}/{video_info['sampled']})")
    print(f"  Saved: {output_path}")
    
    return video_info

def test_all_videos():
    """모든 샘플 비디오 테스트"""
    samples_dir = Path("samples/fake/video")
    output_dir = Path("dev/video_detection_results")
    output_dir.mkdir(exist_ok=True)
    
    video_files = sorted(samples_dir.glob("*.mp4")) + sorted(samples_dir.glob("*.avi"))
    
    print(f"Testing {len(video_files)} videos with 30 frames each...")
    print("="*60)
    
    results = []
    
    for video_path in video_files:
        video_info = visualize_video_detection(video_path, output_dir)
        
        if video_info:
            results.append({
                'file': video_path.name,
                'total_frames': video_info['total_frames'],
                'sampled': video_info['sampled'],
                'detected': video_info['detected'],
                'detection_rate': video_info['detection_rate']
            })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not results:
        print("No videos processed")
        return results
    
    total_videos = len(results)
    total_sampled = sum(r['sampled'] for r in results)
    total_detected = sum(r['detected'] for r in results)
    avg_detection_rate = sum(r['detection_rate'] for r in results) / total_videos
    
    print(f"Total Videos: {total_videos}")
    print(f"Total Frames Sampled: {total_sampled}")
    print(f"Total Frames with Face: {total_detected}")
    print(f"Average Detection Rate: {avg_detection_rate:.1f}%")
    print(f"\nResults saved to: {output_dir}/")
    
    # 상세 결과
    print("\nDetailed Results:")
    print("-"*80)
    print(f"{'Video':<25s} {'Total Frames':<15s} {'Sampled':<10s} {'Detected':<10s} {'Rate':<10s}")
    print("-"*80)
    for r in results:
        status = "[OK]  " if r['detection_rate'] >= 80 else "[WARN]"
        print(f"{status} {r['file']:<20s} {r['total_frames']:<15d} {r['sampled']:<10d} {r['detected']:<10d} {r['detection_rate']:.1f}%")
    
    return results

if __name__ == "__main__":
    print("dlib Video Face Detection Performance Test")
    print("Sampling 30 frames per video (baseline method)")
    print("="*60)
    results = test_all_videos()

