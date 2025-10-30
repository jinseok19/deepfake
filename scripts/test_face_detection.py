#!/usr/bin/env python3
"""
얼굴 인식 테스트 스크립트 - 샘플 이미지에 OpenCV Haar Cascade 적용
"""
import cv2
import requests
from pathlib import Path
import numpy as np

def download_sample_images():
    """샘플 이미지 4장 다운로드"""
    output_dir = Path("test_samples")
    output_dir.mkdir(exist_ok=True)
    
    # 무료 샘플 이미지 URLs (얼굴이 있는 이미지)
    sample_urls = [
        "https://thispersondoesnotexist.com/",  # AI 생성 얼굴 1
        "https://thispersondoesnotexist.com/",  # AI 생성 얼굴 2
        "https://thispersondoesnotexist.com/",  # AI 생성 얼굴 3
        "https://thispersondoesnotexist.com/",  # AI 생성 얼굴 4
    ]
    
    downloaded_files = []
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filepath = output_dir / f"sample_{i+1}.jpg"
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(filepath)
                print(f"다운로드 완료: {filepath}")
        except Exception as e:
            print(f"다운로드 실패 {i+1}: {e}")
    
    return downloaded_files

def detect_faces_with_opencv(image_files, output_dir="test_results"):
    """OpenCV Haar Cascade로 얼굴 감지 후 결과 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # OpenCV Haar Cascade 얼굴 검출기 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    results = []
    for img_file in image_files:
        try:
            # 이미지 읽기
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"이미지 읽기 실패: {img_file}")
                continue
            
            # 그레이스케일로 변환 (얼굴 감지에 더 효율적)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 감지
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # 감지된 얼굴에 사각형 그리기
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # 텍스트 추가
                cv2.putText(img, f"Face Detected", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 결과 저장
            output_file = output_path / f"detected_{img_file.name}"
            cv2.imwrite(str(output_file), img)
            
            results.append({
                'file': img_file.name,
                'faces_detected': len(faces),
                'output': output_file
            })
            
            print(f"처리 완료: {img_file.name} - {len(faces)}개 얼굴 감지")
            
        except Exception as e:
            print(f"처리 실패 {img_file}: {e}")
    
    return results

def main():
    print("="*60)
    print("얼굴 인식 테스트 - OpenCV Haar Cascade 적용")
    print("="*60)
    
    # 1. 샘플 이미지 다운로드
    print("\n[1/2] 샘플 이미지 다운로드 중...")
    image_files = download_sample_images()
    
    if not image_files:
        print("이미지 다운로드 실패. 종료합니다.")
        return
    
    print(f"다운로드 완료: {len(image_files)}개 이미지")
    
    # 2. 얼굴 감지 및 결과 저장
    print("\n[2/2] 얼굴 감지 처리 중...")
    results = detect_faces_with_opencv(image_files)
    
    # 결과 요약
    print("\n" + "="*60)
    print("처리 결과 요약")
    print("="*60)
    for r in results:
        print(f"파일: {r['file']}")
        print(f"  - 감지된 얼굴: {r['faces_detected']}개")
        print(f"  - 결과 파일: {r['output']}")
        print()
    
    print(f"총 {len(results)}개 이미지 처리 완료")
    print(f"결과 저장 위치: test_results/")

if __name__ == "__main__":
    main()

