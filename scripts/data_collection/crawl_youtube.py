#!/usr/bin/env python3
"""
YouTube 얼굴 프레임 추출 스크립트
- 인터뷰, 브이로그 등에서 Real 얼굴 프레임 추출
- 목표: 7,000개
"""

import os
import cv2
import yt_dlp
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from dotenv import load_dotenv


class YouTubeCrawler:
    def __init__(self, output_dir="dataset/real/youtube"):
        load_dotenv()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 비디오 저장 경로
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 메타데이터
        self.metadata = []
        
        # 얼굴 검출기 (OpenCV Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 유튜브 검색 쿼리
        self.search_queries = [
            "interview 2024",
            "vlog face",
            "talking head",
            "podcast interview",
            "celebrity interview",
            "ted talk",
            "news anchor",
            "makeup tutorial"
        ]
    
    def download_video(self, query, max_results=5):
        """YouTube 비디오 검색 및 다운로드"""
        ydl_opts = {
            'format': 'worst',  # 빠른 다운로드를 위해 저화질
            'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'default_search': 'ytsearch' + str(max_results),
        }
        
        video_paths = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=True)
                
                # 검색 결과에서 비디오 정보 추출
                if 'entries' in info:
                    for entry in info['entries'][:max_results]:
                        if entry:
                            video_id = entry.get('id')
                            video_path = self.temp_dir / f"{video_id}.mp4"
                            if video_path.exists():
                                video_paths.append({
                                    'path': video_path,
                                    'id': video_id,
                                    'title': entry.get('title', ''),
                                })
        except Exception as e:
            print(f"   다운로드 실패: {e}")
        
        return video_paths
    
    def extract_faces(self, video_path, video_id, video_title, max_frames=50):
        """비디오에서 얼굴 프레임 추출"""
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 1초마다 샘플링
        frame_interval = fps if fps > 0 else 30
        
        extracted = 0
        frame_idx = 0
        
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # frame_interval마다 처리
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            # 얼굴이 있으면 저장
            for i, (x, y, w, h) in enumerate(faces):
                # 얼굴 영역 크롭
                face_img = frame[y:y+h, x:x+w]
                
                # 파일명 생성
                filename = f"youtube_{video_id}_f{frame_idx}_face{i}.jpg"
                filepath = self.output_dir / filename
                
                # 저장
                cv2.imwrite(str(filepath), face_img)
                
                # 메타데이터
                self.metadata.append({
                    "filename": filename,
                    "path": str(filepath),
                    "label": 0,  # Real
                    "source": "youtube",
                    "video_id": video_id,
                    "video_title": video_title[:100],
                    "frame_index": frame_idx,
                    "face_index": i
                })
                
                extracted += 1
                if extracted >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        return extracted
    
    def save_metadata(self):
        """메타데이터 저장"""
        metadata_dir = self.output_dir.parent.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        import csv
        csv_file = metadata_dir / "youtube_manifest.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.metadata:
                writer = csv.DictWriter(f, fieldnames=self.metadata[0].keys())
                writer.writeheader()
                writer.writerows(self.metadata)
        
        # JSON 저장
        json_file = metadata_dir / "youtube_manifest.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 메타데이터 저장: {csv_file}")
    
    def cleanup_temp(self):
        """임시 파일 삭제"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("✅ 임시 파일 삭제 완료")
    
    def run(self, target_count=7000, videos_per_query=3):
        """크롤링 실행"""
        print("=" * 60)
        print("YouTube 얼굴 프레임 추출")
        print("=" * 60)
        print("⚠️  ffmpeg 설치 필요 (https://ffmpeg.org)")
        
        total_count = 0
        frames_per_video = 100
        
        for query in self.search_queries:
            if total_count >= target_count:
                break
            
            print(f"\n[검색: {query}]")
            
            # 비디오 다운로드
            videos = self.download_video(query, max_results=videos_per_query)
            
            # 각 비디오에서 프레임 추출
            for video_info in tqdm(videos, desc=query):
                video_path = video_info['path']
                video_id = video_info['id']
                video_title = video_info['title']
                
                count = self.extract_faces(
                    video_path, 
                    video_id, 
                    video_title,
                    max_frames=frames_per_video
                )
                
                total_count += count
                
                # 비디오 파일 삭제
                video_path.unlink()
                
                if total_count >= target_count:
                    break
        
        # 메타데이터 저장
        self.save_metadata()
        
        # 임시 파일 정리
        self.cleanup_temp()
        
        print("\n" + "=" * 60)
        print(f"✅ YouTube 크롤링 완료: {total_count}개")
        print("=" * 60)
        
        return total_count


def main():
    parser = argparse.ArgumentParser(description="YouTube 얼굴 프레임 추출")
    parser.add_argument("--output-dir", default="dataset/real/youtube", 
                        help="출력 디렉토리")
    parser.add_argument("--count", type=int, default=7000, 
                        help="수집 목표 개수")
    parser.add_argument("--videos-per-query", type=int, default=3,
                        help="쿼리당 다운로드할 비디오 수")
    
    args = parser.parse_args()
    
    crawler = YouTubeCrawler(args.output_dir)
    crawler.run(target_count=args.count, videos_per_query=args.videos_per_query)


if __name__ == "__main__":
    main()

