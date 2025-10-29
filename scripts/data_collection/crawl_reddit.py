#!/usr/bin/env python3
"""
Reddit AI 이미지 크롤링 스크립트
- r/StableDiffusion, r/midjourney 등에서 AI 생성 이미지 수집
- 목표: 3,000개
"""

import os
import praw
import requests
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from dotenv import load_dotenv


class RedditCrawler:
    def __init__(self, output_dir="dataset/fake/generation/reddit"):
        # .env 파일 로드
        load_dotenv()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reddit API 초기화
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'DeepfakeDataCollector/1.0')
            )
            print("✅ Reddit API 연결 성공")
        except Exception as e:
            print(f"❌ Reddit API 연결 실패: {e}")
            print("→ .env 파일에 REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET 설정 필요")
            raise
        
        # 메타데이터
        self.metadata = []
        
        # AI 이미지 서브레딧 목록
        self.subreddits = [
            'StableDiffusion',
            'midjourney',
            'dalle2',
            'aiArt',
            'Deforum',
            'ArtificialInteligence',
            'MediaSynthesis'
        ]
    
    def download_image(self, url, filename):
        """이미지 다운로드"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            response.raise_for_status()
            
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return str(filepath)
        except Exception as e:
            print(f"   다운로드 실패: {e}")
            return None
    
    def is_image_url(self, url):
        """이미지 URL인지 확인"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        return any(url.lower().endswith(ext) for ext in image_extensions)
    
    def crawl_subreddit(self, subreddit_name, limit=500):
        """서브레딧에서 이미지 크롤링"""
        print(f"\n[{subreddit_name}] 크롤링 시작...")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = subreddit.hot(limit=limit)
            
            count = 0
            for post in tqdm(posts, desc=f"{subreddit_name}", total=limit):
                # 이미지 URL 확인
                if not self.is_image_url(post.url):
                    continue
                
                # 파일명 생성
                filename = f"reddit_{subreddit_name}_{post.id}.jpg"
                
                # 이미지 다운로드
                filepath = self.download_image(post.url, filename)
                if filepath is None:
                    continue
                
                # 메타데이터 저장
                self.metadata.append({
                    "filename": filename,
                    "path": filepath,
                    "label": 1,  # Fake
                    "source": "reddit",
                    "subreddit": subreddit_name,
                    "post_id": post.id,
                    "title": post.title[:100],
                    "url": post.url,
                    "score": post.score,
                    "created_utc": post.created_utc
                })
                
                count += 1
            
            print(f"✅ {subreddit_name}: {count}개 수집")
            return count
            
        except Exception as e:
            print(f"❌ {subreddit_name} 크롤링 실패: {e}")
            return 0
    
    def save_metadata(self):
        """메타데이터 저장"""
        metadata_dir = self.output_dir.parent.parent.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        import csv
        csv_file = metadata_dir / "reddit_manifest.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.metadata:
                writer = csv.DictWriter(f, fieldnames=self.metadata[0].keys())
                writer.writeheader()
                writer.writerows(self.metadata)
        
        # JSON 저장
        json_file = metadata_dir / "reddit_manifest.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 메타데이터 저장: {csv_file}")
        print(f"   총 {len(self.metadata)}개 이미지")
    
    def run(self, target_count=3000):
        """크롤링 실행"""
        print("=" * 60)
        print("Reddit AI 이미지 크롤링")
        print("=" * 60)
        
        total_count = 0
        per_subreddit = target_count // len(self.subreddits) + 100
        
        for subreddit_name in self.subreddits:
            if total_count >= target_count:
                break
            
            count = self.crawl_subreddit(subreddit_name, limit=per_subreddit)
            total_count += count
        
        # 메타데이터 저장
        self.save_metadata()
        
        print("\n" + "=" * 60)
        print(f"✅ Reddit 크롤링 완료: {total_count}개")
        print("=" * 60)
        
        return total_count


def main():
    parser = argparse.ArgumentParser(description="Reddit AI 이미지 크롤링")
    parser.add_argument("--output-dir", default="dataset/fake/generation/reddit", 
                        help="출력 디렉토리")
    parser.add_argument("--count", type=int, default=3000, 
                        help="수집 목표 개수")
    
    args = parser.parse_args()
    
    crawler = RedditCrawler(args.output_dir)
    crawler.run(target_count=args.count)


if __name__ == "__main__":
    main()

