#!/usr/bin/env python3
"""
온라인 갤러리 크롤링 스크립트
- AI 이미지 갤러리에서 Fake 이미지 수집
- Lexica.art, Civitai 등
- 목표: 2,000개
"""

import requests
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import time
from bs4 import BeautifulSoup


class GalleryCrawler:
    def __init__(self, output_dir="dataset/fake/generation/galleries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터
        self.metadata = []
        
        # User Agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def download_image(self, url, filename):
        """이미지 다운로드"""
        try:
            response = requests.get(url, timeout=15, headers=self.headers)
            response.raise_for_status()
            
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return str(filepath)
        except Exception as e:
            print(f"   다운로드 실패: {e}")
            return None
    
    def crawl_lexica(self, count=1000):
        """Lexica.art에서 이미지 크롤링"""
        print(f"\n[Lexica.art] 크롤링 시작...")
        
        # Lexica API (비공식)
        base_url = "https://lexica.art/api/v1/search"
        
        search_queries = [
            "portrait", "face", "person", "human", "character",
            "realistic", "photography", "headshot"
        ]
        
        collected = 0
        
        for query in search_queries:
            if collected >= count:
                break
            
            try:
                # API 요청
                params = {
                    'q': query,
                    'searchMode': 'images',
                    'model': 'lexica-aperture-v3'
                }
                
                response = requests.get(base_url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                images = data.get('images', [])
                
                for img_data in tqdm(images[:count//len(search_queries)], desc=f"Lexica: {query}"):
                    if collected >= count:
                        break
                    
                    img_url = img_data.get('src') or img_data.get('srcSmall')
                    if not img_url:
                        continue
                    
                    img_id = img_data.get('id', f"{collected}")
                    filename = f"lexica_{img_id}.jpg"
                    
                    filepath = self.download_image(img_url, filename)
                    if filepath:
                        self.metadata.append({
                            "filename": filename,
                            "path": filepath,
                            "label": 1,  # Fake
                            "source": "lexica",
                            "query": query,
                            "prompt": img_data.get('prompt', '')[:200],
                            "model": img_data.get('model', 'unknown')
                        })
                        collected += 1
                    
                    time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"   Lexica 크롤링 오류: {e}")
        
        print(f"✅ Lexica: {collected}개 수집")
        return collected
    
    def crawl_civitai(self, count=1000):
        """Civitai에서 이미지 크롤링"""
        print(f"\n[Civitai] 크롤링 시작...")
        
        # Civitai API
        base_url = "https://civitai.com/api/v1/images"
        
        collected = 0
        page = 1
        
        while collected < count:
            try:
                params = {
                    'limit': 100,
                    'page': page,
                    'nsfw': 'false',  # SFW만
                    'sort': 'Most Reactions'
                }
                
                response = requests.get(base_url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                items = data.get('items', [])
                if not items:
                    break
                
                for item in tqdm(items, desc=f"Civitai p{page}"):
                    if collected >= count:
                        break
                    
                    img_url = item.get('url')
                    if not img_url:
                        continue
                    
                    img_id = item.get('id', f"{collected}")
                    filename = f"civitai_{img_id}.jpg"
                    
                    filepath = self.download_image(img_url, filename)
                    if filepath:
                        meta_info = item.get('meta', {})
                        self.metadata.append({
                            "filename": filename,
                            "path": filepath,
                            "label": 1,  # Fake
                            "source": "civitai",
                            "prompt": meta_info.get('prompt', '')[:200],
                            "model": meta_info.get('Model', 'unknown'),
                            "width": item.get('width'),
                            "height": item.get('height')
                        })
                        collected += 1
                    
                    time.sleep(0.3)
                
                page += 1
                time.sleep(1)  # Page 간 대기
                
            except Exception as e:
                print(f"   Civitai 크롤링 오류: {e}")
                break
        
        print(f"✅ Civitai: {collected}개 수집")
        return collected
    
    def save_metadata(self):
        """메타데이터 저장"""
        metadata_dir = self.output_dir.parent.parent.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        import csv
        csv_file = metadata_dir / "galleries_manifest.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.metadata:
                writer = csv.DictWriter(f, fieldnames=self.metadata[0].keys())
                writer.writeheader()
                writer.writerows(self.metadata)
        
        # JSON 저장
        json_file = metadata_dir / "galleries_manifest.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 메타데이터 저장: {csv_file}")
    
    def run(self, target_count=2000):
        """크롤링 실행"""
        print("=" * 60)
        print("온라인 갤러리 크롤링")
        print("=" * 60)
        
        # 각 소스에서 절반씩
        lexica_count = self.crawl_lexica(count=target_count // 2)
        civitai_count = self.crawl_civitai(count=target_count // 2)
        
        total_count = lexica_count + civitai_count
        
        # 메타데이터 저장
        self.save_metadata()
        
        print("\n" + "=" * 60)
        print(f"✅ 갤러리 크롤링 완료: {total_count}개")
        print(f"   - Lexica: {lexica_count}개")
        print(f"   - Civitai: {civitai_count}개")
        print("=" * 60)
        
        return total_count


def main():
    parser = argparse.ArgumentParser(description="온라인 갤러리 크롤링")
    parser.add_argument("--output-dir", default="dataset/fake/generation/galleries",
                        help="출력 디렉토리")
    parser.add_argument("--count", type=int, default=2000,
                        help="수집 목표 개수")
    
    args = parser.parse_args()
    
    crawler = GalleryCrawler(args.output_dir)
    crawler.run(target_count=args.count)


if __name__ == "__main__":
    main()

