#!/usr/bin/env python3
"""
Real 데이터셋 자동 다운로드 스크립트
- FFHQ: 10,000개
- CelebA-HQ: 8,000개
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import gdown
import zipfile
import shutil
import json
import argparse


class RealDatasetDownloader:
    def __init__(self, output_dir="dataset/real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 저장
        self.metadata = {
            "ffhq": [],
            "celebahq": []
        }
    
    def download_ffhq(self, num_images=10000):
        """FFHQ 데이터셋 다운로드"""
        print(f"[FFHQ] {num_images}개 이미지 다운로드 시작...")
        
        ffhq_dir = self.output_dir / "ffhq"
        ffhq_dir.mkdir(exist_ok=True)
        
        # FFHQ 공식 다운로드 (Kaggle)
        # 방법 1: Kaggle API 사용
        kaggle_dataset = "rahulbhalley/ffhq-1024x1024"
        
        try:
            import kaggle
            print("Kaggle API로 다운로드 중...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                kaggle_dataset, 
                path=str(ffhq_dir), 
                unzip=True
            )
            
            # 이미지 파일만 선별
            image_files = list(ffhq_dir.glob("*.png")) + list(ffhq_dir.glob("*.jpg"))
            image_files = sorted(image_files)[:num_images]
            
            # 메타데이터 저장
            for idx, img_path in enumerate(image_files):
                self.metadata["ffhq"].append({
                    "filename": img_path.name,
                    "path": str(img_path),
                    "label": 0,  # Real
                    "dataset": "FFHQ",
                    "index": idx
                })
            
            print(f"✅ FFHQ {len(image_files)}개 다운로드 완료")
            return True
            
        except ImportError:
            print("❌ Kaggle API 미설치. 설치 방법:")
            print("   1. pip install kaggle")
            print("   2. Kaggle API 키 설정 (~/.kaggle/kaggle.json)")
            print("   3. https://www.kaggle.com/docs/api")
            return False
        except Exception as e:
            print(f"❌ FFHQ 다운로드 실패: {e}")
            print("\n대안 방법:")
            print("1. Kaggle에서 수동 다운로드:")
            print("   https://www.kaggle.com/datasets/rahulbhalley/ffhq-1024x1024")
            print(f"2. {ffhq_dir}에 압축 해제")
            return False
    
    def download_celebahq(self, num_images=8000):
        """CelebA-HQ 데이터셋 다운로드"""
        print(f"\n[CelebA-HQ] {num_images}개 이미지 다운로드 시작...")
        
        celebahq_dir = self.output_dir / "celebahq"
        celebahq_dir.mkdir(exist_ok=True)
        
        # Google Drive 다운로드 (공식)
        # CelebA-HQ 1024x1024
        gdrive_id = "1badu11NqxGf6qM3PTTooQDJvQbejgbTv"
        zip_path = celebahq_dir / "celebahq.zip"
        
        try:
            print("Google Drive에서 다운로드 중... (대용량, 시간 소요)")
            gdown.download(
                f"https://drive.google.com/uc?id={gdrive_id}",
                str(zip_path),
                quiet=False
            )
            
            # 압축 해제
            print("압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(celebahq_dir)
            
            # zip 파일 삭제
            zip_path.unlink()
            
            # 이미지 파일 찾기
            image_files = list(celebahq_dir.rglob("*.jpg")) + list(celebahq_dir.rglob("*.png"))
            image_files = sorted(image_files)[:num_images]
            
            # 메타데이터 저장
            for idx, img_path in enumerate(image_files):
                self.metadata["celebahq"].append({
                    "filename": img_path.name,
                    "path": str(img_path),
                    "label": 0,  # Real
                    "dataset": "CelebA-HQ",
                    "index": idx
                })
            
            print(f"✅ CelebA-HQ {len(image_files)}개 다운로드 완료")
            return True
            
        except Exception as e:
            print(f"❌ CelebA-HQ 다운로드 실패: {e}")
            print("\n대안 방법:")
            print("1. 수동 다운로드:")
            print("   https://github.com/tkarras/progressive_growing_of_gans")
            print(f"2. {celebahq_dir}에 이미지 저장")
            return False
    
    def save_metadata(self):
        """메타데이터 저장"""
        metadata_file = self.output_dir.parent / "metadata" / "real_manifest.csv"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 형식으로 저장
        import csv
        
        with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'path', 'label', 'dataset', 'index'])
            writer.writeheader()
            
            # FFHQ
            for item in self.metadata["ffhq"]:
                writer.writerow(item)
            
            # CelebA-HQ
            for item in self.metadata["celebahq"]:
                writer.writerow(item)
        
        # JSON도 저장
        json_file = metadata_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 메타데이터 저장: {metadata_file}")
        print(f"   - FFHQ: {len(self.metadata['ffhq'])}개")
        print(f"   - CelebA-HQ: {len(self.metadata['celebahq'])}개")
        print(f"   - 총: {len(self.metadata['ffhq']) + len(self.metadata['celebahq'])}개")
    
    def run(self, download_ffhq=True, download_celebahq=True, 
            ffhq_count=10000, celebahq_count=8000):
        """전체 다운로드 실행"""
        print("=" * 60)
        print("Real 데이터셋 자동 다운로드")
        print("=" * 60)
        
        success = True
        
        if download_ffhq:
            if not self.download_ffhq(ffhq_count):
                success = False
        
        if download_celebahq:
            if not self.download_celebahq(celebahq_count):
                success = False
        
        # 메타데이터 저장
        self.save_metadata()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ 모든 데이터셋 다운로드 완료!")
        else:
            print("⚠️  일부 데이터셋 다운로드 실패 (수동 다운로드 필요)")
        print("=" * 60)
        
        return success


def main():
    parser = argparse.ArgumentParser(description="Real 데이터셋 자동 다운로드")
    parser.add_argument("--output-dir", default="dataset/real", help="출력 디렉토리")
    parser.add_argument("--ffhq-count", type=int, default=10000, help="FFHQ 다운로드 개수")
    parser.add_argument("--celebahq-count", type=int, default=8000, help="CelebA-HQ 다운로드 개수")
    parser.add_argument("--skip-ffhq", action="store_true", help="FFHQ 건너뛰기")
    parser.add_argument("--skip-celebahq", action="store_true", help="CelebA-HQ 건너뛰기")
    
    args = parser.parse_args()
    
    downloader = RealDatasetDownloader(args.output_dir)
    downloader.run(
        download_ffhq=not args.skip_ffhq,
        download_celebahq=not args.skip_celebahq,
        ffhq_count=args.ffhq_count,
        celebahq_count=args.celebahq_count
    )


if __name__ == "__main__":
    main()

