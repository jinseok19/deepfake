#!/usr/bin/env python3
"""
메타데이터 통합 스크립트
- 모든 크롤링 결과의 메타데이터를 하나로 합침
- CSV, JSON 형식으로 저장
- 통계 생성
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from datetime import datetime


class MetadataMerger:
    def __init__(self, dataset_dir="dataset", output_dir="dataset/metadata"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 파일 경로
        self.manifest_files = [
            self.output_dir / "real_manifest.csv",
            self.output_dir / "youtube_manifest.csv",
            self.output_dir / "reddit_manifest.csv",
            self.output_dir / "galleries_manifest.csv",
            self.output_dir / "flux1_manifest.csv",
        ]
    
    def load_manifests(self):
        """모든 manifest 파일 로드"""
        all_data = []
        
        for manifest_file in self.manifest_files:
            if not manifest_file.exists():
                print(f"⚠️  파일 없음: {manifest_file.name}")
                continue
            
            try:
                df = pd.read_csv(manifest_file)
                all_data.append(df)
                print(f"✅ 로드: {manifest_file.name} ({len(df)}개)")
            except Exception as e:
                print(f"❌ 로드 실패: {manifest_file.name} - {e}")
        
        if not all_data:
            print("\n❌ 로드된 메타데이터가 없습니다.")
            return None
        
        # 모두 합치기
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✅ 총 {len(combined_df)}개 데이터 로드")
        return combined_df
    
    def generate_statistics(self, df):
        """통계 생성"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_count": len(df),
            "real_count": len(df[df['label'] == 0]),
            "fake_count": len(df[df['label'] == 1]),
            "sources": {}
        }
        
        # 소스별 통계
        if 'source' in df.columns or 'dataset' in df.columns:
            source_col = 'source' if 'source' in df.columns else 'dataset'
            
            for source in df[source_col].unique():
                source_df = df[df[source_col] == source]
                stats['sources'][str(source)] = {
                    "count": len(source_df),
                    "real": len(source_df[source_df['label'] == 0]),
                    "fake": len(source_df[source_df['label'] == 1])
                }
        
        # 라벨별 통계
        stats['by_label'] = {
            "0_real": stats['real_count'],
            "1_fake": stats['fake_count'],
            "ratio": f"{stats['fake_count'] / stats['real_count']:.2f}" if stats['real_count'] > 0 else "N/A"
        }
        
        return stats
    
    def save_combined(self, df):
        """통합 데이터 저장"""
        # CSV 저장
        csv_file = self.output_dir / "combined_dataset.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n✅ CSV 저장: {csv_file}")
        
        # JSON 저장
        json_file = self.output_dir / "combined_dataset.json"
        df.to_json(json_file, orient='records', indent=2, force_ascii=False)
        print(f"✅ JSON 저장: {json_file}")
        
        return csv_file, json_file
    
    def save_statistics(self, stats):
        """통계 저장"""
        stats_file = self.output_dir / "dataset_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 통계 저장: {stats_file}")
        
        # 콘솔 출력
        print("\n" + "=" * 60)
        print("데이터셋 통계")
        print("=" * 60)
        print(f"총 데이터: {stats['total_count']:,}개")
        print(f"  - Real: {stats['real_count']:,}개 ({stats['real_count']/stats['total_count']*100:.1f}%)")
        print(f"  - Fake: {stats['fake_count']:,}개 ({stats['fake_count']/stats['total_count']*100:.1f}%)")
        print(f"\n소스별:")
        for source, source_stats in stats['sources'].items():
            print(f"  - {source}: {source_stats['count']:,}개 "
                  f"(Real: {source_stats['real']}, Fake: {source_stats['fake']})")
        print("=" * 60)
        
        return stats_file
    
    def create_train_val_split(self, df, val_ratio=0.2):
        """Train/Validation 분할"""
        from sklearn.model_selection import train_test_split
        
        # 라벨별로 분할 (stratified)
        if 'label' in df.columns:
            train_df, val_df = train_test_split(
                df, 
                test_size=val_ratio, 
                stratify=df['label'],
                random_state=42
            )
            
            # 저장
            train_file = self.output_dir / "train.csv"
            val_file = self.output_dir / "val.csv"
            
            train_df.to_csv(train_file, index=False, encoding='utf-8')
            val_df.to_csv(val_file, index=False, encoding='utf-8')
            
            print(f"\n✅ Train/Val 분할:")
            print(f"   - Train: {len(train_df):,}개 → {train_file}")
            print(f"   - Val: {len(val_df):,}개 → {val_file}")
            
            return train_file, val_file
        else:
            print("⚠️  'label' 컬럼이 없어 분할 불가")
            return None, None
    
    def run(self, create_split=True):
        """메타데이터 병합 실행"""
        print("=" * 60)
        print("메타데이터 통합")
        print("=" * 60)
        
        # 1. 로드
        df = self.load_manifests()
        if df is None:
            return False
        
        # 2. 통합 저장
        self.save_combined(df)
        
        # 3. 통계 생성
        stats = self.generate_statistics(df)
        self.save_statistics(stats)
        
        # 4. Train/Val 분할
        if create_split:
            self.create_train_val_split(df)
        
        print("\n" + "=" * 60)
        print("✅ 메타데이터 통합 완료!")
        print("=" * 60)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="메타데이터 통합")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="데이터셋 루트 디렉토리")
    parser.add_argument("--output-dir", default="dataset/metadata",
                        help="출력 디렉토리")
    parser.add_argument("--no-split", action="store_true",
                        help="Train/Val 분할 안함")
    
    args = parser.parse_args()
    
    merger = MetadataMerger(args.dataset_dir, args.output_dir)
    merger.run(create_split=not args.no_split)


if __name__ == "__main__":
    main()

