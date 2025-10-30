#!/usr/bin/env python3
"""
데이터 수집 마스터 실행 스크립트
- 모든 크롤링 스크립트를 순차적으로 실행
- 진행 상황 모니터링
- 오류 처리
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json


class DataCollectionMaster:
    def __init__(self, output_dir="../../dataset", skip_comfyui=False):
        self.output_dir = output_dir
        self.skip_comfyui = skip_comfyui
        
        # 실행 스크립트 목록
        self.scripts = [
            {
                'name': 'Real 데이터셋 다운로드',
                'script': 'download_real_datasets.py',
                'args': ['--output-dir', f'{output_dir}/real'],
                'skip': False,
                'estimated_time': '3-5시간'
            },
            {
                'name': 'YouTube 크롤링',
                'script': 'crawl_youtube.py',
                'args': ['--output-dir', f'{output_dir}/real/youtube', '--count', '7000'],
                'skip': False,
                'estimated_time': '3-5시간'
            },
            {
                'name': 'Reddit 크롤링',
                'script': 'crawl_reddit.py',
                'args': ['--output-dir', f'{output_dir}/fake/generation/reddit', '--count', '3000'],
                'skip': False,
                'estimated_time': '1-2시간'
            },
            {
                'name': '갤러리 크롤링',
                'script': 'crawl_galleries.py',
                'args': ['--output-dir', f'{output_dir}/fake/generation/galleries', '--count', '2000'],
                'skip': False,
                'estimated_time': '1-2시간'
            },
            {
                'name': 'ComfyUI FLUX.1 생성',
                'script': 'comfyui_automation.py',
                'args': ['--output-dir', f'{output_dir}/fake/generation/flux1', '--count', '3000'],
                'skip': skip_comfyui,
                'estimated_time': '17-20시간'
            },
            {
                'name': '메타데이터 통합',
                'script': 'merge_metadata.py',
                'args': ['--dataset-dir', output_dir],
                'skip': False,
                'estimated_time': '1-2분'
            }
        ]
        
        # 실행 로그
        self.log = {
            'start_time': datetime.now().isoformat(),
            'results': []
        }
    
    def run_script(self, script_info):
        """개별 스크립트 실행"""
        script_name = script_info['name']
        # 스크립트의 절대 경로 생성
        script_dir = Path(__file__).parent
        script_path = str(script_dir / script_info['script'])
        args = script_info['args']
        
        print("\n" + "=" * 70)
        print(f"▶ {script_name}")
        print(f"  스크립트: {script_path}")
        print(f"  예상 시간: {script_info['estimated_time']}")
        print("=" * 70 + "\n")
        
        start_time = datetime.now()
        
        try:
            # Python 스크립트 실행
            cmd = [sys.executable, script_path] + args
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n[OK] {script_name} Complete (Duration: {duration/60:.1f}min)")
            
            # 로그 기록
            self.log['results'].append({
                'name': script_name,
                'script': script_path,
                'status': 'success',
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n[FAIL] {script_name} Failed (Error code: {e.returncode})")
            
            # 로그 기록
            self.log['results'].append({
                'name': script_name,
                'script': script_path,
                'status': 'failed',
                'error_code': e.returncode,
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            })
            
            return False
        
        except Exception as e:
            print(f"\n[ERROR] {script_name} Error: {e}")
            
            self.log['results'].append({
                'name': script_name,
                'script': script_path,
                'status': 'error',
                'error': str(e)
            })
            
            return False
    
    def save_log(self):
        """실행 로그 저장"""
        log_dir = Path(self.output_dir) / "metadata"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log['end_time'] = datetime.now().isoformat()
        
        log_file = log_dir / "collection_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Execution log saved: {log_file}")
    
    def print_summary(self):
        """실행 결과 요약"""
        success_count = sum(1 for r in self.log['results'] if r['status'] == 'success')
        failed_count = sum(1 for r in self.log['results'] if r['status'] != 'success')
        
        total_duration = sum(
            r.get('duration_seconds', 0) 
            for r in self.log['results'] 
            if 'duration_seconds' in r
        )
        
        print("\n\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        print(f"[OK] Success: {success_count}")
        print(f"[FAIL] Failed: {failed_count}")
        print(f"[TIME] Total duration: {total_duration/3600:.1f} hours")
        print("\nDetails:")
        
        for result in self.log['results']:
            status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
            duration = result.get('duration_seconds', 0) / 60
            print(f"  {status_icon} {result['name']}: {duration:.1f}min")
        
        print("=" * 70)
    
    def run(self):
        """전체 실행"""
        print("=" * 70)
        print("DATA COLLECTION AUTOMATION START")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"ComfyUI: {'SKIP' if self.skip_comfyui else 'RUN'}")
        print()
        
        # 스크립트 순차 실행
        for script_info in self.scripts:
            if script_info['skip']:
                print(f"\n[SKIP] {script_info['name']}")
                continue
            
            success = self.run_script(script_info)
            
            # 실패 시 계속 진행 여부 확인 (중요한 스크립트만)
            if not success and script_info['script'] in ['download_real_datasets.py']:
                print("\n[WARN] Important script failed. Continue? (y/n)")
                # 자동화를 위해 계속 진행
                print("-> Automatically continuing...")
        
        # 로그 저장
        self.save_log()
        
        # 결과 요약
        self.print_summary()
        
        print("\nDATA COLLECTION COMPLETE!")
        print(f"\nNext steps:")
        print(f"  1. dataset/metadata/combined_dataset.csv 확인")
        print(f"  2. 데이터 전처리")
        print(f"  3. 모델 파인튜닝")
        print(f"  4. EXP-006 제출")


def main():
    parser = argparse.ArgumentParser(
        description="데이터 수집 마스터 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 실행 (ComfyUI 포함)
  python run_all.py
  
  # ComfyUI 제외
  python run_all.py --skip-comfyui
  
  # 커스텀 출력 경로
  python run_all.py --output-dir /path/to/dataset
        """
    )
    
    parser.add_argument("--output-dir", default="../../dataset",
                        help="데이터셋 출력 디렉토리")
    parser.add_argument("--skip-comfyui", action="store_true",
                        help="ComfyUI 생성 건너뛰기 (17-20시간 절약)")
    
    args = parser.parse_args()
    
    master = DataCollectionMaster(
        output_dir=args.output_dir,
        skip_comfyui=args.skip_comfyui
    )
    
    master.run()


if __name__ == "__main__":
    main()

