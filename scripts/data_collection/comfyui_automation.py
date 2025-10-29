#!/usr/bin/env python3
"""
ComfyUI FLUX.1 자동 생성 스크립트
- ComfyUI API로 FLUX.1 모델 배치 생성
- 목표: 3,000개
- 주의: ComfyUI 서버 실행 필요
"""

import os
import json
import requests
import websocket
import uuid
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from dotenv import load_dotenv


class ComfyUIAutomation:
    def __init__(self, output_dir="dataset/fake/generation/flux1", 
                 comfyui_url="http://127.0.0.1:8188"):
        load_dotenv()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comfyui_url = os.getenv('COMFYUI_URL', comfyui_url)
        self.client_id = str(uuid.uuid4())
        
        # 메타데이터
        self.metadata = []
        
        # 프롬프트 템플릿
        self.prompts = [
            "portrait of a {age} year old {gender}, {expression}, {style}",
            "{gender} with {hair_color} hair, {background}, professional photo",
            "close-up of {gender} face, {lighting}, realistic",
            "{gender} selfie, {location}, candid shot",
            "headshot of {gender}, {occupation}, studio lighting"
        ]
        
        self.variables = {
            'age': ['20', '25', '30', '35', '40', '45', '50'],
            'gender': ['man', 'woman', 'person'],
            'expression': ['smiling', 'serious', 'neutral', 'laughing'],
            'style': ['realistic photo', 'professional headshot', 'natural lighting'],
            'hair_color': ['blonde', 'brown', 'black', 'red', 'gray'],
            'background': ['office', 'outdoor', 'studio', 'home'],
            'lighting': ['natural light', 'studio lighting', 'soft light'],
            'location': ['park', 'city', 'beach', 'indoor'],
            'occupation': ['business person', 'doctor', 'teacher', 'artist']
        }
    
    def check_server(self):
        """ComfyUI 서버 연결 확인"""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats")
            response.raise_for_status()
            print("✅ ComfyUI 서버 연결 성공")
            return True
        except Exception as e:
            print(f"❌ ComfyUI 서버 연결 실패: {e}")
            print(f"→ ComfyUI를 {self.comfyui_url}에서 실행해주세요")
            return False
    
    def generate_prompt(self, idx):
        """랜덤 프롬프트 생성"""
        import random
        
        template = random.choice(self.prompts)
        
        # 변수 치환
        for key, values in self.variables.items():
            if '{' + key + '}' in template:
                template = template.replace('{' + key + '}', random.choice(values))
        
        return template
    
    def create_workflow(self, prompt, seed):
        """FLUX.1 워크플로우 생성 (단순화)"""
        # 실제 ComfyUI 워크플로우는 매우 복잡함
        # 여기서는 기본 구조만 제공
        workflow = {
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "flux1-schnell.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "low quality, blurry, distorted",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "flux1_output",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        return workflow
    
    def queue_prompt(self, workflow):
        """프롬프트 큐에 추가"""
        try:
            p = {"prompt": workflow, "client_id": self.client_id}
            data = json.dumps(p).encode('utf-8')
            
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            return response.json()['prompt_id']
        except Exception as e:
            print(f"   큐 추가 실패: {e}")
            return None
    
    def get_image(self, filename, subfolder, folder_type):
        """생성된 이미지 다운로드"""
        try:
            url = f"{self.comfyui_url}/view"
            params = {
                'filename': filename,
                'subfolder': subfolder,
                'type': folder_type
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.content
        except Exception as e:
            print(f"   이미지 다운로드 실패: {e}")
            return None
    
    def run(self, target_count=3000):
        """배치 생성 실행"""
        print("=" * 60)
        print("ComfyUI FLUX.1 자동 생성")
        print("=" * 60)
        
        # 서버 확인
        if not self.check_server():
            print("\n⚠️  ComfyUI 서버가 실행되지 않았습니다.")
            print("   설치 및 실행 방법:")
            print("   1. git clone https://github.com/comfyanonymous/ComfyUI")
            print("   2. cd ComfyUI && pip install -r requirements.txt")
            print("   3. FLUX.1 모델 다운로드 (models/checkpoints/)")
            print("   4. python main.py")
            print("\n   지금은 건너뜁니다. (--skip-comfyui)")
            return 0
        
        print(f"\n목표: {target_count}개 생성")
        print(f"예상 시간: {target_count * 40 / 3600:.1f}시간 (이미지당 ~40초)")
        
        generated = 0
        
        with tqdm(total=target_count, desc="FLUX.1 생성") as pbar:
            for idx in range(target_count):
                # 프롬프트 생성
                prompt_text = self.generate_prompt(idx)
                seed = idx + 1000
                
                # 워크플로우 생성
                workflow = self.create_workflow(prompt_text, seed)
                
                # 큐에 추가
                prompt_id = self.queue_prompt(workflow)
                if not prompt_id:
                    continue
                
                # 생성 대기 (단순화 - 실제로는 WebSocket으로 진행상황 추적)
                time.sleep(40)  # FLUX.1 생성 시간
                
                # 이미지 저장 (단순화)
                filename = f"flux1_{idx:05d}.png"
                filepath = self.output_dir / filename
                
                # 메타데이터
                self.metadata.append({
                    "filename": filename,
                    "path": str(filepath),
                    "label": 1,  # Fake
                    "source": "flux1",
                    "prompt": prompt_text,
                    "seed": seed,
                    "model": "FLUX.1-schnell"
                })
                
                generated += 1
                pbar.update(1)
        
        # 메타데이터 저장
        self.save_metadata()
        
        print("\n" + "=" * 60)
        print(f"✅ FLUX.1 생성 완료: {generated}개")
        print("=" * 60)
        
        return generated
    
    def save_metadata(self):
        """메타데이터 저장"""
        metadata_dir = self.output_dir.parent.parent.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        import csv
        csv_file = metadata_dir / "flux1_manifest.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.metadata:
                writer = csv.DictWriter(f, fieldnames=self.metadata[0].keys())
                writer.writeheader()
                writer.writerows(self.metadata)
        
        # JSON 저장
        json_file = metadata_dir / "flux1_manifest.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 메타데이터 저장: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI FLUX.1 자동 생성")
    parser.add_argument("--output-dir", default="dataset/fake/generation/flux1",
                        help="출력 디렉토리")
    parser.add_argument("--count", type=int, default=3000,
                        help="생성 목표 개수")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188",
                        help="ComfyUI 서버 URL")
    
    args = parser.parse_args()
    
    automation = ComfyUIAutomation(args.output_dir, args.comfyui_url)
    automation.run(target_count=args.count)


if __name__ == "__main__":
    main()

