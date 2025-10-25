# 개발 환경 설정 가이드

## 🚀 빠른 시작

### 자동 설치 (추천)
```bash
# 가상환경 자동 설정
setup_env.bat

# 설치 완료 후 활성화
venv\Scripts\activate
```

### 수동 설치
```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
venv\Scripts\activate

# 3. 라이브러리 설치
pip install -r requirements.txt
```

---

## ⚙️ 시스템 요구사항

### 필수
- **Python**: 3.9 이상 (3.10 권장)
- **RAM**: 최소 8GB (16GB 권장)
- **디스크**: 10GB 이상

### GPU 사용 시 (선택)
- **NVIDIA GPU**: CUDA 지원 GPU
- **CUDA**: 11.8 또는 12.x
- **cuDNN**: CUDA 버전에 맞는 버전

---

## 🔧 상세 설치 가이드

### 1. Python 설치 확인
```bash
python --version
# Python 3.9.x 이상이어야 함
```

Python이 없다면:
- [Python 공식 사이트](https://www.python.org/downloads/) 다운로드
- 설치 시 **"Add Python to PATH"** 체크 필수!

### 2. PyTorch 설치

#### CUDA 11.8 (대회 환경과 동일, 추천)
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.x
```bash
pip install torch==2.7.1 torchvision==0.22.1
```

#### CPU Only (GPU 없음)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. dlib 설치 (얼굴 검출)

dlib은 Windows에서 설치가 까다로울 수 있습니다.

#### 방법 1: pip으로 직접 설치 (시도)
```bash
pip install dlib
```

#### 방법 2: CMake 설치 후 빌드
설치 실패 시:
1. [CMake 다운로드](https://cmake.org/download/)
2. [Visual Studio Build Tools 다운로드](https://visualstudio.microsoft.com/downloads/)
   - "Desktop development with C++" 워크로드 선택
3. 재시도:
```bash
pip install cmake
pip install dlib
```

#### 방법 3: 미리 빌드된 wheel 사용
```bash
# Python 버전에 맞는 wheel 다운로드
# https://github.com/sachadee/Dlib
pip install dlib-19.24.0-cp310-cp310-win_amd64.whl
```

### 4. 나머지 라이브러리 설치
```bash
pip install transformers==4.30.0
pip install opencv-python==4.10.0.82
pip install numpy==1.26.4 pandas scipy==1.11.4 scikit-learn==1.3.2
pip install tqdm jupyter ipykernel matplotlib
```

---

## ✅ 설치 확인

### GPU 사용 가능 여부 확인
```python
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
```

### 전체 라이브러리 확인
```python
import torch
import torchvision
import transformers
import cv2
import dlib
import numpy as np
import PIL

print("✓ 모든 라이브러리 정상 설치!")
```

---

## 🧪 로컬 테스트

### 샘플 데이터로 추론 테스트
```bash
# 가상환경 활성화
venv\Scripts\activate

# 추론 스크립트 실행 (샘플 데이터)
python improved_inference.py
```

### Jupyter Notebook 실행
```bash
# Jupyter 시작
jupyter notebook

# task_improved.ipynb 열기
```

---

## 🐛 트러블슈팅

### 문제 1: "dlib 설치 실패"
**해결:**
1. Visual Studio Build Tools 설치
2. CMake 설치
3. 또는 미리 빌드된 wheel 사용

### 문제 2: "torch.cuda.is_available() = False"
**원인:** CUDA 버전 불일치 또는 NVIDIA 드라이버 문제

**해결:**
1. NVIDIA 드라이버 업데이트
2. PyTorch 재설치 (CUDA 버전 확인)
```bash
pip uninstall torch torchvision
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

### 문제 3: "ImportError: No module named 'cv2'"
**해결:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless==4.10.0.82
```

### 문제 4: "Microsoft Visual C++ 14.0 is required"
**해결:**
[Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) 설치

---

## 📦 가상환경 관리

### 가상환경 활성화
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 가상환경 비활성화
```bash
deactivate
```

### 가상환경 삭제
```bash
# Windows
rmdir /s /q venv

# Linux/Mac
rm -rf venv
```

### 설치된 패키지 목록 저장
```bash
pip freeze > requirements_freeze.txt
```

---

## 💡 팁

### 1. Jupyter Kernel 등록
```bash
python -m ipykernel install --user --name deepfake --display-name "Python (Deepfake)"
```

### 2. GPU 메모리 확인
```python
import torch
print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 3. 캐시 정리
```bash
pip cache purge
```

---

## 📞 도움말

### 공식 문서
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [dlib 설치 가이드](http://dlib.net/compile.html)
- [OpenCV 설치](https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)

### 커뮤니티
- 대회 Q&A: https://aifactory.space/task/9197/qna
- 운영 문의: cs@aifactory.page

---

**설치 완료 후:**
1. `python improved_inference.py` - 샘플 데이터 테스트
2. `prepare_submission.bat` - 제출 준비
3. `jupyter notebook` - 노트북 열기

**행운을 빕니다! 🍀**

