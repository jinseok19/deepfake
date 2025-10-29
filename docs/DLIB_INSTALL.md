# dlib 로컬 설치 가이드 (Windows)

## 환경
- OS: Windows 10/11
- Python: 3.10
- 프로젝트: 딥페이크 탐지 모델

---

## 🚀 방법 1: Pre-built Wheel (추천! ⭐)

### CMake 없이 설치 가능!

```bash
# 1. 가상환경 활성화 (있으면)
venv\Scripts\activate

# 2. pip 업그레이드
python -m pip install --upgrade pip

# 3. dlib-bin 설치 (Pre-built)
pip install dlib-bin
```

### 또는 직접 wheel 다운로드

```bash
# Python 3.10용 dlib wheel
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp310-cp310-win_amd64.whl
```

### 설치 확인

```bash
python -c "import dlib; print(dlib.__version__)"
```

---

## 🔧 방법 2: CMake로 빌드 (시간 걸림)

### 1단계: CMake 설치

#### 옵션 A: Chocolatey 사용
```bash
# PowerShell (관리자 권한)
choco install cmake
```

#### 옵션 B: 직접 다운로드
1. https://cmake.org/download/ 방문
2. Windows x64 Installer 다운로드
3. 설치 시 "Add CMake to PATH" 체크

### 2단계: Visual Studio Build Tools 설치

1. https://visualstudio.microsoft.com/downloads/ 방문
2. "Build Tools for Visual Studio" 다운로드
3. "C++ build tools" 선택하여 설치

### 3단계: dlib 설치

```bash
# 가상환경 활성화
venv\Scripts\activate

# dlib 설치 (빌드 시작)
pip install dlib
```

⚠️ **빌드 시간**: 10~20분 소요

---

## 🐍 방법 3: Conda 사용 (Anaconda/Miniconda)

```bash
# Conda 환경 생성
conda create -n deepfake python=3.10
conda activate deepfake

# conda-forge에서 dlib 설치
conda install -c conda-forge dlib

# 확인
python -c "import dlib; print(dlib.__version__)"
```

---

## 🔍 설치 확인 및 테스트

### 설치 확인
```bash
python -c "import dlib; print('dlib version:', dlib.__version__)"
```

### 얼굴 검출 테스트
```python
import dlib
import cv2
from PIL import Image
import numpy as np

# 얼굴 검출기 로드
detector = dlib.get_frontal_face_detector()

# 샘플 이미지로 테스트
img_path = "samples/fake/image/sample_image_1.png"
img = Image.open(img_path)
img_np = np.array(img)

# 얼굴 검출
faces = detector(img_np, 1)
print(f"검출된 얼굴 수: {len(faces)}")
```

---

## ❌ 문제 해결

### 문제 1: "CMake not found"
**해결**: CMake 설치 (방법 2 참고)

### 문제 2: "error: Microsoft Visual C++ 14.0 is required"
**해결**: Visual Studio Build Tools 설치 (방법 2 참고)

### 문제 3: 빌드 시간이 너무 오래 걸림
**해결**: 방법 1 (Pre-built Wheel) 사용

### 문제 4: dlib-bin 설치 실패
**해결**: 
```bash
# 최신 pip로 업그레이드
python -m pip install --upgrade pip setuptools wheel

# 재시도
pip install dlib-bin
```

### 문제 5: Wheel이 없다는 에러
**해결**:
```bash
# Python 버전 확인
python --version

# 해당 버전에 맞는 wheel 사용
# Python 3.10: cp310
# Python 3.9: cp39
# Python 3.11: cp311
```

---

## 📋 권장 방법 요약

| 상황 | 권장 방법 | 이유 |
|------|----------|------|
| 빠른 설치 필요 | 방법 1 (Pre-built) | CMake 불필요, 1분 내 완료 |
| Conda 사용자 | 방법 3 (Conda) | 통합 환경 관리 |
| 최신 버전 필요 | 방법 2 (CMake) | 소스 빌드 |
| 시간 여유 없음 | 방법 1 (Pre-built) | 가장 빠름 |

---

## 🎯 추천 순서

1. **먼저 시도**: 방법 1 (Pre-built Wheel)
2. **실패 시**: 방법 3 (Conda) - Anaconda 있으면
3. **마지막**: 방법 2 (CMake 빌드) - 시간 여유 있을 때

---

## 📝 참고 자료

- dlib 공식: http://dlib.net/
- dlib GitHub: https://github.com/davisking/dlib
- Pre-built Wheels: https://github.com/jloh02/dlib/releases

---

**작성일**: 2025.10.29  
**Python 버전**: 3.10.11  
**OS**: Windows 10

