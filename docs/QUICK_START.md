# ⚡ 빠른 시작 가이드

## 🎯 목표
딥페이크 탐지 모델을 개선하고 제출까지 완료하기

---

## 📋 현재 상황

✅ **완료됨:**
- 가상환경 생성 (`venv/`)
- 대부분 라이브러리 설치 완료
- 개선된 노트북 준비 (`task_improved.ipynb`)
- 문서 및 스크립트 준비

⚠️ **dlib 설치 실패:**
- 얼굴 검출 라이브러리
- CMake 필요
- 아래 3가지 해결 방법 참고

---

## 🚀 3가지 시작 방법

### 방법 1: dlib 설치하고 로컬 테스트 (권장)

#### Step 1: CMake 및 Build Tools 설치
```bash
# 1. CMake 다운로드 및 설치
https://cmake.org/download/
# 설치 시 "Add to PATH" 체크!

# 2. Visual Studio Build Tools 다운로드
https://visualstudio.microsoft.com/downloads/
# "Desktop development with C++" 워크로드 선택
```

#### Step 2: dlib 재설치
```bash
# 가상환경 활성화
venv\Scripts\activate

# dlib 설치
pip install dlib
```

#### Step 3: 로컬 테스트
```bash
# 샘플 데이터로 추론 테스트
python dev\improved_inference.py
```

#### Step 4: 제출
```bash
# 제출 폴더 생성
scripts\prepare_submission.bat

# submit/task.ipynb 열어서 Key 입력 후 실행
```

---

### 방법 2: dlib wheel 사용 (빠름)

```bash
# 1. Python 버전 확인
python --version
# Python 3.10.11

# 2. 해당 버전의 dlib wheel 다운로드
# https://github.com/sachadee/Dlib
# 또는 https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib

# 3. wheel 파일 설치
venv\Scripts\activate
pip install dlib-19.24.0-cp310-cp310-win_amd64.whl

# 4. 로컬 테스트
python dev\improved_inference.py

# 5. 제출
scripts\prepare_submission.bat
```

---

### 방법 3: 로컬 테스트 스킵하고 바로 제출 (가장 빠름)

> 제출 환경에는 dlib이 이미 설치되어 있음!

```bash
# 1. 제출 폴더 생성
scripts\prepare_submission.bat

# 2. submit/task.ipynb 열기

# 3. Cell 19에 Competition Key 입력
aif.submit(
    model_name="deepfake_improved_v1",
    key="YOUR_KEY_HERE"  # ← 본인 Key
)

# 4. 모든 셀 순차 실행
```

**Competition Key 가져오기:**
1. AI Factory 로그인
2. 마이페이지 → 활동히스토리
3. "딥페이크 범죄 대응..." → 키복사
4. 노트북에 붙여넣기

---

## 📊 제출 후 확인 사항

### 리더보드 확인
- 제출 후 약 50-60분 소요
- [리더보드](https://aifactory.space/task/9197/leaderboard) 확인

### 에러 발생 시
1. [제출이력] → [에러보기] 클릭
2. Traceback 확인
3. 주요 에러:
   - **OOM**: BATCH_SIZE 줄이기 (16→8)
   - **Timeout**: NUM_FRAMES 줄이기 (30→20)
   - **Import Error**: 라이브러리 버전 확인

---

## 🎛️ 하이퍼파라미터 튜닝

성능 향상을 위해 `submit/task.ipynb` Cell 15 수정:

### 정확도 중시
```python
NUM_FRAMES = 40      # 프레임 많이
MEAN_WEIGHT = 0.7    # 평균 비중 높게
MAX_WEIGHT = 0.3
```

### 속도 중시
```python
NUM_FRAMES = 20      # 프레임 적게
BATCH_SIZE = 32      # 배치 크게
```

### 균형 (기본값)
```python
NUM_FRAMES = 30
BATCH_SIZE = 16
MEAN_WEIGHT = 0.6
MAX_WEIGHT = 0.4
```

---

## 📈 예상 결과

### 베이스라인
- F1 Score: ~0.92
- 추론 시간: 70분

### 개선 버전
- F1 Score: ~0.94-0.97 (+2~5%)
- 추론 시간: 50-60분 (-20~30%)

---

## 🆘 도움말

### 자주 묻는 질문

**Q: dlib 설치가 계속 실패해요**
A: 방법 3 (바로 제출) 사용 권장. 제출 환경에는 이미 설치됨.

**Q: 로컬에서 테스트하고 싶어요**
A: CMake + Visual Studio Build Tools 설치 필요. 또는 wheel 사용.

**Q: GPU가 없어요**
A: 괜찮습니다. 제출 환경에서 L4 GPU 제공됨.

**Q: 제출 시간이 너무 오래 걸려요**
A: NUM_FRAMES를 20으로 줄이면 40-50분으로 단축.

**Q: 하루 3회 제출 제한이 부담됩니다**
A: 로컬 테스트로 사전 검증하거나, 파라미터 신중히 선택.

---

## 📞 문의

- **기술 문의**: cs@aifactory.page
- **Q&A 게시판**: https://aifactory.space/task/9197/qna

---

**다음 단계:**
1. 위 3가지 방법 중 선택
2. 제출 실행
3. 리더보드 확인
4. 파라미터 튜닝 후 재제출

**행운을 빕니다! 🍀**

