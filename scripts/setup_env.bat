@echo off
REM 딥페이크 탐지 모델 - 가상환경 설정 스크립트

echo ======================================
echo 딥페이크 탐지 모델 개발 환경 설정
echo ======================================
echo.

REM Python 버전 확인
echo [1/5] Python 버전 확인...
python --version
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다!
    echo Python 3.9 이상을 설치해주세요.
    pause
    exit /b 1
)
echo.

REM 가상환경 생성
echo [2/5] 가상환경 생성 중...
if exist venv (
    echo 기존 가상환경이 있습니다. 삭제하고 새로 만들까요? [Y/N]
    set /p answer=
    if /i "%answer%"=="Y" (
        rmdir /s /q venv
        python -m venv venv
    )
) else (
    python -m venv venv
)
echo ✓ 가상환경 생성 완료
echo.

REM 가상환경 활성화
echo [3/5] 가상환경 활성화...
call venv\Scripts\activate.bat
echo ✓ 가상환경 활성화 완료
echo.

REM pip 업그레이드
echo [4/5] pip 업그레이드 중...
python -m pip install --upgrade pip
echo.

REM 라이브러리 설치
echo [5/5] 필요한 라이브러리 설치 중...
echo.
echo ※ 주의: PyTorch CUDA 버전을 선택하세요.
echo.
echo [1] CUDA 11.8 (추천 - 대회 환경과 동일)
echo [2] CUDA 12.x
echo [3] CPU only (GPU 없음)
echo.
set /p cuda_choice="선택 (1-3): "

if "%cuda_choice%"=="1" (
    echo CUDA 11.8 버전 설치 중...
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
) else if "%cuda_choice%"=="2" (
    echo CUDA 12.x 버전 설치 중...
    pip install torch==2.7.1 torchvision==0.22.1
) else (
    echo CPU 버전 설치 중...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo 나머지 라이브러리 설치 중...
pip install transformers==4.30.0
pip install opencv-python==4.10.0.82
pip install dlib
pip install numpy==1.26.4 pandas scipy==1.11.4 scikit-learn==1.3.2
pip install tqdm pathlib
pip install jupyter ipykernel matplotlib datasets
echo.

echo ======================================
echo 환경 설정 완료!
echo ======================================
echo.
echo 다음 명령어로 가상환경을 활성화하세요:
echo   venv\Scripts\activate
echo.
echo Jupyter Notebook 실행:
echo   jupyter notebook
echo.
pause

