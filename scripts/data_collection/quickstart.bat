@echo off
REM Windows Quick Start Script for Data Collection
REM ================================================

echo ========================================
echo Data Collection Quick Start (Windows)
echo ========================================
echo.

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
echo OK: Python installed
echo.

REM Install dependencies
echo [2/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK: Dependencies installed
echo.

REM Check .env file
echo [3/5] Checking .env file...
if not exist .env (
    echo WARNING: .env file not found!
    echo Creating template .env file...
    echo REDDIT_CLIENT_ID=DgmKFCNd5jIDZGZmx7HcUg > .env
    echo REDDIT_CLIENT_SECRET=BtY3O1iq_emy_yyA73gdHzUqvtGgig >> .env
    echo REDDIT_USER_AGENT=DeepfakeDataCollector/1.0 >> .env
    echo COMFYUI_URL=http://127.0.0.1:8188 >> .env
    echo OUTPUT_DIR=../../dataset >> .env
    echo FFHQ_COUNT=10000 >> .env
    echo CELEBAHQ_COUNT=8000 >> .env
    echo YOUTUBE_COUNT=7000 >> .env
    echo REDDIT_COUNT=3000 >> .env
    echo GALLERY_COUNT=2000 >> .env
    echo FLUX_COUNT=3000 >> .env
    echo OK: .env file created
) else (
    echo OK: .env file exists
)
echo.

REM Setup Kaggle API
echo [4/5] Setting up Kaggle API...
if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
if exist kaggle.json (
    copy /Y kaggle.json "%USERPROFILE%\.kaggle\kaggle.json" >nul
    echo OK: Kaggle API configured
) else (
    echo WARNING: kaggle.json not found
    echo Download from https://www.kaggle.com/settings
)
echo.

REM Run data collection
echo [5/5] Starting data collection...
echo.
echo WARNING: This will take 25-34 hours!
echo - Real datasets: 3-5 hours
echo - YouTube crawling: 3-5 hours
echo - Reddit crawling: 1-2 hours
echo - Galleries crawling: 1-2 hours
echo - ComfyUI generation: SKIPPED (use --no-skip-comfyui to include)
echo.
echo Press Ctrl+C to cancel, or
pause

python run_all.py --skip-comfyui

echo.
echo ========================================
echo Data Collection Complete!
echo ========================================
echo.
echo Check results in: ..\..\dataset
echo Metadata: ..\..\dataset\metadata\combined_dataset.csv
echo.
pause

