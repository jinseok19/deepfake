@echo off
REM 딥페이크 탐지 모델 제출 준비 스크립트

echo ======================================
echo 딥페이크 탐지 모델 제출 준비
echo ======================================
echo.

REM 1. 제출 폴더 생성
echo [1/4] 제출 폴더 생성 중...
if exist submit rmdir /s /q submit
mkdir submit
mkdir submit\model
echo   완료!
echo.

REM 2. task.ipynb 복사
echo [2/4] task.ipynb 복사 중...
copy task_improved.ipynb submit\task.ipynb >nul
if errorlevel 1 (
    echo   [오류] task_improved.ipynb 파일을 찾을 수 없습니다!
    pause
    exit /b 1
)
echo   완료!
echo.

REM 3. 모델 폴더 복사
echo [3/4] 모델 폴더 복사 중 (시간이 걸릴 수 있습니다)...
xcopy baseline\model submit\model /E /I /Y /Q >nul
if errorlevel 1 (
    echo   [오류] 모델 폴더를 복사할 수 없습니다!
    pause
    exit /b 1
)
echo   완료!
echo.

REM 4. 파일 크기 확인
echo [4/4] 제출 파일 확인 중...
dir submit\task.ipynb | find "task.ipynb"
echo.
echo 모델 파일:
dir submit\model\deep-fake-detector-v2-model\*.* /b
echo.

echo ======================================
echo 제출 준비 완료!
echo ======================================
echo.
echo 다음 단계:
echo 1. submit\task.ipynb 파일을 열기
echo 2. Cell 19에서 YOUR_KEY_HERE를 본인의 Competition Key로 교체
echo 3. 모든 셀을 순서대로 실행
echo 4. 제출 완료 대기 (약 50-60분)
echo.
echo 자세한 내용은 SUBMISSION_GUIDE.md를 참고하세요.
echo.
pause

