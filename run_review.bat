@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_env.bat"
echo === 검수 뷰어 생성 (표본 라벨링) ===
"%PY%" pipeline\review_viewer.py %*
start "" "%~dp0eval\review.html"
pause
