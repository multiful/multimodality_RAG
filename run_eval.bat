@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_env.bat"
echo === 고도화 지표 산출 ===
"%PY%" pipeline\eval_image.py %*
start "" "%~dp0eval\eval_report.html"
pause
