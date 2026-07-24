@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_env.bat"
echo === Phase 2: 이미지 고도화 파이프라인 (규칙+캐시+pHash+VLM) ===
"%PY%" pipeline\s2_image_pipeline.py %*
pause
