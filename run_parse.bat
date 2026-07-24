@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0_env.bat"
echo === Phase 1: MinerU 파싱 (하나 20건) ===
"%PY%" pipeline\s1_parse.py %*
echo.
echo Phase 1 종료. 실패목록: data\parsed\parse_failures.csv
pause
