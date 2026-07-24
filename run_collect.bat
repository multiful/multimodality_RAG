@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === 하나증권 산업분석 리포트 20개 수집 ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect_hana_industry.ps1" -Count 20
pause
