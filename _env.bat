@echo off
rem ── 공통 환경: 연산환경(MinerU·torch)은 pdfex\demo_venv 재사용 ──
rem 로컬에 demo_venv가 있으면 그것을, 없으면 pdfex의 것을 쓴다.
if exist "%~dp0demo_venv\Scripts\python.exe" (
  set "PY=%~dp0demo_venv\Scripts\python.exe"
) else (
  set "PY=C:\Users\wodlf\OneDrive\Desktop\pdfex\demo_venv\Scripts\python.exe"
)
