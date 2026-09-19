@echo off
setlocal
cd /d %~dp0
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
set VISIONID_EPHEMERAL=0
set VISIONID_RESET_ON_STARTUP=0
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
endlocal
