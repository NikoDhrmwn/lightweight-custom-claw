@echo off
title LiteClaw — Nemotron ASR Server
echo.
echo   ========================================
echo   Starting Nemotron ASR Server...
echo   ========================================
echo.

cd /d "%~dp0"

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo   [ERROR] Python is not installed or not in your PATH.
    echo   Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv_asr" (
    echo   [INFO] Creating Python virtual environment for ASR (venv_asr)...
    python -m venv venv_asr
)

call venv_asr\Scripts\activate.bat

:: Install requirements
python -c "import fastapi, uvicorn, nemo, torch" >nul 2>nul
if %errorlevel% neq 0 (
    echo   [INFO] Installing dependencies (fastapi, uvicorn, torch, nemo_toolkit)...
    pip install fastapi uvicorn
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install nemo_toolkit[asr]
)

:: Run the ASR server
python src/voice/asr_server.py

pause
