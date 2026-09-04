@echo off
title LiteClaw — OmniVoice TTS Server
echo.
echo   ========================================
echo   Starting OmniVoice.cpp Server...
echo   ========================================
echo.

cd /d "%~dp0"

:: Check if omnivoice-server.exe exists
if not exist "omnivoice-server.exe" (
    echo   [WARNING] omnivoice-server.exe was not found in the root directory.
    echo   Please place the omnivoice-server.exe binary in this folder.
    echo.
)

:: Check if model exists
if not exist "E:\Qwen3.6\models\omnivoice-base-Q8_0.gguf" (
    echo   [WARNING] E:\Qwen3.6\models\omnivoice-base-Q8_0.gguf was not found.
    echo   Please download it from:
    echo   https://huggingface.co/Serveurperso/OmniVoice-GGUF
    echo   and place it in E:\Qwen3.6\models folder.
    echo.
)

echo   Launching omnivoice-server.exe on port 8090...
omnivoice-server.exe --model E:\Qwen3.6\models\omnivoice-base-Q8_0.gguf --tokenizer E:\Qwen3.6\models\omnivoice-tokenizer-Q8_0.gguf --port 8090
if %errorlevel% neq 0 (
    echo.
    echo   [ERROR] Failed to run omnivoice-server.exe.
)

pause
