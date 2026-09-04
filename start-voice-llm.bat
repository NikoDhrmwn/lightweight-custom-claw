@echo off
title LiteClaw — Gemma 4 E2B LLM Server
echo.
echo   ========================================
echo   Starting llama-server for Voice...
echo   ========================================
echo.

cd /d "%~dp0"

:: Check if llama-server.exe exists
where llama-server >nul 2>nul
if %errorlevel% neq 0 (
    if not exist "llama-server.exe" (
        echo   [WARNING] llama-server.exe was not found in PATH or root directory.
        echo   Please make sure llama.cpp binaries are installed.
        echo.
    )
)

:: Check if model exists
if not exist "E:\Qwen3.6\models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf" (
    echo   [WARNING] E:\Qwen3.6\models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf was not found.
    echo   Please download it from:
    echo   https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF
    echo   and place it in E:\Qwen3.6\models folder.
    echo.
)

echo   Launching llama-server with Gemma 4 E2B on port 8081...
llama-server.exe ^
  -m E:\Qwen3.6\models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf ^
  -c 32768 ^
  -fa ^
  -ngl 99 ^
  -b 2048 ^
  -ub 512 ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --port 8081 ^
  --temp 1.0 ^
  --top-k 64 ^
  --top-p 0.95 ^
  --min-p 0.0 ^
  --repeat-penalty 1.0

if %errorlevel% neq 0 (
    echo.
    echo   [ERROR] Failed to run llama-server.exe.
)

pause
