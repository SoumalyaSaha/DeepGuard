@echo off
title Deepfake Detector Setup
color 0A
echo.
echo  ====================================
echo   Deepfake Detector - Setup ^& Launch
echo  ====================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during install!
    pause
    exit /b 1
)
echo  [OK] Python found

:: Create virtual environment
if not exist venv (
    echo  Creating virtual environment...
    python -m venv venv
    echo  [OK] Virtual env created
) else (
    echo  [OK] Virtual env exists
)

:: Install packages
echo.
echo  Installing packages (first time takes 2-5 minutes)...
echo  Please wait...
echo.
venv\Scripts\pip install --quiet --upgrade pip
venv\Scripts\pip install fastapi "uvicorn[standard]" python-multipart httpx pydantic
venv\Scripts\pip install torch torchvision Pillow numpy opencv-python-headless
venv\Scripts\pip install librosa soundfile transformers timm scipy
echo  [OK] All packages installed

:: Create weights + logs folder
if not exist weights mkdir weights
if not exist logs mkdir logs
echo  [OK] Folders ready

:: Kill anything on our ports
for %%P in (8000 5001 5004 5002 7001) do (
    for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| find ":%%P "') do (
        taskkill /PID %%a /F >nul 2>&1
    )
)

echo.
echo  Starting all services...
echo.

start "Gateway :8000"    /min cmd /c "cd /d %~dp0 && venv\Scripts\uvicorn gateway.main:app --host 0.0.0.0 --port 8000 > logs\gateway.log 2>&1"
timeout /t 2 /nobreak >nul

start "NPR :5001"        /min cmd /c "cd /d %~dp0\models\npr     && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5001 > %~dp0logs\npr.log 2>&1"
start "UFD :5004"        /min cmd /c "cd /d %~dp0\models\ufd     && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5004 > %~dp0logs\ufd.log 2>&1"
start "RawNet :5002"     /min cmd /c "cd /d %~dp0\models\rawnet  && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5002 > %~dp0logs\rawnet.log 2>&1"
start "CrossViT :7001"   /min cmd /c "cd /d %~dp0\models\crossvit && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 7001 > %~dp0logs\crossvit.log 2>&1"
start "DIRE :5005"       /min cmd /c "cd /d %~dp0\models\dire     && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5005 > %~dp0logs\dire.log 2>&1"

echo  [OK] Gateway        -^>  http://localhost:8000
echo  [OK] NPR model      -^>  http://localhost:5001
echo  [OK] UFD model      -^>  http://localhost:5004
echo  [OK] RawNet2        -^>  http://localhost:5002
echo  [OK] CrossViT       -^>  http://localhost:7001
echo.
timeout /t 4 /nobreak >nul

echo  ====================================
echo   All done! Opening API docs...
echo  ====================================
echo.
echo  API:   http://localhost:8000
echo  Docs:  http://localhost:8000/docs
echo.
echo  To stop everything, close this window
echo  and run stop.bat
echo.
start "" "http://localhost:8000/docs"
pause
