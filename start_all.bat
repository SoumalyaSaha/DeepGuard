@echo off
echo Starting DeepGuard...
start "Gateway" cmd /k "cd /d C:\Users\soumalyasaha\OneDrive\Desktop\DeepGuard && venv\Scripts\activate && cd gateway && uvicorn main:app --port 8000"
timeout /t 3 /nobreak >nul
start "NPR Model" cmd /k "cd /d C:\Users\soumalyasaha\OneDrive\Desktop\DeepGuard && venv\Scripts\activate && cd models\npr && uvicorn main:app --port 8001"
echo Both servers starting! Check the two windows that opened.
start "DIRE Model" cmd /k "cd /d C:\Users\soumalyasaha\OneDrive\Desktop\DeepGuard && venv\Scripts\activate && cd models\dire && uvicorn main:app --port 5003"