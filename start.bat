@echo off
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
echo.
echo  Arxivore starting at http://127.0.0.1:8000
echo  Press Ctrl+C to stop.
echo.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
