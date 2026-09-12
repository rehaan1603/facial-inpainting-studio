@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8765 in your browser. Keep this window open while using the app.
"%~dp0.venv\Scripts\python.exe" "%~dp0webapp\server.py"
if errorlevel 1 pause
