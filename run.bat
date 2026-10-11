@echo off
REM ==============================================================================
REM  🌱 ZeroTrace AI — 1-Click Laptop Runner (Windows)
REM  Slashes AI electrical consumption, token waste, and carbon emissions.
REM ==============================================================================

echo.
echo ========================================================================
echo   ZeroTrace AI -- Green AI Inference Gateway and Telemetry Profiler
echo ========================================================================
echo.

REM 1. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.9+ from https://www.python.org/ (ensure "Add Python to PATH" is checked).
    pause
    exit /b 1
)

REM 2. Virtual Environment
cd /d "%~dp0"
if not exist ".venv" (
    echo [*] Setting up virtual environment (.venv)...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM 3. Install requirements
echo [*] Verifying dependencies...
pip install -r requirements.txt --quiet

REM 4. Check .env
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [*] Created .env from template
    )
)

echo.
echo ========================================================================
echo   Launching ZeroTrace AI at http://127.0.0.1:8000
echo   Opening web dashboard in your default browser...
echo   Press Ctrl+C to stop the server.
echo ========================================================================
echo.

start http://127.0.0.1:8000

python server.py
pause
