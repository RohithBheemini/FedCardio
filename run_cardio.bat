@echo off
REM ========================================================
REM FedCardio One-Click Launcher for Windows
REM ========================================================

echo.
echo ========================================================
echo  Starting FedCardio Multi-Hospital Clinical AI Platform
echo ========================================================
echo.

REM Verify Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

REM Display Platform Info
python cardio_cli.py info

REM Launch Streamlit Portal
echo [INFO] Launching FedCardio Web Portal (localhost:8501)...
python -m streamlit run web/app.py --server.port 8501

pause
