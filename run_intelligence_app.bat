@echo off
REM Launcher for AirImpact Dhaka Real-Time Intelligence Platform (app.py)
cd /d "%~dp0"
if exist ".venv\Scripts\streamlit.exe" (
    echo Starting AirImpact Dhaka Platform via project virtual environment...
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo Starting AirImpact Dhaka Platform via python -m streamlit...
    python -m streamlit run app.py
)
pause
