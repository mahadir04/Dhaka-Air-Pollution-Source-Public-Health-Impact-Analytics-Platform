@echo off
REM Launcher for AirImpact Dhaka Streamlit Dashboard
cd /d "%~dp0"
if exist ".venv\Scripts\streamlit.exe" (
    echo Starting Streamlit via project virtual environment...
    ".venv\Scripts\streamlit.exe" run dashboard/app.py
) else (
    echo Starting Streamlit via python -m streamlit...
    python -m streamlit run dashboard/app.py
)
pause
