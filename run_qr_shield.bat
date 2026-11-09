@echo off
setlocal
cd /d "%~dp0"

if exist ".venv311\Scripts\python.exe" (
    set "PYTHON=.venv311\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    echo QR Shield AI virtual environment was not found.
    echo Create one with: python -m venv .venv
    echo Then install the dependencies and PyTorch as described in README.md.
    pause
    exit /b 1
)

"%PYTHON%" -m streamlit run "%~dp0frontend\app.py"
