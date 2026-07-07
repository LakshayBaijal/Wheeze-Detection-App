@echo off
REM Windows: Double-click this file to start the testing app

cd /d "%~dp0"
echo Starting Wheeze Detection Testing App...
echo.
echo If this is your first time, Python packages are installing (takes ~1 min)
echo.

REM Install requirements
pip install -q -r requirements.txt

REM Start the app
echo Opening the app in your browser...
streamlit run src/app.py

REM Keep the window open if there's an error
pause
