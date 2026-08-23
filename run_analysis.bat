@echo off
set TARGET_DIR=%~1

if not "%TARGET_DIR%"=="" goto RUN

echo ====================================================
echo  Plasmon Analysis Pipeline
echo ====================================================
echo  [Usage 1] Drag and drop your experiment folder here.
echo  [Usage 2] Or type the folder name below.
echo ----------------------------------------------------
set /p TARGET_DIR="Enter Folder Name: "

:RUN
if "%TARGET_DIR%"=="" (
    echo [Error] Folder name was not entered.
    pause
    exit /b
)

echo.
echo ====================================================
echo  Starting analysis on: %TARGET_DIR%
echo ====================================================
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0run_pipeline.py" --dir "%TARGET_DIR%" --method peak --min-dist 4 --radius 1 --size 1.0

echo.
echo ====================================================
echo  [STEP 2/2] Running Integrated Voronoi & Excel Summary Analysis...
echo ====================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0run_batch_analysis.py" --dir "%TARGET_DIR%" --sigmas 3 5

echo.
echo ====================================================
echo  All analysis, Voronoi plots, and Excel summary completed!
echo ====================================================
echo.
pause
