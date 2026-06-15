@echo off
:: ============================================================
:: Windows Task Scheduler Setup — Run as Administrator
:: Place this in: AI-Forest-Fire-Detection\scripts\
:: ============================================================

set "SCRIPTS_DIR=%~dp0"
if "%SCRIPTS_DIR:~-1%"=="\" set "SCRIPTS_DIR=%SCRIPTS_DIR:~0,-1%"

set "LAUNCHER=%SCRIPTS_DIR%\run_pipeline_task.bat"

if not exist "%LAUNCHER%" (
    echo ERROR: Cannot find run_pipeline_task.bat at: %LAUNCHER%
    echo Make sure run_pipeline_task.bat is in the same scripts\ folder.
    pause & exit /b 1
)

echo Launcher : %LAUNCHER%
echo Interval : Every 1 hour
echo.

schtasks /delete /tn "FireDetectionPipeline" /f >nul 2>&1

schtasks /create /tn "FireDetectionPipeline" /tr "%LAUNCHER%" /sc HOURLY /mo 1 /ru "%USERNAME%" /rl HIGHEST /f

if %errorlevel%==0 (
    echo.
    echo ============================================================
    echo  SUCCESS - Pipeline runs every 1 hour automatically.
    echo  Log file: AI-Forest-Fire-Detection\pipeline_log.txt
    echo ============================================================
    echo.
    echo  Commands:
    echo    Run now  : schtasks /run /tn "FireDetectionPipeline"
    echo    View log : type "%SCRIPTS_DIR%\..\pipeline_log.txt"
    echo    Remove   : schtasks /delete /tn "FireDetectionPipeline" /f
) else (
    echo ERROR: Right-click and choose "Run as administrator"
)
pause