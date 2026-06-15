@echo off
:: ============================================================
:: Pipeline Launcher — called by Windows Task Scheduler
:: Place this in: AI-Forest-Fire-Detection\scripts\
:: ============================================================

:: Change to project root (one level up from scripts\)
cd /d "%~dp0.."

:: Run the pipeline from the correct directory
python scripts\run_pipeline.py >> pipeline_log.txt 2>&1