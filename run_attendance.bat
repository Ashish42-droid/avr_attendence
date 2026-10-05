@echo off
title AVR Face Recognition Attendance System
cd /d "%~dp0"
echo ======================================================
echo  AVR Face Recognition Attendance System
echo  Connecting to: rtsp://192.168.1.18:554/stream1
echo ======================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" face_attend.py %*
) else (
    python face_attend.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Attendance system exited with an error.
    pause
)
