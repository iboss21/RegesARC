@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>&1 || (
  echo Python 3 is required. Run tools\installers\install-toolchain.bat with the server stopped.
  exit /b 1
)

echo Starting RegesARC Serve Layer...
python reges/serve.py --port 8080 &
set SERVE_PID=%ERRORLEVEL%

if %SERVE_PID% neq 0 (
  echo Failed to start serve layer. Check for port conflicts or errors above.
  exit /b 1
)

echo.
echo Serve layer started on port 8080.
echo Starting chat client...
echo.

python reges/chat.py --port 8080

echo.
echo Shutting down serve layer...
taskkill /f /pid %SERVE_PID% >nul 2>&1

exit /b 0
