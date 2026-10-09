@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  RegesARC Launcher
echo ========================================
echo.

where python >nul 2>&1 || (
  echo ERROR: Python 3 is required but not found in PATH.
  echo Run tools\installers\install-toolchain.bat first, then restart this script.
  pause
  exit /b 1
)

python setup.py --list
echo.

set /p "choice=Download a model? (y/n): "
if /i "%choice%"=="y" goto :download
if /i "%choice%"=="n" goto :skip_download

:download
echo.
echo Available models for download:
python -c "import sys; sys.path.insert(0,'.'); from reges.catalog import CATALOG; [print(f'  {i+1}. {m[\"label\"]} ({m[\"params\"]})') for i,m in enumerate(CATALOG)]"
set /p "model_num=Enter model number (1-11): "
python -c "import sys; sys.path.insert(0,'.'); from reges.install import install_selected; install_selected(%model_num%)"
echo.

:skip_download
echo.
echo Starting chat interface...
start "" python reges/chat.py --model regescore-1.0-35
echo Chat client launched in new window.
pause
exit /b 0
