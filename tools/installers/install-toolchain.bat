@echo off
setlocal
echo RegesARC toolchain installer. Stop any local server on 8080 first.
where gcc >nul 2>&1 && goto :python
if exist C:\msys64\usr\bin\bash.exe (
  C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-make"
  echo Add C:\msys64\mingw64\bin to PATH, then reopen the shell.
  goto :python
)
winget install -e --id MSYS2.MSYS2 --accept-package-agreements --accept-source-agreements
if exist C:\msys64\usr\bin\bash.exe (
  C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-make"
)
:python
where python >nul 2>&1 || winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
echo CUDA is optional and not installed here.
exit /b 0
