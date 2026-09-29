@echo off
chcp 65001 >nul
echo ================================================================
echo  Installing Python 3.11, Git, GitHub CLI and PyCharm (winget)
echo ================================================================
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
winget install -e --id GitHub.cli --accept-package-agreements --accept-source-agreements
winget install -e --id JetBrains.PyCharm.Community --accept-package-agreements --accept-source-agreements
if errorlevel 1 winget install -e --id JetBrains.PyCharm --accept-package-agreements --accept-source-agreements
echo.
echo DONE. Close this window and run 1_setup.bat
echo done > "%~dp0logs_install_done.txt"
pause
