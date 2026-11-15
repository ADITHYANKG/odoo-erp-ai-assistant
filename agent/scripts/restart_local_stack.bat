@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart_local_stack.ps1" %*
pause
