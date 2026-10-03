@echo off
rem AiForPrinter service: watches the printer camera while printing. Window: main.py ui
rem Auto-pause is off by default. The service never stops a print.
rem Console output (start-up line, crashes) goes to %LOCALAPPDATA%\AiForPrinter\watch-console.log,
rem the journal itself to %LOCALAPPDATA%\AiForPrinter\watcher.log.
cd /d "%~dp0.."
if not exist "%LOCALAPPDATA%\AiForPrinter" mkdir "%LOCALAPPDATA%\AiForPrinter"
".venv\Scripts\python.exe" main.py watch >> "%LOCALAPPDATA%\AiForPrinter\watch-console.log" 2>&1
