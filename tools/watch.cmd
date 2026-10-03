@echo off
rem AiForPrinter service: watches the printer camera while printing. Window: main.py ui
rem Auto-pause is off by default. The service never stops a print.
cd /d "%~dp0.."
".venv\Scripts\python.exe" main.py watch
