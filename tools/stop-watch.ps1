# Stops the hidden AiForPrinter service (python main.py watch). Nothing else is touched:
# not camera_server, not n8n, not the printer.
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*main.py watch*' }
if (-not $procs) { Write-Output 'AiForPrinter service is not running'; exit 0 }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output "stopped $($p.ProcessId)" }
