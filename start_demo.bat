@echo off
cd /d C:\Users\pavel\OneDrive\Plocha\diplomka_code

start "Flask app" cmd /k "python run.py"

timeout /t 5 /nobreak >nul

start "Cloudflared tunnel" cmd /k "C:\Cloudflared\bin\cloudflared.exe tunnel --url http://localhost:5000"

pause