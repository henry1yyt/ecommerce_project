@echo off
cd /d %~dp0
python analysis.py
if errorlevel 1 exit /b 1
python verify_project.py
if errorlevel 1 exit /b 1
echo.
echo 完成：请打开 report\电商大促运营分析与退货风险预测.html
pause
