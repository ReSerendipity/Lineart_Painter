@echo off
chcp 65001 >nul
title lineart_painter · AI 临摹重绘工作台
cd /d "%~dp0"
echo 正在启动 lineart_painter 工作台...
echo 服务就绪后会自动打开浏览器: http://127.0.0.1:8765
echo （本窗口保持打开，关闭即退出服务）
start "" /b cmd /c "timeout /t 2 >nul & start "" http://127.0.0.1:8765"
"C:\Python312\python.exe" webui.py
pause
