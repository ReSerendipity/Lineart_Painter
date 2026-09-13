@echo off
chcp 65001 >nul
title lineart_painter · 模型权重下载
cd /d "%~dp0"
echo 正在通过 setup_models.py 下载全部模型权重（hf-mirror.com 国内镜像）...
echo 已存在且大小匹配的文件会自动跳过（支持断点续传）。
echo 用法示例:
echo   download_models.bat            # 下载全部可选模型
echo   download_models.bat --only sd15 cn_anime   # 只下指定项
echo   download_models.bat --skip midas animatediff   # 跳过指定项
echo.
echo 若报错缺少依赖，请先执行: pip install huggingface_hub
python setup_models.py %*
pause
