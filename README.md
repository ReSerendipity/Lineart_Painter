# Lineart Painter

本地运行的 AI 临摹重绘工作台：把图片或视频处理为线稿、固有色、体积明暗、反射光、色线、高光和最终还原结果。项目提供一个无需前端构建工具的本地 Web 界面，也可以直接调用 Python 模块处理单张图片或视频。

## 功能

- 图片和视频输入
- 经典 XDoG 线稿提取
- K-Means 固有色提取与多阶段重绘
- 可选 Anime2Sketch 神经线稿
- 可选 ControlNet + Stable Diffusion 神经上色
- 可选 DDColor 黑白照片上色
- 可选 AnimeGANv2 动漫风格化
- 浏览器查看阶段结果，并导出 PNG / MP4

## 快速开始

### 1. 安装基础依赖

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python
```

视频处理需要 OpenCV 的视频编解码支持。若需要更好的 H.264 兼容性，可额外安装 `ffmpeg` 并确保它位于 `PATH` 中。

### 2. 启动本地 Web 界面

Windows 用户可以双击 `启动程序.cmd`。如果 Python 不在 `C:\Python312\python.exe`，请先修改该启动文件中的 Python 路径。

也可以在项目目录执行：

```bash
python webui.py
```

然后打开 <http://127.0.0.1:8765>，选择图片或视频并运行处理。

## 命令行用法

经典处理不需要神经模型，可直接运行：

```bash
python lineart_painter.py input.png --out out_demo
```

处理视频：

```bash
python lineart_painter.py input.mp4 --out out_video
```

查看全部参数：

```bash
python lineart_painter.py --help
```

## 可选模型

模型权重没有提交到 Git 仓库，因为它们体积很大，并且部分模型受各自上游项目许可证和下载条款约束。程序会从本地目录读取这些权重：

| 功能 | 目录或文件 | 额外依赖 |
| --- | --- | --- |
| Anime2Sketch 神经线稿 | `weights/` | PyTorch |
| ControlNet + Stable Diffusion 上色 | `models/` | PyTorch、Diffusers、Transformers |
| DDColor 黑白照片上色 | `models/ddcolor/` | PyTorch |
| AnimeGANv2 风格化 | `models/animegan/` | ONNX Runtime、OpenCV |

没有模型时，经典 XDoG + K-Means 流程仍可运行。模型缓存、生成结果和用户上传文件均由 `.gitignore` 排除，不应提交到公共仓库。

## 目录结构

```text
Lineart_Painter/
├── webui.py                 # 本地 Web 服务和界面
├── lineart_painter.py       # 核心图片 / 视频处理流程
├── neural.py                # 可选神经模型适配
├── basicsr/                 # DDColor 所需网络组件
├── _ddcolor_*.py            # DDColor 推理组件
├── models/                  # 本地模型缓存，不纳入 Git
├── weights/                 # Anime2Sketch 权重，不纳入 Git
├── uploads/                 # Web 上传文件，不纳入 Git
└── webout/                  # Web 生成结果，不纳入 Git
```

## 注意事项

- 神经上色和神经线稿通常需要较多显存；首次加载模型也会明显更慢。
- Web 服务默认只监听 `127.0.0.1`，仅供本机使用。
- 输入图片、视频和生成结果默认保存在本地项目目录中，请根据需要自行清理。
- 本项目代码以 Apache License 2.0 发布；第三方模型和依赖仍以其各自许可证为准。
