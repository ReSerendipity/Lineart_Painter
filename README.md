# Lineart Painter

本地运行的 AI 临摹重绘工作台：把图片或视频处理为线稿、固有色、体积明暗、反射光、色线、高光和最终还原结果。项目提供一个无需前端构建工具的本地 Web 界面，也可以直接调用 Python 模块处理单张图片或视频。

## 功能

- 图片和视频输入（Web 界面支持多文件批量、串行队列）
- 经典 XDoG 线稿提取
- K-Means 固有色提取与多阶段重绘
- 可选 Anime2Sketch 神经线稿
- 可选 ControlNet + Stable Diffusion 神经上色（5 种变体：anime / standard / canny / scribble / depth）
- 可选 img2img 原图风格混合还原（--neural-strength）
- 可选手绘线稿直接填色（--input-lineart）
- 可选 DDColor 黑白照片上色（图片与视频）
- 可选 AnimeGANv2 动漫风格化（图片与视频）
- 可选 AnimateDiff 视频帧间一致性上色（替代逐帧上色，减少闪烁）
- 浏览器查看阶段结果，并导出 PNG / MP4（自动转 H.264）

## 快速开始

### 1. 安装基础依赖

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python
```

神经功能需要 PyTorch、Diffusers、Transformers、ONNX Runtime（按需）：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install diffusers transformers accelerate safetensors huggingface_hub onnxruntime
```

视频处理需要 OpenCV 的视频编解码支持；H.264 转码由 `imageio-ffmpeg` 提供（`pip install imageio-ffmpeg`）。

### 2. 下载模型权重（一键脚本）

```bash
python setup_models.py              # 下载全部（约 10GB，hf-mirror 镜像）
python setup_models.py --list       # 查看可下载项
python setup_models.py --only sd15 cn_anime anime2sketch   # 只下需要的
```

经典管线（XDoG + K-Means）零模型即可运行；只有用到神经功能才需要对应权重。脚本可断点续传，已下载文件自动跳过。

### 3. 启动本地 Web 界面

Windows 用户可以双击 `启动程序.cmd`。如果 Python 不在 `C:\Python312\python.exe`，请先修改该启动文件中的 Python 路径。

也可以在项目目录执行：

```bash
python webui.py
```

然后打开 <http://127.0.0.1:8765>，选择图片或视频（可多选批量）并运行处理。

### 4. 回归测试

```bash
python smoke_test.py            # 全量（含神经上色与视频，约 3~5 分钟）
python smoke_test.py --fast     # 快速（约 30 秒）
```

## 命令行用法

经典处理不需要神经模型，可直接运行：

```bash
python lineart_painter.py input.png --out out_demo
```

处理视频：

```bash
python lineart_painter.py input.mp4 --out out_video
```

常用神经组合：

```bash
# 神经线稿 + 神经上色（动漫线稿 ControlNet，默认）
python lineart_painter.py input.png --neural-lineart --neural-color

# 用 canny / scribble / depth 变体（先运行 setup_models.py 下载对应权重）
python lineart_painter.py input.png --neural-color --cn canny

# 手绘线稿直接上色
python lineart_painter.py sketch.png --input-lineart --neural-color

# 视频帧间一致性上色（AnimateDiff，需先下载 animatediff 权重）
python lineart_painter.py input.mp4 --animate-diff

# 视频整体应用 照片上色 + 动漫风格化
python lineart_painter.py input.mp4 --photo-color --anime-gan

# 黑白照片上色 / 照片转动漫
python lineart_painter.py old_photo.png --photo-color
python lineart_painter.py photo.png --anime-gan
```

查看全部参数：

```bash
python lineart_painter.py --help
```

小提示：

- 文件不存在或不支持的扩展名会被自动跳过并提示，不需要手动清理。
- 处理失败时默认只打印一句话原因；需要完整堆栈排查时加 `--debug`。

## 可选模型

模型权重没有提交到 Git 仓库，因为它们体积很大，并且部分模型受各自上游项目许可证和下载条款约束。用 `python setup_models.py` 一键下载（来自 hf-mirror 镜像），程序从本地目录读取：

| 功能 | 目录或文件 | 大小 | 额外依赖 |
| --- | --- | --- | --- |
| Anime2Sketch 神经线稿 | `weights/` | 409MB | PyTorch |
| SD1.5 上色基座 | `models/hfhome/` | 2.7GB | PyTorch、Diffusers、Transformers |
| ControlNet anime 线稿 | `models/hfhome/` | 722MB | 同上 |
| ControlNet standard/canny/scribble/depth | `models/cn_*/` | 各 ~690MB | 同上 |
| DDColor 黑白照片上色 | `models/ddcolor/` | 210MB | PyTorch |
| AnimeGANv2 风格化 | `models/animegan/` | 8MB | ONNX Runtime、OpenCV |
| AnimateDiff 视频一致性 | `models/animatediff/` | 1.7GB | PyTorch、Diffusers |
| MiDaS 深度估计（depth 变体依赖） | `models/hfhome/` | 86MB | Transformers |

没有模型时，经典 XDoG + K-Means 流程仍可运行。模型缓存、生成结果和用户上传文件均由 `.gitignore` 排除，不应提交到公共仓库。

## 目录结构

```text
Lineart_Painter/
├── webui.py                 # 本地 Web 服务和界面（支持批量）
├── lineart_painter.py       # 核心图片 / 视频处理流程
├── neural.py                # 神经模型适配（含 AnimateDiff / 深度估计）
├── setup_models.py          # 一键下载全部模型权重
├── smoke_test.py            # 自动化回归测试
├── basicsr/                 # DDColor 所需网络组件
├── _ddcolor_*.py            # DDColor 推理组件
├── _a2s_*.py                # Anime2Sketch 官方源码
├── models/                  # 本地模型缓存，不纳入 Git
├── weights/                 # Anime2Sketch 权重，不纳入 Git
├── uploads/                 # Web 上传文件，不纳入 Git
└── webout/                  # Web 生成结果，不纳入 Git
```

## 注意事项

- 神经上色和神经线稿通常需要较多显存；首次加载模型也会明显更慢。无 GPU 时自动回退 CPU（经典管线 CPU 秒级，神经上色 CPU 每张 2~5 分钟）。
- AnimateDiff 以 16 帧为一段生成，段边界可能有轻微跳变；逐帧神经上色对快速运动镜头有闪烁，AnimateDiff 可显著缓解。
- Web 服务默认只监听 `127.0.0.1`，仅供本机使用。
- 输入图片、视频和生成结果默认保存在本地项目目录中，请根据需要自行清理。
- 本项目代码以 Apache License 2.0 发布；第三方模型和依赖仍以其各自许可证为准。
