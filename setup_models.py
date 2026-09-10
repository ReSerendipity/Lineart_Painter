# -*- coding: utf-8 -*-
"""
setup_models.py — 一键下载 / 校验全部模型权重
================================================
用法：
    python setup_models.py                 # 下载全部可选模型（约 10GB）
    python setup_models.py --list          # 列出可下载项
    python setup_models.py --only sd15 cn_anime anime2sketch   # 只下指定项
    python setup_models.py --skip anime2sketch                 # 跳过指定项

所有权重来自 hf-mirror（国内镜像），已存在且大小匹配的文件自动跳过（可断点续传）。
无需全部下载：经典管线（XDoG + K-Means）零模型即可运行；
只有用到对应神经功能时才需要对应权重。

可选模型（按功能）：
  anime2sketch  神经线稿         409MB  weights/
  sd15          SD1.5 上色基座   ~2.7GB models/hfhome（huggingface 缓存）
  cn_anime      ControlNet 动漫线稿 ~722MB models/hfhome
  cn_standard   ControlNet 标准线稿 ~722MB models/cn_standard/
  cn_canny      ControlNet Canny 边缘 ~1.4GB models/cn_canny/
  cn_scribble   ControlNet 草图   ~1.4GB models/cn_scribble/
  cn_depth      ControlNet 深度   ~1.4GB models/cn_depth/
  ddcolor       DDColor 照片上色  210MB  models/ddcolor/
  animegan      AnimeGANv2 风格化 8MB   models/animegan/
  animatediff   AnimateDiff 运动适配器 ~1.7GB models/animatediff/
  midas         MiDaS 深度估计    ~86MB  models/hfhome（cn_depth 需要）
"""

import os
import sys
import time
import urllib.request

HF_ENDPOINT = "https://hf-mirror.com"
HERE = os.path.dirname(os.path.abspath(__file__))
HF_HOME = os.path.join(HERE, "models", "hfhome")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HOME", HF_HOME)

CN_COMMON = {  # 与 NeuralColorizer 的加载方式一致：config.json + fp16.safetensors
    "cn_standard": ("lllyasviel/control_v11p_sd15_lineart",
                    "models/cn_standard", "线稿 ControlNet(标准)"),
    "cn_canny": ("lllyasviel/control_v11p_sd15_canny",
                 "models/cn_canny", "ControlNet(Canny 边缘)"),
    "cn_scribble": ("lllyasviel/control_v11p_sd15_scribble",
                    "models/cn_scribble", "ControlNet(草图)"),
    "cn_depth": ("lllyasviel/control_v11f1p_sd15_depth",
                 "models/cn_depth", "ControlNet(深度)"),
}


def _fmt_mb(n):
    return "%.0fMB" % (n / 1048576)


def _url_download(url, dest, expect_mb=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  [已存在] %s (%s)" % (dest, _fmt_mb(os.path.getsize(dest))))
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("  [下载] %s -> %s%s" % (
        url.split("/resolve/main/")[-1], dest,
        " (~%s)" % _fmt_mb(expect_mb * 1048576) if expect_mb else ""))
    req = urllib.request.Request(url, headers={"User-Agent": "doubao-agent"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print("  [完成] %s (%s, %.0fs)" % (dest, _fmt_mb(total), time.time() - t0))


def _hf_download(repo_id, local_dir=None, variant=None, cache_dir=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    if local_dir and os.path.isdir(local_dir) and any(
            f.endswith((".safetensors", ".pth", ".bin")) and
            os.path.getsize(os.path.join(local_dir, f)) > 0
            for f in os.listdir(local_dir)):
        print("  [已存在] %s" % local_dir)
        return
    print("  [下载] %s%s%s" % (repo_id, " (fp16)" if variant else "",
                              " -> %s" % local_dir if local_dir else ""))
    kw = {}
    if variant:
        kw["variant"] = variant
    if cache_dir:
        kw["cache_dir"] = cache_dir
    if local_dir:
        kw["local_dir"] = local_dir
    t0 = time.time()
    snapshot_download(repo_id, allow_patterns=[
        "*.json", "*.txt", "*.safetensors", "*.bin", "*.pth", "*.model",
        "*.patch"], **kw)
    print("  [完成] %.0fs" % (time.time() - t0))


def download_all(only=None, skip=()):
    from huggingface_hub import hf_hub_download

    items = [
        ("anime2sketch", "神经线稿 (Anime2Sketch)",
         lambda: _hf_download("gyrojeff/Anime2Sketch",
                              local_dir=os.path.join(HERE, "weights"))),
        ("sd15", "SD1.5 上色基座",
         lambda: _hf_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                              variant="fp16", cache_dir=HF_HOME)),
        ("cn_anime", "ControlNet 动漫线稿",
         lambda: _hf_download(
             "lllyasviel/control_v11p_sd15s2_lineart_anime",
             variant="fp16", cache_dir=HF_HOME)),
    ]
    for name, (repo, local, note) in CN_COMMON.items():
        items.append((name, note,
                      (lambda r=repo, l=local: _download_cn(r, l))))
    items += [
        ("ddcolor", "DDColor 照片上色",
         lambda: _url_download(
             HF_ENDPOINT + "/piddnad/DDColor-models/resolve/main/"
             "ddcolor_paper_tiny.pth",
             os.path.join(HERE, "models", "ddcolor",
                          "ddcolor_paper_tiny.pth"), 210)),
        ("animegan", "AnimeGANv2 风格化",
         lambda: _url_download(
             HF_ENDPOINT + "/akhaliq/AnimeGANv2-ONNX/resolve/main/"
             "face_paint_512_v2_0.onnx",
             os.path.join(HERE, "models", "animegan",
                          "face_paint_512_v2_0.onnx"), 8)),
        ("animatediff", "AnimateDiff 运动适配器",
         lambda: _download_cn("guoyww/animatediff-motion-adapter-v1-5-2",
                              os.path.join(HERE, "models", "animatediff"))),
        ("midas", "MiDaS 深度估计 (cn_depth 依赖)",
         lambda: _hf_download("Intel/dpt-hybrid-midas",
                              cache_dir=HF_HOME)),
    ]

    if only:
        items = [it for it in items if it[0] in only]
    items = [it for it in items if it[0] not in skip]

    print("=" * 62)
    for name, note, _ in items:
        print("  [%s] %s" % (name, note))
    print("=" * 62)
    for name, note, fn in items:
        print(">>> %s (%s)" % (name, note))
        try:
            fn()
        except Exception as e:
            print("  [失败] %s: %s" % (name, e))
    print("全部完成。")


def _download_cn(repo, local):
    from huggingface_hub import hf_hub_download
    if os.path.isdir(local) and any(
            f == "diffusion_pytorch_model.fp16.safetensors" and
            os.path.getsize(os.path.join(local, f)) > 0
            for f in os.listdir(local)):
        print("  [已存在] %s" % local)
        return
    os.makedirs(local, exist_ok=True)
    print("  [下载] %s -> %s" % (repo, local))
    t0 = time.time()
    for fname in ("config.json", "diffusion_pytorch_model.fp16.safetensors"):
        hf_hub_download(repo, fname, local_dir=local)
    print("  [完成] %.0fs" % (time.time() - t0))


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print("可选下载项：")
        for name, note in [
            ("anime2sketch", "神经线稿"),
            ("sd15", "SD1.5 上色基座"),
            ("cn_anime", "ControlNet 动漫线稿"),
            ("cn_standard", "ControlNet 标准线稿"),
            ("cn_canny", "ControlNet Canny"),
            ("cn_scribble", "ControlNet 草图"),
            ("cn_depth", "ControlNet 深度"),
            ("ddcolor", "照片上色"),
            ("animegan", "动漫风格化"),
            ("animatediff", "视频一致性"),
            ("midas", "深度估计"),
        ]:
            print("  %-14s %s" % (name, note))
        return
    only = None
    skip = []
    if "--only" in args:
        only = args[args.index("--only") + 1].split(",")
    if "--skip" in args:
        skip = args[args.index("--skip") + 1].split(",")
    download_all(only=only, skip=skip)


if __name__ == "__main__":
    main()
