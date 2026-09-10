# -*- coding: utf-8 -*-
"""
lineart_painter.py — 图片 -> 线稿 -> 分步填色 -> 还原 自动化流水线
==============================================================
复刻视频中「AI 临摹重绘」的分步流程，全程本地自动处理，无需训练模型：

    01 线稿         提取干净的黑白线稿（XDoG）
    02 固有色       把原图颜色聚成色块，按线稿区域填平色
    03 体积明暗     用低频亮度图给色块叠加明暗
    04 反射光       暗部叠加冷色环境反光
    05 色线         线稿线条染上局部颜色
    06 高光与细节   高光提亮 + 锐化细节
    FINAL           合并输出还原图

依赖：numpy, opencv-python（pip install numpy opencv-python）
用法：
    python lineart_painter.py <图片路径> [--out 输出目录] [--lines 0.02] [--k 10]
    python lineart_painter.py <文件夹路径>   # 批量处理文件夹内所有图片
输出：输出目录下生成 01_lineart.png ... 06_highlight.png、final.png、
      lineart.svg（线稿矢量）和 replay.html（可播放的分步还原演示页）。
"""

import argparse
import base64
import os
import sys
import traceback

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv-python，请先执行: pip install opencv-python")


# ---------------------------------------------------------------- 工具函数
def imread_rgb(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("无法读取图片: " + path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, rgb):
    ok = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1]
    with open(path, "wb") as f:
        f.write(ok.tobytes())


def gaussian(img, sigma):
    if sigma <= 0:
        return img
    k = max(1, int(round(sigma * 3)) * 2 + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


# ---------------------------------------------------------------- 01 线稿 (XDoG)
def clean_binary(binary):
    """去噪（去掉 <20px 连通块）+ 线条轻微加粗，让线稿干净且区域闭合。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            binary[lab == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def extract_lineart(gray01, sigma=2.4, k_sigma=1.6, tau=0.985, eps=-0.10, phi=12.0):
    """XDoG 边缘提取 -> 干净的黑线白底线稿。gray01: 0..1 灰度图。

    返回二值图：255=线条，0=背景。eps 越接近 0，线条越多；越负越少。
    """
    g1 = gaussian(gray01, sigma)
    g2 = gaussian(gray01, sigma * k_sigma)
    dog = g1 - tau * g2                        # difference of gaussians
    mx = float(np.abs(dog).max())
    if mx > 0:
        dog = dog / mx                         # 按最大幅值归一化
    s = 1.0 + np.tanh(phi * (dog - eps))       # XDoG 映射
    s = (s - s.min()) / max(s.max() - s.min(), 1e-9)
    binary = (s < 0.5).astype(np.uint8) * 255  # 深侧边缘 -> 线条
    return clean_binary(binary)                # 255=线条, 0=背景


def line_mask01(binary):
    """返回 0..1 的线条掩码（1=线）。"""
    return (binary > 0).astype(np.float32)


# ---------------------------------------------------------------- 02 固有色
def flat_colors(rgb, k=10, downscale=512, return_centers=False):
    """k-means 颜色量化 -> 色块平涂层（固有色）。"""
    h, w = rgb.shape[:2]
    scale = min(1.0, downscale / float(max(h, w)))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3,
                                    cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    quant = centers[labels.reshape(small.shape[:2])]
    if scale < 1.0:
        quant = cv2.resize(quant, (w, h), interpolation=cv2.INTER_NEAREST)
    # 中值滤波去除量化噪点，让色块更“平”
    quant = cv2.medianBlur(quant, 5)
    if return_centers:
        return quant, centers
    return quant


def quantize_with_centers(rgb, centers, downscale=512):
    """用固定聚类中心对图像做固有色（视频各帧颜色稳定，避免闪烁）。"""
    h, w = rgb.shape[:2]
    scale = min(1.0, downscale / float(max(h, w)))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    pixels = small.reshape(-1, 3).astype(np.float32)
    d = np.sum((pixels[:, None, :] - centers.astype(np.float32)[None, :, :]) ** 2, axis=2)
    labels = d.argmin(axis=1)
    quant = centers[labels.reshape(small.shape[:2])].astype(np.uint8)
    if scale < 1.0:
        quant = cv2.resize(quant, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.medianBlur(quant, 5)


# ---------------------------------------------------------------- 明暗/反射光/高光
def luminance_map(rgb, sigma_ratio=0.05):
    """低频亮度图，0..1。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sigma = max(1.0, min(gray.shape) * sigma_ratio)
    return gaussian(gray, sigma)


def apply_shading(flat, lum):
    m = np.clip(0.62 + 0.75 * lum, 0.30, 1.28)      # 暗部压暗，亮部提亮
    return np.clip(flat.astype(np.float32) * m[..., None], 0, 255).astype(np.uint8)


def reflected_light(shaded, lum):
    """暗部叠加冷色环境反光。"""
    dark = np.clip(1.0 - lum * 1.6, 0.0, 1.0)       # 越暗权重越大
    out = shaded.astype(np.float32).copy()
    out[..., 2] = np.clip(out[..., 2] + dark * 14.0, 0, 255)   # 蓝通道
    out[..., 0] = np.clip(out[..., 0] - dark * 6.0, 0, 255)    # 红通道微降
    return out.astype(np.uint8)


def color_lines(line_binary, rgb, alpha=0.55):
    """线稿染上局部颜色：线 = 原色 * alpha + 黑 * (1-alpha)。"""
    h, w = rgb.shape[:2]
    sigma = max(2.0, min(h, w) * 0.015)
    local = gaussian(rgb.astype(np.float32), sigma)
    ink = line_mask01(line_binary)[..., None]
    colored = (local * alpha) * ink + 0.0 * (1 - ink)
    colored = np.clip(colored, 0, 255).astype(np.uint8)
    # 白底 + 彩色线条
    canvas = np.full((h, w, 3), 255, np.uint8)
    canvas[line_binary > 0] = colored[line_binary > 0]
    return canvas


def add_highlights(shaded, lum, thr=0.80, boost=1.22):
    """高光提亮。"""
    hi = (lum > thr).astype(np.float32)[..., None]
    return np.clip(shaded.astype(np.float32) * (1.0 + (boost - 1.0) * hi), 0, 255).astype(np.uint8)


def unsharp(rgb, sigma=1.2, amount=0.55):
    blur = gaussian(rgb.astype(np.float32), sigma)
    sharp = rgb.astype(np.float32) + amount * (rgb.astype(np.float32) - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- SVG 线稿导出
def export_lineart_svg(line_binary, out_svg):
    """把线稿二值图转成 SVG 路径（轮廓追踪），模拟 VECTOR STUDIO 的矢量线稿。"""
    h, w = line_binary.shape[:2]
    contours, _ = cv2.findContours(line_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(float)
        d = "M{:.1f},{:.1f}".format(pts[0, 0], pts[0, 1])
        for i in range(1, len(pts), 2):   # 隔点采样，减小文件体积
            d += "L{:.1f},{:.1f}".format(pts[i, 0], pts[i, 1])
        d += "Z"
        paths.append(d)
    body = "\n".join(
        '  <path d="%s" fill="none" stroke="#1A1B1C" stroke-width="1.4"/>' % d
        for d in paths
    )
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d">\n<rect width="100%%" height="100%%" fill="#FFFFFF"/>\n%s\n</svg>'
           % (w, h, w, h, body))
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    return len(paths)


# ---------------------------------------------------------------- 主流水线
def process_image(src, out_dir, line_eps=0.02, k=10, neural_lineart=False,
                  neural_color=False, prompt=None, seed=None,
                  a2s_variant="improved", cn_variant="anime", strength=None,
                  input_lineart=False, photo_color=False, anime_gan=False):
    os.makedirs(out_dir, exist_ok=True)
    rgb = imread_rgb(src)
    h, w = rgb.shape[:2]
    gray01 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 01 线稿（神经版：Anime2Sketch；经典版：XDoG；或直接用输入的线稿）
    if input_lineart:
        binary = (gray01 < 0.82).astype(np.uint8) * 255
        line_binary = clean_binary(binary)
    elif neural_lineart:
        from neural import a2s_sketch_binary
        line_binary, gray_sketch = a2s_sketch_binary(rgb, variant=a2s_variant)
        imwrite_rgb(os.path.join(out_dir, "01_lineart_neural_raw.png"),
                    np.stack([gray_sketch] * 3, axis=-1))
    else:
        line_binary = extract_lineart(gray01, eps=line_eps)
    lineart = np.full((h, w, 3), 255, np.uint8)
    lineart[line_binary > 0] = (26, 27, 28)

    # 02 固有色
    flat = flat_colors(rgb, k=k)
    flat_bg = flat.copy()

    # 03 体积明暗
    lum = luminance_map(rgb)
    shaded = apply_shading(flat, lum)

    # 04 反射光
    reflected = reflected_light(shaded, lum)

    # 05 色线
    cl = color_lines(line_binary, rgb)

    # 06 高光与细节
    hl = add_highlights(reflected, lum)
    final = unsharp(hl)
    # 线稿压顶（黑色线稿版）
    final_lines = final.copy()
    final_lines[line_binary > 0] = (26, 27, 28)
    # 彩线版
    final_cl = final.copy()
    final_cl[line_binary > 0] = cl[line_binary > 0]

    stages = [
        ("01_lineart", lineart, "01 线稿"),
        ("02_flatcolor", flat_bg, "02 固有色"),
        ("03_shading", shaded, "03 体积明暗"),
        ("04_reflected", reflected, "04 反射光"),
        ("05_colorline", cl, "05 色线"),
        ("06_highlight", hl, "06 高光与细节"),
        ("final", final_lines, "FINAL 还原图(黑线)"),
        ("final_colorline", final_cl, "FINAL 还原图(彩线)"),
    ]
    # 神经上色还原：ControlNet(lineart) + SD1.5 生成彩色图
    if neural_color:
        from neural import DEFAULT_PROMPT, NeuralColorizer
        colorizer = NeuralColorizer(cn_variant=cn_variant)
        line_gray = np.where(line_binary > 0, 0, 255).astype(np.uint8)  # 白底黑线
        pr = prompt or DEFAULT_PROMPT
        neural_final = colorizer.colorize(
            line_gray, pr, seed=seed, strength=strength,
            init_rgb=rgb if strength else None)
        stages.append(("final_neural", neural_final, "FINAL 神经上色(ControlNet)"))

    # 扩展模型：黑白照片上色（DDColor）
    if photo_color:
        from neural import DDColorizer
        colorized = DDColorizer().colorize(rgb)
        stages.append(("colorized_photo", colorized, "照片上色(DDColor)"))

    # 扩展模型：动漫风格化（AnimeGANv2 ONNX）
    if anime_gan:
        from neural import AnimeGAN
        anime_style = AnimeGAN().stylize(rgb)
        stages.append(("anime_style", anime_style, "动漫风格(AnimeGANv2)"))
    paths = {}
    for name, img, _ in stages:
        p = os.path.join(out_dir, name + ".png")
        imwrite_rgb(p, img)
        paths[name] = p

    svg_path = os.path.join(out_dir, "lineart.svg")
    n_paths = export_lineart_svg(line_binary, svg_path)

    # replay.html：分步播放演示页
    replay_path = os.path.join(out_dir, "replay.html")
    if neural_lineart or neural_color:
        engine_note = ("本页由 lineart_painter.py 自动生成：线稿来自 Anime2Sketch 神经网络"
                       "（%s），上色还原由 ControlNet + Stable Diffusion 1.5 在本地 GPU 完成，"
                       "SVG 线稿见同目录 lineart.svg。" % a2s_variant)
    else:
        engine_note = None
    build_replay(replay_path, stages, rgb, engine_note=engine_note)

    print("完成: %s -> %s  (SVG路径数=%d, 尺寸=%dx%d)" % (src, out_dir, n_paths, w, h))
    return paths, replay_path


# ---------------------------------------------------------------- replay.html
def build_replay(replay_path, stages, original, engine_note=None):
    """把各阶段图以 base64 内嵌，生成可滑动播放的分步演示页（仿视频界面）。"""
    if engine_note is None:
        engine_note = ("本页由 lineart_painter.py 自动生成：所有中间层均为本地算法实时产出"
                       "（XDoG 线稿 + K-Means 固有色 + 亮度明暗），SVG 线稿见同目录 lineart.svg。")

    def b64(rgb):
        ok = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1]
        return base64.b64encode(ok.tobytes()).decode()

    imgs = [(label, b64(img)) for label, img, _ in stages]
    orig_b64 = b64(original)
    steps = ',\n'.join(
        '{label:%s,data:"data:image/png;base64,%s"}' % (label, data)
        for label, data in imgs
    )
    names = [label for label, img, _ in stages]
    html = """<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>AI 临摹重绘 · 分步还原演示</title>
<style>
  body{margin:0;background:#0E1116;color:#E8EAED;font-family:'Segoe UI','PingFang SC',Arial,sans-serif;min-height:100vh}
  .wrap{max-width:1100px;margin:0 auto;padding:20px 16px}
  h1{font-size:18px;margin:0 0 4px;color:#A3D5E8}
  .sub{font-size:12px;color:#7A828E;margin-bottom:16px}
  .stage-tag{font-size:12px;color:#7A828E;margin-bottom:6px}
  .stage-name{font-size:22px;font-weight:600;margin-bottom:10px}
  .canvas-box{position:relative;background:#11151C;border:1px solid #232A34;border-radius:12px;overflow:hidden}
  canvas{display:block;width:100%%;height:auto}
  .bar{display:flex;align-items:center;gap:10px;margin:14px 0 10px}
  .btn{background:#232A34;color:#E8EAED;border:1px solid #343D4A;border-radius:8px;padding:8px 16px;font-size:13px;cursor:pointer}
  .btn:hover{background:#2E3742}
  input[type=range]{flex:1;accent-color:#A3D5E8}
  .steps{display:flex;flex-wrap:wrap;gap:8px}
  .step{font-size:12px;color:#7A828E;padding:6px 10px;border:1px solid #232A34;border-radius:20px;cursor:pointer}
  .step.on{color:#0E1116;background:#A3D5E8;border-color:#A3D5E8}
  .note{font-size:12px;color:#7A828E;margin-top:14px;line-height:1.7}
</style></head>
<body><div class="wrap">
  <h1>AI 临摹重绘 · 自动化分步还原</h1>
  <div class="sub">线稿 → 固有色 → 体积明暗 → 反射光 → 色线 → 高光与细节（拖动进度条或点击步骤播放）</div>
  <div class="stage-tag">STEP <span id="cur">0</span> / <span id="tot">0</span></div>
  <div class="stage-name" id="sname">—</div>
  <div class="canvas-box"><canvas id="cv" width="0" height="0"></canvas></div>
  <div class="bar">
    <button class="btn" id="play">▶ 播放</button>
    <input type="range" id="slider" min="0" max="0" value="0">
    <button class="btn" id="orig">原图</button>
  </div>
  <div class="steps" id="steps"></div>
  <div class="note">%s</div>
</div>
<script>
(function(){
  var STEPS = [%s];
  var ORIG = "data:image/png;base64,%s";
  var cv = document.getElementById('cv'), ctx = cv.getContext('2d');
  var img = new Image(); var cur = STEPS.length - 1, playing = false, timer = null;
  var sname = document.getElementById('sname'), curEl = document.getElementById('cur');
  var totEl = document.getElementById('tot'), slider = document.getElementById('slider');
  var stepsEl = document.getElementById('steps');
  var names = %s;
  totEl.textContent = STEPS.length; slider.max = STEPS.length - 1;
  STEPS.forEach(function(s, i){
    var b = document.createElement('div'); b.className = 'step'; b.textContent = s.label;
    b.onclick = function(){ show(i); };
    stepsEl.appendChild(b);
  });
  function show(i){
    cur = i; slider.value = i; curEl.textContent = (i + 1);
    sname.textContent = STEPS[i].label;
    img.onload = function(){ cv.width = img.width; cv.height = img.height; ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(img,0,0); };
    img.src = STEPS[i].data;
    var bs = stepsEl.children; for (var j=0;j<bs.length;j++){ bs[j].className = 'step' + (j===i?' on':''); }
  }
  function play(){
    if (playing){ playing=false; clearInterval(timer); document.getElementById('play').textContent='▶ 播放'; return; }
    playing=true; document.getElementById('play').textContent='⏸ 暂停';
    timer = setInterval(function(){ if (cur >= STEPS.length-1){ show(0); } else { show(cur+1); } }, 900);
  }
  document.getElementById('play').onclick = play;
  slider.oninput = function(){ show(parseInt(this.value,10)); };
  document.getElementById('orig').onclick = function(){
    cur = -1; slider.value = 0; curEl.textContent = '原图'; sname.textContent = '原图参考';
    img.onload = function(){ cv.width = img.width; cv.height = img.height; ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(img,0,0); };
    img.src = ORIG;
  };
  show(STEPS.length - 1);
})();
</script>
</body></html>
""" % (steps, orig_b64, names, engine_note)
    with open(replay_path, "w", encoding="utf-8") as f:
        f.write(html)
    return replay_path


# ---------------------------------------------------------------- 视频模式
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def process_video(src, out_dir, line_eps=-0.10, k=10, max_side=1280,
                  neural_lineart=False, a2s_variant="improved",
                  neural_color=False, prompt=None, seed=None,
                  cn_variant="anime", strength=None, input_lineart=False):
    """视频 -> 逐帧 线稿/填色/还原 -> 合成输出视频（video_final.mp4 与 video_lineart.mp4）。

    首帧做 k-means 取聚类中心，后续帧复用中心，保证颜色不闪烁。
    --neural-color 时：每帧用 ControlNet+SD 神经上色（较慢，帧间用固定种子保稳）。
    """
    colorizer = None
    pr = prompt
    vseed = seed if seed is not None else 42
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError("无法打开视频: " + src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    scale = min(1.0, max_side / float(max(W, H)))
    ow, oh = int(round(W * scale)), int(round(H * scale))
    if ow % 2:
        ow += 1
    if oh % 2:
        oh += 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_final = cv2.VideoWriter(os.path.join(out_dir, "video_final.mp4"), fourcc, fps, (ow, oh))
    vw_line = cv2.VideoWriter(os.path.join(out_dir, "video_lineart.mp4"), fourcc, fps, (ow, oh))
    centers = None
    previews = {}
    idx = 0
    step = max(1, (total or 300) // 5)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        gray01 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        if input_lineart:
            binary = (gray01 < 0.82).astype(np.uint8) * 255
            line_binary = clean_binary(binary)
        elif neural_lineart:
            from neural import a2s_sketch_binary
            line_binary, _ = a2s_sketch_binary(rgb, variant=a2s_variant)
        else:
            line_binary = extract_lineart(gray01, eps=line_eps)
        if neural_color:
            # 神经上色（每帧 ControlNet+SD，帧间固定种子保持稳定）
            if colorizer is None:
                from neural import DEFAULT_PROMPT, NeuralColorizer
                colorizer = NeuralColorizer(cn_variant=cn_variant)
                pr = prompt or DEFAULT_PROMPT
            line_gray = np.where(line_binary > 0, 0, 255).astype(np.uint8)
            final_lines = colorizer.colorize(
                line_gray, pr, seed=vseed, strength=strength,
                init_rgb=rgb if strength else None)
        else:
            if centers is None:
                flat, centers = flat_colors(rgb, k=k, return_centers=True)
            else:
                flat = quantize_with_centers(rgb, centers)
            lum = luminance_map(rgb)
            shaded = apply_shading(flat, lum)
            reflected = reflected_light(shaded, lum)
            hl = add_highlights(reflected, lum)
            final = unsharp(hl)
            final_lines = final.copy()
            final_lines[line_binary > 0] = (26, 27, 28)
        lineart = np.full((oh, ow, 3), 255, np.uint8)
        lineart[line_binary > 0] = (26, 27, 28)
        vw_final.write(cv2.cvtColor(final_lines, cv2.COLOR_RGB2BGR))
        vw_line.write(cv2.cvtColor(lineart, cv2.COLOR_RGB2BGR))
        if idx % step == 0 or idx == 0:
            previews[idx] = final_lines
        idx += 1
        if idx % 30 == 0:
            print("  帧 %d/%d" % (idx, total or idx))
    cap.release()
    vw_final.release()
    vw_line.release()
    for fi, img in previews.items():
        imwrite_rgb(os.path.join(out_dir, "preview_f%05d.png" % fi), img)
    # ffmpeg 转码：mp4v -> H.264（兼容性更好、体积更小）
    import shutil
    import subprocess
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None
    if ffmpeg_exe:
        for name in ("video_final.mp4", "video_lineart.mp4"):
            srcv = os.path.join(out_dir, name)
            dstv = os.path.join(out_dir, name.replace(".mp4", "_h264.mp4"))
            subprocess.run([ffmpeg_exe, "-y", "-i", srcv, "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", dstv],
                           capture_output=True)
            if os.path.exists(dstv):
                print("  ffmpeg H.264 转码完成: " + dstv)
    else:
        print("  (ffmpeg 未安装，视频保持 mp4v 编码)")
    print("完成视频: %s -> %s (共%d帧, %.1ffps, 输出%dx%d)" % (src, out_dir, idx, fps, ow, oh))
    return out_dir


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="图片 -> 线稿 -> 分步填色 -> 还原 自动化流水线")
    ap.add_argument("input", help="输入图片或文件夹")
    ap.add_argument("--out", default=None, help="输出目录（默认: 输入同目录/out）")
    ap.add_argument("--lines", type=float, default=-0.10,
                    help="线稿阈值 eps（默认-0.10；越接近 0 线条越多，越负越少）")
    ap.add_argument("--k", type=int, default=10, help="固有色聚类色块数量(4-20)")
    ap.add_argument("--maxside", type=int, default=1280,
                    help="视频模式最长边像素（默认1280，越小越快）")
    ap.add_argument("--neural-lineart", action="store_true",
                    help="用 Anime2Sketch 神经线稿替换 XDoG（效果更干净，需 torch）")
    ap.add_argument("--neural-color", action="store_true",
                    help="用 ControlNet+SD1.5 神经上色还原（仅图片模式，首次需下载约4GB权重）")
    ap.add_argument("--prompt", default=None,
                    help="神经上色提示词（默认动漫插画风格）")
    ap.add_argument("--seed", type=int, default=None,
                    help="神经上色随机种子（固定可复现）")
    ap.add_argument("--a2s", default="improved", choices=["default", "improved"],
                    help="Anime2Sketch 权重变体（improved 质量更高）")
    ap.add_argument("--cn", default="anime", choices=["anime", "standard"],
                    help="ControlNet 变体：anime=动漫线稿专用(默认)，standard=通用线稿")
    ap.add_argument("--neural-strength", type=float, default=None, metavar="0-1",
                    help="img2img 混合强度：用原图风格还原（越接近0越贴近原图，0.35~0.6 推荐）")
    ap.add_argument("--input-lineart", action="store_true",
                    help="输入已是线稿（跳过线稿提取，直接填色/上色）")
    ap.add_argument("--photo-color", action="store_true",
                    help="黑白/灰度照片自动上色（DDColor，仅图片）")
    ap.add_argument("--anime-gan", action="store_true",
                    help="照片转动漫风格（AnimeGANv2 ONNX，仅图片）")
    args = ap.parse_args()

    inp = args.input
    if os.path.isdir(inp):
        files = [os.path.join(inp, f) for f in sorted(os.listdir(inp))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp",
                                        ".mp4", ".avi", ".mov", ".mkv"))]
        if not files:
            sys.exit("文件夹内没有图片/视频")
    else:
        files = [inp]

    for f in files:
        try:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTS:
                out = args.out or os.path.join(os.path.dirname(os.path.abspath(f)) or ".", "out_video")
                process_video(f, out, line_eps=args.lines, k=args.k,
                              max_side=args.maxside,
                              neural_lineart=args.neural_lineart,
                              a2s_variant=args.a2s,
                              neural_color=args.neural_color,
                              prompt=args.prompt, seed=args.seed,
                              cn_variant=args.cn,
                              strength=args.neural_strength,
                              input_lineart=args.input_lineart)
            else:
                out = args.out or os.path.join(os.path.dirname(os.path.abspath(f)) or ".", "out")
                process_image(f, out, line_eps=args.lines, k=args.k,
                              neural_lineart=args.neural_lineart,
                              neural_color=args.neural_color,
                              prompt=args.prompt, seed=args.seed,
                              a2s_variant=args.a2s,
                              cn_variant=args.cn,
                              strength=args.neural_strength,
                              input_lineart=args.input_lineart,
                              photo_color=args.photo_color,
                              anime_gan=args.anime_gan)
        except Exception:
            print("处理失败: %s" % f)
            traceback.print_exc()


if __name__ == "__main__":
    main()
