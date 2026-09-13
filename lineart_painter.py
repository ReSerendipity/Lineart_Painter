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


# ---------------------------------------------------------------- 文字区域保护
def detect_text_mask(line_binary, min_area=4, max_area=300,
                     cluster_distance=40, dilate_size=5):
    """从二值线稿中检测文字/对话区域，返回保护遮罩（255=需保护的文字区）。

    原理：文字由大量小面积连通块聚集而成。筛选面积在 [min_area, max_area]
    范围内的连通块，再通过膨胀将邻近的文字笔画合并为文字块。
    """
    # 线条像素为 255，找连通块
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_binary, 8)
    text_components = np.zeros_like(line_binary)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            text_components[labels == i] = 255
    # 膨胀合并邻近文字笔画为文字块
    if dilate_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        text_components = cv2.dilate(text_components, kernel, iterations=2)
    # 再次连通块分析，过滤掉孤立的小点（只保留合并后的大块）
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        text_components, 8)
    mask = np.zeros_like(line_binary)
    for i in range(1, n2):
        area = stats2[i, cv2.CC_STAT_AREA]
        # 合并后的文字块通常 > 200px（至少几个字）
        if area >= 200:
            mask[labels2 == i] = 255
    return mask


def apply_text_protection(final_rgb, original_rgb, text_mask):
    """将文字区域从原图复制回最终结果，避免神经上色污染文字。

    text_mask: 255=文字区域, 0=非文字区域
    """
    if text_mask is None or np.count_nonzero(text_mask) == 0:
        return final_rgb
    mask3 = np.stack([text_mask > 0] * 3, axis=-1)
    result = final_rgb.copy()
    result[mask3] = original_rgb[mask3]
    return result


# ---------------------------------------------------------------- 参考图调色板迁移
def extract_palette(rgb, k=8):
    """从图片中提取主色调色板，返回 k 个 RGB 颜色（按亮度排序）。"""
    h, w = rgb.shape[:2]
    scale = 512.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # 按亮度排序
    lum = centers[:, 0] * 0.299 + centers[:, 1] * 0.587 + centers[:, 2] * 0.114
    order = lum.argsort()
    return centers[order].astype(np.uint8)


def transfer_palette(target_rgb, ref_rgb, k=8):
    """将参考图的调色板迁移到目标图。

    原理：分别提取两图的 k 色调色板（按亮度排序），将目标图中每个颜色
    映射到参考图中相同亮度等级的颜色，实现色彩风格迁移。
    """
    ref_palette = extract_palette(ref_rgb, k)
    tgt_palette = extract_palette(target_rgb, k)
    # 构建映射：目标调色板 -> 参考调色板（按亮度等级一一对应）
    h, w = target_rgb.shape[:2]
    pixels = target_rgb.reshape(-1, 3).astype(np.float32)
    # 对每个像素，找到最近的目标调色板颜色，替换为对应参考颜色
    d = np.sum((pixels[:, None, :] - tgt_palette.astype(np.float32)[None, :, :]) ** 2, axis=2)
    idx = d.argmin(axis=1)
    mapped = ref_palette[idx].reshape(h, w, 3).astype(np.uint8)
    # 轻度模糊融合，避免色块过于生硬
    return cv2.bilateralFilter(mapped, 5, 40, 40)


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
def _parse_cn_variants(cn_variant):
    """将 cn_variant 解析为变体列表（支持逗号分隔字符串）。"""
    if isinstance(cn_variant, str):
        return [v.strip() for v in cn_variant.split(",") if v.strip()]
    return list(cn_variant)


def _make_cond(cn_variant, rgb, gray01, line_binary):
    """按单个 ControlNet 变体生成条件图（uint8 单通道）。"""
    from cond_preprocessors import get_cond
    return get_cond(cn_variant, rgb, gray01, line_binary)


def _make_conds(cn_variant, rgb, gray01, line_binary):
    """按 ControlNet 变体（可多个）生成条件图列表。"""
    variants = _parse_cn_variants(cn_variant)
    return [_make_cond(v, rgb, gray01, line_binary) for v in variants]


def process_image(src, out_dir, line_eps=0.02, k=10, neural_lineart=False,
                  neural_color=False, prompt=None, seed=None,
                  a2s_variant="improved", cn_variant="anime", strength=None,
                  input_lineart=False, photo_color=False, anime_gan=False,
                  cn_scale=None, protect_text=False, reference_color=None,
                  color_hint=None, gen_size=768):
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
        pr = prompt or DEFAULT_PROMPT
        conds = _make_conds(cn_variant, rgb, gray01, line_binary)
        cond_arg = conds[0] if len(conds) == 1 else conds
        # 参考图驱动上色：迁移参考图调色板到固有色，作为 img2img 初始图
        init_img = rgb if strength else None
        eff_strength = strength
        if reference_color and os.path.exists(reference_color):
            ref_bgr = cv2.imread(reference_color, cv2.IMREAD_COLOR)
            if ref_bgr is not None:
                ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                ref_rgb = cv2.resize(ref_rgb, (w, h),
                                     interpolation=cv2.INTER_AREA)
                color_guide = transfer_palette(flat, ref_rgb, k=8)
                stages.append(("ref_color_guide", color_guide,
                               "参考图调色板迁移"))
                init_img = color_guide
                eff_strength = strength if strength else 0.5
        # 用户点选颜色提示：直接作为 img2img 初始图（优先级最高）
        if color_hint and os.path.exists(color_hint):
            hint_bgr = cv2.imread(color_hint, cv2.IMREAD_COLOR)
            if hint_bgr is not None:
                hint_rgb = cv2.cvtColor(hint_bgr, cv2.COLOR_BGR2RGB)
                hint_rgb = cv2.resize(hint_rgb, (w, h),
                                      interpolation=cv2.INTER_LINEAR)
                stages.append(("color_hint", hint_rgb, "用户颜色提示"))
                init_img = hint_rgb
                eff_strength = strength if strength else 0.45
        neural_final = colorizer.colorize(
            cond_arg, pr, seed=seed, strength=eff_strength,
            scale=cn_scale if cn_scale else 0.85,
            size=gen_size,
            init_rgb=init_img)
        # 文字区域保护：检测文字并从原图复制回来
        if protect_text:
            text_mask = detect_text_mask(line_binary)
            if np.count_nonzero(text_mask) > 0:
                neural_final = apply_text_protection(
                    neural_final, rgb, text_mask)
                stages.append(("text_mask", text_mask, "文字保护遮罩"))
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
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _ffmpeg_h264(out_dir, names):
    """把 out_dir 下指定 mp4 用 ffmpeg 转成 H.264（mp4v 兼容性差）。"""
    import shutil
    import subprocess
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None
    if not ffmpeg_exe:
        print("  (ffmpeg 未安装，视频保持 mp4v 编码)")
        return []
    made = []
    for name in names:
        srcv = os.path.join(out_dir, name)
        dstv = os.path.join(out_dir, name.replace(".mp4", "_h264.mp4"))
        if not os.path.exists(srcv):
            continue
        subprocess.run([ffmpeg_exe, "-y", "-i", srcv, "-c:v", "libx264",
                        "-crf", "18", "-pix_fmt", "yuv420p", dstv],
                       capture_output=True)
        if os.path.exists(dstv):
            made.append(dstv)
            print("  ffmpeg H.264 转码完成: " + dstv)
    return made


def process_video(src, out_dir, line_eps=-0.10, k=10, max_side=1280,
                  neural_lineart=False, a2s_variant="improved",
                  neural_color=False, prompt=None, seed=None,
                  cn_variant="anime", strength=None, input_lineart=False,
                  photo_color=False, anime_gan=False, animate_diff=False,
                  cn_scale=None, gen_size=768):
    """视频 -> 逐帧 线稿/填色/还原 -> 合成输出视频（video_final.mp4 与 video_lineart.mp4）。

    首帧做 k-means 取聚类中心，后续帧复用中心，保证颜色不闪烁。
    --neural-color 时：每帧用 ControlNet+SD 神经上色（较慢，帧间用固定种子保稳）。
    --animate-diff 时：改为 AnimateDiff 帧间一致生成（16 帧一段），闪烁显著减少。
    --photo-color / --anime-gan 时：整段视频另输出 DDColor 上色 / 动漫风格版。
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
    vw_final = None
    if not animate_diff:  # AnimateDiff 模式：线稿收集完再生成，最后才写
        vw_final = cv2.VideoWriter(os.path.join(out_dir, "video_final.mp4"),
                                   fourcc, fps, (ow, oh))
    vw_line = cv2.VideoWriter(os.path.join(out_dir, "video_lineart.mp4"),
                              fourcc, fps, (ow, oh))
    centers = None
    previews = {}
    line_frames = [] if animate_diff else None
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
        if animate_diff:
            line_frames.append(np.where(line_binary > 0, 0, 255).astype(np.uint8))
        elif neural_color:
            # 神经上色（每帧 ControlNet+SD，帧间固定种子保持稳定）
            if colorizer is None:
                from neural import DEFAULT_PROMPT, NeuralColorizer
                colorizer = NeuralColorizer(cn_variant=cn_variant)
                pr = prompt or DEFAULT_PROMPT
            conds = _make_conds(cn_variant, rgb, gray01, line_binary)
            cond_arg = conds[0] if len(conds) == 1 else conds
            final_lines = colorizer.colorize(
                cond_arg, pr, seed=vseed, strength=strength,
                scale=cn_scale if cn_scale else 0.85,
                size=gen_size,
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
        if vw_final is not None:
            vw_final.write(cv2.cvtColor(final_lines, cv2.COLOR_RGB2BGR))
        vw_line.write(cv2.cvtColor(lineart, cv2.COLOR_RGB2BGR))
        if idx % step == 0 or idx == 0:
            previews[idx] = lineart if animate_diff else final_lines
        idx += 1
        if idx % 30 == 0:
            print("  帧 %d/%d" % (idx, total or idx))
    cap.release()
    if vw_final is not None:
        vw_final.release()
    vw_line.release()
    if animate_diff:
        # AnimateDiff 帧间一致性生成（16 帧一段；低分辨率输入用 384 提速省显存）
        from neural import DEFAULT_PROMPT, AnimateDiffColorizer
        adc = AnimateDiffColorizer(cn_variant=cn_variant)
        asize = 384 if max_side <= 640 else 512
        outs = adc.colorize_frames(line_frames, prompt or DEFAULT_PROMPT,
                                   seed=vseed, size=asize)
        vw_final = cv2.VideoWriter(os.path.join(out_dir, "video_final.mp4"),
                                   fourcc, fps, (ow, oh))
        for rgb in outs:
            vw_final.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        vw_final.release()
        for fi, img in list(previews.items()):
            if fi < len(outs):
                previews[fi] = outs[fi]
    for fi, img in previews.items():
        imwrite_rgb(os.path.join(out_dir, "preview_f%05d.png" % fi), img)
    # ffmpeg 转码：mp4v -> H.264（兼容性更好、体积更小）
    _ffmpeg_h264(out_dir, ["video_final.mp4", "video_lineart.mp4"])

    # 扩展模型：整段视频逐帧应用 照片上色 / 动漫风格化
    for flag, label, writer_name in (
            (photo_color, "照片上色(DDColor)", "video_photo_color.mp4"),
            (anime_gan, "动漫风格(AnimeGANv2)", "video_anime_style.mp4")):
        if not flag:
            continue
        from neural import DDColorizer, AnimeGAN
        fx = DDColorizer() if label.startswith("照片") else AnimeGAN()
        cap2 = cv2.VideoCapture(src)
        vw = cv2.VideoWriter(os.path.join(out_dir, writer_name), fourcc,
                             fps, (ow, oh))
        n = 0
        while True:
            ok, frame = cap2.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
            out_rgb = fx.colorize(rgb) if label.startswith("照片") \
                else fx.stylize(rgb)
            vw.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            n += 1
            if n % 15 == 0:
                print("  %s 帧 %d/%d" % (label, n, total or n))
        cap2.release()
        vw.release()
        _ffmpeg_h264(out_dir, [writer_name])
    print("完成视频: %s -> %s (共%d帧, %.1ffps, 输出%dx%d)" % (src, out_dir, idx, fps, ow, oh))
    return out_dir


# ---------------------------------------------------------------- 配置文件
CONFIG_VERSION = "1.0"
_CONFIG_KEYS = [
    "lines", "k", "maxside", "neural_lineart", "a2s", "neural_color",
    "prompt", "seed", "cn", "strength", "input_lineart",
    "photo_color", "anime_gan", "animate_diff", "gen_size",
]


def save_config(args, path):
    """将当前参数保存为 JSON 配置文件。"""
    import json
    from datetime import datetime
    cfg = {
        "version": CONFIG_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": {k: getattr(args, k, None) for k in _CONFIG_KEYS},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("配置已保存: %s" % path)


def load_config(path):
    """从 JSON 配置文件加载参数，返回 dict。缺失键保持 None（使用 CLI 默认值）。"""
    import json
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    params = cfg.get("params", {})
    print("已加载配置: %s (版本 %s)" % (path, cfg.get("version", "?")))
    return params


def apply_config_defaults(ap, config_params):
    """将配置文件中的参数设为 argparse 默认值（CLI 显式参数仍优先）。

    对 store_true 类型的参数，配置文件中的 true/false 会覆盖默认值。
    """
    for key, val in config_params.items():
        if val is None:
            continue
        # argparse 使用下划线，CLI 用连字符
        dest = key
        try:
            action = next(a for a in ap._actions if a.dest == dest)
            action.default = val
        except StopIteration:
            pass


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
    ap.add_argument("--cn", default="anime",
                    help="ControlNet 变体：anime=动漫线稿(默认), standard=通用线稿, "
                         "canny=边缘, scribble=草图, depth=深度, manga_line=漫画线稿。"
                         "支持逗号分隔多值堆叠，如 --cn anime,depth")
    ap.add_argument("--cn-scale", default=None,
                    help="多 ControlNet 时各变体的权重（逗号分隔，如 0.85,0.6），"
                         "默认全部 0.85")
    ap.add_argument("--neural-strength", type=float, default=None, metavar="0-1",
                    help="img2img 混合强度：用原图风格还原（越接近0越贴近原图，0.35~0.6 推荐）")
    ap.add_argument("--input-lineart", action="store_true",
                    help="输入已是线稿（跳过线稿提取，直接填色/上色）")
    ap.add_argument("--photo-color", action="store_true",
                    help="黑白/灰度照片自动上色（DDColor，图片与视频均支持）")
    ap.add_argument("--anime-gan", action="store_true",
                    help="照片转动漫风格（AnimeGANv2 ONNX，图片与视频均支持）")
    ap.add_argument("--animate-diff", action="store_true",
                    help="视频用 AnimateDiff 帧间一致性神经上色（替代逐帧上色，需额外权重）")
    ap.add_argument("--low-vram", action="store_true",
                    help="低显存模式：启用 sequential CPU offload，4GB 显存也能跑神经上色（速度较慢）")
    ap.add_argument("--protect-text", action="store_true",
                    help="神经上色时自动检测并保护文字/对话区域，避免上色污染")
    ap.add_argument("--reference-color", default=None, metavar="REF.png",
                    help="参考图驱动上色：迁移参考图的调色板到线稿（自动启用 img2img 混合）")
    ap.add_argument("--color-hint", default=None, metavar="HINT.png",
                    help="颜色提示图：用户点选的颜色标记图，作为 img2img 初始图（优先级最高）")
    ap.add_argument("--pipeline", default=None, metavar="PIPELINE.json",
                    help="使用节点化管线 JSON 定义处理流程（替代默认分步管线）。"
                         "预设: pipelines/default_classic.json, pipelines/neural_color.json, "
                         "pipelines/minimal_lineart_only.json")
    ap.add_argument("--gen-size", type=int, default=768,
                    help="神经上色生成分辨率（默认768，低显存模式自动降为512）")
    ap.add_argument("--debug", action="store_true",
                    help="出错时显示完整堆栈（默认只显示一句话原因）")
    ap.add_argument("--config", default=None, metavar="FILE.json",
                    help="从 JSON 配置文件加载参数（CLI 显式参数优先覆盖）")
    ap.add_argument("--save-config", default=None, metavar="FILE.json",
                    help="将本次参数保存为 JSON 配置文件后退出（不执行处理）")
    args = ap.parse_args()

    # 加载配置文件（在 parse_args 之后应用，使 CLI 参数优先）
    if args.config:
        if not os.path.exists(args.config):
            sys.exit("配置文件不存在: %s" % args.config)
        cfg_params = load_config(args.config)
        # 仅在用户未显式指定该参数时使用配置值
        for key, val in cfg_params.items():
            if val is None:
                continue
            dest = key
            try:
                action = next(a for a in ap._actions if a.dest == dest)
                # 检查用户是否在命令行显式设置了该参数
                if getattr(args, dest, None) == action.default:
                    setattr(args, dest, val)
            except StopIteration:
                pass

    # 保存配置后退出
    if args.save_config:
        save_config(args, args.save_config)
        return

    # 设置显存模式
    if args.low_vram:
        from vram_manager import vram
        vram.low_vram = True
        print("[vram] 低显存模式已启用（sequential CPU offload）")

    inp = args.input
    if os.path.isdir(inp):
        files = [os.path.join(inp, f) for f in sorted(os.listdir(inp))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp",
                                        ".mp4", ".avi", ".mov", ".mkv"))]
        if not files:
            sys.exit("文件夹内没有图片/视频")
    else:
        files = [inp]

    # 解析多 ControlNet 权重
    _cn_scale = None
    if args.cn_scale:
        _cn_scale = [float(s.strip()) for s in args.cn_scale.split(",") if s.strip()]

    for f in files:
        if not os.path.exists(f):
            print("跳过（文件不存在）: %s" % f)
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            print("跳过（不支持的文件类型 %s）: %s" % (ext or "无扩展名", f))
            continue
        try:
            # 节点化管线模式
            if args.pipeline:
                from pipeline_nodes import DAGPipeline
                pipe = DAGPipeline.from_json(args.pipeline)
                out = args.out or os.path.join(
                    os.path.dirname(os.path.abspath(f)) or ".", "out_pipeline")
                pipe.run(f, out,
                         line_eps=args.lines, k=args.k,
                         neural_lineart=args.neural_lineart,
                         a2s_variant=args.a2s,
                         neural_color=args.neural_color,
                         prompt=args.prompt, seed=args.seed,
                         cn_variant=args.cn,
                         strength=args.neural_strength,
                         cn_scale=_cn_scale,
                         protect_text=args.protect_text,
                         reference_color=args.reference_color,
                         color_hint=args.color_hint)
                print("管线完成: %s -> %s" % (os.path.basename(f), out))
                continue
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
                              input_lineart=args.input_lineart,
                              photo_color=args.photo_color,
                              anime_gan=args.anime_gan,
                              animate_diff=args.animate_diff,
                              cn_scale=_cn_scale,
                              gen_size=args.gen_size)
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
                              anime_gan=args.anime_gan,
                              cn_scale=_cn_scale,
                              protect_text=args.protect_text,
                              reference_color=args.reference_color,
                              color_hint=args.color_hint,
                              gen_size=args.gen_size)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            else:
                print("处理失败 %s: %s" % (os.path.basename(f), e))


if __name__ == "__main__":
    main()
