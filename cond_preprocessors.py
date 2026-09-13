# -*- coding: utf-8 -*-
"""
cond_preprocessors.py — ControlNet 条件图预处理器注册表
========================================================
将 lineart_painter._make_cond() 中的硬编码 if-else 重构为插件化注册表。
新增预处理器只需 @register_cond("name") 装饰一个函数即可。

预处理器签名：fn(rgb, gray01, line_binary) -> uint8 灰度图 (HxW, 0..255)
  - rgb:       原图 RGB uint8 (HxWx3)
  - gray01:    灰度图 float32 (HxW, 0..1)
  - line_binary: 二值线稿 uint8 (HxW, 255=线条, 0=背景)
"""

import cv2
import numpy as np

# ---------------------------------------------------------------- 注册表
_COND_REGISTRY = {}


def register_cond(name):
    """装饰器：将函数注册为指定名称的条件图预处理器。"""
    def decorator(fn):
        _COND_REGISTRY[name] = fn
        return fn
    return decorator


def get_cond(name, rgb, gray01, line_binary):
    """按名称获取条件图。name 不区分大小写。"""
    key = name.lower()
    if key not in _COND_REGISTRY:
        raise ValueError(
            "未知条件图预处理器: %s。可用: %s" % (name, list_conds()))
    return _COND_REGISTRY[key](rgb, gray01, line_binary)


def list_conds():
    """返回所有已注册的预处理器名称（排序）。"""
    return sorted(_COND_REGISTRY.keys())


# ---------------------------------------------------------------- 内置预处理器

@register_cond("anime")
def _cond_anime(rgb, gray01, line_binary):
    """动漫线稿 ControlNet：白底黑线，直接使用线稿二值图。"""
    return np.where(line_binary > 0, 0, 255).astype(np.uint8)


@register_cond("standard")
def _cond_standard(rgb, gray01, line_binary):
    """通用线稿 ControlNet：白底黑线。"""
    return np.where(line_binary > 0, 0, 255).astype(np.uint8)


@register_cond("scribble")
def _cond_scribble(rgb, gray01, line_binary):
    """草图 ControlNet：白底黑线（草图风格，线条略粗）。"""
    # 草图变体：线条加粗 1 次，模拟手绘草图质感
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(line_binary, kernel, iterations=1)
    return np.where(thick > 0, 0, 255).astype(np.uint8)


@register_cond("canny")
def _cond_canny(rgb, gray01, line_binary):
    """Canny 边缘 ControlNet：白底黑边缘。"""
    edges = cv2.Canny((gray01 * 255).astype(np.uint8), 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return np.where(edges > 0, 0, 255).astype(np.uint8)


@register_cond("depth")
def _cond_depth(rgb, gray01, line_binary):
    """深度 ControlNet：近白远黑灰度图（MiDaS）。"""
    from neural import depth_map
    return depth_map(rgb)


@register_cond("manga_line")
def _cond_manga_line(rgb, gray01, line_binary):
    """漫画线稿 (lineart_anime / manga-line)：基于 Anime2Sketch 的高质量线稿。

    与默认 anime 预处理器的区别：
    - 使用 Anime2Sketch 神经线稿（improved 变体）作为基底
    - 自适应阈值 + 细化处理，线条更干净、更接近漫画原稿质感
    - 不依赖上游 line_binary 参数（自行提取线稿）
    """
    from neural import a2s_sketch
    gray = a2s_sketch(rgb, variant="improved")
    # 自适应阈值：对局部对比度敏感，保留淡线
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 10)
    # 去噪：去除 <15px 连通块
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (binary == 0).astype(np.uint8) * 255, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 15:
            binary[lab == i] = 255
    return binary.astype(np.uint8)
