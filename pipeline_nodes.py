# -*- coding: utf-8 -*-
"""
pipeline_nodes.py — 节点化处理管线引擎
========================================
将 lineart_painter.py 中的硬编码分步处理流程重构为可配置的 DAG 节点管线。
每个节点封装一个处理步骤，通过 JSON 定义可自由组合、跳过、重排。

用法：
    from pipeline_nodes import DAGPipeline, DEFAULT_PIPELINE
    pipe = DAGPipeline.from_json("my_pipeline.json")
    pipe.run("input.png", "out_dir/", line_eps=-0.10, k=10)

节点 JSON 格式：
    {
      "nodes": [
        {"id": "lineart", "type": "ExtractLineart", "params": {"neural": false}},
        {"id": "flat", "type": "FlatColors", "params": {"k": 10}, "after": ["lineart"]},
        {"id": "final", "type": "SaveOutputs", "after": ["flat", "shading"]}
      ]
    }
"""

import os
import json
from collections import deque

import numpy as np
import cv2


# ---------------------------------------------------------------- 上下文
class NodeContext:
    """节点间共享的状态容器。"""

    def __init__(self):
        self.rgb = None           # 原图 RGB uint8
        self.gray01 = None        # 灰度 float32 0..1
        self.line_binary = None   # 二值线稿 uint8 (255=线,0=背景)
        self.flat = None          # 固有色 RGB
        self.centers = None       # K-Means 聚类中心
        self.shading = None       # 明暗图
        self.lum = None           # 亮度图
        self.reflected = None     # 反射光图
        self.color_lines = None   # 色线图
        self.highlight = None     # 高光图
        self.final = None         # 最终经典管线结果
        self.neural_final = None  # 神经上色结果
        self.stages = []          # (name, image, label) 列表
        self.out_dir = None
        self.params = {}          # 全局参数

    def add_stage(self, name, img, label):
        self.stages.append((name, img, label))


# ---------------------------------------------------------------- 节点基类
class PipelineNode:
    """管线节点基类。子类实现 run(ctx)。"""
    type_name = "Base"

    def __init__(self, node_id, params=None, after=None):
        self.id = node_id
        self.params = params or {}
        self.after = after or []

    def run(self, ctx):
        raise NotImplementedError


# ---------------------------------------------------------------- 节点注册表
_NODE_REGISTRY = {}


def register_node(cls):
    _NODE_REGISTRY[cls.type_name] = cls
    return cls


def get_node_class(type_name):
    if type_name not in _NODE_REGISTRY:
        raise ValueError("未知节点类型: %s，可用: %s" % (type_name, list(_NODE_REGISTRY.keys())))
    return _NODE_REGISTRY[type_name]


# ---------------------------------------------------------------- 具体节点实现

@register_node
class LoadImage(PipelineNode):
    """加载输入图片。"""
    type_name = "LoadImage"

    def run(self, ctx):
        import lineart_painter as lp
        src = ctx.params.get("src")
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("无法读取图片: %s" % src)
        ctx.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = ctx.rgb.shape[:2]
        ctx.params["h"] = h
        ctx.params["w"] = w
        gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        ctx.gray01 = gray
        ctx.add_stage("00_input", ctx.rgb, "输入原图")


@register_node
class ExtractLineart(PipelineNode):
    """线稿提取：XDoG 经典 或 Anime2Sketch 神经。"""
    type_name = "ExtractLineart"

    def run(self, ctx):
        import lineart_painter as lp
        neural = self.params.get("neural", ctx.params.get("neural_lineart", False))
        a2s = self.params.get("a2s_variant", "improved")
        eps = self.params.get("line_eps", ctx.params.get("line_eps", -0.10))
        if neural:
            from neural import a2s_sketch_binary, a2s_sketch
            ctx.line_binary = a2s_sketch_binary(ctx.rgb, variant=a2s)
            ctx.add_stage("01_lineart_neural_raw",
                          a2s_sketch(ctx.rgb, variant=a2s),
                          "01 神经线稿(原始)")
        else:
            xdog = lp.extract_lineart(ctx.gray01, eps=eps)
            ctx.line_binary = lp.clean_binary(xdog)
        ctx.add_stage("01_lineart", ctx.line_binary, "01 线稿")


@register_node
class FlatColors(PipelineNode):
    """K-Means 固有色提取。"""
    type_name = "FlatColors"

    def run(self, ctx):
        import lineart_painter as lp
        k = self.params.get("k", ctx.params.get("k", 10))
        ctx.flat, ctx.centers = lp.flat_colors(ctx.rgb, k=k, return_centers=True)
        ctx.add_stage("02_flatcolor", ctx.flat, "02 固有色(K=%d)" % k)


@register_node
class Shading(PipelineNode):
    """明暗渲染。"""
    type_name = "Shading"

    def run(self, ctx):
        import lineart_painter as lp
        ctx.lum = lp.luminance_map(ctx.rgb)
        ctx.shading = lp.apply_shading(ctx.flat, ctx.lum)
        ctx.add_stage("03_shading", ctx.shading, "03 明暗")


@register_node
class ReflectedLight(PipelineNode):
    """反射光（环境光提亮暗部）。"""
    type_name = "ReflectedLight"

    def run(self, ctx):
        import lineart_painter as lp
        if ctx.shading is None:
            ctx.shading = ctx.flat.copy()
        if ctx.lum is None:
            ctx.lum = lp.luminance_map(ctx.rgb)
        ctx.reflected = lp.reflected_light(ctx.shading, ctx.lum)
        ctx.add_stage("04_reflected", ctx.reflected, "04 反射光")


@register_node
class ColorLines(PipelineNode):
    """色线：将黑色线稿替换为周围深色。"""
    type_name = "ColorLines"

    def run(self, ctx):
        import lineart_painter as lp
        base = ctx.reflected if ctx.reflected is not None else ctx.shading
        if base is None:
            base = ctx.flat.copy()
        ctx.color_lines = lp.color_lines(ctx.line_binary, base)
        ctx.add_stage("05_colorlines", ctx.color_lines, "05 色线")


@register_node
class Highlights(PipelineNode):
    """高光增强。"""
    type_name = "Highlights"

    def run(self, ctx):
        import lineart_painter as lp
        base = ctx.color_lines if ctx.color_lines is not None else ctx.reflected
        if base is None:
            base = ctx.flat.copy()
        if ctx.lum is None:
            ctx.lum = lp.luminance_map(ctx.rgb)
        ctx.highlight = lp.add_highlights(base, ctx.lum)
        ctx.final = lp.unsharp(ctx.highlight)
        ctx.add_stage("06_highlight", ctx.highlight, "06 高光")
        ctx.add_stage("07_final", ctx.final, "07 FINAL 经典还原")


@register_node
class NeuralColor(PipelineNode):
    """神经上色：ControlNet + SD1.5。"""
    type_name = "NeuralColor"

    def run(self, ctx):
        from neural import DEFAULT_PROMPT, NeuralColorizer
        cn = self.params.get("cn_variant", ctx.params.get("cn_variant", "anime"))
        prompt = self.params.get("prompt", ctx.params.get("prompt")) or DEFAULT_PROMPT
        seed = self.params.get("seed", ctx.params.get("seed"))
        strength = self.params.get("strength", ctx.params.get("strength"))
        scale = self.params.get("cn_scale", ctx.params.get("cn_scale")) or 0.85
        colorizer = NeuralColorizer(cn_variant=cn)
        from lineart_painter import _make_conds
        conds = _make_conds(cn, ctx.rgb, ctx.gray01, ctx.line_binary)
        cond_arg = conds[0] if len(conds) == 1 else conds
        init = ctx.rgb if strength else None
        # 参考图上色
        ref = self.params.get("reference_color", ctx.params.get("reference_color"))
        if ref and os.path.exists(ref):
            from lineart_painter import transfer_palette
            ref_bgr = cv2.imread(ref, cv2.IMREAD_COLOR)
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            h, w = ctx.rgb.shape[:2]
            ref_rgb = cv2.resize(ref_rgb, (w, h), interpolation=cv2.INTER_AREA)
            guide = transfer_palette(ctx.flat if ctx.flat is not None else ctx.rgb, ref_rgb)
            init = guide
            strength = strength or 0.5
        # 颜色提示
        hint = self.params.get("color_hint", ctx.params.get("color_hint"))
        if hint and os.path.exists(hint):
            hint_bgr = cv2.imread(hint, cv2.IMREAD_COLOR)
            hint_rgb = cv2.cvtColor(hint_bgr, cv2.COLOR_BGR2RGB)
            h, w = ctx.rgb.shape[:2]
            hint_rgb = cv2.resize(hint_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            init = hint_rgb
            strength = strength or 0.45
        ctx.neural_final = colorizer.colorize(
            cond_arg, prompt, seed=seed, strength=strength,
            scale=scale, init_rgb=init)
        # 文字保护
        if self.params.get("protect_text", ctx.params.get("protect_text", False)):
            from lineart_painter import detect_text_mask, apply_text_protection
            mask = detect_text_mask(ctx.line_binary)
            if np.count_nonzero(mask) > 0:
                ctx.neural_final = apply_text_protection(
                    ctx.neural_final, ctx.rgb, mask)
        ctx.add_stage("final_neural", ctx.neural_final, "FINAL 神经上色")


@register_node
class PhotoColor(PipelineNode):
    """DDColor 黑白照片上色。"""
    type_name = "PhotoColor"

    def run(self, ctx):
        from neural import DDColorizer
        colorizer = DDColorizer()
        result = colorizer.colorize(ctx.rgb)
        ctx.final = result
        ctx.add_stage("final_photo", result, "FINAL 照片上色")


@register_node
class AnimeGAN(PipelineNode):
    """AnimeGANv2 动漫风格化。"""
    type_name = "AnimeGAN"

    def run(self, ctx):
        from neural import AnimeGAN
        gan = AnimeGAN()
        result = gan.transform(ctx.rgb)
        ctx.final = result
        ctx.add_stage("final_anime", result, "FINAL 动漫风格")


@register_node
class SaveOutputs(PipelineNode):
    """保存所有阶段产物 + SVG + replay.html。"""
    type_name = "SaveOutputs"

    def run(self, ctx):
        import lineart_painter as lp
        out_dir = ctx.params.get("out_dir") or ctx.out_dir
        os.makedirs(out_dir, exist_ok=True)
        for name, img, _label in ctx.stages:
            if img is None:
                continue
            if len(img.shape) == 2:
                cv2.imwrite(os.path.join(out_dir, name + ".png"), img)
            else:
                cv2.imwrite(os.path.join(out_dir, name + ".png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # SVG 线稿
        if ctx.line_binary is not None:
            lp.export_lineart_svg(ctx.line_binary,
                                  os.path.join(out_dir, "lineart.svg"))
        # replay.html
        lp.build_replay(os.path.join(out_dir, "replay.html"),
                        ctx.stages, ctx.rgb)


# ---------------------------------------------------------------- DAG 管线引擎
class DAGPipeline:
    """有向无环图管线：按依赖关系拓扑排序执行节点。"""

    def __init__(self, nodes):
        self.nodes = nodes  # list of PipelineNode

    @classmethod
    def from_json(cls, path):
        """从 JSON 文件加载管线定义。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = []
        for nd in data.get("nodes", []):
            node_cls = get_node_class(nd["type"])
            node = node_cls(nd["id"], params=nd.get("params", {}),
                            after=nd.get("after", []))
            nodes.append(node)
        return cls(nodes)

    @classmethod
    def default(cls):
        """返回与 process_image 等价的默认管线。"""
        return cls([
            LoadImage("load"),
            ExtractLineart("lineart", after=["load"]),
            FlatColors("flat", after=["lineart"]),
            Shading("shading", after=["flat"]),
            ReflectedLight("reflected", after=["shading"]),
            ColorLines("colorlines", after=["reflected"]),
            Highlights("highlights", after=["colorlines"]),
            SaveOutputs("save", after=["highlights"]),
        ])

    def topological_sort(self):
        """Kahn 算法拓扑排序。"""
        id_to_node = {n.id: n for n in self.nodes}
        in_degree = {n.id: 0 for n in self.nodes}
        graph = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for dep in n.after:
                if dep in id_to_node:
                    graph[dep].append(n.id)
                    in_degree[n.id] += 1
        queue = deque([nid for nid, d in in_degree.items() if d == 0])
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for next_id in graph[nid]:
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    queue.append(next_id)
        if len(order) != len(self.nodes):
            raise ValueError("管线存在循环依赖")
        return [id_to_node[nid] for nid in order]

    def run(self, src, out_dir, **kwargs):
        """执行管线。kwargs 为全局参数（line_eps, k, neural_lineart 等）。"""
        ctx = NodeContext()
        ctx.out_dir = out_dir
        ctx.params = dict(kwargs)
        ctx.params["src"] = src
        ctx.params["out_dir"] = out_dir
        ordered = self.topological_sort()
        for node in ordered:
            print("[pipeline] 执行节点: %s (%s)" % (node.id, node.type_name))
            node.run(ctx)
        return ctx


# ---------------------------------------------------------------- 默认管线 JSON
DEFAULT_PIPELINE_JSON = {
    "name": "default_classic",
    "description": "经典分步还原管线（与 process_image 等价）",
    "nodes": [
        {"id": "load", "type": "LoadImage"},
        {"id": "lineart", "type": "ExtractLineart", "after": ["load"]},
        {"id": "flat", "type": "FlatColors", "params": {"k": 10}, "after": ["lineart"]},
        {"id": "shading", "type": "Shading", "after": ["flat"]},
        {"id": "reflected", "type": "ReflectedLight", "after": ["shading"]},
        {"id": "colorlines", "type": "ColorLines", "after": ["reflected"]},
        {"id": "highlights", "type": "Highlights", "after": ["colorlines"]},
        {"id": "save", "type": "SaveOutputs", "after": ["highlights"]},
    ],
}

NEURAL_PIPELINE_JSON = {
    "name": "neural_color",
    "description": "神经上色管线：线稿 + ControlNet 上色",
    "nodes": [
        {"id": "load", "type": "LoadImage"},
        {"id": "lineart", "type": "ExtractLineart", "after": ["load"]},
        {"id": "flat", "type": "FlatColors", "params": {"k": 10}, "after": ["lineart"]},
        {"id": "neural", "type": "NeuralColor", "after": ["flat"]},
        {"id": "save", "type": "SaveOutputs", "after": ["neural"]},
    ],
}

MINIMAL_PIPELINE_JSON = {
    "name": "minimal_lineart_only",
    "description": "最简管线：只提取线稿并保存",
    "nodes": [
        {"id": "load", "type": "LoadImage"},
        {"id": "lineart", "type": "ExtractLineart", "after": ["load"]},
        {"id": "save", "type": "SaveOutputs", "after": ["lineart"]},
    ],
}
