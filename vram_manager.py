# -*- coding: utf-8 -*-
"""
vram_manager.py — 显存智能管理模块
====================================
提供模型注册、自动卸载、低显存模式等功能，帮助在有限 GPU 显存下
运行多个神经模型（ControlNet、SD、AnimateDiff、DDColor 等）。

用法：
    from vram_manager import vram
    vram.low_vram = True          # 启用低显存模式
    vram.register("sd_pipe", pipe) # 注册模型
    vram.unload("sd_pipe")         # 卸载单个模型
    vram.unload_all()              # 卸载全部模型
    free = vram.get_free_vram()    # 查询可用显存 (MB)
"""

import gc

# 全局单例
class _VRAMManager:
    def __init__(self):
        self._models = {}       # name -> object
        self.low_vram = False   # 低显存模式标志
        self._torch = None

    def _get_torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                self._torch = False
        return self._torch

    def register(self, name, model):
        """注册一个模型到显存管理器。重复注册覆盖旧引用。"""
        self._models[name] = model
        return model

    def unload(self, name):
        """卸载指定模型：移到 CPU 并删除引用，触发垃圾回收。"""
        if name not in self._models:
            return False
        model = self._models.pop(name)
        torch = self._get_torch()
        if torch and torch.cuda.is_available():
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass
        del model
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def unload_all(self):
        """卸载所有已注册模型。"""
        names = list(self._models.keys())
        for n in names:
            self.unload(n)
        return len(names)

    def unload_except(self, keep_names):
        """卸载除 keep_names 列表外的所有模型。"""
        keep = set(keep_names) if isinstance(keep_names, (list, tuple)) else {keep_names}
        removed = 0
        for n in list(self._models.keys()):
            if n not in keep:
                if self.unload(n):
                    removed += 1
        return removed

    def get_free_vram(self):
        """返回当前可用显存 (MB)，无 CUDA 时返回 None。"""
        torch = self._get_torch()
        if not torch or not torch.cuda.is_available():
            return None
        try:
            free, total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
        except Exception:
            return None

    def get_used_vram(self):
        """返回已分配显存 (MB)。"""
        torch = self._get_torch()
        if not torch or not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            return None

    def ensure_vram(self, required_mb):
        """检查可用显存是否足够，不够则自动卸载非活动模型。

        返回 True 表示当前可用显存 >= required_mb（或无 CUDA）。
        """
        free = self.get_free_vram()
        if free is None:
            return True  # CPU 模式不限制
        if free >= required_mb:
            return True
        # 尝试卸载模型释放空间（按注册顺序逆序卸载，最近注册的先卸）
        for name in reversed(list(self._models.keys())):
            self.unload(name)
            free = self.get_free_vram()
            if free is not None and free >= required_mb:
                return True
        return False

    def apply_low_vram(self, pipe):
        """对 diffusers pipeline 应用低显存优化。

        - enable_attention_slicing(): 注意力切片，减少峰值显存
        - enable_vae_slicing(): VAE 切片，处理大图时省显存
        - enable_sequential_cpu_offload(): 模型层按需加载到 GPU（最省显存但最慢）
        """
        if not self.low_vram:
            return
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            # sequential offload 最省显存但会显著降低速度，仅在低显存模式启用
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

    def list_models(self):
        """返回已注册模型名称列表。"""
        return list(self._models.keys())

    def status(self):
        """返回显存状态摘要字符串。"""
        free = self.get_free_vram()
        used = self.get_used_vram()
        models = self.list_models()
        if free is None:
            return "CPU 模式（无 CUDA），已注册模型: %s" % (models or "无")
        return ("显存: 已用 %dMB / 可用 %dMB | 低显存: %s | 已注册模型: %s"
                % (used or 0, free, self.low_vram, models or "无"))


# 全局单例
vram = _VRAMManager()
