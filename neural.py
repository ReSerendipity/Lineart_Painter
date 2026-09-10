# -*- coding: utf-8 -*-
"""
neural.py — 神经升级模块（供 lineart_painter.py 调用）
==========================================================
1. Anime2Sketch 线稿：pix2pix U-Net，照片/插画 -> 干净手绘风线稿
2. ControlNet(lineart) + Stable Diffusion 1.5：线稿 -> 神经上色还原
   - cn 变体：anime（动漫线稿专用） / standard（通用线稿）
   - 混合模式：strength + 原图 = img2img 还原原图风格

权重来源（hf-mirror，国内镜像）：
  - Anime2Sketch:  gyrojeff/Anime2Sketch (netG.pth / improved.bin) -> weights/
  - ControlNet:    anime = lllyasviel/control_v11p_sd15s2_lineart_anime (fp16)
                   standard = lllyasviel/control_v11p_sd15_lineart (fp16) -> models/cn_standard/
  - SD1.5:         stable-diffusion-v1-5/stable-diffusion-v1-5 (fp16)

用法（在 lineart_painter.py 中）：
    from neural import a2s_sketch_binary, NeuralColorizer
"""

import os

import numpy as np

import cv2

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HF_HOME = os.path.join(MODELS_DIR, "hfhome")
MODEL_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_CN_ANIME = "lllyasviel/control_v11p_sd15s2_lineart_anime"
CN_STANDARD_DIR = os.path.join(MODELS_DIR, "cn_standard")
HF_ENDPOINT = "https://hf-mirror.com"


# =====================================================================
# 一、Anime2Sketch 神经线稿
# =====================================================================
def _load_a2s_module():
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_a2s_model.py")
    spec = importlib.util.spec_from_file_location("a2s_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_A2S_MOD = None


def _a2s():
    global _A2S_MOD
    if _A2S_MOD is None:
        _A2S_MOD = _load_a2s_module()
    return _A2S_MOD


_A2S_NET = None
_A2S_DEVICE = None


def a2s_net(variant="improved", device=None, weights_dir=None):
    """构建并加载 Anime2Sketch 网络（懒加载单例）。variant: 'default' | 'improved'。"""
    global _A2S_NET, _A2S_DEVICE
    if _A2S_NET is not None and _A2S_DEVICE == device and variant == _A2S_NET._variant:
        return _A2S_NET, _A2S_DEVICE
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_dir = weights_dir or WEIGHTS_DIR
    mod = _a2s()
    net = mod.UnetGenerator(3, 1, 8, 64,
                            norm_layer=__import__("functools").partial(
                                torch.nn.InstanceNorm2d, affine=False,
                                track_running_stats=False),
                            use_dropout=False)
    if variant == "default":
        ckpt = torch.load(os.path.join(weights_dir, "netG.pth"),
                          map_location=torch.device("cpu"))
        for key in list(ckpt.keys()):
            if "module." in key:
                ckpt[key.replace("module.", "")] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
    elif variant == "improved":
        ckpt = torch.load(os.path.join(weights_dir, "improved.bin"),
                          map_location=torch.device("cpu"))
        base = net.model.model[1]
        for _ in range(6):
            inc, outc = base.model[5].in_channels, base.model[5].out_channels
            base.model[5] = mod.Upsample(inc, outc)
            base = base.model[3]
        net.load_state_dict(ckpt)
    else:
        raise ValueError("variant must be 'default' or 'improved'")
    net.to(device)
    net.eval()
    net._variant = variant
    _A2S_NET, _A2S_DEVICE = net, device
    return net, device


def a2s_sketch(rgb, variant="improved", load_size=512, device=None):
    """图片 -> Anime2Sketch 灰度线稿（0..255，白底深线，尺寸与原图一致）。"""
    import torch

    net, dev = a2s_net(variant=variant, device=device)
    h, w = rgb.shape[:2]
    img = cv2.resize(rgb, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    t = (torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0) \
        .permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = net(t)
    g = ((out[0, 0].cpu().float() + 1) / 2.0 * 255.0).numpy()
    g = np.clip(g, 0, 255).astype(np.uint8)
    if (g.shape[1], g.shape[0]) != (w, h):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return g


def a2s_sketch_binary(rgb, variant="improved", threshold=200, device=None):
    """神经线稿 -> 干净二值线稿（255=线条, 0=背景），与经典管线输出格式一致。"""
    gray = a2s_sketch(rgb, variant=variant, device=device)
    binary = (gray < threshold).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            binary[lab == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary, gray


# =====================================================================
# 二、ControlNet + SD1.5 神经上色还原
# =====================================================================
class NeuralColorizer:
    """线稿 -> 神经上色（懒加载 diffusers 管线，重复调用只加载一次）。

    cn_variant: 'anime'（动漫线稿专用，默认）| 'standard'（通用线稿）
    """

    def __init__(self, device=None, sd_model=MODEL_SD, cn_variant="anime",
                 hf_home=HF_HOME):
        self.sd_model = sd_model
        self.cn_variant = cn_variant
        self.hf_home = hf_home
        self.device = device
        self._pipe = None
        self._pipe_i2i = None

    # -- 环境：下载走 hf-mirror 镜像 ------------------------------------
    @staticmethod
    def _setup_env():
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        os.environ.setdefault("HF_HOME", HF_HOME)

    def _local_snapshot(self, model_id):
        """直接解析本地 hf 缓存快照目录（不联网，避免代理下 SSL 校验失败）。"""
        import glob
        org, name = model_id.split("/", 1)
        base = os.path.join(self.hf_home, "hub",
                            "models--%s--%s" % (org, name), "snapshots")
        snaps = sorted(glob.glob(os.path.join(base, "*")))
        if not snaps:
            raise FileNotFoundError(
                "未找到 %s 的本地权重缓存（%s）。请确认权重已下载到 models/ 目录"
                "（约 3.5GB，或运行一次带 --neural-color 的下载流程）。"
                % (model_id, base))
        return snaps[-1]

    def _download(self, model_id, patterns):
        from huggingface_hub import snapshot_download
        self._setup_env()
        return snapshot_download(model_id, allow_patterns=patterns,
                                 local_dir_use_symlinks=False)

    def _cn_dir(self):
        if self.cn_variant == "standard":
            if not os.path.exists(os.path.join(CN_STANDARD_DIR,
                                               "diffusion_pytorch_model.fp16.safetensors")):
                raise FileNotFoundError(
                    "标准版 ControlNet 权重未找到（%s）。请先运行下载：\n"
                    "  curl -L -o %s\\diffusion_pytorch_model.fp16.safetensors "
                    "https://hf-mirror.com/lllyasviel/control_v11p_sd15_lineart/"
                    "resolve/main/diffusion_pytorch_model.fp16.safetensors"
                    % (CN_STANDARD_DIR, CN_STANDARD_DIR))
            return CN_STANDARD_DIR
        return self._local_snapshot(MODEL_CN_ANIME)

    def load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel,
                               DPMSolverMultistepScheduler,
                               StableDiffusionControlNetPipeline,
                               UNet2DConditionModel)
        from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import (
            StableDiffusionControlNetImg2ImgPipeline)
        from transformers import CLIPTextModel, CLIPTokenizer
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            cn_dir = self._cn_dir()
            sd_dir = self._local_snapshot(self.sd_model)
        except FileNotFoundError:
            print("[neural] 首次使用：下载权重（约 3.5GB，hf-mirror）...")
            self._setup_env()
            if self.cn_variant == "standard":
                raise
            cn_dir = self._download(
                MODEL_CN_ANIME,
                ["config.json", "*.fp16.safetensors", "README.md"])
            sd_dir = self._download(
                self.sd_model,
                ["**/*.fp16.safetensors", "**/*.json", "**/*.txt",
                 "model_index.json", "feature_extractor/*"])
        print("[neural] 加载 ControlNet(%s) + SD1.5 (fp16) ..." % self.cn_variant)
        controlnet = ControlNetModel.from_pretrained(
            cn_dir, torch_dtype=torch.float16, variant="fp16")
        vae = AutoencoderKL.from_pretrained(
            sd_dir, subfolder="vae", torch_dtype=torch.float16, variant="fp16")
        unet = UNet2DConditionModel.from_pretrained(
            sd_dir, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
        text_encoder = CLIPTextModel.from_pretrained(
            sd_dir, subfolder="text_encoder", torch_dtype=torch.float16,
            variant="fp16")
        tokenizer = CLIPTokenizer.from_pretrained(sd_dir, subfolder="tokenizer")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sd_dir, subfolder="scheduler")
        common = dict(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                      unet=unet, controlnet=controlnet, scheduler=scheduler,
                      safety_checker=None, feature_extractor=None,
                      requires_safety_checker=False)
        self._pipe = StableDiffusionControlNetPipeline(**common)
        self._pipe_i2i = StableDiffusionControlNetImg2ImgPipeline(**common)
        if self.device == "cuda":
            self._pipe.to("cuda")
            self._pipe_i2i.to("cuda")
            self._pipe.enable_attention_slicing()
            self._pipe_i2i.enable_attention_slicing()
        else:
            self._pipe.to("cpu")
            self._pipe_i2i.to("cpu")
        return self._pipe

    def colorize(self, sketch_gray, prompt,
                 negative_prompt=("lowres, bad anatomy, bad hands, "
                                  "text, watermark, signature, blurry, "
                                  "jpeg artifacts, ugly"),
                 steps=25, guidance=7.5, scale=0.85, seed=None, size=512,
                 strength=None, init_rgb=None):
        """线稿（0..255，白底黑线）-> 彩色还原图（RGB uint8，原图尺寸）。

        strength 在 (0,1) 且传入 init_rgb 时启用 img2img 混合模式：
        以原图为底、线稿为结构约束重绘，越接近 0 越贴近原图。
        """
        from PIL import Image
        import torch

        pipe = self.load()
        h, w = sketch_gray.shape[:2]
        ctrl = cv2.resize(sketch_gray, (size, size),
                          interpolation=cv2.INTER_LANCZOS4)
        ctrl_pil = Image.fromarray(np.stack([ctrl] * 3, axis=-1))
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        if strength is not None and init_rgb is not None:
            init = cv2.resize(init_rgb, (size, size),
                              interpolation=cv2.INTER_LANCZOS4)
            init_pil = Image.fromarray(init)
            out = self._pipe_i2i(
                prompt=prompt, negative_prompt=negative_prompt,
                image=init_pil, control_image=ctrl_pil,
                strength=float(strength), num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=scale,
                generator=gen).images[0]
        else:
            out = pipe(prompt=prompt, negative_prompt=negative_prompt,
                       image=ctrl_pil, num_inference_steps=steps,
                       guidance_scale=guidance,
                       controlnet_conditioning_scale=scale,
                       generator=gen).images[0]
        out_np = np.array(out).astype(np.uint8)
        if (out_np.shape[1], out_np.shape[0]) != (w, h):
            out_np = cv2.resize(out_np, (w, h),
                                interpolation=cv2.INTER_LANCZOS4)
        return out_np


# 默认上色提示词（动漫插画风格）
DEFAULT_PROMPT = ("a beautiful anime illustration, clean line art, "
                  "vibrant colors, soft cel shading, detailed, "
                  "masterpiece, best quality")


# =====================================================================
# 三、DDColor 黑白照片上色（CNN，CPU/GPU 都快）
# =====================================================================
_DDC_MOD = None
_DDC_DEVICE = None


def _load_ddcolor_module():
    import importlib.util
    import sys

    base = os.path.dirname(os.path.abspath(__file__))
    if base not in sys.path:
        sys.path.insert(0, base)  # 让 _ddcolor_net 能 import basicsr.*
    path = os.path.join(base, "_ddcolor_net.py")
    spec = importlib.util.spec_from_file_location("ddcolor_net", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ddc():
    global _DDC_MOD
    if _DDC_MOD is None:
        _DDC_MOD = _load_ddcolor_module()
    return _DDC_MOD


class DDColorizer:
    """黑白/灰度照片自动上色（DDColor convnext-tiny @512）。

    权重：models/ddcolor/ddcolor_paper_tiny.pth（hf-mirror piddnad/DDColor-models）
    输入 RGB uint8 -> 输出 RGB uint8（原尺寸）。对彩色输入等效“重新上色”，
    建议配合灰度图使用。
    """

    def __init__(self, device=None, model_path=None):
        import torch

        global _DDC_DEVICE
        base = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "ddcolor",
                                      "ddcolor_paper_tiny.pth")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _DDC_DEVICE = device
        net = _ddc()
        from _ddcolor_pipeline import (build_ddcolor_model,
                                       ColorizationPipeline)
        model = build_ddcolor_model(net.DDColor, model_path=model_path,
                                    input_size=512, model_size="tiny",
                                    device=device)
        self._pipe = ColorizationPipeline(model, input_size=512,
                                          device=device)
        self.device = device

    def colorize(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_bgr = self._pipe.process(bgr)
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# =====================================================================
# 四、AnimeGANv2 动漫风格化（ONNX，验证移动端设备端推理路线）
# =====================================================================
_ANIME_SESS = None


class AnimeGAN:
    """照片 -> 动漫风格（AnimeGANv2 face_paint ONNX @512）。

    权重：models/animegan/face_paint_512_v2_0.onnx（hf-mirror
    akhaliq/AnimeGANv2-ONNX）。纯 onnxruntime 推理（CPU/GPU 均可），
    该 ONNX 权重同样可移植到移动端（ONNX Runtime Mobile）。
    """

    def __init__(self, model_path=None):
        import onnxruntime as ort

        global _ANIME_SESS
        base = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "animegan",
                                      "face_paint_512_v2_0.onnx")
        if _ANIME_SESS is None:
            _ANIME_SESS = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
        sess = _ANIME_SESS
        inp = sess.get_inputs()[0]
        self._name = inp.name
        shape = list(inp.shape)
        self._size = int(shape[2]) if len(shape) >= 3 else 512

    def stylize(self, rgb):
        size = self._size
        img = cv2.resize(rgb, (size, size),
                         interpolation=cv2.INTER_AREA)
        x = (img.astype(np.float32) / 127.5 - 1.0)
        x = x.transpose(2, 0, 1)[None, ...]
        out = _ANIME_SESS.run(None, {self._name: x})[0]
        out = out[0].transpose(1, 2, 0)
        out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
        if out.shape[:2] != rgb.shape[:2]:
            out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_AREA)
        return out
