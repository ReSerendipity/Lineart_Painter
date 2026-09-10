# -*- coding: utf-8 -*-
"""
smoke_test.py — 自动化回归测试（三个引擎 + 全部扩展）
========================================================
把完整管线在真实输入上跑一遍，逐项检查产物存在且非空，输出 PASS/FAIL。

用法：
    python smoke_test.py            # 全量（含神经上色/视频，约 2~4 分钟）
    python smoke_test.py --fast     # 跳过神经上色与视频（约 30 秒）
    python smoke_test.py --only 经典管线 神经线稿 DDColor   # 只跑指定项

退出码：0 = 全部通过；1 = 有失败。产物输出到 smoke_out/（失败时保留现场）。
"""

import os
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "smoke_out")
INPUT = os.path.join(HERE, "demo_input.png")
VIDEO = os.path.join(HERE, "test_video.mp4")

import lineart_painter as lp  # noqa: E402

RESULTS = []


def check(name, fn):
    """执行 fn，检查产物目录里是否有非空文件满足 predicate。"""
    t0 = __import__("time").time()
    try:
        fn()
        elapsed = __import__("time").time() - t0
        RESULTS.append((name, True, "%.1fs" % elapsed))
        print("  [PASS] %-32s %.1fs" % (name, elapsed))
    except Exception as e:
        elapsed = __import__("time").time() - t0
        RESULTS.append((name, False, "%.1fs: %s" % (elapsed, e)))
        print("  [FAIL] %-32s %s" % (name, e))
        traceback.print_exc(limit=2)


def has_png(d, prefix=None):
    for f in os.listdir(d):
        if f.endswith(".png") and (prefix is None or f.startswith(prefix)):
            if os.path.getsize(os.path.join(d, f)) > 10_000:
                return True
    return False


def has_mp4(d):
    return any(f.endswith(".mp4") and
               os.path.getsize(os.path.join(d, f)) > 100_000
               for f in os.listdir(d))


def clean_out(sub):
    d = os.path.join(OUT, sub)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    args = sys.argv[1:]
    fast = "--fast" in args
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1].split(",")

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)

    def want(name):
        return only is None or name in only

    print("=== lineart_painter 回归测试 ===  %s" %
          ("fast 模式" if fast else "全量"))

    if want("经典管线"):
        def _classic():
            d = clean_out("classic")
            lp.process_image(INPUT, d, k=6)
            assert has_png(d, "final"), "缺少 final.png"
            assert os.path.exists(os.path.join(d, "replay.html"))
        check("经典管线 (XDoG+KMeans)", _classic)

    if want("线稿直填"):
        def _lineart_input():
            src = clean_out("lineart_src")
            lp.process_image(INPUT, src, k=4)
            line = os.path.join(src, "01_lineart.png")
            d = clean_out("lineart_in")
            lp.process_image(line, d, input_lineart=True, k=6)
            assert has_png(d, "final")
        check("手绘线稿直接填色 (--input-lineart)", _lineart_input)

    if want("神经线稿") and not fast:
        def _neural_lineart():
            d = clean_out("n_lineart")
            lp.process_image(INPUT, d, neural_lineart=True, k=6)
            assert has_png(d, "01_lineart_neural_raw")
        check("神经线稿 (Anime2Sketch)", _neural_lineart)

    if want("神经上色") and not fast:
        def _neural_color():
            d = clean_out("n_color")
            lp.process_image(INPUT, d, neural_color=True, seed=42, k=6)
            assert has_png(d, "final_neural")
        check("神经上色 (ControlNet+SD1.5)", _neural_color)

    if want("神经双引擎") and not fast:
        def _neural_both():
            d = clean_out("n_both")
            lp.process_image(INPUT, d, neural_lineart=True,
                             neural_color=True, seed=42, k=6)
            assert has_png(d, "final_neural")
        check("神经线稿+神经上色", _neural_both)

    if want("img2img") and not fast:
        def _img2img():
            d = clean_out("i2i")
            lp.process_image(INPUT, d, neural_color=True, seed=42,
                             strength=0.45, k=6)
            assert has_png(d, "final_neural")
        check("img2img 原图混合 (strength=0.45)", _img2img)

    if want("DDColor"):
        def _ddcolor():
            d = clean_out("ddcolor")
            lp.process_image(INPUT, d, photo_color=True, k=4)
            assert has_png(d, "colorized_photo")
        check("照片上色 (DDColor)", _ddcolor)

    if want("AnimeGAN"):
        def _animegan():
            d = clean_out("animegan")
            lp.process_image(INPUT, d, anime_gan=True, k=4)
            assert has_png(d, "anime_style")
        check("动漫风格化 (AnimeGANv2 ONNX)", _animegan)

    if want("视频经典") and not fast:
        def _video():
            d = clean_out("video")
            lp.process_video(VIDEO, d, k=6, max_side=480)
            assert has_mp4(d), "缺少 mp4 产物"
        check("视频经典管线 (75帧)", _video)

    if want("视频扩展") and not fast:
        def _video_ext():
            d = clean_out("video_ext")
            lp.process_video(VIDEO, d, k=6, max_side=480,
                             photo_color=True, anime_gan=True)
            assert os.path.exists(os.path.join(d, "video_photo_color_h264.mp4"))
            assert os.path.exists(os.path.join(d, "video_anime_style_h264.mp4"))
        check("视频 DDColor/AnimeGAN 扩展", _video_ext)

    if want("AnimateDiff") and not fast:
        def _adiff():
            d = clean_out("adiff")
            lp.process_video(VIDEO, d, k=6, max_side=480,
                             animate_diff=True, seed=42)
            assert has_mp4(d), "缺少 mp4 产物"
        check("AnimateDiff 视频一致性 (75帧)", _adiff)

    print("=" * 50)
    fails = [r for r in RESULTS if not r[1]]
    for name, ok, info in RESULTS:
        print("  %s  %s  (%s)" % ("PASS" if ok else "FAIL", name, info))
    print("=" * 50)
    print("结果: %d 通过 / %d 失败" % (len(RESULTS) - len(fails), len(fails)))
    if fails:
        print("失败项: %s" % ", ".join(r[0] for r in fails))
        print("现场保留在 smoke_out/ 目录。")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
