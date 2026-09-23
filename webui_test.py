# -*- coding: utf-8 -*-
"""
webui_test.py — webui.py 请求处理 / 状态回传 / 失败留痕 的最小聚焦测试
=====================================================================
smoke_test.py 只 import lineart_painter，从不触及 webui 的 HTTP 契约；本测试专门
覆盖此前无检查的三处：/api/run 的请求参数解析、/api/status 的作业状态回传，以及
“无在线轮询客户端时仍可事后追溯”的持久关联日志。

断言与退出码风格沿用 smoke_test.py：
    python webui_test.py          # 运行全部聚焦检查

退出码：0 = 全部通过；1 = 有失败（错误入参等回归会被暴露）。
不依赖 numpy/cv2/torch：lineart_painter 与 vram_manager 以桩模块替换。
"""

import io
import json
import os
import sys
import tempfile
import time
import traceback
from urllib.parse import parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# --- 用轻量桩替换重依赖，保证聚焦测试可独立、确定性地导入 webui ---
_lp = __import__("types").ModuleType("lineart_painter")
_lp.VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
_lp.process_image = lambda *a, **k: None
_lp.process_video = lambda *a, **k: None
sys.modules["lineart_painter"] = _lp

_vm = __import__("types").ModuleType("vram_manager")


class _VramStub:
    low_vram = False


_vm.vram = _VramStub()
sys.modules["vram_manager"] = _vm

import webui  # noqa: E402

RESULTS = []


def check(name, fn):
    """执行 fn；捕获断言/异常并记为 PASS/FAIL，风格对齐 smoke_test.check。"""
    t0 = time.time()
    try:
        fn()
        RESULTS.append((name, True, "%.2fs" % (time.time() - t0)))
        print("  [PASS] %-40s %.2fs" % (name, time.time() - t0))
    except Exception as e:
        RESULTS.append((name, False, "%s" % e))
        print("  [FAIL] %-40s %s" % (name, e))
        traceback.print_exc(limit=2)


def make_handler(path, headers=None, body=b""):
    """构造一个不绑套接字的 Handler 实例，直接驱动 do_GET/do_POST 分支。

    仅填充被测分支用到的属性，并以捕获型 _send 替换真实响应写出。
    """
    h = webui.Handler.__new__(webui.Handler)
    h.path = path
    h.headers = dict(headers or {})
    h.rfile = io.BytesIO(body)
    captured = {}

    def _send(code, body, ctype="application/json"):
        captured["code"] = code
        captured["body"] = body

    h._send = _send
    h._captured = captured
    return h


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in f.read().splitlines() if ln.strip()]


# ------------------------------------------------------------- 用例
def case_parse_success():
    """成功入参：类型化解析正确且无错误。"""
    qs = parse_qs(
        "lines=-0.15&k=8&neural_lineart=True&neural_color=False&seed=42"
        "&strength=0.5&gen_size=768&input_lineart=False&photo_color=True"
        "&protect_text=True&prompt=a+b&cn=anime")
    params, err = webui._parse_params(qs)
    assert err is None, "成功入参不应产生错误: %r" % err
    assert params["lines"] == -0.15, params["lines"]
    assert params["k"] == 8 and isinstance(params["k"], int), params["k"]
    assert params["neural_lineart"] is True
    assert params["neural_color"] is False
    assert params["photo_color"] is True
    assert params["protect_text"] is True
    assert params["seed"] == 42
    assert params["strength"] == 0.5
    assert params["gen_size"] == 768
    assert params["maxside"] == 1280


def case_parse_error():
    """错误入参：非法数值应返回结构化错误，而非抛出/静默（旧行为会抛断连接）。"""
    params, err = webui._parse_params(parse_qs("lines=abc"))
    assert err is not None and "参数解析失败" in err, "非法 lines 应返回错误: %r" % err
    _p2, err2 = webui._parse_params(parse_qs("lines=-0.1&gen_size=notanumber"))
    assert err2 is not None, "非法 gen_size 应返回错误"
    # 空串走默认值，不应报错
    _p3, err3 = webui._parse_params(parse_qs("lines=&k=&gen_size="))
    assert err3 is None, "空数值应回落默认值: %r" % err3


def case_status_feedback():
    """/api/status 回传：经真实 HTTP 分支读取 running/done 快照与 job 关联标识。"""
    webui._set_state(state="running", message="处理中…", file="x.png", job="JOB777")
    h = make_handler("/api/status")
    webui.Handler.do_GET(h)
    assert h._captured["code"] == 200, h._captured.get("code")
    d = json.loads(h._captured["body"])
    assert d["state"] == "running" and d["job"] == "JOB777", d
    assert d["message"] == "处理中…", d

    webui._set_state(state="done", images=[{"name": "01 线稿",
                   "url": "/media/01_lineart.png"}], videos=[],
                   out_dir="webout/z", elapsed=2.0)
    h2 = make_handler("/api/status")
    webui.Handler.do_GET(h2)
    d2 = json.loads(h2._captured["body"])
    assert d2["state"] == "done" and len(d2["images"]) == 1, d2
    assert d2["out_dir"] == "webout/z", d2
    # 快照应为拷贝：改返回值不得污染内部状态
    d2["state"] = "hacked"
    assert webui._status_payload()["state"] == "done"


def case_run_error_rejected_and_logged():
    """错误入参经 /api/run：返回 400 结构化错误，并在持久日志留下可重建的关联链。"""
    webui._set_state(state="idle", job=None)
    before = len(read_lines(webui.JOB_LOG))
    h = make_handler("/api/run?lines=abc&k=8",
                     headers={"X-Filename": "bad.png", "Content-Length": "3"},
                     body=b"xyz")
    webui.Handler.do_POST(h)
    assert h._captured["code"] == 400, "错误入参应返回 400，实际 %s" % h._captured.get("code")
    resp = json.loads(h._captured["body"])
    assert resp.get("ok") is False and resp.get("error"), "应回传结构化错误: %s" % resp

    # 事后从持久日志重建“触发 -> 决策边界”，无需在线轮询客户端
    events = [json.loads(ln) for ln in read_lines(webui.JOB_LOG)[before:]]
    rejects = [e for e in events if e["event"] == "reject"]
    assert rejects, "应留下 reject 决策边界记录: %s" % events
    rid = rejects[-1]["job"]
    seq = [e["event"] for e in events if e["job"] == rid]
    assert seq == ["trigger", "reject"], "关联链应可重建 触发->决策边界: %s" % seq
    assert rejects[-1].get("http") == 400, rejects[-1]
    assert any(e["event"] == "trigger" and e["job"] == rid for e in events)


def case_job_log_single_line():
    """持久作业日志：每次作业事件为带 job 标识的单行 JSONL，可按 id 检索。"""
    before = len(read_lines(webui.JOB_LOG))
    webui._log_job("TRACE1", "trigger", file="t.png", bytes=1)
    webui._log_job("TRACE1", "error", file="t.png", error="boom | line2")
    lines = read_lines(webui.JOB_LOG)[before:]
    assert len(lines) == 2, "应写入两条事件: %s" % lines
    assert all("\n" not in ln and "\r" not in ln for ln in lines), "每条须为单行 JSONL"
    rec = json.loads(lines[-1])
    assert rec["job"] == "TRACE1" and rec["event"] == "error", rec
    hits = [json.loads(ln) for ln in lines if json.loads(ln)["job"] == "TRACE1"]
    assert [x["event"] for x in hits] == ["trigger", "error"], hits


def case_access_log_restored():
    """访问日志已恢复：log_message 由静默丢弃改为持久写入。"""
    before = len(read_lines(webui.ACCESS_LOG))

    h = make_handler("/api/status")
    h.address_string = lambda: "127.0.0.1"
    h.log_message('"%s" %s %s', "GET /api/status HTTP/1.1", "200", "-")
    lines = read_lines(webui.ACCESS_LOG)[before:]
    assert len(lines) == 1, "恢复后应持久写入访问日志: %s" % lines
    assert "/api/status" in lines[0] and "200" in lines[0], lines[0]


def main():
    # 将持久日志重定向到临时目录，保证断言隔离且不污染真实 webout/
    tmp = tempfile.mkdtemp(prefix="webui_test_")
    webui.JOB_LOG = os.path.join(tmp, "webui_jobs.jsonl")
    webui.ACCESS_LOG = os.path.join(tmp, "webui_access.log")

    print("=== webui.py 聚焦测试 ===  (日志重定向到 %s)" % tmp)
    check("参数解析-成功入参", case_parse_success)
    check("参数解析-错误入参", case_parse_error)
    check("/api/status 状态回传", case_status_feedback)
    check("/api/run 错误入参被拒并留痕", case_run_error_rejected_and_logged)
    check("持久作业日志(单行/关联)", case_job_log_single_line)
    check("访问日志已恢复", case_access_log_restored)

    print("=" * 56)
    fails = [r for r in RESULTS if not r[1]]
    for name, ok, info in RESULTS:
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                              "" if ok else "  -> %s" % info))
    print("=" * 56)
    print("结果: %d 通过 / %d 失败" % (len(RESULTS) - len(fails), len(fails)))
    if fails:
        print("失败项: %s" % ", ".join(r[0] for r in fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
