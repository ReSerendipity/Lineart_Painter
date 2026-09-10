# -*- coding: utf-8 -*-
"""
webui.py — lineart_painter 本地 Web 界面（双击启动，浏览器操作）
==============================================================
纯 Python 标准库实现，无额外依赖。用法：
    python webui.py            # 默认 http://127.0.0.1:8765
启动后浏览器打开即可：选图/视频 -> 调参数 -> 运行 -> 查看分步结果与视频。

配套双击启动文件：启动程序.cmd
"""

import json
import os
import shutil
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote

import lineart_painter as lp

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
WEB_OUT = os.path.join(HERE, "webout")
PORT = 8765

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WEB_OUT, exist_ok=True)

STAGE_LABELS = {
    "01_lineart": "01 线稿",
    "01_lineart_neural_raw": "01 神经线稿原图",
    "02_flatcolor": "02 固有色",
    "03_shading": "03 体积明暗",
    "04_reflected": "04 反射光",
    "05_colorline": "05 色线",
    "06_highlight": "06 高光与细节",
    "final": "FINAL 还原(黑线)",
    "final_colorline": "FINAL 还原(彩线)",
    "final_neural": "FINAL 神经上色",
    "colorized_photo": "照片上色(DDColor)",
    "anime_style": "动漫风格(AnimeGANv2)",
}

# ---------------- 运行状态 ----------------
_state = {"state": "idle", "message": "", "elapsed": 0.0, "images": [],
          "videos": [], "out_dir": None, "error": None}
_lock = threading.Lock()


def _set_state(**kw):
    with _lock:
        _state.update(kw)


def _run_job(src, out_dir, params):
    t0 = time.time()
    try:
        _set_state(state="running", message="处理中…", error=None)
        ext = os.path.splitext(src)[1].lower()
        if ext in lp.VIDEO_EXTS:
            lp.process_video(
                src, out_dir,
                line_eps=params.get("lines", -0.10),
                k=params.get("k", 10),
                max_side=params.get("maxside", 1280),
                neural_lineart=params.get("neural_lineart", False),
                a2s_variant=params.get("a2s", "improved"),
                neural_color=params.get("neural_color", False),
                prompt=params.get("prompt") or None,
                seed=params.get("seed"),
                cn_variant=params.get("cn", "anime"),
                strength=params.get("strength"))
        else:
            lp.process_image(
                src, out_dir,
                line_eps=params.get("lines", -0.10),
                k=params.get("k", 10),
                neural_lineart=params.get("neural_lineart", False),
                neural_color=params.get("neural_color", False),
                prompt=params.get("prompt") or None,
                seed=params.get("seed"),
                a2s_variant=params.get("a2s", "improved"),
                cn_variant=params.get("cn", "anime"),
                strength=params.get("strength"),
                input_lineart=params.get("input_lineart", False),
                photo_color=params.get("photo_color", False),
                anime_gan=params.get("anime_gan", False))
        # 收集产物
        images, videos = [], []
        for name in sorted(os.listdir(out_dir)):
            if name.endswith(".png"):
                label = next((v for k, v in STAGE_LABELS.items()
                              if name.startswith(k)), name)
                images.append({"name": label, "url": "/media/" + name})
            elif name.endswith(".mp4"):
                videos.append({"name": name, "url": "/media/" + name})
        _set_state(state="done", message="完成",
                   elapsed=time.time() - t0, images=images, videos=videos,
                   out_dir=out_dir)
    except Exception:
        _set_state(state="error", error=traceback.format_exc(),
                   elapsed=time.time() - t0, message="处理失败")


# ---------------- HTTP 处理 ----------------
PAGE = """<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>lineart_painter · AI 临摹重绘工作台</title>
<style>
  body{margin:0;background:#0E1116;color:#E8EAED;font-family:'Segoe UI','PingFang SC',Arial,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:20px 16px}
  h1{font-size:19px;margin:0 0 2px;color:#A3D5E8}
  .sub{font-size:12px;color:#7A828E;margin-bottom:16px}
  .card{background:#151A22;border:1px solid #232A34;border-radius:12px;padding:16px;margin-bottom:14px}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
  .field{display:flex;flex-direction:column;gap:4px}
  .field label{font-size:12px;color:#7A828E}
  select,input[type=number],input[type=text]{background:#0E1116;color:#E8EAED;border:1px solid #343D4A;border-radius:8px;padding:7px 10px;font-size:13px}
  .drop{border:2px dashed #343D4A;border-radius:12px;padding:22px;text-align:center;color:#7A828E;font-size:13px;cursor:pointer}
  .drop.on{border-color:#A3D5E8;color:#A3D5E8}
  .btn{background:#A3D5E8;color:#0E1116;border:none;border-radius:8px;padding:9px 20px;font-size:14px;font-weight:600;cursor:pointer}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .bar{height:6px;background:#232A34;border-radius:3px;overflow:hidden;margin:12px 0 6px}
  .bar i{display:block;height:100%;width:0;background:#A3D5E8;transition:width .4s}
  .status{font-size:12px;color:#7A828E;min-height:18px}
  .grid{display:flex;flex-wrap:wrap;gap:12px}
  .item{flex:1 1 300px;min-width:0;background:#151A22;border:1px solid #232A34;border-radius:10px;padding:10px}
  .item img,.item video{width:100%;height:auto;border-radius:8px;display:block;background:#0E1116}
  .item .lbl{font-size:12px;color:#7A828E;margin-top:8px}
  .err{background:#2A1518;border:1px solid #5C2A30;border-radius:8px;padding:10px;font-size:12px;color:#F3A3A8;white-space:pre-wrap;display:none}
  a.dl{color:#A3D5E8;font-size:12px}
</style></head>
<body><div class="wrap">
  <h1>lineart_painter · AI 临摹重绘工作台</h1>
  <div class="sub">图片/视频 → 线稿 → 填色 → 还原 · 本机运行（神经引擎需要 GPU）</div>

  <div class="card">
    <div class="drop" id="drop">点击选择 或 拖入图片/视频文件</div>
    <input type="file" id="file" accept=".png,.jpg,.jpeg,.bmp,.webp,.mp4,.avi,.mov,.mkv,.webm" hidden>
    <div style="margin-top:12px" class="row">
      <div class="field"><label>线稿引擎</label>
        <select id="neural_lineart"><option value="0">经典 XDoG</option><option value="1">神经 Anime2Sketch</option></select></div>
      <div class="field"><label>上色引擎</label>
        <select id="neural_color"><option value="0">经典 K-Means</option><option value="1">神经 ControlNet+SD</option></select></div>
      <div class="field"><label>ControlNet 变体</label>
        <select id="cn"><option value="anime">anime 动漫</option><option value="standard">standard 通用</option></select></div>
      <div class="field"><label>原图混合强度(0=关)</label>
        <input type="number" id="strength" value="0" min="0" max="1" step="0.05"></div>
      <div class="field"><label>随机种子</label>
        <input type="number" id="seed" value="42" step="1"></div>
    </div>
    <div style="margin-top:10px;display:flex;gap:14px;flex-wrap:wrap;align-items:center">
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="input_lineart" style="width:16px;height:16px">输入已是线稿</label>
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="photo_color" style="width:16px;height:16px">黑白照片上色 DDColor</label>
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="anime_gan" style="width:16px;height:16px">动漫风格化 AnimeGANv2</label>
      <div class="field"><label>色块数 k</label>
        <input type="number" id="k" value="10" min="4" max="20" step="1"></div>
      <div class="field"><label>线稿阈值 eps</label>
        <input type="number" id="lines" value="-0.10" step="0.01"></div>
    </div>
    <div class="field" style="margin-top:12px"><label>上色提示词（留空用默认动漫风格）</label>
      <input type="text" id="prompt" placeholder="a beautiful anime illustration, ..."></div>
    <div style="margin-top:14px"><button class="btn" id="run" disabled>运行</button></div>
    <div class="bar"><i id="bar"></i></div>
    <div class="status" id="status">等待选择文件</div>
    <div class="err" id="err"></div>
  </div>

  <div class="grid" id="result"></div>
</div>
<script>
(function(){
  var drop=document.getElementById('drop'),file=document.getElementById('file'),
      run=document.getElementById('run'),status=document.getElementById('status'),
      bar=document.getElementById('bar'),result=document.getElementById('result'),
      err=document.getElementById('err'),sel=null;
  drop.onclick=function(){file.click();};
  file.onchange=function(){ if(file.files[0]) pick(file.files[0]); };
  ['dragover','dragenter'].forEach(function(e){drop.addEventListener(e,function(ev){ev.preventDefault();drop.classList.add('on');});});
  ['dragleave','drop'].forEach(function(e){drop.addEventListener(e,function(ev){ev.preventDefault();drop.classList.remove('on');});});
  drop.addEventListener('drop',function(ev){ if(ev.dataTransfer.files[0]) pick(ev.dataTransfer.files[0]); });
  function pick(f){ sel=f; drop.textContent='已选择: '+f.name+' ('+(f.size/1048576).toFixed(1)+' MB)'; run.disabled=false; status.textContent='就绪'; }
  function v(id){return document.getElementById(id).value;}
  function qs(){
    return 'lines='+v('lines')+'&k='+v('k')+
      '&neural_lineart='+(v('neural_lineart')==='1')+
      '&neural_color='+(v('neural_color')==='1')+
      '&cn='+v('cn')+'&a2s=improved'+
      '&seed='+(v('seed')||'')+'&strength='+(v('strength')||'')+
      '&prompt='+encodeURIComponent(v('prompt'))+
      '&input_lineart='+document.getElementById('input_lineart').checked+
      '&photo_color='+document.getElementById('photo_color').checked+
      '&anime_gan='+document.getElementById('anime_gan').checked;
  }
  run.onclick=function(){
    if(!sel) return;
    run.disabled=true; result.innerHTML=''; err.style.display='none';
    bar.style.width='5%'; status.textContent='上传中…';
    fetch('/api/run?'+qs(),{method:'POST',body:sel,headers:{'X-Filename':encodeURIComponent(sel.name)}})
      .then(function(r){return r.json();}).then(function(d){
        if(!d.ok){status.textContent='启动失败: '+d.error;run.disabled=false;return;}
        poll();
      }).catch(function(e){status.textContent='上传失败: '+e;run.disabled=false;});
  };
  function poll(){
    fetch('/api/status').then(function(r){return r.json();}).then(function(d){
      if(d.state==='running'){ bar.style.width='30%'; status.textContent=d.message+'（已运行 '+d.elapsed.toFixed(0)+' 秒，神经上色每张约4秒）'; setTimeout(poll,1500); }
      else if(d.state==='done'){ bar.style.width='100%'; status.textContent='完成，用时 '+d.elapsed.toFixed(1)+' 秒';
        var html='';
        d.images.forEach(function(it){ html+='<div class="item"><img src="'+it.url+'"><div class="lbl">'+it.name+'</div></div>'; });
        d.videos.forEach(function(it){ html+='<div class="item"><video controls src="'+it.url+'"></video><div class="lbl"><a class="dl" href="'+it.url+'" download>'+it.name+'</a></div></div>'; });
        result.innerHTML=html; run.disabled=false;
      }
      else if(d.state==='error'){ bar.style.width='0'; status.textContent='处理失败'; err.style.display='block'; err.textContent=d.error; run.disabled=false; }
    }).catch(function(){setTimeout(poll,1500);});
  }
})();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path.startswith("/api/status"):
            with _lock:
                st = dict(_state)
            self._send(200, json.dumps(st))
        elif self.path.startswith("/media/"):
            name = unquote(self.path[len("/media/"):])
            base = _state.get("out_dir") or ""
            p = os.path.join(base, os.path.basename(name))
            if os.path.exists(p):
                ctype = ("video/mp4" if p.endswith(".mp4") else
                         "image/png" if p.endswith(".png") else
                         "application/octet-stream")
                self._send(200, open(p, "rb").read(), ctype)
            else:
                self._send(404, json.dumps({"error": "not found"}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.startswith("/api/run"):
            qs = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            fname = unquote(self.headers.get("X-Filename", "upload.png"))
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            src = os.path.join(UPLOAD_DIR, os.path.basename(fname))
            with open(src, "wb") as f:
                f.write(data)
            out_dir = os.path.join(
                WEB_OUT, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(out_dir, exist_ok=True)
            params = {k: (v[0] if v else "") for k, v in qs.items()}
            params["lines"] = float(params.get("lines", -0.10) or -0.10)
            params["k"] = int(params.get("k", 10) or 10)
            params["maxside"] = 1280
            params["neural_lineart"] = params.get("neural_lineart") == "True"
            params["neural_color"] = params.get("neural_color") == "True"
            params["seed"] = (int(params["seed"])
                              if params.get("seed") else None)
            params["strength"] = (float(params["strength"])
                                  if params.get("strength") else None)
            for flag in ("input_lineart", "photo_color", "anime_gan"):
                params[flag] = params.get(flag) == "True"
            if _state["state"] == "running":
                self._send(429, json.dumps(
                    {"ok": False, "error": "已有任务在运行"}))
                return
            _set_state(state="idle")
            threading.Thread(target=_run_job,
                             args=(src, out_dir, params), daemon=True).start()
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print("lineart_painter 工作台: http://127.0.0.1:%d" % port)
    print("关闭窗口或按 Ctrl+C 退出。")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
