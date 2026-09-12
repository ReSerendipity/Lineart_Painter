# -*- coding: utf-8 -*-
"""
webui.py — lineart_painter 本地 Web 界面（双击启动，浏览器操作）
==============================================================
纯 Python 标准库实现，无额外依赖。用法：
    python webui.py            # 默认 http://127.0.0.1:8765
启动后浏览器打开即可：选图/视频（支持多选批量）-> 调参数 -> 运行 -> 查看分步结果与视频。

配套双击启动文件：启动程序.cmd
"""

import json
import os
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
          "videos": [], "out_dir": None, "error": None, "file": ""}
_lock = threading.Lock()


def _set_state(**kw):
    with _lock:
        _state.update(kw)


def _run_job(src, out_dir, params):
    t0 = time.time()
    fname = os.path.basename(src)
    try:
        _set_state(state="running", message="处理中…", error=None, file=fname)
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
                strength=params.get("strength"),
                input_lineart=params.get("input_lineart", False),
                photo_color=params.get("photo_color", False),
                anime_gan=params.get("anime_gan", False),
                animate_diff=params.get("animate_diff", False))
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
<link rel="icon" href="/favicon.ico">
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
  .drop{border:2px dashed #343D4A;border-radius:12px;padding:18px;text-align:center;color:#7A828E;font-size:13px;cursor:pointer}
  .drop.on{border-color:#A3D5E8;color:#A3D5E8}
  .flist{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
  .fchip{display:flex;gap:8px;align-items:center;background:#0E1116;border:1px solid #343D4A;border-radius:8px;padding:5px 10px;font-size:12px}
  .fchip button{background:none;border:none;color:#EA6668;cursor:pointer;font-size:14px;padding:0 2px}
  .btn{background:#A3D5E8;color:#0E1116;border:none;border-radius:8px;padding:9px 20px;font-size:14px;font-weight:600;cursor:pointer}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .bar{height:6px;background:#232A34;border-radius:3px;overflow:hidden;margin:12px 0 6px}
  .bar i{display:block;height:100%;width:0;background:#A3D5E8;transition:width .4s}
  .status{font-size:12px;color:#7A828E;min-height:18px}
  .group{background:#151A22;border:1px solid #232A34;border-radius:12px;padding:14px;margin-bottom:14px}
  .gtitle{font-size:14px;font-weight:600;color:#A3D5E8;margin-bottom:10px}
  .grid{display:flex;flex-wrap:wrap;gap:12px}
  .item{flex:1 1 280px;min-width:0;background:#0E1116;border:1px solid #232A34;border-radius:10px;padding:10px}
  .item img,.item video{width:100%;height:auto;border-radius:8px;display:block;background:#0E1116}
  .item .lbl{font-size:12px;color:#7A828E;margin-top:8px}
  .err{background:#2A1518;border:1px solid #5C2A30;border-radius:8px;padding:10px;font-size:12px;color:#F3A3A8;white-space:pre-wrap;display:none}
  a.dl{color:#A3D5E8;font-size:12px}
  .hint{font-size:11px;color:#5A6472;font-weight:400;line-height:1.4;margin-top:2px}
  .guide{background:#151A22;border:1px solid #232A34;border-radius:12px;margin-bottom:14px;overflow:hidden}
  .guide summary{cursor:pointer;padding:12px 16px;font-size:14px;font-weight:600;color:#A3D5E8;list-style:none}
  .guide summary::before{content:"▸ ";color:#7A828E}
  .guide[open] summary::before{content:"▾ "}
  .guide .gbody{padding:2px 16px 14px;font-size:12.5px;color:#B9C0CC;line-height:1.7}
  .guide h4{color:#A3D5E8;font-size:12.5px;margin:10px 0 4px}
  .guide table{width:100%;border-collapse:collapse;font-size:12px;margin:4px 0}
  .guide th,.guide td{border:1px solid #232A34;padding:5px 8px;text-align:left;color:#B9C0CC}
  .guide th{color:#E8EAED;background:#0E1116}
  .guide code{background:#0E1116;border:1px solid #232A34;border-radius:4px;padding:1px 5px;color:#94D8C3;font-size:11.5px}
</style></head>
<body><div class="wrap">
  <h1>lineart_painter · AI 临摹重绘工作台</h1>
  <div class="sub">图片/视频 → 线稿 → 填色 → 还原 · 本机运行（神经引擎需要 GPU）· 支持多选批量</div>

  <details class="guide"><summary>使用指南：怎么用 / 每个参数什么意思 / 模型怎么下载</summary><div class="gbody">
    <h4>3 步流程</h4>
    ① 点击上方区域选择或拖入图片/视频（可多选）→ ② 按需要选参数（默认参数可直接运行）→ ③ 点「运行批量」，结果自动显示在下方（每步一张图，视频另有预览）。
    <h4>引擎怎么选（也是“选模型”的关键）</h4>
    <table>
      <tr><th>选项</th><th>含义</th><th>需要模型？</th><th>速度</th></tr>
      <tr><td>线稿 = 经典 XDoG</td><td>传统算法提取线条</td><td>否</td><td>毫秒级</td></tr>
      <tr><td>线稿 = 神经 Anime2Sketch</td><td>AI 提取手绘感线条</td><td>是（约 60MB）</td><td>约 3 秒</td></tr>
      <tr><td>上色 = 经典 K-Means</td><td>聚类算法分块填色</td><td>否</td><td>亚秒</td></tr>
      <tr><td>上色 = 神经 ControlNet+SD</td><td>AI 生成式上色，效果最强</td><td>是（约 4GB）</td><td>10 秒以上</td></tr>
    </table>
    <h4>ControlNet 变体（神经上色时生效）</h4>
    <table>
      <tr><th>变体</th><th>作用</th><th>权重</th></tr>
      <tr><td>anime 动漫线稿（默认）</td><td>针对动漫线稿优化</td><td>已下载</td></tr>
      <tr><td>standard 通用线稿</td><td>普通线稿通用引导</td><td>约 690MB</td></tr>
      <tr><td>canny 边缘</td><td>硬边缘引导，线条更锐</td><td>约 690MB</td></tr>
      <tr><td>scribble 草图</td><td>自由草图风格</td><td>约 690MB</td></tr>
      <tr><td>depth 深度</td><td>按景深结构上色</td><td>约 690MB + MiDaS</td></tr>
    </table>
    <h4>模型下载（在项目目录命令行执行）</h4>
    <code>python setup_models.py --list</code> 查看全部<br>
    <code>python setup_models.py</code> 全部下载（约 8.6GB）<br>
    <code>python setup_models.py --only anime2sketch ddcolor</code> 只下指定项<br>
    未下载的模型选中后会报错提示，下载后即可使用。
    <h4>常用组合速查</h4>
    <table>
      <tr><th>想要的效果</th><th>配置</th></tr>
      <tr><td>什么都不选，直接跑</td><td>默认参数（经典，秒出）</td></tr>
      <tr><td>动漫插画精细还原</td><td>线稿=神经 + 上色=神经 + 变体=anime</td></tr>
      <tr><td>手绘线稿上色</td><td>勾「输入已是线稿」+ 上色=神经</td></tr>
      <tr><td>黑白老照片上色</td><td>勾「黑白照片上色 DDColor」</td></tr>
      <tr><td>照片变动漫</td><td>勾「动漫风格化 AnimeGANv2」</td></tr>
      <tr><td>视频上色不闪烁</td><td>勾「视频一致性 AnimateDiff」+ 上色=神经</td></tr>
    </table>
  </div></details>

  <div class="card">
    <div class="drop" id="drop">点击选择 或 拖入图片/视频文件（可多选）</div>
    <input type="file" id="file" multiple accept=".png,.jpg,.jpeg,.bmp,.webp,.mp4,.avi,.mov,.mkv,.webm" hidden>
    <div class="flist" id="flist"></div>
    <div style="margin-top:12px" class="row">
      <div class="field"><label>线稿引擎</label>
        <select id="neural_lineart"><option value="0">经典 XDoG</option><option value="1">神经 Anime2Sketch</option></select>
        <div class="hint">经典=秒出无需模型；神经=AI 手绘感线条，需下载</div></div>
      <div class="field"><label>上色引擎</label>
        <select id="neural_color"><option value="0">经典 K-Means</option><option value="1">神经 ControlNet+SD</option></select>
        <div class="hint">经典=快速色块；神经=AI 生成式上色，需下载约 4GB</div></div>
      <div class="field"><label>ControlNet 变体</label>
        <select id="cn"><option value="anime">anime 动漫线稿</option><option value="standard">standard 通用线稿</option><option value="canny">canny 边缘</option><option value="scribble">scribble 草图</option><option value="depth">depth 深度</option></select>
        <div class="hint">神经上色的引导条件，未下载的变体需先下载</div></div>
      <div class="field"><label>原图混合强度(0=关)</label>
        <input type="number" id="strength" value="0" min="0" max="1" step="0.05">
        <div class="hint">0=完全按线稿生成；0.35~0.6 混入原图风格</div></div>
      <div class="field"><label>随机种子</label>
        <input type="number" id="seed" value="42" step="1">
        <div class="hint">固定种子可复现相同结果</div></div>
    </div>
    <div style="margin-top:10px;display:flex;gap:14px;flex-wrap:wrap;align-items:center">
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="input_lineart" style="width:16px;height:16px">输入已是线稿<span class="hint" style="display:inline">跳过线稿提取</span></label>
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="photo_color" style="width:16px;height:16px">黑白照片上色 DDColor<span class="hint" style="display:inline">老照片自动上色</span></label>
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="anime_gan" style="width:16px;height:16px">动漫风格化 AnimeGANv2<span class="hint" style="display:inline">照片变动漫风</span></label>
      <label style="font-size:13px;color:#E8EAED;display:flex;gap:6px;align-items:center"><input type="checkbox" id="animate_diff" style="width:16px;height:16px">视频一致性 AnimateDiff<span class="hint" style="display:inline">防闪烁，需 1.7GB</span></label>
      <div class="field"><label>色块数 k</label>
        <input type="number" id="k" value="10" min="4" max="20" step="1">
        <div class="hint">填色用色块数(4-20)，越大越细腻</div></div>
      <div class="field"><label>线稿阈值 eps</label>
        <input type="number" id="lines" value="-0.10" step="0.01">
        <div class="hint">越接近 0 线条越多；越负线条越少</div></div>
    </div>
    <div class="field" style="margin-top:12px"><label>上色提示词（留空用默认动漫风格）</label>
      <input type="text" id="prompt" placeholder="a beautiful anime illustration, ...">
      <div class="hint">英文效果更好，如 watercolor style / cyberpunk</div></div>
    <div style="margin-top:14px"><button class="btn" id="run" disabled>运行批量</button></div>
    <div class="bar"><i id="bar"></i></div>
    <div class="status" id="status">等待选择文件</div>
    <div class="err" id="err"></div>
  </div>

  <div id="result"></div>
</div>
<script>
(function(){
  var drop=document.getElementById('drop'),file=document.getElementById('file'),
      run=document.getElementById('run'),status=document.getElementById('status'),
      bar=document.getElementById('bar'),result=document.getElementById('result'),
      err=document.getElementById('err'),flist=document.getElementById('flist'),
      sel=[],busy=false,queue=[];
  drop.onclick=function(){file.click();};
  file.onchange=function(){ addFiles(file.files); file.value=''; };
  ['dragover','dragenter'].forEach(function(e){drop.addEventListener(e,function(ev){ev.preventDefault();drop.classList.add('on');});});
  ['dragleave','drop'].forEach(function(e){drop.addEventListener(e,function(ev){ev.preventDefault();drop.classList.remove('on');});});
  drop.addEventListener('drop',function(ev){ if(ev.dataTransfer.files.length) addFiles(ev.dataTransfer.files); });
  function addFiles(list){
    for(var i=0;i<list.length;i++){
      var f=list[i],dup=false;
      sel.forEach(function(s){ if(s.name===f.name && s.size===f.size) dup=true; });
      if(!dup) sel.push(f);
    }
    renderFlist();
  }
  function renderFlist(){
    flist.innerHTML='';
    sel.forEach(function(f,idx){
      var d=document.createElement('div'); d.className='fchip';
      d.textContent=f.name+' ('+(f.size/1048576).toFixed(1)+' MB) ';
      var b=document.createElement('button'); b.textContent='×';
      b.onclick=function(){ sel.splice(idx,1); renderFlist(); if(!sel.length){run.disabled=true;} };
      d.appendChild(b); flist.appendChild(d);
    });
    if(sel.length){ run.disabled=false; status.textContent=sel.length+' 个文件待处理'; }
  }
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
      '&anime_gan='+document.getElementById('anime_gan').checked+
      '&animate_diff='+document.getElementById('animate_diff').checked;
  }
  run.onclick=function(){
    if(!sel.length) return;
    run.disabled=true; result.innerHTML=''; err.style.display='none';
    queue=sel.slice(); busy=false;
    status.textContent='准备上传 '+queue.length+' 个文件…'; bar.style.width='3%';
    nextJob();
  };
  function nextJob(){
    if(!queue.length){ bar.style.width='100%'; status.textContent='全部完成'; run.disabled=false; return; }
    var f=queue.shift(), jobT0=Date.now();
    status.textContent='上传 '+f.name+' …'; bar.style.width='5%';
    fetch('/api/run?'+qs(),{method:'POST',body:f,headers:{'X-Filename':encodeURIComponent(f.name)}})
      .then(function(r){return r.json();}).then(function(d){
        if(!d.ok){ status.textContent='启动失败('+f.name+'): '+d.error; run.disabled=false; return; }
        poll(f.name, jobT0);
      }).catch(function(e){ status.textContent='上传失败('+f.name+'): '+e; run.disabled=false; });
  }
  function poll(fname, jobT0){
    fetch('/api/status').then(function(r){return r.json();}).then(function(d){
      var el=(Date.now()-jobT0)/1000;
      if(d.state==='running'){ bar.style.width='30%'; status.textContent='['+fname+'] '+d.message+'（已运行 '+el.toFixed(0)+' 秒）'; setTimeout(function(){poll(fname,jobT0);},1500); }
      else if(d.state==='done'){
        var html='<div class="group"><div class="gtitle">'+fname+' · 用时 '+d.elapsed.toFixed(1)+' 秒</div><div class="grid">';
        d.images.forEach(function(it){ html+='<div class="item"><img src="'+it.url+'"><div class="lbl">'+it.name+'</div></div>'; });
        d.videos.forEach(function(it){ html+='<div class="item"><video controls src="'+it.url+'"></video><div class="lbl"><a class="dl" href="'+it.url+'" download>'+it.name+'</a></div></div>'; });
        html+='</div></div>';
        result.insertAdjacentHTML('beforeend',html);
        bar.style.width='40%';
        status.textContent=(queue.length? '剩余 '+queue.length+' 个文件，继续…' : '完成');
        setTimeout(nextJob, 300);
      }
      else if(d.state==='error'){ bar.style.width='0'; status.textContent='处理失败('+fname+')';
        err.style.display='block'; err.textContent='['+fname+']\\n'+d.error;
        if(queue.length){ status.textContent='跳过 '+fname+'，继续剩余 '+queue.length+' 个…'; setTimeout(nextJob, 300); }
        else run.disabled=false;
      }
    }).catch(function(){setTimeout(function(){poll(fname);},1500);});
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
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/favicon.ico":
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "icons", "lineart-icon.ico")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, f.read(), "image/x-icon")
            else:
                self._send(404, json.dumps({"error": "not found"}))
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
                WEB_OUT, time.strftime("%Y%m%d_%H%M%S_") + os.path.splitext(os.path.basename(fname))[0])
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
            for flag in ("input_lineart", "photo_color", "anime_gan",
                         "animate_diff"):
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
