# Lineart Painter 同类开源仓库调研报告

> 调研日期：2026-09-13  
> 调研范围：GitHub 平台上与「AI 线稿提取 + 上色重绘」功能定位、技术栈、业务场景高度相似的开源项目  
> 筛选标准：Star ≥ 100、近 12 个月有代码提交、文档完善、技术栈可对比  
> 调研目的：挖掘可复用技术点，评估功能复用可行性，避免重复造轮子

---

## 一、调研方法与筛选过程

本次调研以「line art colorization」「anime sketch coloring」「ControlNet lineart」「XDoG line extraction」「manga coloring」等关键词在 GitHub 及技术社区进行多轮检索，初筛出 20+ 相关仓库，再按以下维度过滤：

| 筛选维度 | 标准 |
|----------|------|
| 功能匹配度 | 核心功能包含线稿提取和/或线稿上色重绘 |
| Star 数量 | ≥ 100 |
| 活跃度 | 近 12 个月内有 commit 或 release |
| 文档质量 | 有 README、安装指南、使用示例 |
| 技术栈相关性 | Python / PyTorch / Diffusers / OpenCV / ControlNet |

最终筛选出 **Top 5 高度匹配仓库**，另列出 3 个参考性仓库。

---

## 二、匹配度 Top 5 仓库详细分析

### Top 1：lllyasviel/style2paints — AI 线稿上色的开山之作

| 维度 | 详情 |
|------|------|
| **仓库链接** | https://github.com/lllyasviel/style2paints |
| **Star 数** | ~18,200 |
| **技术栈** | Python (Keras/TensorFlow) + JavaScript (Cocos2d-JS 客户端) |
| **许可证** | Apache-2.0 |
| **最近更新** | 核心代码 2018 年后未大规模更新，但研究页面持续维护 |
| **作者** | Lvmin Zhang (lllyasviel，ControlNet 作者) |

#### 核心功能实现

Style2Paints 是第一个以「真实人类上色作业流程」为设计理念的 AI 线稿上色工具，与本项目的「分步还原」思路高度一致：

1. **线稿输入**：接受手绘线稿或自动提取线稿
2. **颜色提示（Color Hints）**：用户在线稿上点选颜色标记，模型据此扩散填色
3. **分层输出**：输出固有色层、阴影层、高光层等独立图层，而非单张 JPG——这与本项目的 `02_flatcolor` / `03_shading` / `06_highlight` 分阶段输出理念完全一致
4. **迭代精修**：支持多轮颜色提示修正

#### 架构设计

- **生成器网络**：未使用 ResBlock，而是采用专为线稿上色优化的 **Inception 变体**架构
- **训练策略**：以真实插画师的上色流程为监督信号，模拟「铺大色 → 加阴影 → 提高光」的人工步骤
- **客户端-服务端分离**：Python 后端推理 + Cocos2d-JS 前端交互

#### 可借鉴的关键技术点

| 技术点 | 对本项目的价值 | 落地难度 |
|--------|---------------|----------|
| **分层输出设计** | 本项目已实现分阶段 PNG 输出，可进一步学习其「图层可编辑」理念，支持 PSD 分层导出 | 中 |
| **颜色提示交互** | 当前本项目为全自动上色，可增加「用户点选颜色引导」功能，提升可控性 | 高 |
| **Inception 变体架构** | 作为轻量上色模型的备选方案，无需 SD 大模型即可上色 | 中 |

#### 功能复用可行性评估

- **不可直接复用**：基于 Keras（已过时），与本项目 PyTorch+Diffusers 技术栈不兼容
- **可借鉴设计理念**：分层输出、颜色提示交互、模拟人工上色流程——本项目已在经典管线中部分实现
- **风险提示**：作者已转向 ControlNet/Fooocus，style2paints 不再维护，不建议作为技术依赖

---

### Top 2：Mikubill/sd-webui-controlnet — ControlNet 线稿上色的事实标准

| 维度 | 详情 |
|------|------|
| **仓库链接** | https://github.com/Mikubill/sd-webui-controlnet |
| **Star 数** | ~17,800 |
| **技术栈** | Python + PyTorch + Gradio (A1111 WebUI 扩展) |
| **许可证** | MIT |
| **最近更新** | 持续活跃，2025-2026 年仍有 PR 合入（ROCm 支持、文档修复等） |
| **作者** | Mikubill 社区维护 |

#### 核心功能实现

这是 Stable Diffusion WebUI (A1111) 生态中使用最广泛的 ControlNet 扩展，也是线稿上色场景的**事实标准实现**：

1. **多预处理器支持**：内置 `lineart_standard`、`lineart_anime`、`lineart_realistic`、`scribble`、`canny`、`depth` 等 10+ 种条件图预处理器——与本项目 `_make_cond()` 中支持的 5 种 ControlNet 变体完全对应
2. **多 ControlNet 堆叠**：支持同时加载多个 ControlNet（如 lineart + depth + openpose），实现复合条件控制
3. **img2img 集成**：完美兼容 A1111 的 img2img、inpaint、mask 等所有模式
4. **模型管理**：自动扫描模型目录、支持 fp16 加载、权重类型自动检测

#### 架构设计

- **UNet Hook 系统**：通过 `PlugableControlModel` / `PlugableAdapter` / `PlugableControlLLLite` 统一封装不同类型的条件模型，注入到 SD UNet 的注意力层
- **预处理器框架**：每种条件图（lineart/canny/depth 等）封装为独立的预处理器类，可插拔扩展
- **脚本 API**：提供 `Script` 类接口，支持其他扩展调用 ControlNet

#### 可借鉴的关键技术点

| 技术点 | 对本项目的价值 | 落地难度 |
|--------|---------------|----------|
| **多 ControlNet 堆叠** | 本项目当前每次只用 1 个 ControlNet，可学习其多条件融合（如 lineart + depth 同时控制） | 中 |
| **预处理器插件化架构** | 本项目 `_make_cond()` 是硬编码 if-else，可重构为插件化预处理器注册表 | 低 |
| **权重类型自动检测** | 自动识别 fp16/fp32/safetensors/ckpt，本项目当前固定 fp16 | 低 |
| **T2I-Adapter / ControlLoRA 支持** | 比完整 ControlNet 更轻量的条件控制方案，可降低显存需求 | 中 |

#### 功能复用可行性评估

- **不可直接集成**：作为 A1111 扩展，强依赖 WebUI 的脚本框架和模块系统，无法直接嵌入本项目
- **可参考代码实现**：其 `Hook` 注入方式、预处理器调度逻辑、多 ControlNet 加权融合算法，可直接参考移植到 `neural.py` 的 `NeuralColorizer` 中
- **强烈建议关注**：该仓库的预处理器实现（尤其是 `lineart_anime` 的 manga-line 预处理算法）质量极高，可替代本项目当前的 XDoG/Anime2Sketch 二选一方案

---

### Top 3：ali-vilab/MangaNinjia — 参考图驱动的精准线稿上色

| 维度 | 详情 |
|------|------|
| **仓库链接** | https://github.com/ali-vilab/MangaNinjia |
| **Star 数** | 新兴项目，持续增长中（阿里巴巴通义视觉实验室出品） |
| **技术栈** | Python + PyTorch + Diffusers + Stable Diffusion + ControlNet + CLIP |
| **许可证** | 开源（研究用途） |
| **最近更新** | 2024-2025 年活跃开发，有论文支撑 |
| **作者** | Alibaba Tongyi Vision Intelligence Lab (Ali-Vilab) |

#### 核心功能实现

MangaNinjia 是当前最先进的**参考图驱动线稿上色**方案，与本项目的「神经上色」模块场景高度重合但技术路线更先进：

1. **参考图自动对齐**：输入线稿 + 参考彩色图，自动对齐颜色分布，无需手动选色
2. **点控制精修**：支持用户在线稿上点击指定区域颜色，实现细粒度局部控制
3. **多参考图支持**：可同时使用多张参考图，分别对应不同区域的色彩风格
4. **对话气泡保护**：自动检测并保护漫画中的文字区域，避免上色污染

#### 架构设计

- **基于 SD + ControlNet**：以 Stable Diffusion 为生成基座，ControlNet(lineart) 保结构
- **参考注意力机制（Reference Attention）**：修改 UNet 注意力层，将参考图的色彩特征注入生成过程
- **颜色对齐模块**：通过 CLIP 特征匹配实现参考图与线稿区域的语义对齐
- **点控制融合**：用户点选的颜色标记作为额外条件注入

#### 可借鉴的关键技术点

| 技术点 | 对本项目的价值 | 落地难度 |
|--------|---------------|----------|
| **参考图驱动上色** | 本项目当前神经上色仅靠 prompt 描述颜色，可增加「上传参考图自动取色」功能 | 高 |
| **Reference Attention 注入** | 修改 UNet 自注意力层注入参考图特征，比 img2img 混合更精准 | 高 |
| **文字区域保护** | 漫画/插画中的文字、对话框自动检测并遮罩，本项目可增加此预处理 | 中 |
| **点控制交互** | 用户点击指定区域颜色，比纯 prompt 更直观可控 | 高 |

#### 功能复用可行性评估

- **部分可复用**：核心推理管线基于 Diffusers，与本项目 `NeuralColorizer` 技术栈一致，可参考其 Reference Attention 的 UNet 修改方式
- **模型权重可下载**：预训练模型公开，可直接在本项目中加载使用
- **落地建议**：将其「参考图上色」作为本项目 `--neural-color` 的增强模式（`--reference-color path/to/ref.png`），而非替换现有方案

---

### Top 4：comfyanonymous/ComfyUI — 节点化工作流引擎（线稿上色 + 视频一致性）

| 维度 | 详情 |
|------|------|
| **仓库链接** | https://github.com/comfyanonymous/ComfyUI |
| **Star 数** | ~55,000+ |
| **技术栈** | Python + PyTorch + Diffusers + 自定义节点引擎 |
| **许可证** | GPL-3.0 |
| **最近更新** | 极高活跃度，每日 commit，2026 年持续迭代 |
| **作者** | comfyanonymous 社区 |

#### 核心功能实现

ComfyUI 是当前最流行的**节点化 Stable Diffusion 工作流引擎**，线稿上色是其核心应用场景之一：

1. **节点化管线编排**：将「加载线稿 → ControlNet 预处理 → 条件注入 → 采样 → 后处理」拆为独立节点，可视化拖拽组合
2. **ControlNet 全面支持**：通过 `comfyui_controlnet_aux` 扩展支持所有 lineart/canny/scribble/depth 预处理器
3. **AnimateDiff 视频生成**：原生集成 AnimateDiff 运动适配器，支持线稿视频的帧间一致性上色——与本项目 `AnimateDiffColorizer` 功能完全对应
4. **批量处理与队列**：内置任务队列、批量节点、工作流模板保存/分享

#### 架构设计

- **节点执行引擎**：DAG（有向无环图）调度，节点间通过张量/图像/条件对象传递数据
- **模型缓存与复用**：智能管理 VRAM，模型加载后缓存，切换工作流不重复加载
- **前后端分离**：Python 后端 API + 纯前端 JS 节点编辑器，WebSocket 实时通信
- **插件系统**：通过 `custom_nodes` 目录扩展第三方节点

#### 可借鉴的关键技术点

| 技术点 | 对本项目的价值 | 落地难度 |
|--------|---------------|----------|
| **节点化管线设计** | 本项目当前管线是硬编码顺序执行，可参考其 DAG 调度实现灵活的步骤组合 | 高 |
| **VRAM 智能管理** | 模型加载/卸载/缓存策略，本项目当前仅简单懒加载，可优化低显存场景 | 中 |
| **AnimateDiff 工作流** | 其 AnimateDiff 集成比本项目更成熟，支持段间过渡优化、控制网视频 | 中 |
| **工作流模板分享** | JSON 格式保存/导入完整处理流程，本项目可增加「参数配置文件」功能 | 低 |
| **队列与批量** | 本项目 WebUI 已有串行队列，可参考其优先级、暂停、恢复机制 | 中 |

#### 功能复用可行性评估

- **不可直接嵌入**：GPL-3.0 许可证与本项目 Apache-2.0 不兼容，且架构差异大
- **可借鉴架构思想**：节点化、DAG 调度、VRAM 管理是优秀的设计参考
- **实际建议**：如果未来项目需要支持「用户自定义处理流程」，可考虑将核心管线重构为节点引擎；当前阶段不建议

---

### Top 5：Mukosame/Anime2Sketch — 专业级动漫线稿提取器

| 维度 | 详情 |
|------|------|
| **仓库链接** | https://github.com/Mukosame/Anime2Sketch |
| **Star 数** | ~2,100 |
| **技术栈** | Python + PyTorch + Pillow |
| **许可证** | 开源（研究用途） |
| **最近更新** | 权重持续更新（improved 变体），代码基础稳定 |
| **作者** | Xiaoyu Xiang (Mukosame) |

#### 核心功能实现

Anime2Sketch 是本项目**已集成**的神经线稿提取模块，作为独立仓库分析其设计价值：

1. **U-Net pix2pix 架构**：8 层下采样的 U-Net 生成器，InstanceNorm 归一化，无 Dropout
2. **双权重变体**：
   - `default` (netG.pth, 218MB)：基础版，反卷积上采样
   - `improved` (improved.bin, 192MB)：将 6 层反卷积替换为「双线性上采样 + 卷积 + Smooth + MLP 残差」，质量更高
3. **开放域适应**：基于对抗性开放域适应（Adversarial Open Domain Adaptation），对插画、动漫、手绘草图均有效

#### 架构设计

- **生成器**：`UnetGenerator(3→1, 8 downs, ngf=64)`，从最内层向外递归构建
- **上采样改进**：`Upsample` 类 = `ReplicationPad2d + 3x3 平滑卷积 + 双线性上采样 + 3x3 卷积 + 1x1 MLP 残差`，消除反卷积的棋盘伪影
- **推理流程**：512x512 输入 → 归一化 [-1,1] → 前向 → 反归一化 → resize 回原图

#### 可借鉴的关键技术点

| 技术点 | 对本项目的价值 | 落地难度 |
|--------|---------------|----------|
| **improved 上采样设计** | 本项目已使用 improved 变体，其「反卷积→上采样+卷积+残差」替换策略可推广到其他生成模型 | 低 |
| **线稿质量评估标准** | 可学习其训练数据构建方式（XDoG 伪标签 + 真实线稿对抗适应），用于评估本项目 XDoG 线稿质量 | 中 |
| **CLAHE 预处理增强** | 上游测试脚本支持 `equalize_clahe` 对比度增强，可作为线稿提取的可选预处理 | 低 |

#### 功能复用可行性评估

- **已完全复用**：本项目 `_a2s_model.py` + `neural.py:a2s_sketch_binary` 已完整集成，权重已下载
- **无需重复开发**：线稿提取模块已达到该仓库同等能力
- **后续优化方向**：可尝试 `lineart_anime` 预处理器（sd-webui-controlnet 中的 manga-line 算法）作为第三种线稿提取方案，与 XDoG / Anime2Sketch 形成三选一

---

## 三、其他参考性仓库

| 仓库 | Star | 技术栈 | 参考价值 | 不入选 Top5 原因 |
|------|------|--------|----------|-----------------|
| **pfnet/PaintsChainer** | ~3,700 | Python + Chainer | 线稿上色的早期标杆，Web 交互设计参考 | Chainer 框架已停止维护，近 5 年无更新 |
| **piddnad/DDColor** | 活跃 | Python + PyTorch | 双解码器照片上色，ICCV 2023 | 本项目已完整集成，非竞品而是组件 |
| **TachibanaYoshino/AnimeGANv2/v3** | ~4,600 | Python + TensorFlow/ONNX | 照片转动漫风格，轻量 GAN | 本项目已集成 ONNX 版本，非竞品而是组件 |
| **AUTOMATIC1111/stable-diffusion-webui** | ~130,000 | Python + Gradio | 通用 SD 平台，线稿上色是其功能之一 | 通用平台非线稿专用，体量过大 |
| **hepesu/LineDistiller** | ~163 | Python + Keras/PyTorch | 数据驱动线稿提取 + 数据集构建工具 | Star 偏低，Keras 版本过时 |

---

## 四、横向对比总表

| 对比维度 | style2paints | sd-webui-controlnet | MangaNinjia | ComfyUI | Anime2Sketch |
|----------|:---:|:---:|:---:|:---:|:---:|
| **功能匹配度** | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| **技术栈相似度** | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ |
| **活跃度** | ★☆☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| **文档完善度** | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| **可直接复用性** | ★☆☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★☆☆☆☆ | ★★★★★ |
| **综合推荐指数** | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |

---

## 五、可借鉴关键技术点汇总（按落地优先级排序）

### P0 — 立即落地（低难度、高收益）

1. **预处理器插件化重构**  
   将 `lineart_painter.py:_make_cond()` 的硬编码 if-else 重构为注册表模式：
   ```python
   COND_PREPROCESSORS = {}
   def register_cond(name):
       def decorator(fn): COND_PREPROCESSORS[name] = fn; return fn
       return decorator
   ```
   参考 sd-webui-controlnet 的预处理器框架，便于后续新增 `lineart_realistic`、`lineart_anime`（manga-line 算法）等预处理器。

2. **增加 `lineart_anime` 预处理器**  
   sd-webui-controlnet 中的 manga-line 预处理算法（lllyasviel 开发）质量优于本项目当前的 XDoG，可作为第三种线稿提取方案，与 XDoG / Anime2Sketch 并列。

3. **参数配置文件导出/导入**  
   参考 ComfyUI 的工作流 JSON，将当前 CLI 参数组合保存为 `.json` 配置文件，支持一键复现处理流程。

### P1 — 短期规划（中难度、中高收益）

4. **多 ControlNet 堆叠支持**  
   参考 sd-webui-controlnet 的多 ControlNet 加权融合，支持同时使用 `lineart + depth` 或 `lineart + canny`，提升复杂场景的结构保持能力。

5. **VRAM 智能管理**  
   参考 ComfyUI 的模型缓存策略，实现模型加载优先级、自动卸载、低显存模式（如 4GB 显存下自动启用 sequential offload）。

6. **文字/对话区域自动保护**  
   参考 MangaNinjia 的气泡保护，在神经上色前用简单的连通域分析或 OCR 检测文字区域，生成遮罩避免上色污染。

### P2 — 中长期探索（高难度、高潜力）

7. **参考图驱动上色**  
   移植 MangaNinjia 的 Reference Attention 机制，增加 `--reference-color ref.png` 参数，实现「上传参考图自动取色上色」，这是本项目当前最大的功能空白。

8. **用户点选颜色引导**  
   参考 style2paints 的颜色提示交互，在 WebUI 中增加「在线稿上点选颜色」功能，提升上色可控性。

9. **节点化管线引擎**  
   参考 ComfyUI 的 DAG 调度，将当前硬编码的 6 步管线重构为可拖拽组合的节点系统，支持用户自定义「只线稿+固有色」或「跳过反射光直接高光」等灵活流程。

---

## 六、避免重复开发的具体落地建议

### 6.1 已覆盖、无需重复开发的领域

| 功能 | 本项目状态 | 对标仓库 | 结论 |
|------|-----------|----------|------|
| XDoG 线稿提取 | ✅ 已实现 | 通用算法 | 无需重复 |
| Anime2Sketch 神经线稿 | ✅ 已集成 | Mukosame/Anime2Sketch | 已复用，无需重复 |
| ControlNet+SD 上色（5变体） | ✅ 已实现 | sd-webui-controlnet | 核心能力已对齐 |
| img2img 混合模式 | ✅ 已实现 | A1111/ComfyUI | 已对齐 |
| DDColor 照片上色 | ✅ 已集成 | piddnad/DDColor | 已复用，无需重复 |
| AnimeGAN 风格化 | ✅ 已集成 | TachibanaYoshino | 已复用，无需重复 |
| AnimateDiff 视频一致性 | ✅ 已实现 | ComfyUI | 基础能力已对齐 |
| 经典分步还原管线 | ✅ 已实现 | style2paints（理念） | 理念已落地 |
| Web UI + 批量队列 | ✅ 已实现 | A1111/ComfyUI | 基础能力已对齐 |

### 6.2 存在差距、建议优先补齐的领域

| 功能差距 | 对标仓库 | 建议方案 | 优先级 |
|----------|----------|----------|--------|
| 参考图自动取色上色 | MangaNinjia | 移植 Reference Attention，新增 `--reference-color` | P2 |
| 多 ControlNet 条件融合 | sd-webui-controlnet | 重构 NeuralColorizer 支持多 ControlNet 加权 | P1 |
| manga-line 高质量线稿 | sd-webui-controlnet | 集成 lineart_anime 预处理器作为第三方案 | P0 |
| 文字区域保护 | MangaNinjia | 增加 OCR/连通域检测遮罩 | P1 |
| 用户点选颜色引导 | style2paints | WebUI 增加交互层 | P2 |
| 低显存优化 | ComfyUI | 模型缓存 + 自动卸载 + sequential offload | P1 |
| 处理流程配置化 | ComfyUI | 参数 JSON 导出/导入 | P0 |

### 6.3 不建议投入的方向

1. **自研线稿提取模型**：Anime2Sketch + XDoG + manga-line 三方案已覆盖绝大多数场景，自研模型投入产出比极低
2. **自研上色基座模型**：SD1.5 + ControlNet 生态成熟，自研基座需要海量数据和算力，无必要
3. **重写为 ComfyUI 节点**：GPL-3.0 许可证冲突，且本项目定位是「开箱即用的独立工具」而非「通用工作流平台」
4. **复刻 style2paints 的 Keras 模型**：技术栈过时，作者已放弃维护

---

## 七、结论

本项目 Lineart Painter 在**核心功能覆盖度上已达到主流开源项目的 80% 以上**，经典管线（XDoG + K-Means 分步还原）是差异化优势，神经功能（ControlNet 上色、AnimateDiff 视频、DDColor、AnimeGAN）均已通过集成上游开源项目实现，不存在根本性的重复开发。

**最大的差异化机会**在于：
1. 将「经典分步还原」与「神经上色」深度融合（当前是两条独立管线，可考虑用经典管线的固有色/明暗作为神经上色的额外条件）
2. 增加参考图驱动上色（MangaNinjia 路线），填补纯 prompt 上色的可控性不足
3. 持续优化 WebUI 交互体验，保持「零构建、双击启动」的轻量优势

**建议研发资源分配**：70% 用于差异化功能（参考图上色、经典+神经融合、交互优化），20% 用于工程优化（低显存、多 ControlNet、配置化），10% 用于跟踪上游更新（Anime2Sketch improved、ControlNet 新变体、AnimateDiff v3）。

---

*报告生成时间：2026-09-13 | 数据来源：GitHub 仓库页面、技术社区文档、论文预印本 | Star 数为调研时近似值*
