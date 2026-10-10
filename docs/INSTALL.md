# Installation & dependencies / 安装与完整依赖

Version / 版本：2.9.0。This is an extension, not a standalone generator. / 这是扩展，不是独立生成程序。

## Install / 安装

Back up workflows before upgrading. Install **one copy** only; duplicate folders can cause node/route conflicts. / 更新前备份工作流，仅安装一份，避免重复注册。

From your actual `ComfyUI/custom_nodes` directory:

```powershell
git clone https://github.com/EnvirCat/ComfyUI-Minimax-FullStack.git
```

Use **ComfyUI's Python**, not an unrelated system Python. Windows portable example, run from the portable root / 便携版根目录执行：

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-Minimax-FullStack\requirements.txt
```

Virtual environment / 虚拟环境：

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-Minimax-FullStack/requirements.txt
```

The numpy/Pillow/av/aiohttp packages are **only the extension's direct Python imports**. Install the host, custom nodes and weights below separately. Do not replace ComfyUI's Torch/CUDA stack or blindly force a different llama.cpp wheel.

这四个包只是本插件直接依赖，不包含上游节点和模型。不要为了本插件擅自替换宿主 Torch/CUDA 或 llama.cpp 构建。

## Host / 宿主

Use an up-to-date [ComfyUI](https://github.com/Comfy-Org/ComfyUI) containing native H3 and current VIDEO APIs. This extension uses Python 3.10+; release tests used Windows portable Python 3.13. A modern Chromium browser is recommended. VRAM/RAM depend on model, resolution, duration and offloading; low-VRAM options are not a universal hardware guarantee.

需要包含原生 H3、当前 VIDEO 接口，以及 Qwen Image 2.1（如使用）的新版 ComfyUI。本插件 Python 3.10+，本次测试为便携版 3.13。显存与内存随模型、分辨率、时长、卸载策略变化，不承诺某个显存容量一定可运行。

Check these native IDs in `/object_info` or the node library:

| Capability / 功能 | Native interfaces / 原生接口 |
| --- | --- |
| H3 | `MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, `MiniMaxH3SigmaShift`, `CLIPLoader` type `minimax`, native sampling and video/audio VAE decode |
| Video / 视频 | `CreateVideo`, `SaveVideo`, `comfy_api.latest.InputImpl.VideoFromFile` |
| Attention / 注意力 | `ModelAttentionBackend`; `BlockSparseAttention` when that mode is selected |
| Qwen images / Qwen 生图 | `TextEncodeQwenImage21`, `QwenImage21Cache`, matching native encoder/VAE support |
| Assets / 资产 | Current ComfyUI asset subsystem and its host dependencies; `--enable-assets` or the extension's explicit opt-in |

If missing, update **ComfyUI and its requirements** through its normal procedure. Updating only this extension cannot add missing host APIs. / 缺失时更新宿主及宿主依赖，单独更新本插件无法补齐原生接口。

## Custom nodes / 上游节点

Install through Manager or the linked project's instructions, then restart. This extension calls installed APIs, not bundled implementations.

| Project / 项目 | When needed / 何时需要 | Interfaces / 接口 |
| --- | --- | --- |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Default preview, low-VRAM attention, FF chunking; selected Sage modes / 默认预览、低显存、FF 分块或 Sage | `ModelPreviewOverrideKJ`, `MiniMaxLowVRAMAttention`, `MiniMaxChunkFeedForward`; selected `PathchSageAttentionKJ` / `MiniMaxH3MemoryEfficientSageAttentionPatch` |
| [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) | Local prompt/vision; text API with local material analysis / 本地提示词、视觉或纯文本 API 的本地看图 | Compatible GGUF, matching mmproj, loader/storage lifecycle and upstream llama-cpp runtime |
| [comfyui-krea2-conditioning — huwhitememes](https://github.com/huwhitememes/comfyui-krea2-conditioning) | Krea2 images / Krea2 生图 | `ConditioningKrea2Rebalance` with `preset`, `per_layer_weights`, `multiplier`, `renormalize`. Older original interfaces are not assumed compatible. / 旧接口不自动兼容 |
| [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) | RIFE enabled / 开启插帧 | `RIFE VFI` and selected checkpoint such as `rife47.pth` |
| [ComfyUI-DLSS5-Enhancer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer) | DLSS only / 仅 DLSS 放大 | `DLSS5Settings`, `DLSS5EnhanceImages`; upstream runtime, GPU and driver requirements also apply |
| [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) | Optional external image batch / 可选批量图片 | `Load Image Batch` |
| [Comfyroll](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes), [Custom Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) | Optional text input/display / 可选文本输入显示 | `CR Prompt Text`, `Show Text` |

Lanczos needs no DLSS dependency. If optional nodes are absent, disable their features. Without KJNodes, disable preview/low-VRAM/chunk-FF and use supported native attention; the standard path may use substantially more memory.

Lanczos 无需 DLSS。未安装可选依赖时关掉对应功能。没有 KJNodes 可关闭预览、低显存和 FF 分块，选支持的原生注意力，但内存需求可能显著增加。

## Model files / 模型文件

Download from trusted upstream sources and observe licenses. Family links do **not** promise every quantization works with the native loader.

| Under `ComfyUI/models` / 目录 | What to install / 安装内容 |
| --- | --- |
| `diffusion_models/` | [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) REF2VA and/or FL2VA-compatible weights. Reference and base/keyframe branches must match the requested mode. / REF 与基础关键帧分支不要随意互换 |
| `text_encoders/` | Matching H3 text encoder, typically `minimax_h3*`, using `CLIPLoader` type `minimax` |
| `vae/` | H3 video VAE (`h3_video*`) and audio VAE (`h3_audio*`) / 两者分别选择 |
| `loras/` | Compatible H3 video/character LoRAs; matching distilled/DMAD adapter for its low-step schedule / LoRA 与低步数方案配套 |
| `vae_approx/` | `taeh3.safetensors` for tiny-decoder sampling preview / tiny 预览需要 |
| LLM folder registered by llama-cpp_vlm / 上游注册的 LLM 目录 | Local GGUF + family-matched mmproj for vision; choose the matching handler / 本地 GGUF 与同系列视觉投影文件、对应 handler |

**Four steps are not a universal H3 setting.** Match the checkpoint, adapter and schedule. Wrong VAE/encoder/branch can cause errors or bad output even with plausible filenames. / **4 步不是通用参数**，应按配套模型／蒸馏 LoRA 设置；普通模型使用上游推荐步数。

Reference-image generation additionally needs / 生图额外需要：

| Engine / 引擎 | Model set / 模型组 |
| --- | --- |
| [Krea2](https://huggingface.co/Comfy-Org/Krea-2) | Krea2 Turbo, matching Qwen3-VL 4B text encoder, `qwen_image_vae.safetensors`, **Krea2 TurboSDA LoRA**. The helper uses TurboSDA in its initial lower-resolution stage; final sampling defaults to 8 steps. / 初始阶段的 TurboSDA 不是可省略的人物 LoRA |
| [Qwen Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | Native-loader-compatible Qwen Image 2.1 weights, matching encoder (tested with Qwen3-VL 8B) and Qwen Image 2.1 VAE; default 25 steps |

Image LoRAs must match the image engine; H3 LoRAs belong to video settings. Personal character weights are not distributed. / 生图与视频 LoRA 分开配置，且需要系列兼容；本仓库不附带私人角色权重。

## Assets, restart & updates / 资产、重启与更新

Restart the backend after Python/node changes; refreshing the browser only reloads frontend files. Keep the console visible. Do not interrupt active jobs to upgrade.

- Enable native Assets with `--enable-assets`, or **Settings → Enable native asset registration**, then restart when idle.
- The button stores an explicit local preference in `ComfyUI/user/h3_fullstack/assets.json`; `prestartup_script.py` applies it on next start. No ComfyUI core file is patched.
- ComfyUI scans old outputs in the background; a large library can take time. Native entries require files inside ComfyUI's output root. External directories remain accessible via this extension's own preview.

原生资产栏需要开启资产系统；本插件按钮仅保存本机明确启用的偏好，下次启动生效。旧文件由 ComfyUI 后台扫描，图库大时需等待。输出根目录外的文件可用本插件预览，但不承诺出现在原生资产栏。

For a git installation / git 安装更新，以下从便携版根目录执行：

```powershell
git -C .\ComfyUI\custom_nodes\ComfyUI-Minimax-FullStack pull --ff-only
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-Minimax-FullStack\requirements.txt
```

Adjust paths to your installation. Back up custom edits; do not resolve a dirty checkout with a destructive reset. / 路径按实际环境调整，自行修改先备份，不要强制重置。
