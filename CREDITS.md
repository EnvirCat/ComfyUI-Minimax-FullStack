# Credits / 致谢

Thank you to the authors and communities whose work makes this integrated workspace possible. This extension calls installed node APIs; it does not bundle or claim ownership of their implementations.

感谢以下项目的作者与社区。本节点通过已安装的接口进行编排，不打包或宣称拥有上游实现。请访问、支持原项目，并遵守各自许可。

| Project / 项目 | Contribution / 用途 |
| --- | --- |
| [MiniMax H3 — MiniMax](https://github.com/MiniMax-AI/MiniMax-H3) | H3 model ecosystem and official prompting guidance / H3 模型生态与官方提示词规范 |
| [Official base prompt guide / 官方基础文档](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) | T2V/keyframe structure used as the basis of the compact base contract / 基础与关键帧规范来源 |
| [Official reference prompt guide / 官方参考文档](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) | Subject mapping, retention and six-field reference structure / 主体映射、保留关系与六段结构来源 |
| [ComfyUI — Comfy Org and contributors](https://github.com/Comfy-Org/ComfyUI) | Native H3 loading/conditioning/sampling, audio/video APIs, resolution formula and attention selection / 原生 H3、音视频与尺寸/注意力接口 |
| [ComfyUI-KJNodes — kijai](https://github.com/kijai/ComfyUI-KJNodes) | Model Preview Override, H3 Low VRAM/Chunk Feed Forward and selected Sage patches / 预览、低显存、FF 分块与 Sage 补丁 |
| [ComfyUI-llama-cpp_vlm — lihaoyun6](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) | Local GGUF/VLM loader, multimodal inference and native model lifecycle / 本地加载、多模态推理与模型生命周期 |
| [ComfyUI-llama-cpp — kijai](https://github.com/kijai/ComfyUI-llama-cpp) | Upstream lineage of the local LLM node family / 本地 LLM 节点的上游来源 |
| [llama-cpp-python — JamePeng](https://github.com/JamePeng/llama-cpp-python) and [llama.cpp — ggml-org](https://github.com/ggml-org/llama.cpp) | Underlying local inference ecosystem; install compatible builds via the upstream instructions / 本地推理运行时生态 |
| [ComfyUI-Frame-Interpolation — Fannovel16 and contributors](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) | Optional RIFE VFI interpolation / 可选 RIFE 插帧 |
| [ComfyUI-DLSS5-Enhancer — Blueforcer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer) | Optional DLSS image-batch enhancement; runtime and hardware requirements remain upstream / 可选 DLSS 增强，运行时与硬件要求由上游规定 |
| [ComfyUI-Yusu-MiniMaxH3-Unified — yusu-02 / 在下鱼酥](https://github.com/yusu-02/ComfyUI-Yusu-MiniMaxH3-Unified) | Unified H3 reference-panel/workflow inspiration, not a runtime dependency / 统一参考面板与工作流设计参考，非运行依赖 |
| [WAS Node Suite — WASasquatch and contributors](https://github.com/WASasquatch/was-node-suite-comfyui) | Optional external Load Image Batch input / 外接批量图片入口 |
| [Comfyroll Custom Nodes — Suzie1 and contributors](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) | Optional CR Prompt Text STRING input / 外接文本输入 |
| [ComfyUI-Custom-Scripts — pythongosssss](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) | Optional Show Text display; this extension scopes its fallback renderer to its own connected output / 外接文本显示，兼容渲染仅作用于本节点连接的输出 |

All names belong to their respective authors. Model checkpoints, LoRAs, tiny decoders, proprietary runtimes and third-party documentation are not distributed here. Compatibility with a node is not an endorsement. If an attribution is incomplete, please open an issue so it can be corrected.

所有名称归相应作者所有。本仓库不分发模型、LoRA、tiny 解码器、专有运行时或第三方完整文档。接口兼容不代表背书；如有遗漏，欢迎提出 issue 补充。
