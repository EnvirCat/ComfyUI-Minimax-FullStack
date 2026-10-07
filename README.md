# ComfyUI-Minimax-FullStack

### From creative intent to finished video. One H3 workspace.

**Full-stack video creation:** bring reference assets, model loading, LoRAs, prompting, sampling, preview, optional interpolation/upscaling and video saving into one user-facing ComfyUI node.

**Start with an idea, not a professionally written prompt.** A local multimodal LLM interprets your creative intent together with the contents and order of your reference images and sampled video frames, then builds a structured MiniMax H3 prompt following the official base/reference prompt-writing guidance.

**从想法到成片，一站式完成 H3 视频生成。** 不必先写专业提示词：输入创作意图，本地多模态模型结合参考素材的内容与顺序，按照 H3 官方提示词规范，组织完整的画面、动作、镜头与声音描述。

[中文完整介绍](README.zh-CN.md) · [Credits / 致谢](CREDITS.md) · [Validation](docs/VALIDATION.md) · Version **2.2.0**

This is an independent community extension, not an official MiniMax product. It orchestrates installed ComfyUI nodes; model weights and third-party node implementations are not bundled. Prompt optimization uses a **local** model, not a hosted LLM API.

## Node screenshots

Real ComfyUI interface captures with an empty reference panel and a harmless coffee-video intent. They demonstrate the interface, not a new generation benchmark.

![Fixed nine-image reference grid with individual connection points](docs/images/references.jpg)

![One video panel and two audio panels](docs/images/video-audio.jpg)

![Intent, Chinese prompt option and optimization controls](docs/images/parameters.jpg)

![Duration, aspect ratio, resolution and compact LoRA rows](docs/images/controls.jpg)

## What makes it different

- **One workspace, end to end.** Auto / T2V / I2V / REF, main model and encoder/VAE selection, dynamic LoRA rows, independent sampling controls, one attention backend, optional memory controls, live sampling preview and a seekable finished-video player.
- **Intent-aware, reference-aware prompting.** For example: “A relaxed portrait video of a woman beside a street.” Supply identity photos, a street video and an accessory image. The local vision model receives their numbered order and infers their intended relationships; you can override an assignment directly in the intent. No separate Role form is needed.
- **Official-guide structure.** A compact runtime contract derived from the official [base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) and [reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) preserves the respective three/six fields, reference labels, shot progression, dialogue conventions, ambience and music sections. Format and nonexistent-reference checks reject malformed results instead of silently replacing your input.
- **Review before generation.** Optimize only, inspect/copy the result, then click **Use Optimized Prompt**. Subsequent video runs reuse that reviewed text without running the LLM again. An `Optimized Prompt` STRING output supports downstream display/text nodes.
- **Chinese or English descriptions.** Check **使用中文提示词** for full Simplified Chinese descriptive prose. Field names, reference labels and protocol markers remain unchanged; original dialogue and visible text remain verbatim. Both languages request the same content coverage, but identical video quality is not guaranteed.
- **Deliberate VRAM handoff.** Successful prompt-only runs may keep the local LLM/projector warm while clearing conversation state. Before any video-generation path—including optimization-off reuse—both are released through their native owner. Failed optimization releases them too; failed cleanup prevents H3 model loading.
- **Batch-friendly inputs.** Each of the nine image cards has its own IMAGE connection point, including compatibility with an installed **Load Image Batch** node. The prompt accepts an external STRING such as **CR Prompt Text**. Seed-based Prompt Variation proposes compatible alternatives; it does not guarantee uniqueness.
- **Optional finishing.** RIFE interpolation; DLSS enhancement or explicitly non-neural Lanczos resizing. Upscaling saves both original-size and upscaled MP4s, preserving the generated audio.

## Install

1. Use a ComfyUI build that provides native MiniMax H3 conditioning, `CLIPLoader` type `minimax`, `MiniMaxH3SigmaShift`, audio decoding and VIDEO saving APIs.
2. Install the feature dependencies below, following each project's own platform-specific instructions.
3. In `ComfyUI/custom_nodes`, run:

```bash
git clone https://github.com/EnvirCat/ComfyUI-Minimax-FullStack.git
```

4. Restart the backend after queued jobs finish, then refresh the browser. Search for **MiniMax H3 Full Stack** (`H3FullStack`) or open [the starter workflow](examples/intent-to-video.json).

Do not install this alongside an older copy named `ComfyUI-H3-FullStack`: they register the same node IDs and routes. Keep a backup, then replace the old installation rather than loading both copies.

### Feature dependencies

| Feature | Required project / node |
| --- | --- |
| H3 model loading, conditioning, sampling, attention selection and saving | [ComfyUI](https://github.com/Comfy-Org/ComfyUI), with native H3 support |
| Live preview / default Low VRAM and feed-forward chunking / Sage options | [KJNodes](https://github.com/kijai/ComfyUI-KJNodes): `ModelPreviewOverrideKJ`, `MiniMaxLowVRAMAttention`, `MiniMaxChunkFeedForward`; relevant Sage patches when selected |
| Local intent optimizer | [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm): `llama_cpp_model_loader`, `llama_cpp_instruct_adv`, native storage API; a compatible local llama.cpp runtime |
| Optional interpolation | [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation), `RIFE VFI` |
| Optional neural enhancement | [ComfyUI-DLSS5-Enhancer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer), `DLSS5EnhanceImages`, plus its supported runtime/GPU/driver |
| Optional external image/text/display nodes | WAS Node Suite, Comfyroll and Custom Scripts; [links and thanks](CREDITS.md) |

Missing optional features do not install themselves. Disable the corresponding options or install the required project. The default preview and memory controls require KJNodes. This release requires the **named node APIs**, not just any historical version of each package; arbitrary older releases are not claimed compatible.

### Models

| ComfyUI model folder | Contents |
| --- | --- |
| `models/diffusion_models` | Matching H3 FL2VA for T2V/I2V and Ref2VA for REF; only use one checkpoint for both if it explicitly supports both |
| `models/text_encoders` | Compatible MiniMax H3 text encoder |
| `models/vae` | H3 video VAE and H3 audio VAE |
| `models/loras` | Optional compatible LoRAs; DMAD is an ordinary selectable LoRA |
| `models/LLM` | Local instruct GGUF plus its matching vision `mmproj` when reading visual references |
| `models/vae_approx` | Compatible tiny VAE for live preview, when used |

Weights are not downloaded by this extension. Obtain them from the respective publishers and follow their licenses. Select the correct GGUF chat handler and matching projector; not every language model is a vision model.

## Workflow

The fixed **References** area has a 3×3 image grid, one visible video panel and two audio panels. Click or drag files to upload; multi-select fills free slots in selection order. Replace/delete use compact icons. T2V dims and ignores references without clearing them. I2V activates only the first two image slots for first/last-frame conditioning. Auto chooses T2V without active media and REF with media.

**Parameters** begins with the intent/prompt, language and optimization switches, the selectable optimized result, Duration (Seconds), Aspect Ratio, Megapixels and LoRAs. Less frequently changed model/sampling/acceleration/optimizer/output controls follow in collapsible groups. The main References panel remains open. **Preview** shows sampling when enabled, then the finished MP4 with native pause/seek controls; upscaled/original versions share the player area.

### Intent → review → video

1. Enter an idea, attach references if needed and enable **Optimize Prompt**.
2. Disable **Generate After Optimization** to generate only the prompt. For immediate end-to-end generation leave it enabled.
3. Inspect the **Optimized Prompt** box or connect its native STRING output to a text node.
4. Click **Use Optimized Prompt** and run. The LLM is skipped and any previously retained weights are unloaded before H3 loads. Click **Use Input Prompt** to return to your input, or enable optimization to create another version.

Simply disabling optimization also automatically reuses the latest result when the original local intent is unchanged. Changes to intent, media, duration, aspect/mode or language invalidate automatic reuse; the old text remains readable and can be explicitly selected. Changing the language switch does **not** translate an already reviewed result—optimize again.

### Resolution, timing and speed

- **Duration** is target video length in seconds. Native H3 output is 24 fps; frame counts round upward to `17k + 5`, so a 5-second request becomes 124 frames / approximately 5.167 seconds.
- Set **Aspect Ratio** and **Megapixels** for size; dimensions follow the core Resolution Selector calculation and round to **Multiple** (normally 32). The resulting width × height is displayed. The 16 MP control ceiling is validation, **not** a supported H3/hardware maximum. Start small and follow the model's published resolution limits and available VRAM.
- DMAD is added in **LoRAs**, with an independent weight. Set **Steps**, sampler and scheduler to the LoRA/model publisher's recommendations; this node does not silently couple them to a preset.
- Select one attention backend. Low VRAM attention and feed-forward chunking are separate memory controls. Kernel availability and hardware compatibility determine whether Sage/Sol work; unsupported selections fail rather than silently switching algorithms.
- Optimizer defaults use an 8192 context, 1536 output tokens, temperature 0.65, no thinking and bounded visual samples. Each supplied image/video sample reserves 1024 visual tokens in the tested Qwen-VL path. Large reference sets may require a larger context; truncation requires a larger output budget. A matching prompt-only model can stay warm; H3 still receives a clean handoff.

### Outputs and saved files

Three public ports: **Video** (`VIDEO`), **Upscaled Video** (`VIDEO`) and **Optimized Prompt** (`STRING`). Prompt-only runs block VIDEO branches cleanly. With upscaling disabled, the upscaled branch is blocked, not a `None` video sent to Save Video.

The node saves MP4s itself. **Output Directory** is relative to ComfyUI's output folder unless an absolute directory is selected; **Filename** is a prefix. Timestamp, unique suffix and seed prevent overwrites. Upscaling writes `_base.mp4` and `_upscaled.mp4`; a failed upscale keeps the already saved base file. Additional external Save Video nodes can create extra copies.

Prompt/generation metadata is saved locally, including user intent and reference mapping. Workflow properties can contain prompt text and result paths. Review them before sharing a workflow. Uploaded assets remain local; remove-card actions do not delete source files. File/upload routes are intended for loopback use, not public hosting.

## Honest boundaries

The optimizer sees images and **three sampled frames per reference video**, not every frame. Audio references currently provide a numbered metadata map; the optimizer does **not** listen to them, transcribe speech or infer voice timbre. H3 itself can receive enabled audio references. Clear intentions and compatible references still matter; subject grouping, role inference and reference adherence can be imperfect.

The runtime contracts follow the official documents but are compact, project-maintained instructions—not MiniMax's hosted context-processing service, official Context-IR model or a verbatim redistribution of the full documents. Local model quality affects prompt quality. Larger contexts/high resolutions/long clips may exhaust memory. Review generated prompts before expensive batches.

## Validation and development

**41 Python offline tests + 16 frontend behavior tests passed** for 2.2.0. No new GPU generation was queued for the Chinese/lifecycle changes. Prior local smoke tests covered H3 generation, visual prompting, RIFE, DLSS and player controls; see [tested versus untested combinations](docs/VALIDATION.md).

Run using your ComfyUI Python environment, with its backend directory on the import path:

```bash
# For a checkout outside custom_nodes, set COMFYUI_ROOT to your backend directory.
python -m unittest discover -s tests -p "test_*.py"
node --test tests/test_frontend.cjs
node --check web/fullstack.js
```

The configuration-only tests do not need ComfyUI: `python -m unittest discover -s tests -p test_config.py`. Pipeline tests mock model loading and use tiny temporary video fixtures; no weights are downloaded and no generation jobs are submitted.

## Thanks and license

Built on the work of MiniMax, ComfyUI and the node authors linked in [CREDITS.md](CREDITS.md). Please support the upstream projects. This repository's original extension code is MIT-licensed; third-party nodes, kernels, model weights and documentation retain their respective terms. No endorsement or ownership of their work is implied.
