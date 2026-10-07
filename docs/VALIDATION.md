# Validation scope / 验收范围

Release 2.2.0, 2026-10-07. Tested in an existing Windows ComfyUI environment with native H3 nodes, KJNodes and llama-cpp-vlm. This is not a promise of universal platform/package-version compatibility.

## Current release

- 41 offline Python tests passed: config/frame alignment, invalid inputs, graph wiring, LoRAs, mutually exclusive attention paths, native DynamicCombo argument handling, prompt-only blocked video branches, actual tiny temporary MP4 saving, reference maps, compact contracts, tokenizer budget and model lifecycle.
- 16 Node frontend tests passed: prompt display, connected Show Text, native video controls, version selection, late preview events, history isolation, reviewed-prompt reuse, invalidation and Chinese-switch persistence.
- JavaScript syntax check passed.
- The restarted local backend reported 2.2.0 with an empty queue. A separate harmless starter workflow loaded the real frontend; its Chinese checkbox toggled off/on correctly and a compact DMAD LoRA row accepted its model selection. Screenshots were captured without running generation or saving edits to the user's workflow.
- Local model loading and vision inference are mocked in the new Chinese/lifecycle tests. No new GPU jobs were submitted for these changes. Actual Chinese-versus-English video quality and full-load VRAM handoff are **not** benchmarked.
- Successful prompt-only retention, matching-config reuse, changed-budget reload, error cleanup, cleanup-before-direct-video and blocking video loading on cleanup failure are verified at the API/ordering level.

## Earlier local smoke tests

Earlier versions successfully exercised low-resolution H3 T2V/reference generation, one-image vision prompting, one-image plus one-video prompting, RIFE 2× interpolation, Lanczos and DLSS 2× dual-video saving with audio. Harmless existing-video fixtures verified pause/seek and optimized text display without re-running a model.

These are functional smoke tests, not controlled speed or visual-quality benchmarks. No general speedup ratio is advertised.

## Not fully tested

Nine-image maximum-reference inference; all Sage/Sol kernels and driver combinations; maximum-resolution/long-duration clips; arbitrary third-party package revisions; non-Windows operation; audible transcription (not implemented); guaranteed identity adherence or unique batch prompts; native mouse-drag port gestures across all frontend revisions.

## Run the tests

Use the same Python environment as ComfyUI. A plugin checkout under `ComfyUI/custom_nodes` discovers the backend through its parent directories. For an independent checkout set `COMFYUI_ROOT` to the backend path, then run:

```bash
python -m unittest discover -s tests -p "test_*.py"
node --test tests/test_frontend.cjs
node --check web/fullstack.js
```

The tests do not download models or enqueue generation. Pipeline tests import ComfyUI and create tiny video files in temporary directories. Configuration-only tests can run without ComfyUI:

```bash
python -m unittest discover -s tests -p test_config.py
```

## 中文说明

本次发布通过 41 项 Python 离线测试、16 项前端行为测试和 JS 语法检查。中文生成与 LLM 生命周期使用 mock 验证格式、调用参数及释放顺序，没有新增 GPU 生成任务，也没有中英文画质或高负载显存基准。历史实测包括低分辨率 H3、本地视觉提示词、插帧、两种放大和音轨保留；不能将功能冒烟测试宣传为普遍提速或画质保证。九图满负载、高分辨率长视频、全部内核/驱动组合以及跨平台仍需用户环境验证。
