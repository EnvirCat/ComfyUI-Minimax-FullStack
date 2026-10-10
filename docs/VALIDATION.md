# Validation / 验收范围 · 2.9.0

Date / 日期：2026-10-10。Environment / 环境：Windows portable ComfyUI, embedded Python 3.13, Chromium browser. Results below distinguish automated checks from actual generation and untested integrations.

## Automated / 自动化

- **221 Python unit tests**: configuration, H3 graph wiring, local-model lifecycle/worker failures, context/recovery, reference contracts and prompt quality checks, image/overall-reference planning, MP4 output/cancellation, and API transport/routing.
- **86 frontend tests**: reference wiring and apply behavior, compact panels, prompt review/recovery, player timeline, one-click stage queueing, attached output closure and API modal/key isolation.
- API cases use a local fake HTTP SSE/JSON server: hidden reasoning, truncation, auth-error redaction, cancellation, visual opt-in, OpenAI token fields, endpoint validation and Windows DPAPI scope. No real key is in the fixtures.

## Live export check / 实际导出验收

A minimal neutral 128×128 five-frame batch and stereo silent audio ran through installed **RIFE VFI → H3FSFinish → native SaveVideo** for base and Lanczos 2× output. H3FSFinish and both attached saves returned standard output entries with native asset IDs. The files contain ten video frames at 48fps, preserved audio and correct base/upscaled dimensions. This smoke test exercises finishing/export, **not H3 diffusion or artistic quality**.

The native asset subsystem was enabled and prior base/upscaled MP4s were discovered by its output scan and verified through the native asset API. The browser API settings entry was checked; the native Assets sidebar still requires separate frontend verification. Screenshots are from a neutral/reference-workflow verification view, not private user generations.

A first test submission used the wrong dynamic-combo API shape for the current native SaveVideo and failed; the fixture was corrected to the native flat `format` / `format.codec` API inputs and rerun. The successful test used a new seed to avoid reusing cached finishing results.

## Not claimed / 未承诺的范围

- No real paid API provider or remote vision model was called for this release; text-test success on a user's account does not establish vision or end-to-end video support.
- No new full-resolution H3 generation was submitted for this export repair; it does not promise a measured speedup for every long clip, DLSS setting, disk or device.
- DLSS uses its upstream implementation; this release's new streaming path was exercised with Lanczos, not DLSS.
- Unit tests do not prove facial identity, exact image recreation, Chinese/English perceptual equivalence, untested quantization compatibility or the quality of generated prompts on every local model.
- Crash recovery reuses complete saved stages, not partially sampled diffusion states. Files must still exist and inputs must match.

## Reproduce / 复现

From the plugin directory, with the real ComfyUI Python/environment:

```powershell
$env:COMFYUI_ROOT = 'C:\path\to\ComfyUI'
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/test_frontend.cjs tests/test_image_generator_frontend.cjs tests/test_api_frontend.cjs
```

The Python tests import host ComfyUI APIs; they are not standalone in an empty Python environment. Node tests need a Node.js runtime; it is a test-only dependency, not required for normal plugin use.

An explicit live test is available:

```powershell
python tests/validate_video_export.py --submit
```

It checks that the queue is empty, intentionally creates tiny test videos under `output/H3FullStack/export-smoke`, requires RIFE, and does not sample H3 or use private media. Run only when you want those test outputs.
