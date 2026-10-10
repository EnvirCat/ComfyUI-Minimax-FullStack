# Changelog

## 2.9.0 — 2026-10-10

- Local / OpenAI-compatible API prompt backend shared by the main workbench and attached image generator; explicit visual-upload opt-in, local visual fallback, endpoint-scoped Windows-encrypted keys or environment keys. No DeepSeek preset.
- Standard saved-video metadata for native Assets; explicit local asset-system opt-in with idle restart, without patching ComfyUI core.
- Include explicitly attached downstream video/text outputs in video and one-click queue submissions, not in image/prompt-only review.
- Incremental MP4 encoding and streaming Lanczos, file-backed native VIDEO outputs and compatible remux saves; preserve audio, CRF and final duration.
- Post-interpolation encoding/upscale progress and early base-video preview; cancelled encodes do not publish partial MP4s.
- Shared prompt backend overrides stale planner values in overall-reference image presets.
- Expanded bilingual introduction, exact dependency matrix, Chinese/English manuals, API/privacy guide and attribution.

## Included changes since the prior public 2.2.0 release

- Compact tabbed Chinese-first workbench, editable/reviewable prompts and final-video seeking.
- Attachable reference-image generator with editable three-view character-card template, explicit Apply for occupied slots and prompt reuse on generation-parameter changes.
- Overall-reference image-to-video chain with per-slot person/object/scene choice, aspect override, checkpoints and human review stages.
- Stronger reference coverage, visual-style separation, prompt validation and isolated local-model lifecycle.

Older public history remains in Git. Intermediate local version numbers do not imply independently published releases.
