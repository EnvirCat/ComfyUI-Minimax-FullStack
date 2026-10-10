# User manual · 2.9.0

[Installation](INSTALL.md) · [API setup & privacy](API.md) · [中文完整手册](MANUAL.zh-CN.md)

## Workbench

Add **MiniMax H3 Full Stack**. Mode, duration, aspect, megapixels, intent and video LoRAs stay at the top. **Materials**, **Prompt / Preview**, and **Settings** separate everyday actions from folded model, attention and post-processing controls. Image LoRAs and H3 video LoRAs are independent.

## Existing references to video

1. Upload/drop images, video and audio. Use the actual reference labels and order, not invented `<Picture N>` identifiers.
2. Write the intent: subject, how each reference is used, action, camera, timing and sound.
3. Choose REF with its compatible REF2VA checkpoint, or Auto for material-based routing. Base/keyframe modes need the matching branch.
4. Enable prompt optimization and select **Generate video prompt** first. This does not generate a complete video.
5. Inspect/edit the full prompt in **Prompt / Preview**. Choose the reviewed-prompt option to use the approved text, then **Generate video**.

Turning optimization off uses the current input text; it does not universally mean “use the previous optimized draft.” Explicitly choose the reviewed draft, and recheck it after changing references/intent. Protocol checks help, but cannot verify every visual fact or the final result's quality.

Chinese descriptions are the default; H3 protocol field names and reference labels remain unchanged. UI language is selectable. Dialogue is not automatically translated.

## Overall-reference chain

Enable overall reference in Materials, optionally supply a style image, select the slots to generate and give each a subject intent. Set each slot's person/object/scene/composition type explicitly where needed. Per-slot aspect defaults to video aspect.

Without person-reference enabled, the image is used for medium, color, lighting and optical style only. With it enabled, person material can learn hair, clothing, accessories and makeup, without copying the background/pose. Facial geometry is omitted by default so identity can come from a LoRA. This is textual analysis followed by generation, not mandatory image editing or pixel-identical identity transfer.

Select Krea2/Qwen Image 2.1 and its matching weights, image LoRAs, steps and resolution. **Generate material · pause for review** stops after images. Inspect them, change the top-level intent if desired, then generate the video prompt from those actual materials. **One-click video** continues through material, prompt, H3 generation and post-processing.

Disconnect external image generators from selected slots before asking the overall batch to regenerate them. Afterward, attach an image generator to revise an individual image.

## Attached reference-image generator

Use a reference slot's upper-left port or add **H3 Reference Image Generator**. Fan its normal `IMAGE` output to multiple slots. Empty slots accept the result automatically; occupied slots require an explicit Apply action.

| Constraint | Behavior |
| --- | --- |
| Recreate | Needs a reference. Reverse-describes visible subject, clothing, pose, composition, background and lighting, then generates from text. Not a pixel-copy guarantee. |
| Character card | One composite with front, side and back views. Template is editable; default full-body framing includes the complete head and feet with margins, without facial-anatomy descriptions. |
| Style + intent | Style only; the user's intent decides the subject. With no reference it uses the supplied text directly. |

Design and edit the prompt first, then generate/reroll. A valid draft is reused when changing LoRA, weight, seed or diffusion model; changing intent, reference or constraint calls for redesign. Multi-person scenes with a character LoRA can contaminate the second identity; separate character assets plus explicit H3 reference composition are safer than assuming automatic regional LoRA binding.

## Prompt model

Open **Settings → Prompt model · Local / API**, or the same button in the image helper. Default is local. Select matching GGUF, mmproj and handler in local settings. For API URL/model/key, explicit visual opt-in, compatible providers and privacy, read [API.md](API.md).

A text-only API cannot see images. Use local visual analysis or explicitly enable a vision-capable API. The text connection test proves neither vision support nor full video compatibility. Local prompt models are released before video sampling; this does not make oversized diffusion jobs safe.

## Playback and output

Sampling preview is intermediate and may not contain the entire duration. Final MP4 playback supports pause and seeking. RIFE completion is followed by encoding and optional upscaling; stage progress shows this work. The base file becomes available before upscaling finishes.

H3 uses 24fps and frame counts on the `5 + 17n` grid. A requested 10s becomes 243 frames / 10.125s. RIFE 2x may report 485 frames; the extension holds one final frame to make 486 at 48fps, preserving duration and the original audio. A requested 2s becomes 56 frames / about 2.333s.

Default files are under `output/H3FullStack/`: timestamped `_base.mp4`, optional `_upscaled.mp4`, a parameter/prompt JSON report, `prompts/`, `diagnostics/`, and reference-image result folders. Output path/prefix are configurable. A base video remains on disk even if later upscale fails.

Attach native **Save Video** to the VIDEO sockets and **Show Text** to Optimized Prompt. Video/one-click execution includes explicitly connected downstream branches; image/prompt review does not. File-backed VIDEO allows compatible Auto saves to remux, while custom codec settings may re-encode.

Native Assets needs `--enable-assets` or **Enable native asset registration**, followed by an idle restart. Background scanning can recover prior output files. External directories outside ComfyUI's output root are not guaranteed native registration.

## Recovery and troubleshooting

Save the workflow and material. Completed overall-reference stages are stored; matching intent, material order/file state, mode, duration and language can reuse a ready prompt. A crashed diffusion pass cannot resume from an arbitrary sampling frame.

- Empty Assets but MP4 exists: check asset-system enablement and output root; allow background scan time.
- Empty attached saves: refresh the 2.9.0 frontend, verify VIDEO connections and run a video step, not only prompt design.
- Slow after interpolation: watch encoding/upscale progress and backend logs; large frames, DLSS and disk I/O can remain expensive. Test without upscale to isolate.
- Objects become people: explicitly set slot type, remove character LoRA from object/scene material and inspect the draft.
- Invalid/truncated prompts: inspect full text and diagnostics, fix context/handler, or edit before approving.
- Missing nodes/weights: use the [complete dependency matrix](INSTALL.md); encoders, VAEs and LoRAs must match their engine.
- API errors: check endpoint-scoped key, account quota/permissions and model ID. Raw remote error bodies are not displayed.

For workflow connectivity testing, start with 2s and 0.1 MP. This is not a production-quality preset. Include version, failing stage and sanitized diagnostics in bug reports, never keys or private media.
