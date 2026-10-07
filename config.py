"""Small, dependency-free configuration contract shared by API and tests."""
import copy
import json
import math
import re
from pathlib import Path

DEFAULTS = {
    "version": 2, "mode": "auto", "fl2va_model": "", "ref2va_model": "",
    "text_encoder": "", "video_vae": "", "audio_vae": "", "loras": [],
    "resolution": "megapixels", "aspect": "16:9", "megapixels": 1.0, "multiple": 32,
    "width": 1344, "height": 768,
    "seconds": 5.0, "ref_image_size": "match", "media": [],
    "profile": "custom", "dmad_lora": "", "steps": 4,
    "sampler": "euler", "scheduler": "simple", "shift_video": 12.0, "shift_audio": 2.0,
    "attention": "comfy kitchen attention", "low_vram": True, "head_chunks": 4,
    "chunk_ff": True, "ff_chunks": 2, "sol_tau": 1.3, "sol_start": 0.2, "sol_end": 1.0,
    "preview": True, "preview_frames": 8, "preview_fps": 8, "preview_size": 768,
    "tiny_vae": "taeh3.safetensors", "preview_audio": False,
    "rife": False, "rife_model": "rife47.pth", "rife_multiplier": 2,
    "upscale": False, "upscale_method": "DLSS", "upscale_factor": 2.0,
    "output_dir": "H3FullStack", "filename": "H3", "crf": 18,
    "optimize_on_run": False, "generate_after_optimization": True,
    "use_chinese_prompt": False,
    "use_reviewed_prompt": False, "reviewed_prompt": "",
    "llm_model": "", "llm_mmproj": "None",
    "llm_vision": True, "llm_handler": "Qwen3.8", "llm_ctx": 8192,
    "llm_vram": 14, "llm_tokens": 1536, "llm_temperature": 0.65,
    "prompt_variation": True,
}
ASPECTS = {k: tuple(map(int, k.split(":"))) for k in
           ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9", "9:21", "4:5", "5:4")}
SHORT_EDGES = {"360P": 352, "480P": 480, "540P": 544, "640P": 640, "720P": 736, "768P": 768}
LIMITS = {"image": 9, "video": 3, "audio": 2}
EXTENSIONS = {"image": {".png", ".jpg", ".jpeg", ".webp"},
              "video": {".mp4", ".mov", ".mkv", ".webm"},
              "audio": {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}}

def number(value, label, minimum, maximum, integer=False):
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(n) or not minimum <= n <= maximum or (integer and n != int(n)):
        raise ValueError(f"{label} must be between {minimum} and {maximum}" + (" and an integer" if integer else ""))
    return int(n) if integer else n

def geometry(c):
    if c["resolution"] == "megapixels":
        a, b = ASPECTS[c['aspect']]
        mp = number(c['megapixels'], 'Megapixels', .1, 16)
        multiple = number(c['multiple'], 'Multiple', 32, 128, True)
        if multiple % 32: raise ValueError('H3 Multiple must be divisible by 32')
        scale = math.sqrt(mp * 1024 * 1024 / (a * b))
        w = round(a * scale / multiple) * multiple
        h = round(b * scale / multiple) * multiple
    elif c["resolution"] == "custom":
        w = number(c["width"], "Width", 128, 4096, True)
        h = number(c["height"], "Height", 128, 4096, True)
        if w % 32 or h % 32:
            raise ValueError("H3 dimensions must be multiples of 32")
    else:
        if c["resolution"] not in SHORT_EDGES or c["aspect"] not in ASPECTS:
            raise ValueError("Unknown resolution or aspect ratio")
        a, b = ASPECTS[c["aspect"]]
        short = SHORT_EDGES[c["resolution"]]
        # Same native area cap as H3; ultrawide never silently exceeds 1 MP.
        scale = min(short / min(a, b), math.sqrt(1344 * 768 / (a * b)))
        w, h = max(32, round(a * scale / 32) * 32), max(32, round(b * scale / 32) * 32)
        while w * h > 1344 * 768:
            if w > h: w -= 32
            else: h -= 32
    seconds = number(c["seconds"], "Duration", 0.2, 30)
    requested = max(5, math.ceil(seconds * 24 - 1e-8))
    frames = requested + (5 - requested) % 17
    return w, h, frames

def parse_config(raw):
    try:
        supplied = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        raise ValueError("Invalid settings JSON") from None
    if not isinstance(supplied, dict):
        raise ValueError("Settings must be an object")
    c = copy.deepcopy(DEFAULTS)
    c.update({k: v for k, v in supplied.items() if k in DEFAULTS})
    c['mode'] = {'Auto':'auto', 'T2V':'text', 'I2V':'first_last', 'REF':'reference'}.get(c['mode'], c['mode'])
    # Migrate the previous preset once. From v2 onward steps are independent
    # and DMAD is an ordinary editable LoRA, never an implicit model patch.
    if supplied.get('version', 1) < 2:
        if c['profile'] == 'DMAD 4-step' and c['dmad_lora'] and not any(x.get('name') == c['dmad_lora'] for x in c['loras']):
            c['loras'].insert(0, {'name': c['dmad_lora'], 'weight': 1., 'enabled': True})
        if c['profile'] == 'standard': c.update(steps=20, sampler='res_multistep', scheduler='simple')
        if c['resolution'] != 'megapixels':
            w, h, _ = geometry(c)
            c.update(resolution='megapixels', megapixels=max(.1,w*h/(1024*1024)), multiple=32)
        c['llm_vision'] = True
    c.update(version=2, profile='custom', preview_audio=False)
    if c["mode"] not in ("auto", "text", "first_frame", "first_last", "reference"):
        raise ValueError("Unknown H3 mode")
    if c["profile"] not in ("DMAD 4-step", "standard", "custom"):
        raise ValueError("Unknown sampling profile")
    if c["attention"] not in ("pytorch attention", "comfy kitchen attention", "sage", "memory_efficient_sage", "sol-attn"):
        raise ValueError("Unknown attention backend")
    if c["upscale_method"] not in ("DLSS", "Lanczos"):
        raise ValueError("Unknown upscale method")
    if c["ref_image_size"] not in ("match", "max"):
        raise ValueError("Unknown reference image sizing")
    if not isinstance(c["loras"], list) or not isinstance(c["media"], list):
        raise ValueError("LoRA and references must be lists")
    clean_loras = []
    for item in c["loras"]:
        if not isinstance(item, dict): raise ValueError("Invalid LoRA row")
        name = str(item.get("name", ""))
        if name and name != "none":
            clean_loras.append({"name": name, "weight": number(item.get("weight", 1), "LoRA weight", -5, 5),
                                "enabled": bool(item.get("enabled", True))})
    c["loras"] = clean_loras
    for kind, limit in LIMITS.items():
        if sum(x.get("kind") == kind for x in c["media"] if isinstance(x, dict)) > limit:
            raise ValueError(f"Maximum {limit} {kind} references")
    occupied = set()
    counts = {kind:0 for kind in LIMITS}
    for item in c["media"]:
        if not isinstance(item, dict) or item.get("kind") not in LIMITS or not item.get("path"):
            raise ValueError("Invalid reference configuration")
        if Path(item["path"]).suffix.lower() not in EXTENSIONS[item["kind"]]:
            raise ValueError("Reference file extension mismatch")
        kind = item['kind']
        slot = number(item.get('slot', counts[kind]), 'Reference slot', 0, LIMITS[kind]-1, True)
        if (kind, slot) in occupied: raise ValueError('Duplicate reference slot')
        occupied.add((kind, slot)); counts[kind] += 1; item['slot'] = slot
        item["start"] = number(item.get("start", 0), "Reference start", 0, 86400)
        item["end"] = number(item.get("end", 0), "Reference end", 0, 86400)
        if item["end"] and item["end"] <= item["start"]: raise ValueError("Reference end must be after start")
    for key, lo, hi, integer in (
        ("steps", 1, 100, True), ("head_chunks", 1, 56, True), ("ff_chunks", 1, 64, True),
        ("rife_multiplier", 2, 4, True), ("preview_frames", 1, 32, True), ("preview_fps", 1, 24, True),
        ("preview_size", 128, 2048, True), ("upscale_factor", 1.5, 3, False), ("crf", 0, 40, True),
        ("llm_ctx", 8192, 32768, True), ("llm_vram", 1, 24, True), ("llm_tokens", 256, 4096, True),
        ("llm_temperature", 0, 1.5, False), ("shift_video", 1, 30, False), ("shift_audio", 1, 30, False),
        ("sol_tau", 0, 4, False), ("sol_start", 0, 1, False), ("sol_end", 0, 1, False)):
        c[key] = number(c[key], key, lo, hi, integer)
    if c["sol_start"] >= c["sol_end"]: raise ValueError("Sparse attention start must be before end")
    if c["llm_tokens"] >= c["llm_ctx"]: raise ValueError("LLM output budget must be smaller than context")
    geometry(c)
    prefix = str(c["filename"]).strip()
    if not prefix or re.search(r'[<>:"/\\|?*\x00-\x1f]', prefix) or prefix in (".", ".."):
        raise ValueError("Filename contains a path or reserved Windows characters")
    if prefix.split('.')[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1,10)), *(f"LPT{i}" for i in range(1,10))}:
        raise ValueError("Reserved Windows filename")
    c["filename"] = prefix.rstrip(". ")
    if not c['filename']: raise ValueError('Filename must not be empty')
    return c

def route_mode(c, image_count, video_count, audio_count):
    has_media = image_count + video_count + audio_count > 0
    mode = ("reference" if has_media else "text") if c["mode"] == "auto" else c["mode"]
    if mode == "text": return mode  # Disabled references remain stored, but are not used.
    if mode == "reference" and not has_media: raise ValueError("REF requires images, video or audio")
    if mode in ("first_frame", "first_last"):
        if video_count or audio_count: raise ValueError("I2V accepts images only")
        maximum = 1 if mode == "first_frame" else 2
        if not 1 <= image_count <= maximum: raise ValueError(f"This mode requires 1–{maximum} images")
    return mode

def effective_sampling(c):
    return c["steps"], c["sampler"], c["scheduler"]

def output_directory(c, root):
    path = Path(str(c["output_dir"]).strip() or "H3FullStack")
    if not path.is_absolute():
        if ".." in path.parts: raise ValueError("Relative output directory cannot contain ..")
        path = Path(root) / path
    return path.resolve()
