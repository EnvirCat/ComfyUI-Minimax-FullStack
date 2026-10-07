"""Bounded local media decoding. Videos are explicitly resampled to H3's 24 fps."""
from pathlib import Path
import math
import av
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths
from .config import EXTENSIONS

def resolve(path):
    p = Path(path)
    if not p.is_absolute():
        if '..' in p.parts: raise ValueError('Reference path cannot escape the input directory')
        p = Path(folder_paths.get_input_directory()) / p
    p = p.resolve()
    if not p.is_file(): raise FileNotFoundError(f'Reference file not found: {p}')
    return p

def probe(path):
    p = resolve(path)
    kind = next((k for k, exts in EXTENSIONS.items() if p.suffix.lower() in exts), None)
    if not kind: raise ValueError('Unsupported media format')
    if kind == 'image':
        with Image.open(p) as img:
            w, h = ImageOps.exif_transpose(img).size
        return {'kind': kind, 'width': w, 'height': h, 'duration': 0, 'has_audio': False}
    with av.open(str(p)) as c:
        duration = float(c.duration / av.time_base) if c.duration else 0
        for s in c.streams:
            if s.duration and s.time_base: duration = max(duration, float(s.duration * s.time_base))
        if not duration or not math.isfinite(duration): raise ValueError('Cannot read media duration')
        v = c.streams.video[0] if c.streams.video else None
        if kind == 'video' and v is None: raise ValueError('File has no video track')
        if kind == 'audio' and not c.streams.audio: raise ValueError('File has no audio track')
        return {'kind': kind, 'duration': duration, 'has_audio': bool(c.streams.audio),
                'width': v.width if v else 0, 'height': v.height if v else 0}

def window(item):
    meta = probe(item['path'])
    start = float(item.get('start', 0))
    end = min(meta['duration'], float(item.get('end', 0)) or start + 15)
    if not 0.2 <= end - start <= 15.001:
        raise ValueError('Each reference clip must be 0.2–15 seconds; end 0 selects up to 15 seconds')
    return start, end, meta

def image_tensor(item):
    with Image.open(resolve(item['path'])) as img:
        img = ImageOps.exif_transpose(img).convert('RGB')
        img.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255).unsqueeze(0)

def audio_tensor(item):
    start, end, _ = window(item)
    chunks = []
    rate = 48000
    with av.open(str(resolve(item['path']))) as c:
        if not c.streams.audio: raise ValueError('Media has no audio track')
        s = c.streams.audio[0]
        resampler = av.AudioResampler(format='fltp', layout='stereo', rate=rate)
        for frame in c.decode(s):
            ts = float(frame.pts * frame.time_base) if frame.pts is not None else 0
            if ts >= end: break
            for out in resampler.resample(frame):
                at = float(out.pts * out.time_base) if out.pts is not None else ts
                arr = out.to_ndarray()
                left = max(0, round((start - at) * rate))
                right = min(arr.shape[-1], round((end - at) * rate))
                if right > left: chunks.append(arr[:, left:right].copy())
    if not chunks: raise ValueError('Selected segment contains no audio')
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0).float()
    return {'waveform': waveform, 'sample_rate': rate}

def video_tensor(item, max_frames):
    start, end, meta = window(item)
    total = min(max_frames, max(5, math.ceil((end - start) * 24)))
    desired = start + np.arange(total) / 24
    result, current = [], None
    w, h = meta['width'], meta['height']
    scale = min(1., math.sqrt(1344 * 768 / (w * h)))
    target = (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32))
    with av.open(str(resolve(item['path']))) as c:
        s = c.streams.video[0]
        for frame in c.decode(s):
            at = float(frame.pts * frame.time_base) if frame.pts is not None else float(frame.time or 0)
            if at < start - 1 / 24: continue
            if current is None: current = frame.to_image().convert('RGB').resize(target, Image.Resampling.LANCZOS)
            while len(result) < total and desired[len(result)] <= at:
                result.append(np.array(current, dtype=np.float32) / 255)
            current = frame.to_image().convert('RGB').resize(target, Image.Resampling.LANCZOS)
            if len(result) >= total or at >= end: break
    if current is None: raise ValueError('Selected segment contains no video frames')
    while len(result) < total: result.append(np.array(current, dtype=np.float32) / 255)
    return torch.from_numpy(np.stack(result)), audio_tensor(item) if item.get('use_audio') and meta['has_audio'] else None

def prepare(c, frames, reference_image=None, reference_video=None, reference_audio=None, image_inputs=None):
    images, videos, paired, audios, descriptions = {}, {}, {}, {}, []
    if c['mode'] == 'text':
        return {'images':images, 'videos':videos, 'paired':paired, 'audios':audios, 'descriptions':descriptions}
    image_inputs = image_inputs or {}
    maximum = 1 if c['mode'] == 'first_frame' else 2 if c['mode'] == 'first_last' else 9
    for kind in ('image', 'video', 'audio'):
        if c['mode'] in ('first_frame','first_last') and kind != 'image': continue
        slots = {x['slot']:x for x in c['media'] if x['kind'] == kind}
        if kind == 'image':
            for i in range(maximum):
                tensor = image_inputs.get(f'image_{i+1}')
                if tensor is None and i not in slots: continue
                item = slots.get(i)
                tensor = image_tensor(item) if tensor is None else tensor
                # A connected IMAGE batch fills consecutive active image slots.
                for frame in tensor:
                    if len(images) >= maximum: raise ValueError(f'This mode supports at most {maximum} images')
                    index = len(images); images[f'ref_image_{index}'] = frame.unsqueeze(0)
                    descriptions.append({'kind':'image', 'name':Path(item['path']).name if item else f'Image {i+1}',
                                         'role':str(item.get('role','')) if item else ''})
            continue
        for _, item in sorted(slots.items()):
            target = {'image': images, 'video': videos, 'audio': audios}[kind]
            index = len(target)
            slot = f'ref_{kind}_{index}'
            if kind == 'image': target[slot] = image_tensor(item)
            elif kind == 'audio': target[slot] = audio_tensor(item)
            else:
                target[slot], soundtrack = video_tensor(item, frames)
                if soundtrack is not None: paired[f'ref_video_audio_{index}'] = soundtrack
            descriptions.append({'kind': kind, 'label': f'<{dict(image="Picture",video="Video",audio="Audio")[kind]} {index + 1}>',
                                 'name': Path(item['path']).name, 'role': str(item.get('role', ''))})
    if reference_image is not None:
        for frame in reference_image:
            i = len(images); images[f'ref_image_{i}'] = frame.unsqueeze(0)
    if reference_video is not None and c['mode'] not in ('first_frame','first_last'):
        from comfy_extras.nodes_video import GetVideoComponents
        vf, va, fps = GetVideoComponents.execute(reference_video)[:3]
        count = min(frames, math.ceil(len(vf) * 24 / float(fps)))
        indices = (torch.arange(count) * float(fps) / 24).long().clamp(max=len(vf) - 1)
        i = len(videos); videos[f'ref_video_{i}'] = vf[indices]
        if va is not None: paired[f'ref_video_audio_{i}'] = va
    if reference_audio is not None and c['mode'] not in ('first_frame','first_last'): audios[f'ref_audio_{len(audios)}'] = reference_audio
    if len(images) > maximum or len(videos) > 3 or len(audios) > 2:
        raise ValueError(f'Reference limit: {maximum} images, 3 videos, 2 audio files')
    return {'images': images, 'videos': videos, 'paired': paired, 'audios': audios, 'descriptions': descriptions}

def label_manifest(media):
    lines = []
    audio_index = 0
    def role(kind, index):
        descriptions=[x for x in media['descriptions'] if x['kind']==kind]
        if index>=len(descriptions): return ''
        item=descriptions[index]
        return f"; file {item['name']}; user-specified role: {item['role']}" if item['role'] else f"; file {item['name']}"
    for i in range(len(media['images'])): lines.append(f'<Picture {i+1}>: image reference'+role('image',i))
    for i, k in enumerate(media['videos']):
        if f'ref_video_audio_{k.rsplit("_",1)[-1]}' in media['paired']:
            audio_index += 1
            lines.append(f'<Audio {audio_index}>: soundtrack paired with <Video {i+1}>')
        lines.append(f'<Video {i+1}>: video reference, not preserved output frames'+role('video',i))
    for i,_ in enumerate(media['audios']):
        audio_index += 1; lines.append(f'<Audio {audio_index}>: standalone audio reference'+role('audio',i))
    return '\n'.join(lines) or 'No references. Text-to-video only.'
