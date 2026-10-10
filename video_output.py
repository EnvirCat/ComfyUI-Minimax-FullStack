"""Bounded frame-by-frame MP4 writing, including optional streaming Lanczos."""
from fractions import Fraction
from pathlib import Path
import math
import uuid
import av
import numpy as np
from PIL import Image

def encode_mp4(images,audio,fps,path,crf,progress=None,cancel=None,resize=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.partial-'+uuid.uuid4().hex+'.mp4')
    width,height=resize or (images.shape[2],images.shape[1]);count=len(images)
    rate=Fraction(str(fps));waveform=None
    try:
        with av.open(str(temporary),mode='w',format='mp4',options={'movflags':'+faststart'}) as container:
            stream=container.add_stream('libx264',rate=rate);stream.width=width;stream.height=height;stream.pix_fmt='yuv420p'
            stream.options={'crf':str(crf),'preset':'medium'}
            # Match Comfy's SDR BT.709 output; this path handles 8-bit RGB only.
            stream.codec_context.color_primaries=1;stream.codec_context.color_trc=1;stream.codec_context.colorspace=1;stream.codec_context.color_range=1
            audio_stream=None
            if audio is not None:
                sample_rate=int(audio['sample_rate']);waveform=audio['waveform'][0,:,:math.ceil(sample_rate*count/rate)].detach().float().cpu().contiguous().numpy()
                layout={1:'mono',2:'stereo',6:'5.1'}.get(waveform.shape[0])
                if not layout: raise ValueError('Unsupported audio channel count')
                audio_stream=container.add_stream('aac',rate=sample_rate,layout=layout)
            for index,pixels in enumerate(images):
                if cancel: cancel()
                # NumPy avoids tiny CPU Torch operations launching a full thread pool
                # for every frame. Never materialize the entire upscaled float batch.
                pixels=pixels.detach().cpu().numpy()
                rgb=np.clip(pixels*255,0,255).astype(np.uint8)
                if resize: rgb=np.asarray(Image.fromarray(rgb).resize(resize,Image.Resampling.LANCZOS))
                frame=av.VideoFrame.from_ndarray(rgb,format='rgb24').reformat(format='yuv420p',dst_colorspace=av.video.reformatter.Colorspace.ITU709)
                frame.color_primaries=1;frame.color_trc=1;frame.colorspace=1;frame.color_range=1
                frame.pts=index;frame.time_base=1/rate
                for packet in stream.encode(frame): container.mux(packet)
                if progress and (index==0 or (index+1)%24==0 or index+1==count): progress(index+1,count)
            for packet in stream.encode(None): container.mux(packet)
            if audio_stream is not None and waveform.shape[-1]:
                frame=av.AudioFrame.from_ndarray(waveform,format='fltp',layout=layout);frame.sample_rate=sample_rate;frame.pts=0
                for packet in audio_stream.encode(frame): container.mux(packet)
                for packet in audio_stream.encode(None): container.mux(packet)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # Only the new partial file owned by this call.

def saved_entries(files,root):
    root=Path(root).resolve();entries=[]
    for file in files:
        path=Path(file['path']).resolve()
        if path.is_relative_to(root) and path.is_file():
            entries.append({'filename':path.name,'subfolder':path.parent.relative_to(root).as_posix(),'type':'output'})
    return entries
