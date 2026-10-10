"""Contain native llama/MTMD faults; never share its CUDA context with ComfyUI."""
from __future__ import annotations
import base64
from collections import deque
import io
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

PREFIX='H3FS_NATIVE:'
LOG=logging.getLogger('H3FullStack')


def encode_visual(frames):
    from PIL import Image
    import numpy as np
    result=[]
    for frame in frames:
        values=frame[0].detach().cpu().numpy()
        stream=io.BytesIO()
        Image.fromarray(np.clip(values*255,0,255).astype(np.uint8)).save(stream,format='PNG')
        result.append(base64.b64encode(stream.getvalue()).decode('ascii'))
    return result


def watch_process(process,payload,notify,interrupted,timeout=900):
    """Drain both pipes, forward progress, wait for *exit*, and cancel only our child."""
    events=queue.Queue();tail=deque(maxlen=12);result=None;failure=None;closed=set()
    def reader(stream,kind):
        try:
            for line in stream: events.put((kind,line.rstrip()))
        finally:
            events.put((kind,None))
    threads=[threading.Thread(target=reader,args=(process.stdout,'out'),daemon=True),
             threading.Thread(target=reader,args=(process.stderr,'err'),daemon=True)]
    for thread in threads: thread.start()
    started=time.monotonic();repeated=0
    try:
        process.stdin.write(json.dumps(payload,ensure_ascii=False)+'\n');process.stdin.close()
        while process.poll() is None or len(closed)<2:
            interrupted()
            if time.monotonic()-started>timeout: raise RuntimeError('本地提示词模型超时，已停止隔离任务；ComfyUI 与已完成素材保留，可重试。')
            try: kind,line=events.get(timeout=.25)
            except queue.Empty: continue
            if line is None: closed.add(kind);continue
            if kind=='out' and line.startswith(PREFIX):
                message=json.loads(line[len(PREFIX):])
                if message['type']=='result': result=message['value']
                elif message['type']=='error': failure=message['value']
                elif message['type']=='progress': notify(message['event'],message['payload'])
            elif line:
                tail.append(line)
                # Native hybrid attention can emit thousands of identical warnings.
                # Preserve a sample and count, without burying useful backend logs.
                if 'find_slot: non-consecutive' in line:
                    repeated+=1
                    if repeated==1: LOG.warning('本地模型（隔离进程）: %s',line)
                else: LOG.info('本地模型: %s',line)
        code=process.wait()
        if repeated: LOG.warning('本地模型非连续 token 警告共 %s 次，已隔离处理',repeated)
        if code!=0:
            LOG.error('本地模型子进程异常退出：%s；最近日志：%s',code,'\n'.join(tail))
            raise RuntimeError(f'本地提示词模型进程异常退出（代码 {code}），已隔离，ComfyUI 不会因此退出。已完成素材保留，请重试当前步骤。')
        if failure: raise ValueError(failure)
        if result is None: raise RuntimeError('本地模型未返回完整结果；已停止本步骤，不会继续生成视频。')
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
        for thread in threads: thread.join(timeout=2)
        for stream in (process.stdout,process.stderr): stream.close()


def run_native(task,c,request,seed,owner,**extra):
    from . import nodes as m
    import folder_paths
    payload={'task':task,'config':c,'system':request[0],'user':request[1],
             'visual':encode_visual(request[2]),'seed':seed,'owner':str(owner),
             'comfy_root':str(Path(folder_paths.__file__).resolve().parent),
             'models_dir':folder_paths.models_dir,'output_dir':folder_paths.get_output_directory(),
             'paths':{k:[list(paths),list(extensions)] for k,(paths,extensions) in folder_paths.folder_names_and_paths.items()},**extra}
    # Serial GPU ownership; no native LLM is ever loaded in the parent on this path.
    with m.LLM_LOCK:
        m.mm.unload_all_models();m.gc.collect();m.mm.soft_empty_cache()
        m.notify('h3fs_status',owner,text='正在启动隔离的本地提示词模型…')
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        process=subprocess.Popen([sys.executable,'-u','-s',str(Path(__file__).with_name('native_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',errors='replace',
            cwd=payload['comfy_root'],env={**os.environ,'PYTHONUTF8':'1'},**options)
        return watch_process(process,payload,lambda event,data:m.notify(event,owner,**data),m.mm.throw_exception_if_processing_interrupted)
