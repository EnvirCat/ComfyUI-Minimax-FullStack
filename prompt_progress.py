"""Durable prompt checkpoints; failures remain inspectable without approving them."""
from datetime import datetime
import logging


def input_key(c):
    from .reference_studio import signature
    from .media import resolve
    sources=[]
    for x in c.get('media',[]):
        path=resolve(x['path'])
        sources.append([x['kind'],x['slot'],x['path'],path.stat().st_mtime_ns if path.is_file() else None,x.get('start'),x.get('end'),x.get('use_audio')])
    return signature([sorted(sources,key=lambda x:(x[0],x[1])),c['mode'],c['seconds'],c['use_chinese_prompt']])


def publish(m,c,intent,text,owner,stage,attempt=0,warnings=()):
    value={'text':text,'original':intent,'stage':stage,'attempt':attempt,
           'warnings':list(warnings),'optimized':stage=='ready','updated':datetime.now().isoformat(),
           'media_paths':[x['path'] for x in c.get('media',[]) if x.get('path')], 'input_key':input_key(c),
           'mode':c['mode'],'seconds':c['seconds'],'use_chinese_prompt':c['use_chinese_prompt']}
    job_id=c.get('global_reference',{}).get('job_id')
    if job_id:
        try:
            from .reference_studio import read_job,save_job
            job=read_job(job_id)
            if job:
                job['video_prompt']=value;save_job(job)
        except (ValueError,OSError):
            logging.getLogger('H3FullStack').exception('Could not persist video prompt checkpoint')
    m.notify('h3fs_prompt_candidate',owner,**value)
    return value
