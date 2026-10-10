"""OpenAI-compatible prompt transport; no SDK, credentials or remote error bodies in graphs."""
from __future__ import annotations
import base64
import copy
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PRESETS = {
    'openai': {'label':'OpenAI', 'base_url':'https://api.openai.com/v1', 'model':'', 'auth':True},
    'qwen': {'label':'Qwen / DashScope', 'base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model':'', 'auth':True},
    'gemini': {'label':'Gemini (OpenAI compatible)', 'base_url':'https://generativelanguage.googleapis.com/v1beta/openai', 'model':'', 'auth':True},
    'moonshot': {'label':'Moonshot / Kimi', 'base_url':'https://api.moonshot.ai/v1', 'model':'', 'auth':True},
    'openrouter': {'label':'OpenRouter', 'base_url':'https://openrouter.ai/api/v1', 'model':'', 'auth':True},
    'lmstudio': {'label':'LM Studio', 'base_url':'http://127.0.0.1:1234/v1', 'model':'', 'auth':False},
    'ollama': {'label':'Ollama', 'base_url':'http://127.0.0.1:11434/v1', 'model':'', 'auth':False},
    'custom': {'label':'Custom / 自定义', 'base_url':'', 'model':'', 'auth':True},
}
API_DEFAULTS = {'provider':'openai', 'base_url':PRESETS['openai']['base_url'], 'model':'',
                'vision':False, 'local_vision':True, 'timeout':180, 'token_field':'auto', 'key_env':'', 'send_temperature':True}
_LOCK = threading.RLock()

class APIError(ValueError):
    pass

def endpoint(url):
    if not isinstance(url, str) or len(url)>2048 or re.search(r'[\s\\\x00-\x1f]',url):
        raise APIError('API 地址无效')
    try:
        p=urllib.parse.urlsplit(url); port=p.port
        if not p.hostname or p.username is not None or p.password is not None or p.query or p.fragment:
            raise ValueError()
        if p.scheme not in ('https','http') or p.scheme=='http' and p.hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError()
        if port is not None and not 1<=port<=65535: raise ValueError()
        if any(x in ('.','..') for x in p.path.split('/')): raise ValueError()
    except ValueError:
        raise APIError('API 地址必须为 HTTPS；仅本机回环地址允许 HTTP，不能包含密钥、查询参数或跳转路径') from None
    path=p.path.rstrip('/')
    if not path.endswith('/chat/completions'): path+='/chat/completions'
    return urllib.parse.urlunsplit((p.scheme,p.netloc.lower(),path,'',''))

def settings(value):
    if not isinstance(value,dict): raise APIError('API 设置必须为对象')
    if set(value)-set(API_DEFAULTS): raise APIError('API 设置包含未知字段；密钥不能存入工作流')
    c={**copy.deepcopy(API_DEFAULTS),**value}
    if c['provider'] not in PRESETS: raise APIError('未知 API 供应商')
    if not isinstance(c['base_url'],str): raise APIError('API 地址必须为文本')
    if c['base_url']: endpoint(c['base_url'])
    if not isinstance(c['model'],str) or len(c['model'])>200 or re.search(r'[\x00-\x1f]',c['model']): raise APIError('API 模型名称无效')
    c['model']=c['model'].strip()
    for k in ('vision','local_vision','send_temperature'):
        if not isinstance(c[k],bool): raise APIError(f'{k} 必须为布尔值')
    if isinstance(c['timeout'],bool) or not isinstance(c['timeout'],(int,float)) or not 10<=c['timeout']<=600: raise APIError('API 超时必须为 10–600 秒')
    if c['token_field'] not in ('auto','max_tokens','max_completion_tokens'): raise APIError('未知 API 输出预算字段')
    if not isinstance(c['key_env'],str) or c['key_env'] and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,100}',c['key_env']): raise APIError('密钥环境变量名称无效')
    return c

def _root():
    import folder_paths
    return Path(folder_paths.get_user_directory())/'h3_fullstack'/'credentials'

def _identifier(c):
    return hashlib.sha256((c['provider']+'\n'+endpoint(c['base_url'])).encode()).hexdigest()

class _Blob(ctypes.Structure):
    _fields_=[('cbData',wintypes.DWORD),('pbData',ctypes.POINTER(ctypes.c_ubyte))]

def _dpapi(data,protect):
    if os.name!='nt': raise APIError('此平台不写入密钥文件；请设置密钥环境变量')
    crypt=ctypes.WinDLL('crypt32',use_last_error=True); kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    source=ctypes.create_string_buffer(data); incoming=_Blob(len(data),ctypes.cast(source,ctypes.POINTER(ctypes.c_ubyte))); outgoing=_Blob()
    fn=crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    fn.argtypes=[ctypes.POINTER(_Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(_Blob)]
    fn.restype=wintypes.BOOL; kernel.LocalFree.argtypes=[ctypes.c_void_p]; kernel.LocalFree.restype=ctypes.c_void_p
    try:
        if not fn(ctypes.byref(incoming),None,None,None,None,1,ctypes.byref(outgoing)): raise APIError('Windows 本机密钥加密/解密失败，请重新保存密钥')
        return ctypes.string_at(outgoing.pbData,outgoing.cbData)
    finally:
        ctypes.memset(source,0,len(data))
        if outgoing.pbData:
            ctypes.memset(outgoing.pbData,0,outgoing.cbData);kernel.LocalFree(outgoing.pbData)

def save_key(c,key):
    c=settings(c)
    if not isinstance(key,str) or not 1<=len(key)<=4096 or re.search(r'[\s\x00-\x1f]',key): raise APIError('API 密钥格式无效')
    encrypted=_dpapi(key.encode(),True)
    with _LOCK:
        root=_root();root.mkdir(parents=True,exist_ok=True)
        target=root/(_identifier(c)+'.dpapi');temp=root/(uuid.uuid4().hex+'.tmp')
        try:
            temp.write_bytes(encrypted);os.replace(temp,target)
        finally: temp.unlink(missing_ok=True)

def clear_key(c):
    with _LOCK: (_root()/(_identifier(settings(c))+'.dpapi')).unlink(missing_ok=True)

def key_for(c):
    c=settings(c)
    if c['key_env']:
        key=os.environ.get(c['key_env'],'')
    else:
        with _LOCK:
            path=_root()/(_identifier(c)+'.dpapi')
            key=_dpapi(path.read_bytes(),False).decode() if path.is_file() else ''
    if key and (len(key)>4096 or re.search(r'[\s\x00-\x1f]',key)): raise APIError('密钥格式无效，请重新设置')
    return key

def key_status(c):
    return {'has_key':bool(key_for(c)), 'storage':'windows-dpapi' if os.name=='nt' else 'environment-only'}

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

def image_url(frame):
    from PIL import Image
    import numpy as np
    if hasattr(frame,'detach'): frame=frame.detach().cpu().numpy()
    pixels=np.asarray(frame)
    if pixels.ndim==4: pixels=pixels[0]
    if pixels.ndim!=3 or pixels.shape[-1] not in (3,4): raise APIError('API 视觉输入必须为 RGB 图片')
    if pixels.dtype!=np.uint8: pixels=np.clip(pixels*255,0,255).astype(np.uint8)
    img=Image.fromarray(pixels).convert('RGB');img.thumbnail((768,768))
    out=io.BytesIO();img.save(out,format='JPEG',quality=90)
    return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode('ascii')

def completion(c,system,user,visual=(),tokens=1536,temperature=.65,progress=None,cancel=None,schema=None):
    c=settings(c)
    if not c['model'] or not c['base_url']: raise APIError('请填写 API 模型名称和接口地址')
    if visual and not c['vision']: raise APIError('该 API 未启用看图；请开启 API 看图或本地视觉分析')
    key=key_for(c)
    if not key and PRESETS[c['provider']]['auth']: raise APIError('尚未设置此接口的 API 密钥')
    if schema: system+='\nReturn only a JSON object matching this schema, without Markdown:\n'+json.dumps(schema,ensure_ascii=False)
    content=([{'type':'text','text':user}]+[{'type':'image_url','image_url':{'url':image_url(x)}} for x in visual]) if visual else user
    payload={'model':c['model'],'messages':[{'role':'system','content':system},{'role':'user','content':content}], 'stream':True}
    token_field=c['token_field']
    if token_field=='auto': token_field='max_completion_tokens' if c['provider']=='openai' else 'max_tokens'
    payload[token_field]=int(tokens)
    # Reasoning-only details never enter UI/logs; these families reject temperature.
    fixed_temperature=(c['provider']=='openai' and re.match(r'^(o[134](?:-|$)|gpt-[56](?:\.|-|$))',c['model'])) or (c['provider']=='moonshot' and re.match(r'^kimi-k[23]',c['model']))
    if c['send_temperature'] and not fixed_temperature: payload['temperature']=temperature
    headers={'Content-Type':'application/json','Accept':'text/event-stream'}
    if key: headers['Authorization']='Bearer '+key
    request=urllib.request.Request(endpoint(c['base_url']),data=json.dumps(payload,ensure_ascii=False).encode(),headers=headers,method='POST')
    if cancel: cancel()
    started=time.monotonic();pieces=[];finish=None;size=0;events=0
    def consume(part):
        nonlocal finish,events
        choices=part.get('choices',[])
        if not choices: return
        choice=choices[0];delta=choice.get('delta',choice.get('message',{}))
        if delta.get('refusal'): raise APIError('API 未接受该请求')
        text=delta.get('content')
        if isinstance(text,str): pieces.append(text)
        finish=choice.get('finish_reason') or finish;events+=1
        if progress and events%32==0: progress(events)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request,timeout=min(c['timeout'],60)) as response:
            if response.headers.get('Content-Type','').split(';')[0].strip()=='text/event-stream':
                data=[]
                for line in response:
                    if cancel: cancel()
                    if time.monotonic()-started>c['timeout']: raise APIError('API 请求总时长超过限制，请稍后重试')
                    size+=len(line)
                    if size>16*1024*1024: raise APIError('API 响应超出安全上限')
                    line=line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'): data.append(line[5:].lstrip())
                    elif not line and data:
                        event='\n'.join(data);data=[]
                        if event=='[DONE]': break
                        consume(json.loads(event))
                if data and '\n'.join(data)!='[DONE]': consume(json.loads('\n'.join(data)))
            else:
                body=response.read(16*1024*1024+1)
                if len(body)>16*1024*1024: raise APIError('API 响应超出安全上限')
                consume(json.loads(body))
    except urllib.error.HTTPError as error:
        status=error.code;error.close()
        reason={401:'密钥无效或未授权',403:'接口权限不足',404:'地址或模型未找到',429:'配额不足或请求过于频繁'}.get(status,'服务错误或接口不兼容')
        raise APIError(f'API HTTP {status}：{reason}。请检查设置；没有自动重发收费请求') from None
    except (urllib.error.URLError,TimeoutError,OSError):
        raise APIError('API 连接失败或超时，请检查地址/网络；没有自动重发收费请求') from None
    except (ValueError,KeyError,TypeError,UnicodeError,AttributeError) as error:
        if isinstance(error,APIError): raise
        raise APIError('API 返回格式异常，需 OpenAI Chat Completions 兼容接口') from None
    raw=re.sub(r'<think>.*?(?:</think>|$)','', ''.join(pieces),flags=re.S|re.I).strip()
    if not finish: raise APIError('API 流未完整结束，草稿未应用，请重试')
    if finish=='length': raise APIError('API 输出被截断，请提高输出预算或检查模型的思考预算')
    if finish!='stop' or not raw: raise APIError('API 没有返回完整文本，草稿未应用')
    return raw,{'finish_reason':finish,'backend':'api','provider':c['provider'],'model':c['model'],'max_tokens':tokens}
