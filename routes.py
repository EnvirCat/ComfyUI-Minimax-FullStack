import json
import asyncio
import uuid
from pathlib import Path
from aiohttp import web
import folder_paths
import nodes
from server import PromptServer
from .config import EXTENSIONS
from .media import probe
from .nodes import defaults, filenames, SAVED_FILES

def local_request(request):
    # This plugin never exposes arbitrary local paths over a network endpoint.
    if request.remote not in ('127.0.0.1', '::1', None):
        raise web.HTTPForbidden(text='H3 FullStack local-only endpoint')

@PromptServer.instance.routes.get('/h3fs/config')
async def config(request):
    from .image_generator import generator_defaults, LEGACY_CARD_TEMPLATE
    from .llm_api import PRESETS, API_DEFAULTS
    return web.json_response({'defaults': defaults(), 'models': {k: filenames(k) for k in
                              ('diffusion_models', 'text_encoders', 'vae', 'vae_approx', 'loras', 'LLM')},
                              'available': list(nodes.NODE_CLASS_MAPPINGS), 'version': '2.9.0',
                              'generator_defaults': generator_defaults(), 'generator_legacy_card_template':LEGACY_CARD_TEMPLATE,
                              'api_presets':PRESETS,'api_defaults':API_DEFAULTS})

@PromptServer.instance.routes.post('/h3fs/assets/enable')
async def enable_assets(request):
    value=await api_body(request)
    if value.get('enabled') is not True: raise web.HTTPBadRequest(text='Explicit enabled=true required')
    target=Path(folder_paths.get_user_directory())/'h3_fullstack'/'assets.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps({'enable_assets':True}),encoding='utf-8');temporary.replace(target)
    finally: temporary.unlink(missing_ok=True)
    return web.json_response({'restart_required':not PromptServer.instance.asset_manager.enabled})

def api_request(request):
    from urllib.parse import urlsplit
    local_request(request)
    host=urlsplit('//'+request.host).hostname
    if host not in ('127.0.0.1','localhost','::1'): raise web.HTTPForbidden(text='Loopback Host required')
    origin=request.headers.get('Origin')
    if origin:
        try:
            parsed=urlsplit(origin)
            valid=parsed.scheme in ('http','https') and parsed.netloc.lower()==request.host.lower()
        except ValueError: valid=False
        if not valid: raise web.HTTPForbidden(text='Cross-origin API settings request denied')
    if request.content_type!='application/json': raise web.HTTPBadRequest(text='JSON request required')
    if request.content_length and request.content_length>16384: raise web.HTTPRequestEntityTooLarge(max_size=16384,actual_size=request.content_length)

async def api_body(request):
    api_request(request)
    body=await request.content.read(16385)
    if len(body)>16384: raise web.HTTPRequestEntityTooLarge(max_size=16384,actual_size=len(body))
    try: value=json.loads(body)
    except (ValueError,UnicodeError): raise web.HTTPBadRequest(text='Invalid API settings JSON')
    if not isinstance(value,dict): raise web.HTTPBadRequest(text='API settings must be an object')
    return value

@PromptServer.instance.routes.post('/h3fs/api/credentials')
async def api_credentials(request):
    from .llm_api import settings, key_status, save_key, clear_key, APIError
    value=await api_body(request)
    try:
        c=settings(value.get('settings',{}));action=value.get('action','status')
        if action=='save': await asyncio.to_thread(save_key,c,value.get('key'))
        elif action=='clear': await asyncio.to_thread(clear_key,c)
        elif action!='status': raise APIError('未知密钥操作')
        result=await asyncio.to_thread(key_status,c)
        return web.json_response(result)
    except APIError as error: return web.json_response({'error':str(error)},status=400)

@PromptServer.instance.routes.post('/h3fs/api/test')
async def api_test(request):
    from .llm_api import completion, settings, APIError
    value=await api_body(request)
    try:
        c=settings(value.get('settings',{}))
        raw,_=await asyncio.to_thread(completion,c,'Reply only with: 连接成功 / Connected','Test connection. Do not call tools.',(),64,0)
        return web.json_response({'ok':True,'reply':raw[:200],'note':'仅验证文本连接；未上传图片、未提交生成队列'})
    except APIError as error: return web.json_response({'error':str(error)},status=400)

@PromptServer.instance.routes.get('/h3fs/image-result/{identifier}')
async def image_result(request):
    local_request(request)
    from .image_generator import read_result
    try: result = read_result(request.match_info['identifier'])
    except ValueError: raise web.HTTPBadRequest(text='Invalid image result ID')
    if result is None: raise web.HTTPNotFound(text='Image result not found')
    return web.json_response(result)

@PromptServer.instance.routes.get('/h3fs/reference-job/{job_id}')
async def reference_job(request):
    local_request(request)
    from .reference_studio import read_job, public_job
    try:
        job=read_job(request.match_info['job_id'])
    except ValueError:
        raise web.HTTPBadRequest(text='Invalid job ID')
    if job is None: raise web.HTTPNotFound(text='Reference job not found')
    return web.json_response(public_job(job))

@PromptServer.instance.routes.post('/h3fs/upload')
async def upload(request):
    local_request(request)
    root = Path(folder_paths.get_input_directory()).resolve() / 'h3_fullstack'
    root.mkdir(parents=True, exist_ok=True)
    reader = await request.multipart()
    part = await reader.next()
    if part is None or not part.filename: raise web.HTTPBadRequest(text='No uploaded file')
    name = Path(part.filename).name
    ext = Path(name).suffix.lower()
    if not any(ext in v for v in EXTENSIONS.values()): raise web.HTTPBadRequest(text='Unsupported file format')
    target = root / (uuid.uuid4().hex + ext)
    try:
        size = 0
        with target.open('xb') as f:
            while chunk := await part.read_chunk(65536):
                size += len(chunk)
                if size > 512 * 1024 * 1024: raise ValueError('Maximum file size is 512MB')
                f.write(chunk)
        meta = probe(str(target))
    except Exception as e:
        target.unlink(missing_ok=True)  # only this upload's newly-created file
        return web.json_response({'error': str(e)}, status=400)
    return web.json_response({**meta, 'path': f'h3_fullstack/{target.name}', 'name': name,
                              'start': 0, 'end': min(15, meta['duration']) if meta['duration'] else 0,
                              'use_audio': False, 'role': ''})

@PromptServer.instance.routes.get('/h3fs/file/{token}')
async def file(request):
    local_request(request)
    path = SAVED_FILES.get(request.match_info['token'])
    if path is None or not path.is_file(): raise web.HTTPNotFound(text='Preview file not found')
    return web.FileResponse(path)
