import json
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
    return web.json_response({'defaults': defaults(), 'models': {k: filenames(k) for k in
                              ('diffusion_models', 'text_encoders', 'vae', 'vae_approx', 'loras', 'LLM')},
                              'available': list(nodes.NODE_CLASS_MAPPINGS), 'version': '2.2.0'})

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
