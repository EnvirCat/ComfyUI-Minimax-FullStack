"""Private local subprocess entry point; accepts data, never code, over stdin."""
import base64
import importlib
import io
import json
import os
from pathlib import Path
import sys
import types


def main():
    # Keep a dedicated protocol handle. Python prints *and native C stdout* become
    # logs, not protocol, even when a dependency writes directly to fd 1.
    wire=os.fdopen(os.dup(sys.stdout.fileno()),'w',encoding='utf-8')
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno());sys.stdout=sys.stderr
    def emit(kind,**data):
        wire.write('H3FS_NATIVE:'+json.dumps({'type':kind,**data},ensure_ascii=False)+'\n');wire.flush()
    request=json.loads(sys.stdin.readline())
    sys.path.insert(0,request['comfy_root'])
    import folder_paths
    folder_paths.models_dir=request['models_dir'];folder_paths.set_output_directory(request['output_dir'])
    # Import only the required extension modules, not either extension __init__
    # and not the full custom-node startup / server / routes.
    def package(name,directory):
        module=types.ModuleType(name);module.__path__=[str(directory)];sys.modules[name]=module
        return importlib.import_module(name+'.nodes')
    try:
        native=package('h3fs_native_loader',Path(request['comfy_root'])/'custom_nodes'/'ComfyUI-llama-cpp_vlm')
        for key,(paths,extensions) in request['paths'].items():
            folder_paths.folder_names_and_paths[key]=(paths,set(extensions))
        m=package('h3fs_native_fullstack',Path(__file__).parent)
        m.ISOLATE_NATIVE=False
        m.nodes.NODE_CLASS_MAPPINGS['llama_cpp_model_loader']=native.llama_cpp_model_loader
        m.notify=lambda event,owner,**payload:emit('progress',event=event,payload=payload)
        import numpy as np
        import torch
        from PIL import Image
        visual=[torch.from_numpy(np.asarray(Image.open(io.BytesIO(base64.b64decode(v))).convert('RGB')).copy()).float().unsqueeze(0)/255 for v in request['visual']]
        c=request['config'];seed=request['seed'];owner=request['owner']
        if request['task']=='optimize':
            m.optimizer_request=lambda *args:(request['system'],request['user'],visual)
            m.label_manifest=lambda media:request['labels']
            media={k:{name:None for name in values} for k,values in request['media_keys'].items()}
            value=m.optimize(c,request['prompt'],media,request['mode'],seed,owner)
        elif request['task']=='design':
            value=m.design_completion(c,request['system'],request['user'],visual,seed,owner,request['tokens'],schema=request.get('schema'))
        else: raise ValueError('Unknown local inference task')
        # Release *before* result and exit; parent additionally waits for exit.
        m.release_optimizer()
        emit('result',value=value)
    except Exception as error:
        import traceback
        traceback.print_exc();emit('error',value=str(error))
    finally: wire.close()


if __name__=='__main__': main()
