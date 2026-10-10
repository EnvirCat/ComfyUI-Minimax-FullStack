"""Explicit opt-in live export smoke test; no H3 sampling or private media.

Run with ComfyUI's Python: validate_video_export.py --submit
Requires this extension, native EmptyImage/EmptyAudio/SaveVideo and RIFE VFI.
Creates tiny neutral videos in output/H3FullStack/export-smoke.
"""
import argparse
import json
import time
import urllib.request


def request(base, path, body=None):
    data=None if body is None else json.dumps(body).encode('utf-8')
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--base',default='http://127.0.0.1:8188')
    args=parser.parse_args()
    if not args.submit:
        parser.error('Pass --submit to intentionally queue the tiny export test.')
    queue=request(args.base,'/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Queue must be empty; no active job will be interrupted.')
    settings={'rife':True,'rife_multiplier':2,'rife_model':'rife47.pth',
              'upscale':True,'upscale_method':'Lanczos','upscale_factor':2,
              'output_dir':'H3FullStack/export-smoke','filename':'export-smoke'}
    graph={
        '1':{'class_type':'EmptyImage','inputs':{'width':128,'height':128,'batch_size':5,'color':8421504}},
        '2':{'class_type':'EmptyAudio','inputs':{'duration':1.0,'sample_rate':48000,'channels':2}},
        '3':{'class_type':'H3FSFinish','inputs':{'images':['1',0],'audio':['2',0],
            'settings':json.dumps(settings),'prompt':'Neutral export pipeline test','seed':int(time.time()),'owner_id':'3'}},
        '7':{'class_type':'SaveVideo','inputs':{'video':['3',0],'filename_prefix':'H3FullStack/export-smoke/attached-base',
            'format':'auto','format.codec':'auto'}},
        '8':{'class_type':'SaveVideo','inputs':{'video':['3',1],'filename_prefix':'H3FullStack/export-smoke/attached-upscaled',
            'format':'auto','format.codec':'auto'}},
    }
    reply=request(args.base,'/prompt',{'prompt':graph,'client_id':'h3fs-export-smoke'})
    prompt_id=reply['prompt_id']
    print(json.dumps({'prompt_id':prompt_id},ensure_ascii=False),flush=True)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        history=request(args.base,'/history/'+prompt_id)
        if prompt_id in history:
            item=history[prompt_id]
            print(json.dumps({'status':item['status'],'outputs':item['outputs']},ensure_ascii=False),flush=True)
            if item['status']['status_str']!='success':
                raise RuntimeError('Export smoke test failed')
            for node in ('3','7','8'):
                if not item['outputs'].get(node,{}).get('images'):
                    raise RuntimeError('Missing standard saved-video entry: '+node)
            return
        time.sleep(2)
    raise TimeoutError('Export smoke test did not finish within 180 seconds')


if __name__=='__main__':
    main()
