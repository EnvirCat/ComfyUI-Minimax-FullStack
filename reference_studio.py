"""Serial reference production with durable drafts/results and explicit stages."""
from __future__ import annotations
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import folder_paths
from comfy_execution.graph_utils import GraphBuilder, ExecutionBlocker
from .config import parse_config
from .reference_config import image_size, validate_designed_prompt, prompt_units
from .reference_prompts import (CONTRACT_VERSION, STYLE_SCHEMA, STYLE_SYSTEM, IMAGE_SYSTEM,
                                parse_style, medium_description, asset_contract, validate_anchor, content_reference, scene_scale_rule)
from .reference_learning import learn_profile, override_for, compose_prompt, character_trigger, triggers_for, validate_no_face
from .media import image_tensor


def runtime():
    from . import nodes
    return nodes


def job_root():
    return Path(folder_paths.get_output_directory()) / 'H3FullStack' / 'reference-studio'


def job_path(job_id):
    if not re.fullmatch(r'[a-f0-9]{32}',job_id): raise ValueError('Invalid reference job ID')
    return job_root() / '.jobs' / (job_id+'.json')


def read_job(job_id):
    path = job_path(job_id)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def save_job(job):
    path = job_path(job['id'])
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_job(job_id, intent):
    job = read_job(job_id)
    if job: return job
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', str(intent)).strip('. ')[:24] or '参考素材'
    if name.split('.')[0].upper() in ('CON','PRN','AUX','NUL'): name='项目-'+name
    return {'id':job_id, 'directory':f'{datetime.now():%Y%m%d-%H%M%S}-{name}-{job_id[:6]}',
            'intent':intent,'cards':{},'style':'','style_key':'','status':'ready'}


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def register_file(path):
    m=runtime()
    token=uuid.uuid4().hex
    m.SAVED_FILES[token]=Path(path)
    m.persist_preview_index()
    return '/h3fs/file/'+token


def public_job(job):
    # Preview URLs are re-registered after a backend restart; no arbitrary path route.
    result = json.loads(json.dumps(job))
    for row in result['cards'].values():
        if row.get('path') and Path(row['path']).is_file(): row['preview_url']=register_file(row['path'])
    return result


def progress(job, owner, text, done=0, total=1, stage='design'):
    job.update(status=text,stage=stage,done=done,total=total)
    save_job(job)
    runtime().notify('h3fs_reference_progress',owner,job_id=job['id'],text=text,done=done,total=total,stage=stage)


def effective_prompt(c,card,row):
    base=card['text'] if not card['design'] else card['draft'] or row.get('draft',row.get('prompt',''))
    kind=row.get('kind',asset_contract(card)[0])
    return compose_prompt(base,row.get('reference_description',''),kind,card.get('camera',''),c['use_chinese_prompt'],triggers_for(c,kind),scene_scale_rule(card,c['use_chinese_prompt']))


def design(c, intent, job, owner, allow_design=True,learn_only=False):
    m=runtime()
    cards=[row for row in c['reference_cards'] if row['enabled']]
    # Resolve through the same bounded image reader. The style file is not an H3 reference label.
    from .media import resolve
    # Both Krea2 and Qwen keep their native text-to-image paths.
    for card in cards:
        reference,_=content_reference(c,card)
        if reference:
            resolve(reference)
    style_key=signature([5,c['style_text'],c['style_image'],c['style_medium'],c['style_override'],
                         resolve(c['style_image']).stat().st_mtime_ns if c['style_image'] else 0])
    if allow_design and job.get('style_key') != style_key:
        style=c['style_text'].strip()
        report={}
        if c['style_override'].strip():
            style=c['style_override'].strip()+ ('\n用户画风约束：'+style if style else '')
            if c['style_medium']!='auto': style=medium_description(c['style_medium'],c['use_chinese_prompt'])+'；'+style
        elif c['style_image'] and (learn_only or any(row['design'] for row in cards)):
            progress(job,owner,'正在分析画风参考…',0,len(cards)+1,'style')
            caption=m.design_completion(c,STYLE_SYSTEM,
                '先判断实际可见媒介并给出证据；柔焦不等于插画。仅提取可复用的光线、色调、光学质感和景深，不提取人物、衣物或物品材质。 /no_think',
                [image_tensor({'path':c['style_image']})],c['image_seed'],owner,768,schema=STYLE_SCHEMA)
            report=parse_style(caption,c['style_medium'])
            style=medium_description(report['medium'],c['use_chinese_prompt'])+'；'+report['style']+ ('\n用户画风约束：'+style if style else '')
        elif c['style_medium']!='auto':
            style=medium_description(c['style_medium'],c['use_chinese_prompt'])+'；'+style
        job.update(style=style,style_key=style_key,style_report=report,contract_version=CONTRACT_VERSION)
        save_job(job)
    shared=job.setdefault('shared_profiles',{})
    person_needed=learn_only or any(asset_contract(card)[0]=='person' for card in cards)
    scene_needed=learn_only or any(asset_contract(card)[0]=='scene' for card in cards)
    if c['strict_person_reference'] and person_needed:
        if not c['style_image']: raise ValueError('请先上传人物参考图，再开启人物严格参考')
        shared['person']=learn_profile(m,c,job,c['style_image'],'person',owner,progress,c['person_profile_override'])
    if c['scene_reference'] and scene_needed:
        shared['scene']=learn_profile(m,c,job,c['scene_reference'],'scene',owner,progress,c['scene_profile_override'])
    if learn_only:
        for card in cards:
            reference,kind=content_reference(c,card)
            if reference: learn_profile(m,c,job,reference,kind,owner,progress,override_for(c,card,kind))
        job['contract_version']=CONTRACT_VERSION
        progress(job,owner,'参考学习完成，可修改人物 / 场景设定；尚未生图',stage='review')
        m.notify('h3fs_reference_result',owner,job=public_job(job))
        return
    for i,card in enumerate(cards):
        m.mm.throw_exception_if_processing_interrupted()
        key=str(card['slot']); old=job['cards'].get(key,{})
        kind,constraint=asset_contract(card)
        reference,ref_kind=content_reference(c,card)
        reference_time=resolve(reference).stat().st_mtime_ns if reference else 0
        profile=learn_profile(m,c,job,reference,ref_kind,owner,progress,override_for(c,card,ref_kind))['description'] if reference else ''
        design_key=signature([CONTRACT_VERSION,intent,card['text'],kind,job['style'],c['use_chinese_prompt'],image_size(c,card),reference,reference_time,profile,card['camera']])
        if not card['text'].strip(): raise ValueError(f'请填写第 {card["slot"]+1} 张参考图的物象名称或原文提示词')
        if not card['design']:
            prompt=card['text']  # Verbatim means EXACT, including whitespace; never append inferred style.
        elif card['draft'].strip():
            prompt=validate_anchor(validate_designed_prompt(card['draft'],c['use_chinese_prompt']),card)
        elif old.get('design_key') == design_key and old.get('draft'):
            prompt=old['draft']
        else:
            if not allow_design: raise ValueError(f'第 {card["slot"]+1} 张图需要先设计提示词')
            progress(job,owner,f'第 {card["slot"]+1} 张素材图片提示词正在生成…',i,len(cards),'design')
            system=IMAGE_SYSTEM+(' Write Simplified Chinese.' if c['use_chinese_prompt'] else ' Write English.')
            reserved=prompt_units(compose_prompt('',profile,ref_kind,card['camera'],c['use_chinese_prompt'],triggers_for(c,kind),scene_scale_rule(card,c['use_chinese_prompt'])))
            limit=max(40,512-reserved)
            if profile: system+=' A learned TEXT subject profile will be appended by the program. Only design the scene, composition and pose; do not redefine the appearance or wardrobe. Do not copy the source background, pose or view.'
            w,h=image_size(c,card)
            user=f'唯一主角 / 目标物象：{card["text"]}\n素材类型：{kind}\n强约束：{constraint}\n画风：{job["style"] or "自然、与意图相符"}\n画幅：{w}×{h}\n仅作氛围背景的信息，不能覆盖上述主角和构图：{intent}\n只设计这一张独立素材。 /no_think'
            visual=[]
            user+=f'\n正文最多 {limit} 中文字 / 英文词，参考设定由程序另行追加。'
            if profile: user+='\n可复用参考设定（排除参考背景 / 姿态 / 视角）：'+profile
            if card['camera']: user+='\n用户指定拍摄视角：'+card['camera']
            else: user+='\n拍摄视角：根据物象与意图灵活安排，不固定为参考图视角。'
            prompt=m.design_completion(c,system,user,visual,c['image_seed']+card['slot'],owner,1024)
            try:
                if kind=='person': validate_no_face(prompt)
                prompt=validate_anchor(validate_designed_prompt(prompt,c['use_chinese_prompt']),card)
                validate_designed_prompt(compose_prompt(prompt,profile,ref_kind,card['camera'],c['use_chinese_prompt'],triggers_for(c,kind),scene_scale_rule(card,c['use_chinese_prompt'])),c['use_chinese_prompt'])
            except ValueError as error:
                progress(job,owner,f'第 {card["slot"]+1} 张提示词偏离物象，正在纠正…',i,len(cards),'design')
                corrected=user+'\n上次输出不合格：'+str(error)+'\n请重新写，首句明确物象；独立场景/物品不加入人物。'
                prompt=m.design_completion(c,system,corrected,visual,c['image_seed']+card['slot']+101,owner,1024)
                if kind=='person': validate_no_face(prompt)
                prompt=validate_anchor(validate_designed_prompt(prompt,c['use_chinese_prompt']),card)
        final=compose_prompt(prompt,profile,ref_kind,card['camera'],c['use_chinese_prompt'],triggers_for(c,kind),scene_scale_rule(card,c['use_chinese_prompt']))
        if card['design']: validate_designed_prompt(final,c['use_chinese_prompt'])
        job['cards'][key]={**old,'slot':card['slot'],'kind':kind,'draft':prompt,'prompt':final,'design_key':design_key,'text':card['text'],
                           'content_reference':reference,'reference_kind':ref_kind,'reference_description':profile,'strict_reference':bool(reference)}
        job['contract_version']=CONTRACT_VERSION
        save_job(job)
    progress(job,owner,'图片提示词已就绪，可修改后再生图',len(cards),len(cards),'review')
    m.notify('h3fs_reference_result',owner,job=public_job(job))


def generation_key(c,card,prompt):
    from .media import resolve
    reference,kind=content_reference(c,card)
    version=resolve(reference).stat().st_mtime_ns if reference else 0
    return signature([CONTRACT_VERSION,prompt,card['text'],card.get('kind','auto'),card.get('camera',''),reference,kind,version,override_for(c,card,kind),c['style_text'],c['style_image'],c['style_medium'],c['style_override'],image_size(c,card),c['image_engine'],c['image_model'],c['image_clip'],c['image_vae'],
                      c['image_steps'],c['image_cfg'],c['image_loras'],c['image_seed'],card['revision']])


def result_bundle(job,c):
    generated_slots={card['slot'] for card in c['reference_cards'] if card['enabled']}
    media=[dict(x) for x in c['media'] if x['kind']=='image' and x['slot'] not in generated_slots]
    for card in c['reference_cards']:
        row=job['cards'].get(str(card['slot']),{})
        if card['enabled'] and row.get('path') and Path(row['path']).is_file():
            media.append({'kind':'image','slot':card['slot'],'path':row['path'],
                          'name':Path(row['path']).name,'preview_url':register_file(row['path']),'start':0,'end':0})
    ready=all(any(x['slot']==card['slot'] for x in media) and
              job['cards'][str(card['slot'])].get('text')==card['text'] and
              job['cards'][str(card['slot'])].get('generation_key')==generation_key(c,card,
                  effective_prompt(c,card,job['cards'][str(card['slot'])]))
              for card in c['reference_cards'] if card['enabled'])
    return {'media':media,'job_id':job['id'],'ready':ready}


def output_images(job):
    """Native Comfy output contract: history, Assets registration and image viewers."""
    base=Path(folder_paths.get_output_directory()).resolve()
    result=[]
    for row in sorted(job['cards'].values(),key=lambda x:x['slot']):
        if not row.get('path'): continue
        path=Path(row['path']).resolve()
        if not path.is_file() or not path.is_relative_to(base): continue
        result.append({'filename':path.name,'subfolder':path.parent.relative_to(base).as_posix(),'type':'output'})
    return result


def image_graph(graph,c,card,prompt,owner):
    m=runtime()
    def add(kind,**kw):
        m.node_class(kind)
        n=graph.node(kind,**kw); n.set_override_display_id(owner); return n
    qwen=c['image_engine']=='qwen21'
    model=c['image_model']; clip=c['image_clip']; vae=c['image_vae']
    for category,name in [('diffusion_models',model),('text_encoders',clip),('vae',vae)]: m.require_file(category,name)
    source=add('UNETLoader',unet_name=model,weight_dtype='default').out(0)
    for row in c['image_loras']:
        if row['enabled'] and row['weight']:
            if not c.get('_generator_explicit_loras') and asset_contract(card)[0] in ('scene','object') and character_trigger(row['name']): continue
            m.require_file('loras',row['name'])
            source=add('LoraLoaderModelOnly',model=source,lora_name=row['name'],strength_model=row['weight']).out(0)
    encoder=add('CLIPLoader',clip_name=clip,type='qwen_image' if qwen else 'krea2',device='default').out(0)
    decoder=add('VAELoader',vae_name=vae).out(0)
    if qwen:
        enc=add('TextEncodeQwenImage21',clip=encoder,vae=decoder,prompt=prompt,negative_prompt='',resolution=0)
        pos,neg=enc.out(0),enc.out(1)
        source=add('QwenImage21Cache',model=source,device='auto',dtype='default').out(0)
    else:
        pos=add('CLIPTextEncode',clip=encoder,text=prompt).out(0)
        pos=add('ConditioningKrea2Rebalance',conditioning=pos,preset='detail',per_layer_weights=', '.join(['1']*12),multiplier=1.,renormalize=True).out(0)
        neg=add('ConditioningZeroOut',conditioning=pos).out(0)
    w,h=image_size(c,card)
    latent=add('EmptyLatentImage',width=w,height=h,batch_size=1).out(0)
    seed=(c['image_seed']+card['slot']+card['revision']*9973)%(2**53)
    if not qwen:
        # Native Krea2 two-stage schedule, not a generic sampler mislabeled as Krea.
        speed=next((x for x in m.filenames('loras') if 'krea2_turbo_sda' in x.lower()),'')
        m.require_file('loras',speed)
        fast=add('LoraLoaderModelOnly',model=source,lora_name=speed,strength_model=1.).out(0)
        latent=add('EmptyLatentImage',width=max(32,round(w/1.5/8)*8),height=max(32,round(h/1.5/8)*8),batch_size=1).out(0)
        latent=add('KSampler',model=fast,seed=seed,steps=2,cfg=c['image_cfg'],sampler_name='euler_ancestral',scheduler='simple',positive=pos,negative=neg,latent_image=latent,denoise=1.).out(0)
        latent=add('LatentUpscale',samples=latent,upscale_method='nearest-exact',width=w,height=h,crop='disabled').out(0)
    samples=add('KSampler',model=source,seed=seed,steps=c['image_steps'],cfg=c['image_cfg'],sampler_name='euler',scheduler='simple',positive=pos,negative=neg,latent_image=latent,denoise=1.).out(0)
    return add('VAEDecode',samples=samples,vae=decoder).out(0)


class H3ReferenceAssistant:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'settings':('STRING',{'default':json.dumps(runtime().defaults(),ensure_ascii=False)}),'intent':('STRING',{'multiline':True}),
                            'action':(['analyze','design','images','all','review'],),'job_id':('STRING',{'default':''})},
                'optional':{'previous':('H3FS_REFERENCES',),'owner_id':('STRING',{'default':''}),
                            **{f'image_{i}':('IMAGE',) for i in range(1,10)}},
                'hidden':{'unique_id':'UNIQUE_ID'}}
    RETURN_TYPES=('H3FS_REFERENCES','STRING')+('IMAGE',)*9
    RETURN_NAMES=('参考素材','图片提示词')+tuple(f'图片 {i}' for i in range(1,10))
    FUNCTION='run'
    CATEGORY='EverythingIsParty/MiniMax H3'
    OUTPUT_NODE=True
    @classmethod
    def IS_CHANGED(cls,**kw): return float('nan')
    def run(self,settings,intent,action,job_id='',previous=None,owner_id='',unique_id='',**image_inputs):
        m=runtime();c=parse_config(settings);owner=owner_id or str(unique_id)
        job_id=job_id or c['reference_job'] or uuid.uuid4().hex
        job=create_job(job_id,intent)
        enabled=[x for x in c['reference_cards'] if x['enabled']]
        if not enabled and action!='analyze': raise ValueError('请至少启用一张自动参考图')
        try:
            if previous is None and action!='review':
                design(c,intent,job,owner,allow_design=action!='images',learn_only=action=='analyze')
                job['intent']=intent
                save_job(job)
            if action in ('images','all'):
                m.release_optimizer()
                for i,card in enumerate(enabled):
                    row=job['cards'].get(str(card['slot']),{})
                    prompt=effective_prompt(c,card,row)
                    if not prompt.strip(): raise ValueError(f'第 {card["slot"]+1} 张图尚无提示词，请先设计')
                    if card['design']: validate_anchor(validate_designed_prompt(prompt,c['use_chinese_prompt']),card)
                    key=generation_key(c,card,prompt)
                    if row.get('generation_key')==key and row.get('path') and Path(row['path']).is_file(): continue
                    progress(job,owner,f'第 {card["slot"]+1} 张图片正在生成…',i,len(enabled),'images')
                    graph=GraphBuilder()
                    image=image_graph(graph,c,card,prompt,owner)
                    commit=graph.node('H3FSReferenceCommit',images=image,job_id=job_id,slot=card['slot'],
                                      prompt=prompt,generation_key=key,owner_id=owner)
                    commit.set_override_display_id(owner)
                    next_node=graph.node('H3ReferenceAssistant',settings=json.dumps(c,ensure_ascii=False),intent=intent,
                                         action=action,job_id=job_id,previous=commit.out(0),owner_id=owner,**image_inputs)
                    next_node.set_override_display_id(owner)
                    return {'result':tuple(next_node.out(i) for i in range(11)),'expand':graph.finalize()}
                progress(job,owner,'参考图已全部生成，可审核后继续视频',len(enabled),len(enabled),'complete')
            bundle=result_bundle(job,c)
            if action=='analyze': bundle['ready']=False
            bundle['images']={k:v for k,v in image_inputs.items() if v is not None}
            m.notify('h3fs_reference_result',owner,job=public_job(job))
            images={x['slot']:image_tensor(x) for x in bundle['media']}
            images.update({int(k.split('_')[1])-1:v for k,v in bundle['images'].items()})
            text=json.dumps({k:v.get('prompt','') for k,v in job['cards'].items()},ensure_ascii=False,indent=2)
            return {'ui':{'text':[text],'images':output_images(job)},'result':(bundle,text)+tuple(images.get(i,ExecutionBlocker(None)) for i in range(9))}
        except Exception as error:
            progress(job,owner,str(error),job.get('done',0),max(1,len(enabled)),'error')
            raise


class H3FSReferenceCommit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'job_id':('STRING',),'slot':('INT',),
                            'prompt':('STRING',),'generation_key':('STRING',),'owner_id':('STRING',)}}
    RETURN_TYPES=('H3FS_REFERENCES',)
    FUNCTION='run'
    CATEGORY='EverythingIsParty/MiniMax H3/Internal'
    def run(self,images,job_id,slot,prompt,generation_key,owner_id):
        job=read_job(job_id)
        if job is None: raise ValueError('Reference job no longer exists')
        directory=job_root()/job['directory'];directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'参考图-{slot+1:02d}-{datetime.now():%H%M%S}-{uuid.uuid4().hex[:4]}.png'
        array=np.clip(images[0].detach().cpu().numpy()*255,0,255).astype(np.uint8)
        from PIL.PngImagePlugin import PngInfo
        metadata=PngInfo();metadata.add_text('image_prompt',prompt);metadata.add_text('h3fs_job',job_id)
        Image.fromarray(array).save(path,pnginfo=metadata)
        job['cards'][str(slot)].update(prompt=prompt,path=str(path),generation_key=generation_key)
        save_job(job)
        runtime().notify('h3fs_reference_result',owner_id,job=public_job(job))
        return {'ui':{'images':[x for x in output_images(job) if x['filename']==path.name]},'result':({'job_id':job_id},)}


NODE_CLASS_MAPPINGS={'H3ReferenceAssistant':H3ReferenceAssistant,'H3FSReferenceCommit':H3FSReferenceCommit}
NODE_DISPLAY_NAME_MAPPINGS={'H3ReferenceAssistant':'H3 参考图自动生成助手'}
