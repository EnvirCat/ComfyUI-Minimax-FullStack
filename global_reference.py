"""Shared reference batch: design once, serial T2I, durable review, then H3."""
from __future__ import annotations
import copy
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from comfy_execution.graph_utils import GraphBuilder, ExecutionBlocker


def validate_global(raw):
    if not isinstance(raw, dict): raise ValueError('总体参考设置必须为对象')
    g = {'enabled':False,'image':'','person':False,'action':'video','job_id':'','cards':[], 'image_settings':{},'profiles':{}, **copy.deepcopy(raw)}
    for k in ('enabled','person'):
        if not isinstance(g[k],bool): raise ValueError(f'总体参考 {k} 必须是勾选值')
    if g['action'] not in ('images','all','video','prompt'): raise ValueError('未知总体参考步骤')
    for k in ('image','job_id'):
        if not isinstance(g[k],str): raise ValueError('总体参考路径与记录需为文本')
    if g['job_id'] and not re.fullmatch('[a-f0-9]{32}',g['job_id']): raise ValueError('无效的总体参考记录')
    if not isinstance(g['image_settings'],dict) or not isinstance(g['profiles'],dict): raise ValueError('总体生图参数需为对象')
    if not isinstance(g['cards'],list) or len(g['cards'])>9: raise ValueError('总体参考最多九张图')
    seen=set()
    for row in g['cards']:
        if not isinstance(row,dict) or type(row.get('slot')) is not int or not 0<=row['slot']<9 or row['slot'] in seen: raise ValueError('无效或重复图片位置')
        seen.add(row['slot'])
        if not isinstance(row.get('enabled',False),bool) or not isinstance(row.get('text',''),str): raise ValueError('图片勾选和意向格式错误')
        if len(row.get('text',''))>4096: raise ValueError('图片意向过长')
        if not isinstance(row.get('draft',''),str) or len(row.get('draft',''))>8192: raise ValueError('图片提示词格式错误或过长')
        if type(row.get('revision',0)) is not int or not 0<=row.get('revision',0)<=1000000: raise ValueError('图片重抽次数无效')
        from .config import ASPECTS
        row.setdefault('kind','auto');row.setdefault('aspect','video')
        if row['kind'] not in ('auto','person','object','scene'): raise ValueError('未知图片素材类型')
        if row['aspect'] not in ('video',*ASPECTS): raise ValueError('未知图片画幅')
    return g


def runtime():
    from . import nodes
    return nodes


def image_config(c,card=None):
    from .image_generator import generator_defaults, settings
    g=c['global_reference']
    # Diffusion parameters are independent; share only local planner configuration.
    # The shared planner selector is authoritative. Old image presets may carry
    # stale llm_* keys; they must not silently revert an API choice to local.
    image={**generator_defaults(),'aspect':'video',**g['image_settings'],**{k:v for k,v in c.items() if k.startswith('llm_')}}
    if image.get('aspect','video')=='video': image['aspect']=c['aspect']
    if card is not None:
        image['aspect']=c['aspect'] if card.get('aspect','video')=='video' else card['aspect']
    return settings(image)


def status(job,owner,text,done=0,total=1,stage='design'):
    from .reference_studio import save_job
    job.update(status=text,stage=stage,done=done,total=total)
    save_job(job)
    runtime().notify('h3fs_global_progress',owner,job_id=job['id'],text=text,done=done,total=total,stage=stage)


def plan(c,intent,job,owner):
    from .reference_studio import signature, save_job, public_job
    from .reference_prompts import STYLE_SCHEMA, STYLE_SYSTEM, parse_style, medium_description, asset_contract, validate_anchor
    from .reference_learning import PROFILE_SCHEMA, PROFILE_SYSTEM, parse_profile, validate_no_face, triggers_for
    from .reference_config import validate_designed_prompt
    from .media import image_tensor,resolve
    m=runtime();g=c['global_reference'];im=image_config(c)
    cards=[x for x in g['cards'] if x.get('enabled')]
    if not cards: raise ValueError('请至少勾选一张要生成的图片')
    if not intent.strip(): raise ValueError('请填写视频的意图／提示词')
    if g['person'] and not g['image']: raise ValueError('参考人物需要上传总体参考图')
    ref=image_tensor({'path':g['image']}) if g['image'] else None
    person_needed=g['person'] and any(asset_contract(x)[0]=='person' for x in cards)
    key=signature([3,g['image'],resolve(g['image']).stat().st_mtime_ns if ref is not None else 0,person_needed])
    if job.get('shared_key')!=key and not all(x.get('draft','').strip() for x in cards):
        style='与用户意图相符的自然摄影';profile=''
        if ref is not None:
            status(job,owner,'正在学习总体参考的画风与光影…',0,len(cards)+1)
            report=parse_style(m.design_completion(im,STYLE_SYSTEM,'只分析画风、色调、光线与光学质感。',[ref],im['image_seed'],owner,768,schema=STYLE_SCHEMA))
            style=medium_description(report['medium'],True)+'；'+report['style']
            if person_needed:
                status(job,owner,'正在学习人物发型、服装与妆容；不迁移背景或五官…',0,len(cards)+1)
                report=parse_profile(m.design_completion(im,PROFILE_SYSTEM,'目标素材类型：person。用简体中文；身份由 LoRA 控制。',[ref],im['image_seed'],owner,1024,schema=PROFILE_SCHEMA),'person',True)
                profile=report['description']
        job.update(shared_key=key,style=style,person_profile=profile);save_job(job)
    for i,card in enumerate(cards):
        m.mm.throw_exception_if_processing_interrupted()
        text=card.get('text','').strip()
        if not text: raise ValueError(f'请填写图片 {card["slot"]+1} 的意向')
        kind,contract=asset_contract(card)
        # Unrecognized nouns never inherit a person merely because the video has one.
        profile=job.get('person_profile','') if kind=='person' and g['person'] else ''
        local=image_config(c,card)
        # Keep completed automatic designs compatible with earlier releases.
        parts=[3,text,kind,intent,job['style'],profile,local['aspect']]
        if card.get('draft','').strip(): parts.append(card['draft'].strip())
        design_key=signature(parts)
        full_head=kind=='person' and not re.search(r'特写|极近景|局部|close[ -]?up|macro',text,re.I)
        if full_head: design_key=signature([design_key,'half-body-complete-head-v2'])
        row=job['cards'].get(str(card['slot']),{})
        if row.get('design_key')==design_key and row.get('prompt'): continue
        if card.get('draft','').strip():
            prompt=card['draft'].strip();validate_anchor(prompt,card)
            if kind=='person': validate_no_face(prompt)
            job['cards'][str(card['slot'])]={**row,'slot':card['slot'],'kind':kind,'aspect':card.get('aspect','video'),'text':text,'prompt':prompt,'draft':prompt,'design_key':design_key}
            save_job(job);m.notify('h3fs_global_result',owner,job=public_job(job),complete=False);continue
        status(job,owner,f'图片 {card["slot"]+1}：正在设计提示词…',i,len(cards),'design')
        system=('你是参考素材设计师。输出一段中文文生图提示词，不超过512字。当前图片意向是唯一主体，'
                '视频总意图只决定氛围和用途，不能把院子、蛋糕改成人物镜头。每张图是独立素材而非视频分镜。'
                '不描述五官和脸型，脸部身份由LoRA控制，可以描述妆容。只输出提示词，不写解释、标题或编辑指令。')
        if full_head:
            system+=('本次是可辨认人物的半身人像素材。用120–180字的简洁提示词，先写人物、动作与取景，'
                     '再写发型、可见上衣与妆容，最后用一句话概括光影。半身取景不能写裙摆、腿、鞋、袜等画外部位，'
                     '否则会把注意力引向服装局部。人物是主角，完整头部和视线可见；不要写特写、细节特写或冗长的材质/光学形容词。')
        user=f'当前图片意向：{text}\n明确素材类型：{kind}\n主体要求：{contract}\n视频总意图（不是此图的主体）：{intent}\n画风与光影：{job["style"]}\n画幅：{local["aspect"]}'
        if full_head: user+='\n人物构图要求：中景半身人像，从头顶到腰部完整入镜，人物位于画面中央，头顶上方保留浅色背景。按用户意向安排动作。只描述本景别可见的服装部分。'
        if profile: user+='\n必须保持的人物服装发型设定（不带原背景、姿势或视角）：'+profile
        for attempt in range(2):
            correction='' if not attempt else ('\n上一次违反素材约束：'+str(error)+'。重新只设计当前物象。\n'
                '忽略视频意图里的人物、人物景别及人物动作；当前图是独立素材，人物服装与发型不参与。' if kind in ('scene','object') else '\n请修正上次问题：'+str(error))
            request=user if not attempt or kind not in ('scene','object') else f'当前图片意向：{text}\n主体要求：{contract}\n画风与光影：{job["style"]}\n画幅：{local["aspect"]}'
            prompt=m.design_completion(local,system,request+correction,[],im['image_seed']+card['slot']+attempt*101,owner,1536)
            try:
                prompt=validate_designed_prompt(prompt,True);validate_anchor(prompt,card)
                if kind=='person': validate_no_face(prompt)
                break
            except ValueError as issue:
                error=issue
                if attempt: raise ValueError(f'图片 {card["slot"]+1}（{kind}）提示词两次未满足素材约束：{issue}。请修改此框意向或类型后重试；已完成图片保留。') from issue
                status(job,owner,f'图片 {card["slot"]+1}：正在纠正素材类型串台…',i,len(cards),'design')
        if full_head:
            prompt=re.sub('近景特写|特写','半身近景',prompt)
            framing='中景半身人像，从头顶到腰部完整入镜，人物位于画面中央，头顶上方保留背景。'
            # Framing is a geometric requirement, not an invented face description.
            if not prompt.startswith(framing): prompt=framing+prompt
            prompt=validate_designed_prompt(prompt,True)
        if kind in ('person','auto'): validate_no_face(prompt)
        triggers=triggers_for(im,kind)
        if triggers: prompt=triggers+', '+prompt
        job['cards'][str(card['slot'])]={**row,'slot':card['slot'],'kind':kind,'aspect':card.get('aspect','video'),'text':text,'prompt':prompt,'draft':prompt,'design_key':design_key}
        save_job(job)
        m.notify('h3fs_global_result',owner,job=public_job(job),complete=False)
    job['intent']=intent;save_job(job)


def bundle(job,c):
    from .reference_studio import register_file
    slots={x['slot'] for x in c['global_reference']['cards'] if x.get('enabled')}
    media=[dict(x) for x in c['media'] if x['kind']!='image' or x['slot'] not in slots]
    for slot in sorted(slots):
        row=job['cards'].get(str(slot),{})
        if not row.get('path') or not Path(row['path']).is_file(): raise ValueError(f'图片 {slot+1} 尚未完成')
        media.append({'kind':'image','slot':slot,'path':row['path'],'name':Path(row['path']).name,'preview_url':register_file(row['path']),'start':0,'end':0})
    return {'media':media,'ready':True,'job_id':job['id'],'global':True}


class H3GlobalReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'settings':('STRING',),'intent':('STRING',),'owner_id':('STRING',),'job_id':('STRING',)},'optional':{'previous':('H3FS_REFERENCES',)}}
    RETURN_TYPES=('H3FS_REFERENCES',)
    FUNCTION='run'
    CATEGORY='EverythingIsParty/MiniMax H3/Internal'
    def run(self,settings,intent,owner_id,job_id,previous=None):
        from .config import parse_config
        from .reference_studio import create_job, save_job, public_job, image_graph, signature
        c=parse_config(settings);g=c['global_reference'];m=runtime();job=create_job(job_id,intent)
        cards=[x for x in g['cards'] if x.get('enabled')]
        try:
            if previous is None: plan(c,intent,job,owner_id)
            im=image_config(c);m.release_optimizer()
            for i,card in enumerate(cards):
                row=job['cards'][str(card['slot'])]
                # Registered identity LoRAs never contaminate object/scene assets.
                from .reference_learning import character_trigger
                local=image_config(c,card)
                local['image_seed']=(local['image_seed']+card.get('revision',0))%(2**64)
                if row['kind']!='person':
                    local['image_loras']=[x for x in local['image_loras'] if not character_trigger(x['name'])]
                else:
                    from .reference_learning import triggers_for
                    plain=re.sub(r'^\s*(?:(?:CML|XNJ)\b\s*[,，]\s*)+','',row['prompt'],flags=re.I).strip()
                    trigger=triggers_for(local,row['kind'])
                    row['prompt']=(trigger+', ' if trigger else '')+plain
                    save_job(job)
                key=signature([row['prompt'],card['slot'],{k:local[k] for k in ('image_engine','image_model','image_clip','image_vae','image_loras','image_steps','image_cfg','image_megapixels','image_seed','aspect')}])
                if row.get('generation_key')==key and row.get('path') and Path(row['path']).is_file(): continue
                status(job,owner_id,f'图片 {card["slot"]+1}：正在生图…',i,len(cards),'images')
                graph=GraphBuilder();image=image_graph(graph,local,{**card,'aspect':local['aspect'],'revision':0},row['prompt'],owner_id)
                finish=graph.node('H3GlobalCommit',images=image,job_id=job_id,slot=card['slot'],generation_key=key,owner_id=owner_id)
                finish.set_override_display_id(owner_id)
                child=graph.node('H3GlobalReferences',settings=settings,intent=intent,owner_id=owner_id,job_id=job_id,previous=finish.out(0))
                child.set_override_display_id(owner_id)
                return {'result':(child.out(0),),'expand':graph.finalize()}
            result=bundle(job,c);status(job,owner_id,'素材已完成；可以检查图片、修改总意图，再生成视频提示词',len(cards),len(cards),'complete')
            m.notify('h3fs_global_result',owner_id,job=public_job(job),media=result['media'],complete=True)
            return (result,)
        except Exception as error:
            status(job,owner_id,str(error),job.get('done',0),max(1,len(cards)),'error');raise


class H3GlobalCommit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'job_id':('STRING',),'slot':('INT',),'generation_key':('STRING',),'owner_id':('STRING',)}}
    RETURN_TYPES=('H3FS_REFERENCES',)
    FUNCTION='run'
    CATEGORY=H3GlobalReferences.CATEGORY
    def run(self,images,job_id,slot,generation_key,owner_id):
        from .reference_studio import read_job,save_job,public_job,output_images
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        import numpy as np
        import folder_paths
        job=read_job(job_id)
        if job is None: raise ValueError('总体参考任务记录丢失')
        name=re.sub(r'[<>:"/\\|?*\x00-\x1f]','',job['intent']).strip('. ')[:24] or '素材'
        directory=Path(folder_paths.get_output_directory())/'H3FullStack'/'总体参考图'/f'{name}-{job_id[:6]}'
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'图片-{slot+1:02d}-{datetime.now():%H%M%S}-{uuid.uuid4().hex[:4]}.png'
        row=job['cards'][str(slot)];metadata=PngInfo();metadata.add_text('image_prompt',row['prompt'])
        Image.fromarray(np.clip(images[0].detach().cpu().numpy()*255,0,255).astype(np.uint8)).save(path,pnginfo=metadata)
        row.update(path=str(path),generation_key=generation_key);save_job(job)
        runtime().notify('h3fs_global_result',owner_id,job=public_job(job),complete=False)
        return {'ui':{'images':[x for x in output_images(job) if x['filename']==path.name]},'result':({'job_id':job_id},)}


NODE_CLASS_MAPPINGS={'H3GlobalReferences':H3GlobalReferences,'H3GlobalCommit':H3GlobalCommit}
