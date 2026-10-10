"""Reference images teach editable text profiles; diffusion remains text-to-image."""
import hashlib
import json
import re
from .media import resolve, image_tensor
from .reference_config import prompt_units, validate_designed_prompt
from .reference_prompts import portable_style

PROFILE_SCHEMA={'type':'object','properties':{
    'subject_present':{'type':'boolean'}, 'features':{'type':'string'},
    'wardrobe_and_accessories':{'type':'string'}, 'visual_style':{'type':'string'},
    'uncertainty':{'type':'string'}},
    'required':['subject_present','features','wardrobe_and_accessories','visual_style','uncertainty'],
    'additionalProperties':False}
PROFILE_SYSTEM="""Learn a reusable TEXT profile from the attached reference image, NOT an edit instruction.
Return only the required JSON. Report subject_present false if the requested subject is absent.
For person: features are visible hair, visible build and makeup styling;
face identity is handled by the user's LoRA. NEVER describe face shape, bone structure, eye/nose/mouth shape,
facial proportions or ethnicity, and never prescribe a different identity. Makeup colors and techniques are allowed.
wardrobe_and_accessories are ONLY visible clothes, colors, cuts and accessories. Describe the visible construction
(layered ruffles, pleats, seams, buttons) accurately. A flowing ruffled fabric edge is NOT automatically a scarf,
ribbon, bow or necktie; only claim a tie/bow when its knot or attachment is visible. Do not simplify ruffles into ribbons.
Never invent unseen shoes,
lower-body clothing, accessories or measured dimensions. No background, furniture, location, pose, gesture,
gaze, expression, camera angle, framing or shot distance in the person profile. Those must remain flexible.
For scene: features describe architecture, layout, fixed furnishings, materials and colors; exclude people.
For object: features describe the isolated object's visible shape, proportions, material, color and detail.
visual_style describes photographic/drawn medium, light, palette and optical texture, NOT background content
or indoor/outdoor/studio location. Lighting must remain usable in a new setting.
Do not infer identity names or ethnicity. uncertainty records unseen/ambiguous details, which must NOT become
generation requirements. All description fields together at most 240 Chinese characters / English words.
Use the requested language. The result will guide a NEW text-to-image generation, not copy the original frame."""


def validate_no_face(text):
    anatomy = (r'脸型|鹅蛋脸|瓜子脸|圆脸|五官|骨相|脸部结构|面部比例|脸部轮廓|面部轮廓|'
               r'颧骨|下颌|下巴|眉骨|鼻梁|鼻型|眼型|双眼皮|单眼皮|大眼睛|小眼睛|杏眼|丹凤眼|'
               r'嘴型|唇形|薄唇|厚唇|樱桃小嘴|'
               r'\b(?:oval face|round face|face shape|facial proportions|facial anatomy|facial structure|'
               r'bone structure|cheekbones?|jawline|chin|nose shape|'
               r'(?:large|small|wide|almond[- ]shaped) eyes|(?:thin|full|plump) lips)\b')
    if re.search(anatomy,text,re.I):
        raise ValueError('不应描述脸型或五官；身份由用户 LoRA 控制，可描述妆容')
    return text


def parse_profile(raw,kind,chinese):
    try:
        report=json.loads(raw)
        if not isinstance(report,dict) or not isinstance(report['subject_present'],bool): raise ValueError('参考学习格式不完整')
        for key in ('features','wardrobe_and_accessories','visual_style','uncertainty'):
            if not isinstance(report[key],str): raise ValueError('参考学习描述必须是文字')
        report['visual_style']=portable_style(report['visual_style'])
        if not report['subject_present'] or not report['features'].strip():
            raise ValueError('参考图中没有可识别的目标主体，请换图或填写人工设定')
        fields=['features','wardrobe_and_accessories','visual_style'] if kind=='person' else ['features','visual_style']
        description='；'.join(report[k].strip() for k in fields if report[k].strip())
        if prompt_units(description)>260: raise ValueError('参考设定过长，请精简为可复用特征')
        validate_designed_prompt(description,chinese)
        if kind=='person' and re.search(r'背景|沙发|墙壁|窗户|窗边|房间|卧室|坐在|站在|躺在|俯拍|仰拍|镜头|视角|\b(?:background|sofa|bedroom|sitting|standing|camera|framing)\b',report['features']+'；'+report['wardrobe_and_accessories'],re.I):
            raise ValueError('人物设定混入了背景、姿态或视角，应仅提炼人物外观与服装')
        if kind=='person': validate_no_face(description)
        return {**report,'description':description,'kind':kind}
    except (KeyError,TypeError,json.JSONDecodeError):
        raise ValueError('参考学习格式不完整，请重新学习或填写人工设定') from None


def override_for(c,card,kind):
    if card.get('reference_description','').strip(): return card['reference_description'].strip()
    if card.get('reference_image') and card.get('strict_reference'): return ''
    return c['person_profile_override'].strip() if kind=='person' else c['scene_profile_override'].strip() if kind=='scene' else ''


def learn_profile(m,c,job,path,kind,owner,progress,override=''):
    source=resolve(path)
    key=hashlib.sha256(json.dumps([5,str(source),source.stat().st_mtime_ns,kind,c['use_chinese_prompt'],override],ensure_ascii=False).encode()).hexdigest()
    cache=job.setdefault('learned_profiles',{})
    if key not in cache:
        if override:
            description=validate_designed_prompt(override,c['use_chinese_prompt'])
            if prompt_units(description)>260: raise ValueError('参考设定请控制在 260 字 / 词以内，为构图和意图留出空间')
            report={'description':description,'kind':kind,'manual':True}
        else:
            progress(job,owner,{'person':'正在学习人物特征与服装（排除背景）…','scene':'正在学习场景结构与材质…'}.get(kind,'正在学习物品特征…'),stage='learn')
            user=f'目标素材类型：{kind}。'+('使用简体中文。' if c['use_chinese_prompt'] else 'Write English.')+' /no_think'
            visual=[image_tensor({'path':path})]
            raw=m.design_completion(c,PROFILE_SYSTEM,user,visual,c['image_seed'],owner,1024,schema=PROFILE_SCHEMA)
            try: report=parse_profile(raw,kind,c['use_chinese_prompt'])
            except ValueError as error:
                raw=m.design_completion(c,PROFILE_SYSTEM,user+'\n上次格式 / 内容错误：'+str(error)+'；请纠正，仅保留可见且可复用的特征。',visual,c['image_seed']+101,owner,1024,schema=PROFILE_SCHEMA)
                report=parse_profile(raw,kind,c['use_chinese_prompt'])
        cache[key]={**report,'source':path}
    return cache[key]


def character_trigger(name):
    match=re.search(r'(?<![a-z0-9])(CML|XNJ)(?![a-z0-9])',name,re.I)
    return match.group(1).upper() if match else ''


def triggers_for(c,kind):
    if kind in ('scene','object'): return ''
    return ', '.join(dict.fromkeys(character_trigger(row['name']) for row in c['image_loras']
                                  if row['enabled'] and row['weight'] and character_trigger(row['name'])))


def compose_prompt(base,profile,kind,camera,chinese,triggers='',asset_rule=''):
    pieces=[base]
    if profile:
        title={'person':'人物设定','scene':'场景设定','object':'物品设定'}.get(kind,'主体设定') if chinese else {'person':'Person profile','scene':'Scene profile','object':'Object profile'}.get(kind,'Subject profile')
        rule=('遵循以下可见特征与服装设定重新生图；人物参考不含背景、姿态或拍摄视角，场景和动作按主提示词安排。' if chinese else
              'Generate a NEW image using the visible attributes and wardrobe below. A person reference does not constrain background, pose or camera angle; use the main prompt for setting and action.') if kind=='person' else ('沿用以下主体结构与材质，拍摄视角可按主提示词变化，不加入参考图中无关的人物。' if chinese else 'Reuse the subject structure and materials below; camera angle may follow the main prompt. Do not transfer unrelated people.')
        pieces.append(rule+'\n'+title+'：'+profile)
    if camera.strip(): pieces.append(('摄影视角：' if chinese else 'Camera: ')+camera.strip())
    if asset_rule: pieces.append(asset_rule)
    if triggers and not all(token in base for token in triggers.split(', ')): pieces.append(triggers)
    return '\n\n'.join(pieces)
