"""One optional reference, one image, ordinary IMAGE fan-out. No video side effects."""
from __future__ import annotations
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy_execution.graph_utils import GraphBuilder, ExecutionBlocker
from .config import number, parse_config, ASPECTS
from .media import image_tensor
from .reference_studio import image_graph, register_file
from .reference_prompts import STYLE_SCHEMA, STYLE_SYSTEM, parse_style, medium_description, asset_contract
from .reference_learning import validate_no_face, triggers_for

MODES = ('recreate', 'character_card', 'style_intent')
LEGACY_CARD_TEMPLATE = ('一张横向排版的人物三视图卡片，三个等宽分区从左到右为正视图、严格侧视图、背视图。'
                        '每个分区均为同一成年人物的站立全身像，包括完整双脚与鞋底，比例与身高一致。'
                        '人物只占画面高度的80%，头顶上方与双脚下方各留10%空白，双脚距离画面底边有清楚的空隙。'
                        '三视图的发型、服装剪裁、颜色、材质、配饰完全一致；手臂自然垂放，姿态便于辨认。'
                        '简洁浅灰背景，均匀柔和照明，无场景、道具、文字、水印或额外人物。')
CARD_TEMPLATE = ('一张横向排版的人物三视图卡片，三个等宽分区从左到右为正视图、严格侧视图、背视图。'
                 '每个分区均为同一成年人物的站立全身像，包括完整双脚与鞋底，比例与身高一致。'
                 '远景全身构图，三个分区都完整包含头顶、整个头部、躯干、双腿、双脚与鞋底。'
                 '人物只占分区画面高度的70%，头顶上方与鞋底下方各留15%空白；头顶与双脚均不接触边缘。'
                 '头部与身体保持正常人体比例，三格从头到脚使用相同取景范围。'
                 '三视图的发型、服装剪裁、颜色、材质、配饰完全一致；手臂自然垂放，姿态便于辨认。'
                 '简洁浅灰背景，均匀柔和照明，无场景、道具、文字、水印或额外人物。')


def runtime():
    from . import nodes
    return nodes


def generator_defaults():
    m = runtime()
    c = m.defaults()
    pick = lambda cat, test: next((x for x in m.filenames(cat) if test(x.lower())), '')
    c.update(image_engine='krea2', image_steps=8, image_seed=42,
             image_model=pick('diffusion_models', lambda x: 'krea2_turbo' in x),
             image_clip=pick('text_encoders', lambda x: 'qwen3vl_4b' in x),
             image_vae=pick('vae', lambda x: x.endswith('qwen_image_vae.safetensors')),
             image_loras=[], image_megapixels=1., image_cfg=1.,
             generator_version=1, constraint='style_intent', aspect='9:16',
             reference_path='', reference_name='', card_template=CARD_TEMPLATE,
             draft='', draft_key='', draft_manual=False, no_face_description=True, action='generate', owner_id='', nonce='',
             result_id='', force_regenerate=False)
    return c


def settings(raw):
    raw = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(raw, dict): raise ValueError('生图设置必须是对象')
    value = {**generator_defaults(), **raw}
    c = parse_config({**value, 'auto_reference':False, 'reference_one_click':False, 'reference_cards':[]})
    value.setdefault('instance_id','')
    for k in ('constraint', 'reference_path', 'reference_name', 'card_template', 'draft', 'draft_key', 'action', 'owner_id', 'nonce', 'instance_id', 'result_id'):
        c[k] = value[k]
        if not isinstance(c[k], str): raise ValueError(f'{k} 必须是文本')
    if c['constraint'] not in MODES: raise ValueError('未知的参考约束')
    if c['action'] not in ('design', 'generate'): raise ValueError('未知生图操作')
    if c['aspect'] not in ASPECTS: raise ValueError('未知图片比例')
    if not c['card_template'].strip() or len(c['card_template']) > 4096: raise ValueError('人物卡片模板需为 1–4096 字')
    if c['card_template'].strip()==LEGACY_CARD_TEMPLATE:
        c['card_template']=CARD_TEMPLATE  # Upgrade only the previously shipped default, not custom templates.
    if len(c['draft']) > 12000: raise ValueError('图片提示词过长')
    if not isinstance(value.get('no_face_description',True), bool): raise ValueError('面部描述开关必须为布尔值')
    c['no_face_description'] = value.get('no_face_description', True)
    if not isinstance(value.get('draft_manual',False),bool): raise ValueError('手动提示词标记必须为布尔值')
    c['draft_manual'] = value.get('draft_manual',False)
    if not isinstance(value.get('force_regenerate',False),bool): raise ValueError('重跑标记必须为布尔值')
    c['force_regenerate'] = value.get('force_regenerate',False)
    c['_generator_explicit_loras'] = True
    return c


def prompt_key(c, intent, reference=None):
    key = {k:c[k] for k in ('constraint','reference_path','card_template','no_face_description')}
    key['intent'] = intent
    # LoRA/weight/seed/model are generation controls, not a reason to run vision again.
    # Connected image contents, not just graph links, participate in draft validity.
    if reference is not None:
        key['reference_pixels'] = hashlib.sha256(reference.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(key,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def clean_prompt(text):
    text = re.sub(r'^```[^\n]*\n|\n```$', '', text.strip()).strip()
    if not text or not re.search(r'[\u4e00-\u9fff]',text): raise ValueError('反推结果为空或不是中文，请重新分析')
    if len(text) > 5000: raise ValueError('反推结果异常过长，请重试或手动输入提示词')
    return text


def design(c, intent, reference, owner, seed):
    m = runtime()
    mode = c['constraint']
    has_reference = reference is not None
    if mode == 'recreate' and not has_reference:
        raise ValueError('“完全复刻”需要一张参考图；无参考图请选“参考画风＋输入意向”')
    if not intent.strip() and not has_reference: raise ValueError('请输入意向／提示词或上传参考图')
    no_face = '不描述脸型、五官、面部比例或身份相似度；脸部身份由用户的 LoRA 决定，妆容可以描述。' if c['no_face_description'] else ''
    if mode == 'recreate':
        system = ('你是摄影与图像提示词分析师。根据实际参考图片反推一段可直接文生图的简体中文提示词。'
                  '完整记录可见主体、服装剪裁与材质、配饰、姿势、表情、拍摄角度、景别、构图与人物位置、'
                  '背景空间与可见物体、光源方向、色温、曝光、景深和真实视觉媒介。'
                  '不编造不可见细节，不改服装、不增加物体或故事；柔焦摄影不等于数字插画。'
                  '不得写分析、解释、序号、标题、JSON或“参考图”。只输出完整生图提示词。' + no_face)
        return clean_prompt(m.design_completion(c,system,'反推图片原本的画面。用户补充（不替换原画面）：'+intent,[reference],seed,owner,1536))
    if mode == 'character_card':
        description = intent.strip()
        if has_reference:
            system = ('学习图片中成年人物的发型、发色、服装的结构、颜色、材质、层次、配饰及妆容与摄影画风。'
                      '严格排除背景、场景、道具、原人物姿势、镜头构图。不可见部位不臆造花纹或装饰。'
                      '分辨纵向荷叶边、褶皱与系结；只有明确可见成对蝴蝶结环与结心才写蝴蝶结。'
                      '不能把垂坠褶边改为盘扣或中式服装。不可见的鞋袜款式不描述。'
                      '只输出一段简体中文人物服装与造型描述，不写故事或图像编辑指令。' + no_face)
            description = clean_prompt(m.design_completion(c,system,'人物三视图造型分析。用户补充：'+intent,[reference],seed,owner,1024))
        return c['card_template'].strip() + '\n人物设定：' + description
    # Style is analyzed without the requested subject, then used as text only.
    # This deliberately prevents a portrait reference turning a cake/courtyard into another portrait.
    if not intent.strip(): raise ValueError('“参考画风＋输入意向”需要填写要生成的主体')
    if not has_reference: return intent  # Literal text-to-image, no hidden rewriting or LLM call.
    report = parse_style(m.design_completion(c,STYLE_SYSTEM,'只分析光线、色调、媒介与光学质感。',[reference],seed,owner,768,schema=STYLE_SCHEMA))
    kind, contract = asset_contract({'text':intent, 'kind':'auto'})
    system = ('将用户意向写成一段可直接用于文生图的简体中文提示词，512字以内。'
              '用户指定的主体是画面的主角，不添加未要求的人物、手或人像。'
              '严格遵循提供的画风，但不要从画风推断服装、主体或场景。'
              '院子等场景是可进入的实景空间，不能变成桌面微缩模型。'
              '只输出提示词，不写解释或编辑指令。' + no_face)
    prompt = clean_prompt(m.design_completion(c,system,
                        f'用户意向：{intent}\n主体约束：{contract}\n画风：{medium_description(report["medium"],True)}；{report["style"]}',
                        [],seed,owner,1280))
    return prompt


def result_root():
    return Path(folder_paths.get_output_directory()) / 'H3FullStack' / '参考图生图'


def generation_key(c,prompt,seed):
    fields = ('image_engine','image_model','image_clip','image_vae','image_loras','image_steps','image_megapixels','image_cfg','aspect')
    return hashlib.sha256(json.dumps({'version':1,'prompt':prompt,'seed':seed,**{k:c[k] for k in fields}},
                                    sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def result_path(identifier):
    if not re.fullmatch(r'[a-f0-9]{32}', identifier): raise ValueError('Invalid image result ID')
    return result_root() / '.records' / (identifier+'.json')


def read_result(identifier):
    p = result_path(identifier)
    if not p.is_file(): return None
    result = json.loads(p.read_text(encoding='utf-8'))
    if Path(result['path']).is_file(): result['preview_url'] = register_file(result['path'])
    return result


class H3ImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'settings':('STRING',{'default':json.dumps(generator_defaults(),ensure_ascii=False)}),
                            'intent':('STRING',{'multiline':True,'default':''}),
                            'seed':('INT',{'default':42,'min':0,'max':2**53-1})},
                'optional':{'reference':('IMAGE',)},'hidden':{'unique_id':'UNIQUE_ID'}}
    RETURN_TYPES = ('IMAGE','STRING')
    RETURN_NAMES = ('图片','生图提示词')
    FUNCTION = 'run'
    CATEGORY = 'EverythingIsParty/MiniMax H3'
    OUTPUT_NODE = True
    DESCRIPTION = '独立文生图节点。一个图片输出可接多个图片框；已有图须点击应用才替换。'

    def run(self, settings, intent, seed, reference=None, unique_id=''):
        c = globals()['settings'](settings)
        owner = c['owner_id'] or str(unique_id)
        seed = number(seed,'Seed',0,2**53-1,True)
        if reference is None and c['reference_path']: reference = image_tensor({'path':c['reference_path']})
        if reference is not None and (reference.ndim!=4 or len(reference)!=1): raise ValueError('参考图请只输入一张图片')
        key = prompt_key(c,intent,reference)
        try:
            runtime().notify('h3fs_generator_progress',owner,text='正在设计图片提示词…',done=0,total=2)
            reviewed = c['draft'].strip() and (c['draft_key']==key or c['draft_manual'])
            prompt = c['draft'] if reviewed else design(c,intent,reference,owner,seed)
            if reviewed and c['constraint']=='character_card' and '\n人物设定：' in prompt:
                prompt=c['card_template'].strip()+'\n人物设定：'+prompt.split('\n人物设定：',1)[1]
            if not reviewed and reference is not None and c['no_face_description']: validate_no_face(prompt)
            kind = 'person' if c['constraint']=='character_card' else asset_contract({'text':intent,'kind':'auto'})[0]
            triggers = triggers_for(c,kind)
            if kind not in ('scene','object'):
                # Old analyzed text may carry the previous identity's token.
                # Change identity tokens in code, never analyze the photograph again.
                prompt = re.sub(r'^\s*(?:(?:CML|XNJ)\b\s*[,，]\s*)+','',prompt,flags=re.I).strip()
            if triggers and not all(re.search(r'\b'+trigger+r'\b',prompt,re.I) for trigger in triggers.split(', ')):
                prompt = triggers + ', ' + prompt
            runtime().notify('h3fs_generator_prompt',owner,text=prompt,draft_key=key)
            if c['action']=='design':
                return {'ui':{'text':[prompt]},'result':(ExecutionBlocker(None),prompt)}
            if c['result_id'] and not c['force_regenerate']:
                previous = read_result(c['result_id'])
                previous_key = (previous.get('generation_key') or generation_key(previous['settings'],previous['prompt'],previous['settings']['image_seed'])) if previous else ''
                if previous and Path(previous['path']).is_file() and previous_key==generation_key(c,prompt,seed):
                    image = image_tensor({'path':previous['path']})
                    image._h3fs_generated = {'id':previous['id'],'path':previous['path']}
                    previous['source_instance_id'] = c['instance_id']
                    runtime().notify('h3fs_generator_result',owner,result=previous)
                    runtime().notify('h3fs_generator_progress',owner,text='使用已生成图片，不重跑生图',done=2,total=2)
                    return {'ui':{'images':[previous['image']],'text':[prompt]},'result':(image,prompt)}
            runtime().release_optimizer()
            runtime().notify('h3fs_generator_progress',owner,text='正在生成图片…',done=1,total=2)
            c['image_seed'] = seed
            graph = GraphBuilder()
            image = image_graph(graph,c,{'slot':0,'revision':0,'aspect':c['aspect'],'text':intent,'kind':'auto'},prompt,owner)
            finish = graph.node('H3FSImageCommit', images=image, prompt=prompt, settings=json.dumps(c,ensure_ascii=False),
                                owner_id=owner, identifier=uuid.uuid4().hex, draft_key=key)
            finish.set_override_display_id(owner)
            return {'result':(finish.out(0),finish.out(1)),'expand':graph.finalize()}
        except Exception as error:
            runtime().notify('h3fs_generator_progress',owner,text=str(error),done=0,total=2,error=True)
            raise


class H3FSImageCommit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'prompt':('STRING',),'settings':('STRING',),
                            'owner_id':('STRING',),'identifier':('STRING',),'draft_key':('STRING',)}}
    RETURN_TYPES = ('IMAGE','STRING')
    FUNCTION = 'run'
    CATEGORY = 'EverythingIsParty/MiniMax H3/Internal'

    def run(self, images, prompt, settings, owner_id, identifier, draft_key):
        record = result_path(identifier)
        directory = result_root() / f'{datetime.now():%Y-%m-%d}'
        directory.mkdir(parents=True,exist_ok=True)
        path = directory / f'参考图-{datetime.now():%H%M%S}-{identifier[:6]}.png'
        c = json.loads(settings)
        meta = PngInfo()
        meta.add_text('h3fs_generation',json.dumps({'prompt':prompt,'settings':c},ensure_ascii=False))
        Image.fromarray(np.clip(images[0].detach().cpu().numpy()*255,0,255).astype(np.uint8)).save(path,pnginfo=meta)
        output = Path(folder_paths.get_output_directory()).resolve()
        native = {'filename':path.name,'subfolder':path.parent.relative_to(output).as_posix(),'type':'output'}
        result = {'id':identifier,'path':str(path),'name':path.name,'prompt':prompt,'draft_key':draft_key,
                  'settings':c,'image':native,'preview_url':register_file(path),
                  'generation_key':generation_key(c,prompt,c['image_seed']), 'source_instance_id':c['instance_id']}
        record.parent.mkdir(parents=True,exist_ok=True)
        record.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        # This remains an ordinary torch IMAGE for any other ComfyUI consumer.
        # The main node recognizes provenance and does not silently replace occupied slots.
        images._h3fs_generated = {'id':identifier,'path':str(path)}
        runtime().notify('h3fs_generator_result',owner_id,result=result)
        runtime().notify('h3fs_generator_progress',owner_id,text='生成完成；空框已自动填入，已有图可点击应用',done=2,total=2)
        return {'ui':{'images':[native],'text':[prompt]},'result':(images,prompt)}


NODE_CLASS_MAPPINGS = {'H3ImageGenerator':H3ImageGenerator,'H3FSImageCommit':H3FSImageCommit}
NODE_DISPLAY_NAME_MAPPINGS = {'H3ImageGenerator':'H3 参考图生图','H3FSImageCommit':'H3 图片保存（内部）'}
