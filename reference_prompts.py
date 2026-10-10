"""Small, independently validated contracts for reference assets, not video shots."""
import json
import re

CONTRACT_VERSION = 5
MEDIA = ('photography', 'illustration', '3d', 'unknown')
STYLE_SCHEMA = {'type':'object','properties':{
    'medium':{'type':'string','enum':list(MEDIA)},
    'evidence':{'type':'string'},'style':{'type':'string'},
    'confidence':{'type':'number','minimum':0,'maximum':1}},
    'required':['medium','evidence','style','confidence'],'additionalProperties':False}

STYLE_SYSTEM = """Analyze the attached image's visual medium and reusable appearance, NOT its subject.
Return the requested JSON. Distinguish photographic / photoreal imagery from illustration using visible evidence:
real lens depth, continuous skin/material texture, natural illumination and optical details versus drawn outlines,
visible brush strokes or flat graphic shading. Soft focus, warm tones, smooth skin or a beautiful woman are NOT
evidence of illustration. Do not invent brush strokes. If uncertain use unknown and state the uncertainty.
evidence: concrete visible cues, at most 80 Chinese characters. style: at most 160 Chinese characters describing
lighting, palette, contrast, optical texture and depth only. Texture means image grain, sharpness or diffusion,
NOT skin, hair, fabric or any depicted material. Describe diffuse/soft/warm light without prescribing an indoor,
outdoor or studio location. No person, identity, clothing, props, room layout or portrait
composition. Do not transfer the reference subject. All prose in Simplified Chinese; enum values stay English."""


def parse_style(raw, medium='auto'):
    try:
        value=json.loads(raw)
        if not isinstance(value,dict): raise ValueError()
        if value['medium'] not in MEDIA: raise ValueError()
        if not isinstance(value['style'],str) or not value['style'].strip(): raise ValueError()
        if not isinstance(value['evidence'],str): raise ValueError()
        confidence=float(value['confidence'])
        if not 0 <= confidence <= 1: raise ValueError()
    except (KeyError,TypeError,ValueError):
        raise ValueError('画风分析格式不完整；请重试分析或手动填写画风，不会使用猜测的画风') from None
    if medium != 'auto': value['medium']=medium
    elif confidence < .65: value['medium']='unknown'
    # Neutral appearance fields cannot smuggle a contradictory medium into later prompts.
    value['style']=re.sub(r'(?:数字|数码)?插画(?:风格|风)?|(?:digital )?illustration|摄影(?:风格)?|photograph(?:y|ic)?|笔触[^，。；\n]*', '', value['style'],flags=re.I).strip(' ，。;')
    value['style']=re.sub(r'[^，。；\n]*?(?:肤质|皮肤|纱|裙|服装|衣服|发丝|头发|skin|hair|fabric|dress)[^，。；\n]*[，。；\n]?', '', value['style'],flags=re.I).strip(' ，。;')
    value['style']=portable_style(value['style'])
    value['confidence']=confidence
    return value


def portable_style(text):
    # The same lighting can be used in a new location. Do not turn a courtyard
    # into a studio still life just because its style source was photographed indoors.
    return re.sub(r'室内|室外|摄影棚|\b(?:indoor|outdoor|studio)\b', '', text, flags=re.I).strip()


def scene_scale_rule(card,chinese):
    if not card.get('design',True) or asset_contract(card)[0]!='scene': return ''
    if miniature_requested(card['text']): return ''
    return ('现实生活尺度的实景环境，摄像机位于环境内部；连续地面与建筑空间延伸至画面边缘，充满画幅。'
            '不是桌面微缩景、沙盘、模型底座或产品棚拍。保持主提示词指定的视角。' if chinese else
            'A real, life-size environment photographed from within it. Continuous ground and architectural '
            'space extend to the image edges and fill the frame, not a tabletop miniature, diorama, model base '
            'or studio product shot. Keep the camera angle specified in the main prompt.')


def miniature_requested(text):
    positive=re.sub(r'(?:不是|不要|不含|没有|禁止|无)[^，。；\n]*|\b(?:not|no|without)\b[^,.\n]*','',text,flags=re.I)
    return bool(re.search(r'微缩|沙盘|迷你|模型|玩具|miniature|diorama|toy|scale model',positive,re.I))


def medium_description(medium,chinese):
    labels={'photography':('写实摄影，真实光学和自然材质，不是插画','photographic realism, natural materials and optical light, not illustration'),
            'illustration':('插画','illustration'),'3d':('三维渲染','3D rendering'),
            'unknown':('仅沿用以下光线和色调，不推断媒介','reuse only the following light and color; do not infer a medium')}
    return labels[medium][0 if chinese else 1]


def asset_kind(name, explicit='auto'):
    if explicit!='auto': return explicit
    # Recognize clear everyday nouns. Unfamiliar/composite requests remain unrestricted,
    # instead of pretending a keyword table understands every possible subject.
    text=name.strip().lower()
    text=re.sub(r'(?:没有|不含|不要|禁止|不出现|无)(?:任何)?(?:人物|人像|人手|人群|人员)|\b(?:no|without)\s+(?:people|persons?|humans?)\b','',text)
    if re.search(r'美女|女孩|女人|男人|女性|男性|女主|男主|模特|人物|情侣|女友|男友|漫路|小凝姐|\bcml\b|\bxnj\b|woman|girl|\bman\b|\bperson\b|couple',text): return 'person'
    if re.search(r'院子|小院|庭院|房间|自习室|卧室|客厅|花园|街道|教室|厨房|courtyard|garden|room|street',text): return 'scene'
    if re.search(r'蛋糕|杯子|项链|轿车|跑车|汽车|书本|椅子|饰品|cake|cup\b|necklace|car\b|book\b',text): return 'object'
    return 'auto'


def asset_contract(card):
    kind=asset_kind(card['text'],card.get('kind','auto'))
    constraints={'scene':'真实生活尺度、可进入的场景实景图，摄像机置于环境内部，空间延伸到画面边缘；不是桌面微缩模型或沙盘。完整呈现场景的空间、结构和材质。默认没有人物、人手或肖像；不能把场景当作人像的虚化背景。',
                 'object':'独立物品参考图；物品占据视觉中心且细节可辨认。默认没有人物、人手或人像；不能画人物拿着、吃着或穿戴这件物品。',
                 'person':'独立人物参考图；明确成年人物的外观、服装和姿态。',
                 'auto':'严格围绕用户指定的物象；只有物象名称本身要求人物时才加入人物。'}
    if kind=='scene' and miniature_requested(card['text']):
        constraints['scene']='按用户明确要求的微缩 / 模型尺度构图，完整呈现场景结构与材质；默认无人，不作为人像背景。'
    return kind,constraints[kind]


def content_reference(c,card):
    kind=asset_kind(card['text'],card.get('kind','auto'))
    if card.get('reference_image') and card.get('strict_reference'): return card['reference_image'],kind
    if card.get('strict_reference') and not card.get('reference_image'):
        shared=c['style_image'] if kind=='person' else c['scene_reference'] if kind=='scene' else ''
        if not shared: raise ValueError('严格内容参考已开启，请上传此图的内容参考或对应公共人物 / 场景参考图')
        return shared,kind
    if kind=='person' and c['strict_person_reference']:
        if not c['style_image']: raise ValueError('人物严格参考已开启，请先上传人物参考图')
        return c['style_image'],kind
    if kind=='scene' and c['scene_reference']: return c['scene_reference'],kind
    return '',kind


def validate_anchor(prompt,card):
    kind,_=asset_contract(card)
    # Reviewed drafts may change visual details, but cannot silently turn a standalone
    # cake/courtyard into the VLOG protagonist. Composite nouns explicitly permit people.
    if kind not in ('scene','object'): return prompt
    positive=re.sub(r'(?:无|没有|不含|不包含|不出现|不要|不带|不加入|不迁移|移除)[^，。；\n]*|\b(?:no|without|remove)\b[^,.\n]*','',prompt,flags=re.I)
    if re.search(r'美女|女性|女人|女孩|女郎|少女|男人|男孩|人物|人像|双手|她|(?<!其|吉|利|他)他|\bwoman\b|\bgirl\b|\bman\b|\bportrait\b|\bholding\b',positive,re.I):
        raise ValueError('独立场景 / 物品提示词混入了人物：请重新设计，或将素材类型明确改为人物 / 组合画面')
    noun=card['text'].strip()
    aliases={'杯子':('杯','cup'),'小院子':('小院','庭院','courtyard'),'院子':('院','courtyard'),
             '蛋糕':('蛋糕','cake'),'美女':('女性','女人','woman'),'书本':('书','book')}
    if len(noun)<=12 and not any(x.lower() in prompt.lower() for x in aliases.get(noun,(noun,))):
        raise ValueError(f'提示词没有明确描写目标物象“{noun}”')
    return prompt


IMAGE_SYSTEM = """Design exactly ONE reusable reference ASSET image, not a video frame or storyboard.
Priority: requested asset name + asset contract > explicit style > optional video context.
The video intent is context for atmosphere only. NEVER transplant its protagonist, portrait framing, clothing
or actions into an independent object/scene. A courtyard means the courtyard itself; cake means cake itself.
For a scene: a REAL life-size, inhabitable environment viewed FROM WITHIN the space. Continuous ground and
architecture extend to the image boundaries, not a centered cut-out of a courtyard on a studio table.
No miniature, model base, diorama, toy scale or tilt-shift look unless the asset NAME explicitly asks for it.
Readable spatial layout, architecture, materials and light, NOT a blurred backdrop behind a woman.
Apply transferable light/palette, never transplant 'indoor/studio' locations into an outdoor asset.
For an object: specific shape, material, color and readable detail, NOT a woman holding/eating the object.
For a person: an adult, hair, wardrobe, makeup, pose and framing. Face identity is controlled by the user's LoRA;
NEVER describe face shape, facial anatomy, bone structure or eye/nose/mouth shapes. Makeup is allowed.
Style-only references do not constrain appearance.
When a learned subject profile is supplied, appearance and wardrobe are assembled programmatically;
do not redesign or contradict them. Do NOT copy a reference background, pose or camera angle.
Do not add irrelevant objects, dialogue, logos or readable text. Keep exact supplied LoRA triggers.
Return only the final image prompt, no titles or reasoning, within 512 Chinese characters / English words.
All descriptive prose must be Simplified Chinese when requested, otherwise English."""
