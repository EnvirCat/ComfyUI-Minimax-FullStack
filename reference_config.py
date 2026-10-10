"""Independent, bounded settings for the reference image assistant."""
import math
import re
from .config import number

ASPECTS = ('video', '16:9', '9:16', '1:1', '4:3', '3:4')


def validate_reference_config(c):
    for key in ('auto_reference', 'reference_one_click', 'strict_person_reference'):
        if not isinstance(c[key], bool): raise ValueError(f'{key} must be a boolean')
    if c['image_engine'] not in ('qwen21', 'krea2'): raise ValueError('Unknown image engine')
    if c['reference_job'] and not re.fullmatch(r'[a-f0-9]{32}', c['reference_job']):
        raise ValueError('Invalid reference job ID')
    for key,lo,hi,integer in (('image_steps',1,100,True),('image_megapixels',.1,4,False),
                              ('image_cfg',0,20,False),('image_seed',0,2**53-1,True)):
        c[key] = number(c[key], key, lo, hi, integer)
    if not isinstance(c['reference_cards'], list) or len(c['reference_cards']) > 9:
        raise ValueError('Maximum nine reference cards')
    seen = set()
    cards = []
    for row in c['reference_cards']:
        if not isinstance(row,dict): raise ValueError('Invalid reference card')
        slot = number(row.get('slot'), 'Image slot', 0, 8, True)
        if slot in seen: raise ValueError('Duplicate reference generation slot')
        seen.add(slot)
        aspect = row.get('aspect','video')
        if aspect not in ASPECTS: raise ValueError('Unknown image aspect')
        kind=row.get('kind','auto')
        if kind not in ('auto','person','scene','object'): raise ValueError('Unknown asset kind')
        if 'strict_reference' in row and not isinstance(row['strict_reference'],bool):
            raise ValueError('strict_reference must be a boolean')
        cards.append({'slot':slot, 'enabled':bool(row.get('enabled',False)),
                      'design':bool(row.get('design',True)), 'text':str(row.get('text','')),
                      'draft':str(row.get('draft','')), 'aspect':aspect, 'kind':kind,
                      'reference_image':str(row.get('reference_image','')), 'strict_reference':bool(row.get('strict_reference',False)),
                      'reference_description':str(row.get('reference_description','')), 'camera':str(row.get('camera','')),
                      'revision':number(row.get('revision',0),'Reroll count',0,100000,True)})
    if c.get('reference_prompt_version',0)<4:
        for card in cards:
            if card['design']: card['draft']=''
    c['reference_prompt_version']=4
    c['reference_cards'] = sorted(cards,key=lambda x:x['slot'])
    if not isinstance(c['image_loras'],list) or len(c['image_loras']) > 16:
        raise ValueError('Maximum sixteen image LoRAs')
    c['image_loras'] = [{'name':str(x['name']), 'weight':number(x.get('weight',1),'Image LoRA weight',-5,5),
                         'enabled':bool(x.get('enabled',True))} for x in c['image_loras'] if isinstance(x,dict) and x.get('name')]
    if c['style_medium'] not in ('auto','photography','illustration','3d','unknown'): raise ValueError('Unknown style medium')
    for key in ('style_image','style_text','style_override','scene_reference','person_profile_override','scene_profile_override','image_model','image_clip','image_vae'):
        if not isinstance(c[key],str): raise ValueError(f'{key} must be text')


def image_size(c, card):
    aspect = c['aspect'] if card['aspect'] == 'video' else card['aspect']
    a,b = map(int,aspect.split(':'))
    scale = math.sqrt(c['image_megapixels']*1024*1024/(a*b))
    return max(32,round(a*scale/32)*32), max(32,round(b*scale/32)*32)


def prompt_units(text):
    # Count every Han character and every non-Han word, including mixed prompts.
    return len(re.findall(r'[\u4e00-\u9fff]', text)) + len(re.findall(r'[^\W\u4e00-\u9fff]+',text,re.UNICODE))


def validate_designed_prompt(text, chinese):
    if not text.strip(): raise ValueError('Image prompt was empty')
    if prompt_units(text) > 512: raise ValueError('Designed image prompt exceeds 512 characters/words; original draft preserved')
    if chinese and not re.search(r'[\u4e00-\u9fff]',text): raise ValueError('Chinese image prompt requested but not produced')
    return text.strip()
