"""Normalize presentation only; never invent missing H3 prompt content."""
from __future__ import annotations

import json
import re

BASE_FIELDS = ('integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music')
REF_FIELDS = ('subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
              'overall_soundscape', 'non_diegetic_music')
ALIASES = {
    'subject_definitions': ('主体定义', '主题定义', '对象定义'),
    'summary': ('摘要', '概述', '总结'),
    'retention_analysis': ('保留分析', '保留关系分析', '保留性分析'),
    'detailed_description': ('详细描述', '详细说明'),
    'integrated_multimodal_description': ('综合多模态描述', '集成多模态描述', '整合多模态描述'),
    'overall_soundscape': ('整体声景', '总体声景', '整体音景', '总体音景'),
    'non_diegetic_music': ('非叙事音乐', '非画内音乐', '非剧情音乐', '非场景音乐'),
}


def required_fields(mode):
    return REF_FIELDS if mode == 'reference' else BASE_FIELDS


class PromptStructureError(ValueError):
    def __init__(self, *, missing=(), empty=(), duplicate=(), detail=''):
        self.missing, self.empty, self.duplicate = tuple(missing), tuple(empty), tuple(duplicate)
        problems = [f'{label}: {", ".join(values)}' for label, values in
                    (('missing', missing), ('empty', empty), ('duplicate', duplicate)) if values]
        if detail:
            problems.append(detail)
        super().__init__('Incomplete H3 prompt structure (' + '; '.join(problems) + '); original input preserved')


def _canonical_label(label):
    label = re.sub(r'[\s-]+', '_', label.strip()).casefold()
    for name, aliases in ALIASES.items():
        if label == name or label in aliases:
            return name
    return label


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        name = _canonical_label(key)
        if name in result:
            raise PromptStructureError(duplicate=(name,))
        result[name] = value
    return result


def _render(sections, fields, prefix=''):
    missing = [name for name in fields if name not in sections]
    empty = [name for name in fields if name in sections and
             (not isinstance(sections[name], str) or not sections[name].strip())]
    if missing or empty:
        raise PromptStructureError(missing=missing, empty=empty)
    normalized=dict(sections)
    for name in ('detailed_description','integrated_multimodal_description'):
        if name in normalized:
            # Some local models emit the same explicit shot number with angle
            # brackets. Canonicalize presentation, never add or renumber shots.
            # Quoted visible text and dialogue remain literal user content.
            normalized[name]=re.sub(r'(<d>.*?(?:</d>|$)|"[^"]*"|“[^”]*”|「[^」]*」|『[^』]*』)|<Shot\s+(\d+)>',
                lambda m:m[1] if m[1] is not None else f'[Shot {m[2]}]',normalized[name],flags=re.S|re.I)
    body = '\n\n'.join(f'{name}: {normalized[name].strip()}' for name in fields)
    return (prefix.strip() + '\n\n' if prefix.strip() else '') + body


def normalize_prompt(raw, mode):
    """Accept equivalent Markdown/Chinese headers or a complete JSON envelope.

    Dialogue, asset labels, shot descriptions and keyframe instructions remain
    untouched; only explicit shot-marker brackets are canonicalized. A missing
    section is an error, not a fabricated N/A fallback.
    """
    if not isinstance(raw, str):
        raise PromptStructureError(detail='model returned non-text content')
    text = re.sub(r'<think>.*?(?:</think>|$)', '', raw, flags=re.S | re.I).strip().lstrip('\ufeff')
    fence = re.fullmatch(r'```[^\n]*\n(.*?)\n?```', text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    fields = required_fields(mode)
    if text.startswith('{'):
        try:
            data = json.loads(text, object_pairs_hook=_json_pairs)
        except json.JSONDecodeError as error:
            raise PromptStructureError(detail='invalid or truncated JSON response') from error
        if not isinstance(data, dict):
            raise PromptStructureError(detail='expected one prompt object')
        prefix = data.pop('alignment_instruction', '')
        if not isinstance(prefix, str):
            raise PromptStructureError(detail='non-text alignment instruction')
        extra = set(data) - set(fields)
        if extra:
            raise PromptStructureError(detail='unexpected fields: ' + ', '.join(sorted(extra)))
        if mode == 'reference' and prefix:
            raise PromptStructureError(detail='unexpected keyframe instruction in REF mode')
        return _render(data, fields, prefix)

    names = []
    for name in fields:
        names.append(re.escape(name).replace('_', r'[ _-]+'))
        names.extend(re.escape(alias) for alias in ALIASES[name])
    # Anchored to whole field labels, not keywords appearing within a sentence.
    header = re.compile(
        r'^[ \t]*(?:\#{1,6}[ \t]+)?(?:[-*][ \t]+|\d+[.)、][ \t]+)?'
        r'(?:\*\*|__|`)?(?P<label>' + '|'.join(names) + r')[ \t]*'
        r'(?:\*\*|__|`)?[ \t]*(?:[:：][ \t]*(?:\*\*|__|`)?[ \t]*|$)',
        flags=re.M | re.I)
    # A quoted sign or multiline dialogue can itself contain a field-like line.
    protected = [(match.start(), match.end()) for match in
                 re.finditer(r'<d>.*?(?:</d>|$)|"[^"]*"', text, flags=re.S | re.I)]
    matches = [match for match in header.finditer(text) if
               not any(start <= match.start() < end for start, end in protected)]
    sections, duplicates = {}, []
    for index, match in enumerate(matches):
        name = _canonical_label(match.group('label'))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if name in sections:
            duplicates.append(name)
        sections[name] = text[match.end():end].strip()
    if duplicates:
        raise PromptStructureError(duplicate=duplicates)
    prefix = text[:matches[0].start()].strip() if matches else ''
    # REF/T2V have no instruction preamble; remove only LLM framing before fields.
    if mode in ('reference', 'text'):
        prefix = ''
    return _render(sections, fields, prefix)


def recovery_contract(mode):
    """An internal constrained envelope, rendered back to official H3 text."""
    names = list(required_fields(mode))
    if mode in ('first_frame', 'first_last'):
        names.insert(0, 'alignment_instruction')
    schema = {'type': 'object', 'properties': {name: {'type': 'string', 'minLength': 1} for name in names},
              'required': names, 'additionalProperties': False}
    instruction = ('\nFORMAT RECOVERY: For this response only, transport the complete finished prompt as one JSON object. '
                   'This envelope overrides plain-text formatting, not the H3 content rules or requested prose language. '
                   'Use the exact English property names below with non-empty string values. Include every section. '
                   'Do not include Markdown, explanations, reasoning, or an English duplicate. Preserve all reference labels '
                   'and the full visual/audio coverage; avoid unnecessary repetition and leave room for both sound sections. '
                   'Use escaped newlines inside strings. The node will render the values into the official three/six sections. '
                   'When alignment_instruction is listed, put the required opening/ending keyframe instruction there. '
                   'Required JSON properties: ' + ', '.join(names) + '.')
    return instruction, schema
