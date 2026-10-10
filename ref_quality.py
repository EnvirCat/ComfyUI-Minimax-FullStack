"""Evidence coverage checks, not a claim to measure cinematic quality."""
import re
from .prompt_format import REF_FIELDS, PromptStructureError


def sections(text):
    headers = list(re.finditer(r'(?m)^(' + '|'.join(REF_FIELDS) + r'):\s*', text))
    return {m[1]: text[m.end():headers[i+1].start() if i+1 < len(headers) else len(text)].strip()
            for i, m in enumerate(headers)}


def inspect_ref(text, labels, seconds, chinese):
    data = sections(text)
    definitions = data.get('subject_definitions', '')
    body = data.get('detailed_description', '')
    subjects = list(re.finditer(r'(<Subject \d+>)\s*[:：]', definitions))
    sources, unused = set(), []
    for i, match in enumerate(subjects):
        label = match[1]
        definition = definitions[match.end():subjects[i+1].start() if i+1 < len(subjects) else len(definitions)]
        refs = set(re.findall(r'<(?:Picture|Video) \d+>', definition))
        if label in body:
            sources.update(refs)
        else:
            unused.append(label)
    sources.update(re.findall(r'<(?:Picture|Video) \d+>', body))
    expected = set(re.findall(r'<(?:Picture|Video) \d+>', labels))
    issues, warnings = [], []
    if unused:
        issues.append('Defined visible carriers absent from the actual timeline: ' + ', '.join(unused))
    if expected - sources:
        issues.append('Reference sources with no active shot contribution: ' + ', '.join(sorted(expected-sources)))
    # Only meaningful clip-length requirements; do not demand 4 cuts in a short clip.
    size = len(re.findall(r'[\u4e00-\u9fff]', body)) if chinese else len(re.findall(r'\b[\w-]+\b', body))
    if seconds >= 6 and size < (240 if chinese else 170):
        warnings.append('镜头正文偏短，建议补充动作发展、反应、运镜与同期声；字数不是生成格式的硬性门槛。' if chinese else 'Shot description is brief; consider more action, reaction, camera and physical sound detail. Length alone is not a format error.')
    shots = list(re.finditer(r'\[Shot (\d+)\]', body))
    if not shots or [int(m[1]) for m in shots] != list(range(1, len(shots)+1)):
        issues.append('Shots must start at [Shot 1] and use consecutive numbers')
    times = [int(a)*60+float(b) for a,b in re.findall(r'At (\d{2}):(\d{2}\.\d{3})', body)]
    if len(times) != max(0, len(shots)-1) or any(t >= seconds or t <= 0 for t in times) or times != sorted(set(times)):
        issues.append('Later shots require strictly increasing At MM:SS.mmm cut times inside the actual duration')
    return {'issues': issues, 'warnings': warnings, 'active_sources': sorted(sources), 'body_size': size,
            'body_measure': 'Chinese characters' if chinese else 'words', 'shot_count': len(shots)}


def validate_ref(text, labels, seconds, chinese):
    report = inspect_ref(text, labels, seconds, chinese)
    if report['issues']:
        raise PromptStructureError(detail='REF coverage: ' + '; '.join(report['issues']))
    return report
