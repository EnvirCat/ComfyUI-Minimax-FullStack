def ref_prompt(count=1,chinese=True):
    source=', '.join(f'<Picture {i}>' for i in range(1,count+1))
    content='陶瓷咖啡杯在晨光中，镜头保持静止。' if chinese else 'A ceramic cup in morning light, the camera remains static.'
    data={'subject_definitions':f'<Subject 1>: {content} Sources: {source}.',
          'summary':'[reference generation] '+content,
          'retention_analysis':'<Subject 1>: fully_preserved in [Shot 1].',
          'detailed_description':'[Shot 1] <Subject 1> '+content,
          'overall_soundscape':content,'non_diegetic_music':'N/A'}
    return '\n\n'.join(f'{k}: {v}' for k,v in data.items())
