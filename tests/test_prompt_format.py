"""No model loading: presentation variants of the official H3 format."""
import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('h3_prompt_format',Path(__file__).parents[1]/'prompt_format.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class PromptFormatTests(unittest.TestCase):
    def sections(self,mode='reference'):
        return {name:'[Shot 1] 咖啡杯在晨光中。' for name in f.required_fields(mode)}
    def canonical(self,mode='reference'):
        return '\n\n'.join(f'{key}: {value}' for key,value in self.sections(mode).items())
    def test_canonical_sections_unchanged(self):
        for mode in ('text','reference'):
            self.assertEqual(f.normalize_prompt(self.canonical(mode),mode),self.canonical(mode))
    def test_bold_headers_and_fullwidth_colons(self):
        raw='\n'.join(f'**{key}：** {value}' for key,value in self.sections().items())
        self.assertEqual(f.normalize_prompt(raw,'reference'),self.canonical())
    def test_markdown_headings_numbered_and_title_case(self):
        raw='\n'.join(f'### {index}. {key.replace("_"," ").title()}: {value}' for index,(key,value) in enumerate(self.sections().items(),1))
        self.assertEqual(f.normalize_prompt(raw,'reference'),self.canonical())
    def test_chinese_headers_keep_english_protocol_names(self):
        raw='\n'.join(f'{f.ALIASES[key][0]}：{value}' for key,value in self.sections().items())
        self.assertEqual(f.normalize_prompt(raw,'reference'),self.canonical())
    def test_code_fence_and_standalone_headers(self):
        raw='```markdown\n'+'\n'.join(f'## {key}\n{value}' for key,value in self.sections().items())+'\n```'
        self.assertEqual(f.normalize_prompt(raw,'reference'),self.canonical())
    def test_json_preserves_dialogue_quotes_and_labels(self):
        sections=self.sections();sections['detailed_description']='[Shot 1] <Subject 1> (S1) says <d>[Chinese] "你好！"</d> near "营业中".'
        expected='\n\n'.join(f'{key}: {value}' for key,value in sections.items())
        self.assertEqual(f.normalize_prompt(json.dumps(sections,ensure_ascii=False),'reference'),expected)

    def test_angle_shot_markers_are_canonical_without_renumbering(self):
        data=self.sections();data['detailed_description']='<Shot 1> <Subject 1> 在 <Picture 1> 的小院。<Shot 3> At 00:01.000 镜头继续。'
        result=f.normalize_prompt(json.dumps(data,ensure_ascii=False),'reference')
        self.assertIn('[Shot 1] <Subject 1> 在 <Picture 1>',result);self.assertIn('[Shot 3] At 00:01.000',result);self.assertNotIn('[Shot 2]',result)

    def test_shot_markers_in_dialogue_or_visible_text_stay_literal(self):
        data=self.sections();data['detailed_description']='<Shot 1> <d>[Chinese] <Shot 2></d> 标牌写着 "<Shot 3>"，还有「<Shot 4>」。'
        result=f.normalize_prompt(json.dumps(data,ensure_ascii=False),'reference')
        self.assertIn('[Shot 1] <d>[Chinese] <Shot 2></d>',result);self.assertIn('"<Shot 3>"',result);self.assertIn('「<Shot 4>」',result)
    def test_json_chinese_property_aliases(self):
        sections={f.ALIASES[key][0]:value for key,value in self.sections().items()}
        self.assertEqual(f.normalize_prompt(json.dumps(sections,ensure_ascii=False),'reference'),self.canonical())
    def test_missing_sections_not_filled_with_fake_na(self):
        raw=self.canonical().split('overall_soundscape:',1)[0]
        with self.assertRaisesRegex(f.PromptStructureError,'missing: overall_soundscape, non_diegetic_music'):
            f.normalize_prompt(raw,'reference')
    def test_empty_sections_rejected(self):
        for raw in (self.canonical().replace('non_diegetic_music: [Shot 1] 咖啡杯在晨光中。','non_diegetic_music:'),
                    json.dumps({**self.sections(),'non_diegetic_music':''})):
            with self.assertRaisesRegex(f.PromptStructureError,'empty: non_diegetic_music'):
                f.normalize_prompt(raw,'reference')
    def test_na_is_valid_explicit_value(self):
        raw=self.canonical('text').replace('non_diegetic_music: [Shot 1] 咖啡杯在晨光中。','non_diegetic_music: N/A')
        self.assertEqual(f.normalize_prompt(raw,'text'),raw)
    def test_duplicate_fields_rejected(self):
        with self.assertRaisesRegex(f.PromptStructureError,'duplicate: summary'):
            f.normalize_prompt(self.canonical()+'\nsummary: Second prompt.','reference')
        with self.assertRaisesRegex(f.PromptStructureError,'duplicate: summary'):
            f.normalize_prompt('{"summary":"one","summary":"two"}','reference')
    def test_truncated_json_and_open_thinking_not_accepted(self):
        for raw in ('{"summary":"a cup', '<think>'+self.canonical()):
            with self.assertRaises(f.PromptStructureError):f.normalize_prompt(raw,'reference')
    def test_dialogue_field_like_lines_are_not_headers(self):
        sections=self.sections('text')
        sections['integrated_multimodal_description']='[Shot 1] <d>[English]\noverall_soundscape: exact spoken words.\n</d>'
        raw='\n\n'.join(f'{key}: {value}' for key,value in sections.items())
        self.assertEqual(f.normalize_prompt(raw,'text'),raw)
    def test_keyframe_instruction_retained_in_text_and_json(self):
        prefix='For the target video, at 0.00 seconds, <Picture 1> (from [Shot 1]) is fully referenced.'
        expected=prefix+'\n\n'+self.canonical('first_frame')
        self.assertEqual(f.normalize_prompt(expected,'first_frame'),expected)
        raw=json.dumps({'alignment_instruction':prefix,**self.sections('first_frame')},ensure_ascii=False)
        self.assertEqual(f.normalize_prompt(raw,'first_frame'),expected)
    def test_recovery_schema_requires_only_official_fields(self):
        for mode in ('text','reference','first_frame','first_last'):
            instruction,schema=f.recovery_contract(mode)
            self.assertTrue(set(f.required_fields(mode))<=set(schema['required']))
            self.assertEqual('alignment_instruction' in schema['required'],mode in ('first_frame','first_last'))
            self.assertFalse(schema['additionalProperties']);self.assertIn('not the H3 content rules',instruction)
    def test_extra_json_fields_and_non_text_values_not_silently_dropped(self):
        for changes in ({'reasoning':'a story'},{'summary':{'text':'a cup'}}):
            with self.assertRaises(f.PromptStructureError):
                f.normalize_prompt(json.dumps({**self.sections(),**changes}),'reference')


if __name__=='__main__':unittest.main()
