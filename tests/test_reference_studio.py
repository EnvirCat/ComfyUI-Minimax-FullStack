import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from test_pipeline import m
from ref_fixture import ref_prompt

r=importlib.import_module('fullstack_test.reference_studio')
rc=importlib.import_module('fullstack_test.reference_config')
q=importlib.import_module('fullstack_test.ref_quality')


class ReferenceSettingsTests(unittest.TestCase):
    def test_chinese_defaults_and_explicit_english(self):
        self.assertTrue(m.parse_config({})['use_chinese_prompt'])
        self.assertEqual(m.parse_config({})['ui_language'],'zh')
        self.assertFalse(m.parse_config({'version':3,'use_chinese_prompt':False})['use_chinese_prompt'])
    def test_aspect_presets_and_video_match(self):
        c=m.parse_config({'aspect':'9:16','image_megapixels':.5})
        for aspect in rc.ASPECTS:
            w,h=rc.image_size(c,{'aspect':aspect})
            self.assertEqual(w%32,0);self.assertEqual(h%32,0)
            if aspect=='video':self.assertLess(w,h)
    def test_bad_settings(self):
        for row in ({'reference_job':'../../bad'}, {'image_engine':'unknown'}, {'image_steps':0},
                    {'image_megapixels':float('nan')},{'auto_reference':'yes'},
                    {'reference_cards':[{'slot':9}]},{'reference_cards':[{'slot':0},{'slot':0}]},
                    {'reference_cards':[{'slot':0,'aspect':'2:3'}]}):
            with self.subTest(row=row),self.assertRaises(ValueError):m.parse_config(row)
    def test_512_mixed_units(self):
        self.assertEqual(rc.prompt_units('中文 cup gold'),4)
        self.assertEqual(rc.validate_designed_prompt('中'*512,True),'中'*512)
        with self.assertRaises(ValueError):rc.validate_designed_prompt('中'*513,True)
        with self.assertRaises(ValueError):rc.validate_designed_prompt(' '.join(['cup']*513),False)
        with self.assertRaises(ValueError):rc.validate_designed_prompt('English only',True)


class CoverageTests(unittest.TestCase):
    def test_chinese_subject_colon_preserves_merged_reference_coverage(self):
        text=ref_prompt(5).replace('<Subject 1>:','<Subject 1>：')
        self.assertEqual(q.inspect_ref(text,'<Picture 1> <Picture 2> <Picture 3> <Picture 4> <Picture 5>',5,True)['issues'],[])
    def test_merged_sources_count_as_active_not_one_cut_per_picture(self):
        report=q.inspect_ref(ref_prompt(5),'<Picture 1> <Picture 2> <Picture 3> <Picture 4> <Picture 5>',5,True)
        self.assertEqual(report['issues'],[]);self.assertEqual(report['shot_count'],1)
    def test_defined_machine_but_never_in_shot_is_not_coverage(self):
        text=ref_prompt().replace('\n\nsummary:', '\n<Subject 2>: 咖啡机，来自 <Picture 4>。\n\nsummary:')
        report=q.inspect_ref(text,'<Picture 1> <Picture 4>',5,True)
        self.assertTrue(any('<Subject 2>' in x for x in report['issues']))
        self.assertTrue(any('<Picture 4>' in x for x in report['issues']))
    def test_short_body_for_ten_seconds_and_bad_cut_times(self):
        report=q.inspect_ref(ref_prompt(),' <Picture 1>',10,True)
        self.assertEqual(report['issues'],[]);self.assertTrue(report['warnings'])
        q.validate_ref(ref_prompt(),'<Picture 1>',10,True)
        for cut in ('[Shot 3] At 00:02.000,','[Shot 2] At 00:12.000,','[Shot 2] '):
            text=ref_prompt().replace('overall_soundscape:',cut+' <Subject 1> 移动。\n\noverall_soundscape:')
            self.assertTrue(q.inspect_ref(text,'<Picture 1>',5,True)['issues'])


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h3fs-reference-test-');self.addCleanup(self.tmp.cleanup)
        self.out=patch.object(m.folder_paths,'get_output_directory',return_value=self.tmp.name);self.out.start();self.addCleanup(self.out.stop)
        self.notify=patch.object(m,'notify');self.events=self.notify.start();self.addCleanup(self.notify.stop)
        self.release=patch.object(m,'release_optimizer');self.release.start();self.addCleanup(self.release.stop)
        self.native=patch.object(m,'node_class',return_value=object);self.native.start();self.addCleanup(self.native.stop)
        self.files=patch.object(m,'require_file');self.files.start();self.addCleanup(self.files.stop)
        self.interrupt=patch.object(m.mm,'throw_exception_if_processing_interrupted');self.interrupt.start();self.addCleanup(self.interrupt.stop)
        self.c=m.parse_config({'image_model':'qwen.safetensors','image_clip':'clip.safetensors','image_vae':'vae.safetensors',
                              'reference_cards':[{'slot':0,'enabled':True,'design':False,'text':' 原文：杯子\n保持原样 ','aspect':'1:1'}]})
        self.id='a'*32
    def run_aux(self,action='images',**kw):
        return r.H3ReferenceAssistant().run(json.dumps(self.c,ensure_ascii=False),'居家 VLOG',action,self.id,owner_id='1',**kw)
    def test_raw_prompt_stays_verbatim_even_with_style(self):
        self.c['style_text']='油画、蓝色'
        with patch.object(m,'design_completion') as llm:
            graph=self.run_aux()['expand'];llm.assert_not_called()
        encode=next(x for x in graph.values() if x['class_type']=='TextEncodeQwenImage21')
        self.assertEqual(encode['inputs']['prompt'],self.c['reference_cards'][0]['text'])
        self.assertEqual(sum(x['class_type']=='KSampler' for x in graph.values()),1)
        next_node=next(x for x in graph.values() if x['class_type']=='H3ReferenceAssistant')
        self.assertEqual(next_node['inputs']['previous'][1],0)
        self.assertTrue(all(x['override_display_id']=='1' for x in graph.values()))
    def test_intent_design_propagates_style_and_chinese(self):
        self.c['reference_cards'][0].update(design=True,text='杯子');self.c['style_text']='奶油色电影柔光'
        with patch.object(m,'design_completion',return_value='米白陶瓷杯，柔和晨光，单一主体。') as llm:
            result=self.run_aux('design')
        self.assertIn('奶油色电影柔光',llm.call_args.args[2]);self.assertIn('居家 VLOG',llm.call_args.args[2])
        self.assertIn('Simplified Chinese',llm.call_args.args[1]);self.assertEqual(llm.call_args.args[6],1024)
        self.assertFalse(result['result'][0]['ready'])
        self.assertTrue(r.read_job(self.id)['cards']['0']['prompt'])
    def test_images_requires_design_or_reviewed_draft_without_llm(self):
        self.c['reference_cards'][0].update(design=True,text='杯子')
        with patch.object(m,'design_completion') as llm,self.assertRaisesRegex(ValueError,'先设计'):
            self.run_aux();llm.assert_not_called()
        self.c['reference_cards'][0]['draft']='我的人工修订：釉面陶瓷杯。'
        with patch.object(m,'design_completion') as llm:
            graph=self.run_aux()['expand'];llm.assert_not_called()
        self.assertEqual(next(x['inputs']['prompt'] for x in graph.values() if x['class_type']=='TextEncodeQwenImage21'),'我的人工修订：釉面陶瓷杯。')
    def test_failed_design_preserves_other_completed_materials(self):
        job=r.create_job(self.id,'居家 VLOG');job['cards']['5']={'slot':5,'prompt':'既有图','path':'kept.png'};r.save_job(job)
        self.c['reference_cards'][0].update(design=True,text='杯子')
        with patch.object(m,'design_completion',side_effect=RuntimeError('native error')),self.assertRaises(RuntimeError):self.run_aux('all')
        job=r.read_job(self.id);self.assertEqual(job['cards']['5']['path'],'kept.png');self.assertEqual(job['stage'],'error')
    def test_serial_checkpoint_cache_and_selective_reroll(self):
        graph=self.run_aux()['expand'];commit=next(x['inputs'] for x in graph.values() if x['class_type']=='H3FSReferenceCommit')
        bundle=r.H3FSReferenceCommit().run(torch.zeros(1,32,32,3),**{k:v for k,v in commit.items() if k!='images'})
        self.assertEqual(bundle['ui']['images'][0]['type'],'output')
        result=self.run_aux(previous=bundle['result'][0]);self.assertNotIn('expand',result);self.assertTrue(result['result'][0]['ready'])
        self.assertEqual(result['result'][2].shape,(1,32,32,3))
        self.assertEqual(result['ui']['images'],r.output_images(r.read_job(self.id)))
        self.c['reference_cards'][0]['revision']=1
        self.assertIn('expand',self.run_aux())
        path=Path(r.read_job(self.id)['cards']['0']['path']);self.assertTrue(path.is_file());self.assertIn('参考图-01-',path.name)
    def test_generation_key_detects_changed_inputs(self):
        card=self.c['reference_cards'][0];key=r.generation_key(self.c,card,'prompt')
        for change in ({'image_steps':9},{'image_model':'other'},{'style_text':'新画风'},{'image_seed':99}):
            self.assertNotEqual(r.generation_key({**self.c,**change},card,'prompt'),key)
        self.assertNotEqual(r.generation_key(self.c,{**card,'aspect':'9:16'},'prompt'),key)
        self.assertNotEqual(r.generation_key(self.c,{**card,'text':'新物象'},'prompt'),key)
    def test_job_path_and_no_blank_slots_are_generated(self):
        with self.assertRaises(ValueError):r.job_path('../bad')
        self.c['reference_cards']=[]
        with self.assertRaisesRegex(ValueError,'至少启用'):self.run_aux()
    def test_different_model_families_have_native_graphs(self):
        self.c['image_engine']='krea2';self.c['image_steps']=8
        with patch.object(m,'filenames',return_value=['Krea2/krea2_turbo_sda.safetensors']):graph=self.run_aux()['expand']
        self.assertEqual(sum(x['class_type']=='KSampler' for x in graph.values()),2)
        self.assertTrue(any(x['class_type']=='LatentUpscale' for x in graph.values()))
    def test_one_click_fullstack_expands_assistant_before_video(self):
        self.c.update(auto_reference=True,reference_one_click=True,reference_job=self.id)
        result=m.H3FullStack().run(json.dumps(self.c),'居家 VLOG',42,unique_id='1')
        self.assertEqual([x['class_type'] for x in result['expand'].values()],['H3ReferenceAssistant','H3FullStack'])
        child=list(result['expand'].values())[1]['inputs'];self.assertNotIn('reference_image',child)
        self.assertEqual(child['reference_assets'][1],0)
    def test_review_stage_cannot_feed_unfinished_assets_to_video(self):
        with self.assertRaisesRegex(ValueError,'尚未完成'):
            m.H3FullStack().run('{}','居家 VLOG',42,reference_assets={'media':[],'ready':False})
