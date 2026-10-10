import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from test_pipeline import m

g=importlib.import_module('fullstack_test.image_generator')
media=importlib.import_module('fullstack_test.media')


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h3fs-generator-');self.addCleanup(self.tmp.cleanup)
        for obj,name,value in [(m.folder_paths,'get_output_directory',self.tmp.name),(m.folder_paths,'get_input_directory',self.tmp.name)]:
            p=patch.object(obj,name,return_value=value);p.start();self.addCleanup(p.stop)
        for name in ('notify','release_optimizer','node_class','require_file','persist_preview_index'):
            p=patch.object(m,name);p.start();self.addCleanup(p.stop)
        self.c=g.settings({'image_engine':'qwen21','image_model':'qwen_image_2.1.safetensors','image_clip':'clip.safetensors','image_vae':'vae.safetensors','image_steps':25})
        self.ref=torch.zeros(1,32,32,3)
    def run_node(self,intent='我的原文：蛋糕，米色背景。',**changes):
        c={**self.c,**changes};return g.H3ImageGenerator().run(json.dumps(c,ensure_ascii=False),intent,42,unique_id='3')
    def test_plain_text_no_reference_is_literal_and_never_calls_llm(self):
        with patch.object(m,'design_completion') as llm:out=self.run_node()
        llm.assert_not_called();graph=out['expand']
        self.assertEqual(next(n['inputs']['prompt'] for n in graph.values() if n['class_type']=='TextEncodeQwenImage21'),'我的原文：蛋糕，米色背景。')
        self.assertFalse(any('MiniMax' in n['class_type'] or n['class_type']=='H3FullStack' for n in graph.values()))
    def test_design_only_has_no_diffusion_graph(self):
        out=self.run_node(action='design');self.assertNotIn('expand',out)
        self.assertIsInstance(out['result'][0],m.ExecutionBlocker);self.assertEqual(out['result'][1],'我的原文：蛋糕，米色背景。')
    def test_recreate_requires_one_reference_not_edit_model(self):
        with self.assertRaisesRegex(ValueError,'需要一张'):self.run_node(constraint='recreate')
        c={**self.c,'constraint':'recreate'}
        with patch.object(m,'design_completion',return_value='奶油色褶边上衣，侧坐姿，暖色柔光写真。') as llm:
            out=g.H3ImageGenerator().run(json.dumps(c),'',42,self.ref,'3')
        self.assertIn('背景空间',llm.call_args.args[1]);self.assertIn('不描述脸型',llm.call_args.args[1])
        self.assertEqual([n['class_type'] for n in out['expand'].values()].count('EmptyLatentImage'),1)
        self.assertFalse(any('Edit' in n['class_type'] for n in out['expand'].values()))
    def test_card_is_one_sheet_with_editable_three_view_template(self):
        c={**self.c,'constraint':'character_card','card_template':'同一张图左右排版：正视图、侧视图、背视图。'}
        with patch.object(m,'design_completion',return_value='黑色长发、奶油色褶边服装，柔光摄影。') as llm:
            prompt=g.design(c,'CML',self.ref,'3',42)
        self.assertTrue(prompt.startswith(c['card_template']));self.assertIn('严格排除背景',llm.call_args.args[1]);self.assertIn('不描述脸型',llm.call_args.args[1])
        self.assertNotIn('参考图',prompt)
    def test_style_vision_never_sees_target_then_text_design_never_sees_person_reference(self):
        style=json.dumps({'medium':'photography','style':'暖色柔光，浅景深','confidence':.95,'evidence':'自然光学过渡'})
        with patch.object(m,'design_completion',side_effect=[style,'一座现实尺度的小院子，花草与砖墙，柔和晨光，无人物。']) as llm:
            prompt=g.design(self.c,'小院子',self.ref,'3',42)
        calls=llm.call_args_list
        self.assertNotIn('小院子',calls[0].args[2]);self.assertEqual(calls[0].args[3],[self.ref]);self.assertEqual(calls[1].args[3],[])
        self.assertIn('默认没有人物',calls[1].args[2]);self.assertIn('小院子',prompt)
    def test_reviewed_draft_skips_analysis_but_changed_reference_invalidates_it(self):
        c={**self.c,'constraint':'recreate','draft':'人工确认：奶油色衣服，柔光。'};c['draft_key']=g.prompt_key(c,'',self.ref)
        with patch.object(m,'design_completion') as llm:
            out=g.H3ImageGenerator().run(json.dumps(c),'',42,self.ref,'3')
        llm.assert_not_called();self.assertIn('人工确认',next(n['inputs']['prompt'] for n in out['expand'].values() if n['class_type']=='TextEncodeQwenImage21'))
        self.assertNotEqual(c['draft_key'],g.prompt_key(c,'',torch.ones_like(self.ref)))
    def test_commit_native_assets_metadata_and_provenance_survive_reload(self):
        result=g.H3FSImageCommit().run(self.ref,'中文提示词',json.dumps(self.c),'3','a'*32,'key')
        image=result['result'][0];self.assertIs(image,self.ref);self.assertEqual(image._h3fs_generated['id'],'a'*32)
        record=g.read_result('a'*32);self.assertTrue(Path(record['path']).is_file());self.assertEqual(result['ui']['images'][0],record['image'])
        with Image.open(record['path']) as im:self.assertEqual(json.loads(im.info['h3fs_generation'])['prompt'],'中文提示词')
        self.assertEqual(record['settings']['image_steps'],25)
    def test_empty_fanout_slots_accept_image_existing_slot_keeps_saved_image(self):
        images=torch.ones(1,32,32,3);images._h3fs_generated={'id':'a'*32,'path':'candidate.png'}
        existing=Path(self.tmp.name)/'original.png';Image.new('RGB',(32,32),(0,0,0)).save(existing)
        c=m.parse_config({'mode':'reference','media':[{'kind':'image','slot':0,'path':str(existing)}]})
        result=media.prepare(c,124,image_inputs={'image_1':images,'image_2':images,'image_3':images})
        self.assertEqual(len(result['images']),3);self.assertEqual(float(result['images']['ref_image_0'].sum()),0)
        self.assertGreater(float(result['images']['ref_image_1'].sum()),0);self.assertGreater(float(result['images']['ref_image_2'].sum()),0)
    def test_ordinary_native_image_inputs_still_override_upload(self):
        existing=Path(self.tmp.name)/'original.png';Image.new('RGB',(32,32),(0,0,0)).save(existing)
        c=m.parse_config({'mode':'reference','media':[{'kind':'image','slot':0,'path':str(existing)}]})
        result=media.prepare(c,124,image_inputs={'image_1':torch.ones(1,32,32,3)})
        self.assertGreater(float(result['images']['ref_image_0'].sum()),0)
    def test_validates_bad_controls_and_record_path(self):
        for change in ({'constraint':'invalid'},{'action':'video'},{'card_template':''},{'image_steps':0},{'aspect':'7:13'},{'no_face_description':'yes'}):
            with self.subTest(change=change),self.assertRaises(ValueError):g.settings(change)
        with self.assertRaises(ValueError):g.read_result('../../private')
    def test_qwen_and_krea_graphs_use_native_t2i_with_requested_seed_and_loras(self):
        from comfy_execution.graph_utils import GraphBuilder
        with patch.object(m,'filenames',return_value=['krea2_turbo_sda.safetensors']):
            for engine in ('qwen21','krea2'):
                c={**self.c,'image_engine':engine,'image_seed':81,'image_loras':[{'name':'CML.safetensors','weight':.7,'enabled':True}]}
                graph=GraphBuilder();g.image_graph(graph,c,{'slot':0,'revision':0,'aspect':'1:1','text':'蛋糕'},'原文', '3');rows=list(graph.finalize().values())
                self.assertTrue(any(n['class_type']=='LoraLoaderModelOnly' and n['inputs']['strength_model']==.7 for n in rows))
                self.assertTrue(all(n['inputs']['seed']==81 for n in rows if n['class_type']=='KSampler'))
                self.assertTrue(all(n['inputs']['batch_size']==1 for n in rows if n['class_type']=='EmptyLatentImage'))

    def test_standard_video_run_reuses_completed_image_without_llm_or_diffusion(self):
        c={**self.c,'image_seed':42}
        g.H3FSImageCommit().run(self.ref,'蛋糕，米色背景',json.dumps(c),'3','b'*32,'key')
        c.update(result_id='b'*32,draft='蛋糕，米色背景',draft_manual=True)
        with patch.object(m,'design_completion') as llm,patch.object(g,'image_graph') as graph:
            out=g.H3ImageGenerator().run(json.dumps(c),'蛋糕',42,unique_id='3')
        llm.assert_not_called();graph.assert_not_called();self.assertNotIn('expand',out)
        self.assertEqual(out['result'][0]._h3fs_generated['id'],'b'*32)
        with patch.object(m,'design_completion') as llm:
            out=g.H3ImageGenerator().run(json.dumps({**c,'force_regenerate':True}),'蛋糕',42,unique_id='3')
        self.assertIn('expand',out);llm.assert_not_called()

    def test_changed_generation_parameters_invalidate_result_reuse_not_reviewed_prompt(self):
        c={**self.c,'image_seed':42}
        g.H3FSImageCommit().run(self.ref,'蛋糕，米色背景',json.dumps(c),'3','c'*32,'key')
        c.update(result_id='c'*32,draft='蛋糕，米色背景',draft_manual=True,image_steps=26)
        with patch.object(m,'design_completion') as llm:out=g.H3ImageGenerator().run(json.dumps(c),'蛋糕',42,unique_id='3')
        llm.assert_not_called();self.assertIn('expand',out)
