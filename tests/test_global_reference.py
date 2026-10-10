import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from test_pipeline import m

g=importlib.import_module('fullstack_test.global_reference')
studio=importlib.import_module('fullstack_test.reference_studio')
generator=importlib.import_module('fullstack_test.image_generator')


class GlobalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h3fs-global-test-');self.addCleanup(self.tmp.cleanup)
        for name in ('get_output_directory','get_input_directory'):
            p=patch.object(m.folder_paths,name,return_value=self.tmp.name);p.start();self.addCleanup(p.stop)
        for name in ('notify','release_optimizer','node_class','require_file','persist_preview_index'):
            p=patch.object(m,name);p.start();self.addCleanup(p.stop)
        self.im={'image_engine':'qwen21','image_model':'qwen_image_2.1.safetensors','image_clip':'clip.safetensors','image_vae':'vae.safetensors','aspect':'video','image_steps':25,
                 'image_loras':[{'name':'CML.safetensors','weight':1,'enabled':True},{'name':'style.safetensors','weight':.5,'enabled':True}]}
        self.c=m.parse_config({'mode':'reference','aspect':'16:9','global_reference':{'enabled':True,'action':'images','image_settings':self.im,'cards':[{'slot':0,'enabled':True,'text':'小院子'},{'slot':2,'enabled':True,'text':'蛋糕'}]}})
        self.ref=Path(self.tmp.name)/'portrait.png';Image.new('RGB',(32,32),'white').save(self.ref)
    def test_validation_rejects_duplicate_slots_and_unknown_actions(self):
        for bad in ({'action':'delete'},{'person':'yes'},{'job_id':'../secret'},{'cards':[{'slot':0},{'slot':0}]}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):g.validate_global(bad)
        for bad in ({'cards':[{'slot':0,'kind':'bad'}]},{'cards':[{'slot':0,'aspect':'bad'}]}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):g.validate_global(bad)
    def test_slot_types_override_nouns_and_each_aspect_defaults_to_video(self):
        card=g.validate_global({'cards':[{'slot':0,'text':'小院子'}]})['cards'][0]
        self.assertEqual(card['aspect'],'video');self.assertEqual(card['kind'],'auto')
        self.assertEqual(g.image_config(self.c,card)['aspect'],'16:9')
        card['aspect']='9:16';self.assertEqual(g.image_config(self.c,card)['aspect'],'9:16')
    def test_shared_planner_wins_over_stale_image_preset(self):
        self.c['llm_backend']='api'
        self.c['llm_api']={'provider':'openai','model':'test','base_url':'https://api.openai.com/v1'}
        self.c['global_reference']['image_settings'].update(llm_backend='local',llm_api={})
        planned=g.image_config(self.c)
        self.assertEqual(planned['llm_backend'],'api')
        self.assertEqual(planned['llm_api']['model'],'test')
    def test_person_global_never_injects_profile_into_explicit_other_type(self):
        self.c['global_reference'].update(image=str(self.ref),person=True,cards=[{'slot':0,'enabled':True,'text':'庭院','kind':'scene','aspect':'9:16'},{'slot':1,'enabled':True,'text':'蛋糕','kind':'object','aspect':'1:1'}])
        job=studio.create_job('1'*32,'居家人物近景VLOG')
        style=json.dumps({'medium':'photography','style':'暖色柔光','confidence':.9,'evidence':'自然照明'})
        with patch.object(m,'design_completion',side_effect=[style,'庭院实景，其他花草围绕砖墙，柔光。','独立蛋糕，柔光。']) as llm:
            g.plan(self.c,'人物在小院近景玩耍',job,'1')
        self.assertEqual(llm.call_count,3);self.assertNotIn('必须保持的人物',llm.call_args_list[1].args[2]);self.assertNotIn('CML',job['cards']['0']['prompt'])
        self.assertEqual(job['cards']['0']['aspect'],'9:16');self.assertEqual(job['cards']['1']['aspect'],'1:1')
    def test_bad_object_prompt_is_corrected_once_not_immediately_failed(self):
        self.c['global_reference']['cards']=self.c['global_reference']['cards'][:1]
        with patch.object(m,'design_completion',side_effect=['一位美女在小院子微笑。','小院子实景，绿植与石板路，暖色柔光。']) as llm:
            job=studio.create_job('2'*32,'人物VLOG');g.plan(self.c,'人物近景',job,'1')
        self.assertEqual(llm.call_count,2);self.assertNotIn('视频总意图',llm.call_args.args[2]);self.assertNotIn('美女',job['cards']['0']['prompt'])
    def test_scene_aspect_reaches_real_graph_size_not_just_ui(self):
        self.c['global_reference']['cards']=[{'slot':0,'enabled':True,'text':'小院子','kind':'scene','aspect':'9:16'}]
        with patch.object(m,'design_completion',return_value='真实小院子，绿植与石板路。'):
            out=g.H3GlobalReferences().run(json.dumps(self.c),'VLOG','1','3'*32)
        latent=next(n['inputs'] for n in out['expand'].values() if n['class_type']=='EmptyLatentImage')
        self.assertLess(latent['width'],latent['height'])
    def test_image_model_params_independent_video_aspect_supported(self):
        im=g.image_config(self.c);self.assertEqual(im['aspect'],'16:9');self.assertEqual(im['image_steps'],25);self.assertNotEqual(im['image_steps'],self.c['steps'])

    def test_manual_image_prompts_and_retakes_do_not_relearn_reference_or_call_llm(self):
        self.c['global_reference'].update(image=str(self.ref),person=True)
        for card in self.c['global_reference']['cards']:card['draft']=card['text']+'独立素材，柔和照明，无人物。'
        job=studio.create_job('8'*32,'居家VLOG')
        with patch.object(m,'design_completion') as llm:
            g.plan(self.c,'居家VLOG',job,'1');g.plan(self.c,'居家VLOG',job,'1')
        llm.assert_not_called();self.assertEqual(job['cards']['0']['prompt'],self.c['global_reference']['cards'][0]['draft'])

    def test_completed_automatic_prompts_remain_compatible_without_manual_draft(self):
        job=studio.create_job('7'*32,'居家VLOG')
        with patch.object(m,'design_completion',side_effect=['小院子实景，无人物。','独立蛋糕，柔光。']):g.plan(self.c,'居家VLOG',job,'1')
        old=studio.signature([3,'小院子','scene','居家VLOG',job['style'],'','16:9'])
        self.assertEqual(job['cards']['0']['design_key'],old)
        with patch.object(m,'design_completion') as llm:g.plan(self.c,'居家VLOG',job,'1')
        llm.assert_not_called()

    def test_interrupted_one_click_reuses_ready_prompt_only_for_unchanged_inputs(self):
        from fullstack_test.prompt_progress import publish,input_key
        self.c['global_reference'].update(action='all',job_id='9'*32)
        self.c['media']=[{'kind':'image','slot':0,'path':str(self.ref),'start':0,'end':0}]
        job=studio.create_job('9'*32,'VLOG');studio.save_job(job)
        publish(m,self.c,'VLOG','已完成视频提示词','1','ready')
        fake={'images':{'ref_image_0':'image'},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        bundle={'global':True,'media':self.c['media'],'ready':True,'job_id':'9'*32}
        with patch.object(m,'prepare',return_value=fake),patch.object(m,'optimize') as llm,patch.object(m,'validate_dependencies'):
            out=m.H3FullStack().run(json.dumps(self.c),'VLOG',42,reference_assets=bundle)
        llm.assert_not_called();self.assertIn('expand',out)
        self.c['seconds']=11;self.assertNotEqual(input_key(self.c),studio.read_job('9'*32)['video_prompt']['input_key'])
    def test_style_analyzed_once_and_objects_do_not_receive_portrait(self):
        self.c['global_reference']['image']=str(self.ref);job=studio.create_job('a'*32,'居家VLOG')
        style=json.dumps({'medium':'photography','style':'暖色柔光，浅景深','confidence':.9,'evidence':'自然光学'})
        with patch.object(m,'design_completion',side_effect=[style,'现实尺度的小院子，砖墙花草。','独立的一块蛋糕，瓷盘，柔和光线。']) as llm:
            g.plan(self.c,'居家VLOG',job,'1');g.plan(self.c,'居家VLOG',job,'1')
        self.assertEqual(llm.call_count,3)
        self.assertEqual(llm.call_args_list[1].args[3],[]);self.assertNotIn('CML',job['cards']['0']['prompt']);self.assertIn('小院子',job['cards']['0']['prompt'])
    def test_person_profile_only_for_person_and_excludes_face(self):
        self.c['global_reference'].update(image=str(self.ref),person=True,cards=[{'slot':0,'enabled':True,'text':'美女'},{'slot':1,'enabled':True,'text':'蛋糕'}])
        job=studio.create_job('b'*32,'短片');style=json.dumps({'medium':'photography','style':'柔光','confidence':.9,'evidence':'自然照明'})
        profile=json.dumps({'subject_present':True,'features':'黑色长发，淡妆','wardrobe_and_accessories':'奶油色褶边无袖上衣','visual_style':'柔光摄影','uncertainty':''})
        with patch.object(m,'design_completion',side_effect=[style,profile,'成年女性，黑色长发，奶油色褶边上衣。','独立蛋糕，柔光。']) as llm:g.plan(self.c,'短片',job,'1')
        self.assertIn('奶油色褶边',llm.call_args_list[2].args[2]);self.assertNotIn('奶油色褶边',llm.call_args_list[3].args[2]);self.assertTrue(job['cards']['0']['prompt'].startswith('CML, '))
        self.assertIn('半身取景不能写裙摆、腿、鞋、袜',llm.call_args_list[2].args[1]);self.assertIn('头顶到腰部',job['cards']['0']['prompt'])
    def test_serial_expansion_skips_identity_lora_for_objects(self):
        with patch.object(m,'design_completion',side_effect=['现实尺度的小院子，花草。','独立蛋糕，柔光。']):
            out=g.H3GlobalReferences().run(json.dumps(self.c),'居家VLOG','1','c'*32)
        rows=list(out['expand'].values());self.assertEqual(sum(x['class_type']=='H3GlobalCommit' for x in rows),1)
        loras=[x['inputs']['lora_name'] for x in rows if x['class_type']=='LoraLoaderModelOnly'];self.assertEqual(loras,['style.safetensors'])
        self.assertFalse(any('MiniMaxH3' in x['class_type'] for x in rows));self.assertTrue(any(x['class_type']=='H3GlobalReferences' for x in rows))
    def test_commit_one_directory_native_assets_and_completed_reuse(self):
        self.c['global_reference']['cards']=self.c['global_reference']['cards'][:1]
        with patch.object(m,'design_completion',return_value='现实尺度的小院子，花草。'):
            out=g.H3GlobalReferences().run(json.dumps(self.c),'居家VLOG','1','d'*32)
        commit=next(x['inputs'] for x in out['expand'].values() if x['class_type']=='H3GlobalCommit')
        saved=g.H3GlobalCommit().run(torch.zeros(1,32,32,3),commit['job_id'],commit['slot'],commit['generation_key'],commit['owner_id'])
        self.assertIn('总体参考图/居家VLOG-',saved['ui']['images'][0]['subfolder'])
        with patch.object(m,'design_completion') as llm:
            result=g.H3GlobalReferences().run(json.dumps(self.c),'居家VLOG','1','d'*32)
        llm.assert_not_called();self.assertEqual(result[0]['ready'],True);self.assertEqual(result[0]['media'][0]['slot'],0)
    def test_main_pause_blocks_videos_and_does_not_optimize(self):
        self.c['global_reference']['job_id']='e'*32
        job=studio.create_job('e'*32,'VLOG');studio.save_job(job)
        bundle={'global':True,'media':[{'kind':'image','slot':0,'path':str(self.ref)}],'ready':True,'job_id':'e'*32}
        with patch.object(m,'optimize') as llm,patch.object(m,'validate_dependencies') as deps:
            out=m.H3FullStack().run(json.dumps(self.c),'VLOG',42,unique_id='1',reference_assets=bundle)
        llm.assert_not_called();deps.assert_not_called();self.assertNotIn('expand',out);self.assertIsInstance(out['result'][0],m.ExecutionBlocker)
    def test_main_batch_rejects_selected_external_inputs_without_deleting_media(self):
        with self.assertRaisesRegex(ValueError,'已接线'):m.H3FullStack().run(json.dumps(self.c),'VLOG',42,image_1=torch.zeros(1,32,32,3))
    def test_changed_video_intent_uses_saved_images_without_replanning(self):
        self.c['global_reference']['action']='prompt';self.c['media']=[{'kind':'image','slot':0,'path':str(self.ref)}]
        fake={'images':{'ref_image_0':'image'},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        with patch.object(g,'plan') as plan,patch.object(m,'prepare',return_value=fake),patch.object(m,'optimize',return_value='新的H3提示词') as llm:
            out=m.H3FullStack().run(json.dumps(self.c),'修改后的新意图',42,unique_id='1')
        plan.assert_not_called();self.assertEqual(llm.call_args.args[1],'修改后的新意图');self.assertEqual(out['result'][2],'新的H3提示词')
    def test_one_click_enters_h3_optimizer_after_materials(self):
        self.c['global_reference']['action']='all';self.c['optimize_on_run']=False
        fake={'images':{'ref_image_0':'image'},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        bundle={'global':True,'media':[{'kind':'image','slot':0,'path':str(self.ref)}],'ready':True,'job_id':'f'*32}
        with patch.object(m,'prepare',return_value=fake),patch.object(m,'optimize',return_value='完整视频提示词') as llm,patch.object(m,'validate_dependencies'):
            out=m.H3FullStack().run(json.dumps(self.c),'新意图',42,reference_assets=bundle)
        llm.assert_called_once();self.assertTrue(any(x['class_type']=='H3FSConditioning' for x in out['expand'].values()))
    def test_lora_change_does_not_invalidate_image_prompt(self):
        c=generator.settings({**self.im,'aspect':'9:16'});c['constraint']='recreate';ref=torch.zeros(1,32,32,3)
        key=generator.prompt_key(c,'',ref);c.update(draft='CML, 奶油色服装，柔光。',draft_key=key,image_loras=[{'name':'XNJ.safetensors','weight':.8,'enabled':True}])
        self.assertEqual(key,generator.prompt_key(c,'',ref))
        with patch.object(m,'design_completion') as llm:out=generator.H3ImageGenerator().run(json.dumps(c),'',42,ref,'3')
        llm.assert_not_called();text=next(x['inputs']['prompt'] for x in out['expand'].values() if x['class_type']=='TextEncodeQwenImage21');self.assertTrue(text.startswith('XNJ, '));self.assertNotIn('CML',text)
    def test_default_card_template_keeps_head_and_feet_with_30_percent_margin(self):
        self.assertIn('整个头部',generator.CARD_TEMPLATE);self.assertIn('70%',generator.CARD_TEMPLATE);self.assertIn('各留15%',generator.CARD_TEMPLATE)
        generator.validate_no_face(generator.CARD_TEMPLATE)
        custom=generator.LEGACY_CARD_TEMPLATE+'按用户要求使用蓝色背景。'
        self.assertEqual(generator.settings({'card_template':custom})['card_template'],custom)
        self.assertEqual(generator.settings({'card_template':generator.LEGACY_CARD_TEMPLATE})['card_template'],generator.CARD_TEMPLATE)
    def test_preview_rate_matches_full_grid_duration_without_loading_audio(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from fractions import Fraction
        wrapper=MagicMock();model=MagicMock()
        with patch.object(m,'node_class',return_value=SimpleNamespace(__module__='fake_preview')),patch.object(m.importlib,'import_module',return_value=SimpleNamespace(_PreviewOverrideWrapper=wrapper)):
            m.H3FSTimelinePreview().run(model,384,243,'taeh3.safetensors','1')
        kw=wrapper.call_args.kwargs;self.assertEqual(kw['preview_frames'],72);self.assertEqual(kw['preview_fps'],Fraction(1728,243));self.assertEqual(kw['preview_frames']/kw['preview_fps'],Fraction(243,24));self.assertNotIn('audio_vae',kw)
