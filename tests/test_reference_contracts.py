import importlib
import json
import unittest
from unittest.mock import patch
studio=importlib.import_module('test_reference_studio')
r,m=studio.r,studio.m

p=importlib.import_module('fullstack_test.reference_prompts')
l=importlib.import_module('fullstack_test.reference_learning')


def learned(**changes):
    return json.dumps({'subject_present':True,'features':'深棕色长发，淡粉色唇妆',
                       'wardrobe_and_accessories':'米白色无袖荷叶边上衣','visual_style':'暖色写实摄影',
                       'uncertainty':'鞋子不可见',**changes},ensure_ascii=False)


class AssetContractTests(unittest.TestCase):
    def test_scene_defaults_to_real_life_scale_but_explicit_miniatures_are_allowed(self):
        card={'text':'小院子','design':True}
        self.assertIn('现实生活尺度',p.scene_scale_rule(card,True))
        self.assertIn('环境内部',p.scene_scale_rule(card,True))
        self.assertEqual(p.scene_scale_rule({'text':'小院子','design':False},True),'')
        self.assertEqual(p.scene_scale_rule({'text':'微缩小院子','design':True},True),'')
        self.assertIn('微缩 / 模型尺度',p.asset_contract({'text':'微缩小院子'})[1])
        self.assertIn('现实生活尺度',p.scene_scale_rule({'text':'小院子，不要微缩模型','design':True},True))

    def test_transferable_style_does_not_prescribe_source_location(self):
        raw=json.dumps({'medium':'photography','evidence':'真实纹理','style':'暖色室内漫射光，低对比，浅景深','confidence':.9})
        self.assertEqual(p.parse_style(raw)['style'],'暖色漫射光，低对比，浅景深')
        self.assertNotIn('studio',p.portable_style('warm studio light'))

    def test_nouns_are_independent_assets_not_vlog_protagonists(self):
        for noun,kind in [('美女','person'),('小院子','scene'),('蛋糕','object'),('项链','object'),('自习室','scene'),('女人拿着蛋糕','person'),('陌生物象','auto')]:
            self.assertEqual(p.asset_kind(noun),kind)

    def test_explicit_kind_wins_and_negated_people_do_not_create_person_assets(self):
        for noun,kind in [('成年女性','person'),('成年男性','person'),('女主','person'),('人物模特','person'),('没有人物的小院子','scene'),('不含人物的蛋糕','object'),('courtyard without people','scene'),('没有人物的蛋糕，旁边另有成年女性','person')]:
            with self.subTest(noun=noun): self.assertEqual(p.asset_kind(noun),kind)
        self.assertEqual(p.asset_kind('人物参考，小院子','scene'),'scene')

    def test_courtyard_and_cake_reject_accidental_woman(self):
        for noun,prompt in [('小院子','一位年轻女性坐在小院子，人物近景。'),('蛋糕','一位美女拿着蛋糕。')]:
            with self.assertRaisesRegex(ValueError,'混入'):p.validate_anchor(prompt,{'text':noun})
        p.validate_anchor('蛋糕独立摆在桌上，无人物或人手。',{'text':'蛋糕'})
        p.validate_anchor('一位女性拿着蛋糕。',{'text':'蛋糕','kind':'person'})
        with self.assertRaisesRegex(ValueError,'目标物象'):p.validate_anchor('一只陶瓷杯，窗边光线。',{'text':'蛋糕'})

    def test_style_evidence_and_explicit_medium_cannot_conflict(self):
        raw=json.dumps({'medium':'illustration','evidence':'暖色柔焦','style':'数字插画风格，暖色柔焦，笔触细腻。','confidence':.5})
        report=p.parse_style(raw)
        self.assertEqual(report['medium'],'unknown');self.assertNotIn('插画',report['style']);self.assertNotIn('笔触',report['style'])
        self.assertEqual(p.parse_style(raw,'photography')['medium'],'photography')
        for bad in ('not json','[]','{"medium":"wrong"}'):
            with self.assertRaises(ValueError):p.parse_style(bad)


class AssetFlowTests(unittest.TestCase):
    setUp=studio.StudioTests.setUp
    run_aux=studio.StudioTests.run_aux
    def test_scene_scale_rule_reaches_actual_encoder_and_effective_prompt(self):
        self.c['reference_cards'][0].update(text='小院子',design=True,draft='现实小庭院，石板路和围墙，平视全景。')
        with patch.object(m,'design_completion') as llm: graph=self.run_aux()['expand'];llm.assert_not_called()
        encoder=next(x for x in graph.values() if x['class_type']=='TextEncodeQwenImage21')
        self.assertIn('现实生活尺度',encoder['inputs']['prompt'])
        job=r.read_job(self.id)
        self.assertEqual(encoder['inputs']['prompt'],r.effective_prompt(self.c,self.c['reference_cards'][0],job['cards']['0']))

    def test_bounded_correction_and_anchor_persisted(self):
        self.c['reference_cards'][0].update(design=True,text='小院子')
        with patch.object(m,'design_completion',side_effect=['一位女性坐在小院子。','小院子有白墙与绿植，石板地面清晰可见，无人物。']) as llm:
            self.run_aux('design')
        self.assertEqual(llm.call_count,2)
        row=r.read_job(self.id)['cards']['0'];self.assertEqual(row['kind'],'scene');self.assertNotIn('女性',row['prompt'])

    def test_twice_wrong_prompt_stops_instead_of_producing_wrong_image(self):
        self.c['reference_cards'][0].update(design=True,text='蛋糕')
        with patch.object(m,'design_completion',return_value='年轻美女拿着蛋糕。') as llm,self.assertRaisesRegex(ValueError,'混入'):
            self.run_aux('all')
        self.assertEqual(llm.call_count,2);self.assertEqual(r.read_job(self.id)['stage'],'error')

    def test_review_only_registers_images_without_llm_or_diffusion(self):
        graph=self.run_aux()['expand'];commit=next(x['inputs'] for x in graph.values() if x['class_type']=='H3FSReferenceCommit')
        import torch
        r.H3FSReferenceCommit().run(torch.zeros(1,32,32,3),**{k:v for k,v in commit.items() if k!='images'})
        with patch.object(m,'design_completion') as llm,patch.object(r,'image_graph') as image:
            result=self.run_aux('review');llm.assert_not_called();image.assert_not_called()
        self.assertEqual(len(result['ui']['images']),1)

    def test_old_design_cache_is_not_trusted(self):
        self.c['reference_cards'][0].update(design=True,text='蛋糕')
        job=r.create_job(self.id,'居家 VLOG');job['cards']['0']={'slot':0,'text':'蛋糕','prompt':'美女拿着蛋糕。','design_key':'old'};r.save_job(job)
        with patch.object(m,'design_completion',return_value='蛋糕独立置于白瓷盘，草莓奶油层次清楚，无人物。') as llm:self.run_aux('design')
        self.assertEqual(llm.call_count,1);self.assertEqual(r.read_job(self.id)['contract_version'],p.CONTRACT_VERSION)

    def test_style_texture_cannot_become_cake_fabric(self):
        raw=json.dumps({'medium':'photography','evidence':'自然景深','style':'暖色柔光。材质呈半透明纱感与光滑肤质，背景轻度虚化。','confidence':.85})
        report=p.parse_style(raw);self.assertNotIn('纱',report['style']);self.assertNotIn('肤质',report['style']);self.assertIn('暖色柔光',report['style'])

    def test_reviewed_wrong_draft_does_not_start_diffusion(self):
        self.c['reference_cards'][0].update(design=True,text='蛋糕',draft='美女拿着蛋糕。')
        with patch.object(r,'image_graph') as image,self.assertRaisesRegex(ValueError,'混入'):self.run_aux()
        image.assert_not_called()

    def test_native_output_rejects_paths_outside_output(self):
        self.assertEqual(r.output_images({'cards':{'0':{'slot':0,'path':__file__}}}),[])

    def test_shared_person_reference_does_not_leak_into_cake_or_courtyard(self):
        self.c.update(strict_person_reference=True,style_image='person.png',scene_reference='courtyard.png')
        self.assertEqual(p.content_reference(self.c,{'text':'美女'}),('person.png','person'))
        self.assertEqual(p.content_reference(self.c,{'text':'蛋糕'}),('','object'))
        self.assertEqual(p.content_reference(self.c,{'text':'小院子'}),('courtyard.png','scene'))
        self.assertEqual(p.content_reference(self.c,{'text':'美女','reference_image':'own.png','strict_reference':True}),('own.png','person'))

    def test_strict_without_reference_is_rejected(self):
        self.c['reference_cards'][0].update(text='美女',strict_reference=True)
        with self.assertRaisesRegex(ValueError,'请上传'):self.run_aux()

    def test_qwen_strict_profile_keeps_text_to_image_not_edit(self):
        card=self.c['reference_cards'][0];card.update(reference_image='cup.png',strict_reference=True)
        graph=r.GraphBuilder();r.image_graph(graph,self.c,card,'杯子','1');nodes=graph.finalize()
        encoder=next(x for x in nodes.values() if x['class_type']=='TextEncodeQwenImage21')
        self.assertNotIn('image_1',encoder['inputs']);self.assertNotIn('images',encoder['inputs'])
        sample=next(x for x in nodes.values() if x['class_type']=='KSampler')
        self.assertEqual(sample['inputs']['latent_image'][1],0);self.assertTrue(any(x['class_type']=='EmptyLatentImage' for x in nodes.values()))
        self.assertFalse(any(x['class_type']=='H3FSReferenceImage' for x in nodes.values()))

    def test_strict_scene_lock_does_not_trip_no_people_guard(self):
        prompt=l.compose_prompt('小院子具有米白墙面。','米黄墙面，石板路','scene','',True)
        p.validate_anchor(prompt,{'text':'小院子'})

    def test_uncertain_hidden_details_are_not_generation_requirements(self):
        report=l.parse_profile(learned(), 'person',True)
        self.assertNotIn('鞋子',report['description']);self.assertIn('唇妆',report['description'])

    def test_learning_sees_pixels_but_image_design_uses_only_reusable_text(self):
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        self.c['reference_cards'][0].update(text='美女',design=True,reference_image=str(source),strict_reference=True)
        with patch.object(m,'design_completion',side_effect=[learned(),'一位成年女性在新设计的庭院散步，侧面中景，柔和光线。']) as llm:self.run_aux('design')
        self.assertEqual(len(llm.call_args_list[0].args[3]),1);self.assertEqual(llm.call_args_list[1].args[3],[])
        row=r.read_job(self.id)['cards']['0'];self.assertIn('无袖荷叶边',row['prompt']);self.assertNotIn('脸型',row['reference_description']);self.assertNotIn('鞋子',row['prompt'])

    def test_krea_accepts_strict_text_profile_without_any_edit_conditioning(self):
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        self.c['image_engine']='krea2';self.c['reference_cards'][0].update(reference_image=str(source),strict_reference=True,reference_description='白色陶瓷杯，圆柱形杯身')
        with patch.object(m,'design_completion') as llm,patch.object(m,'filenames',return_value=['Krea2/krea2_turbo_sda.safetensors']):graph=self.run_aux('all')['expand']
        llm.assert_not_called();self.assertEqual(sum(x['class_type']=='KSampler' for x in graph.values()),2)
        encoder=next(x for x in graph.values() if x['class_type']=='CLIPTextEncode');self.assertIn('白色陶瓷杯',encoder['inputs']['text'])

    def test_content_file_change_invalidates_image_cache(self):
        import os
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        card=self.c['reference_cards'][0];card.update(reference_image=str(source),strict_reference=True)
        before=r.generation_key(self.c,card,'杯子');stat=source.stat()
        os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
        self.assertNotEqual(before,r.generation_key(self.c,card,'杯子'))

    def test_bad_strict_toggle_cannot_silently_enable_reference(self):
        with self.assertRaisesRegex(ValueError,'boolean'):
            m.parse_config({'reference_cards':[{'slot':0,'strict_reference':'false'}]})

    def test_strict_raw_prompt_remains_verbatim(self):
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        original='  保持参考图中的杯子，侧面构图。\n'
        self.c['reference_cards'][0].update(text=original,design=False,reference_image=str(source),strict_reference=True,reference_description='白色陶瓷杯，圆柱形杯身')
        with patch.object(m,'design_completion') as llm:
            graph=self.run_aux()['expand'];llm.assert_not_called()
        encoder=next(x for x in graph.values() if x['class_type']=='TextEncodeQwenImage21')
        self.assertTrue(encoder['inputs']['prompt'].startswith(original));self.assertIn('白色陶瓷杯',encoder['inputs']['prompt']);self.assertNotIn('image_1',encoder['inputs'])

    def test_old_edit_loader_is_no_longer_registered(self):
        self.assertNotIn('H3FSReferenceImage',r.NODE_CLASS_MAPPINGS)

    def test_person_profiles_reject_background_and_facial_identity_but_allow_makeup(self):
        for features in ('黑色长发，坐在沙发上','鹅蛋脸，细长鼻梁','oval face, black hair',
                         '黑色长发，大眼睛，薄唇','面部轮廓柔和，双眼皮','black hair, almond-shaped eyes, full lips'):
            with self.assertRaises(ValueError):l.parse_profile(learned(features=features),'person',True)
        self.assertIn('唇妆',l.parse_profile(learned(),'person',True)['description'])
        self.assertNotIn('室内',l.parse_profile(learned(visual_style='室内暖色柔光'),'person',True)['description'])

    def test_analyze_alone_works_before_filling_objects_and_never_starts_diffusion(self):
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        self.c.update(reference_cards=[],style_image=str(source),strict_person_reference=True,style_override='写实摄影')
        with patch.object(m,'design_completion',return_value=learned()) as llm,patch.object(r,'image_graph') as images:
            result=self.run_aux('analyze');images.assert_not_called()
        self.assertEqual(llm.call_count,1);self.assertFalse(result['result'][0]['ready'])
        self.assertIn('荷叶边',r.read_job(self.id)['shared_profiles']['person']['description'])

    def test_profile_cache_is_reused_and_manual_override_is_authoritative(self):
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        self.c.update(reference_cards=[],style_image=str(source),strict_person_reference=True,style_override='写实摄影')
        with patch.object(m,'design_completion',return_value=learned()) as llm:
            self.run_aux('analyze');self.run_aux('analyze');self.assertEqual(llm.call_count,1)
            self.c['person_profile_override']='深色长发，淡粉色唇妆，米白色无袖上衣'
            self.run_aux('analyze');self.assertEqual(llm.call_count,1)
        self.assertEqual(r.read_job(self.id)['shared_profiles']['person']['description'],self.c['person_profile_override'])

    def test_camera_can_change_without_copying_the_reference_view(self):
        result=l.compose_prompt('成年女性在新庭院中散步。','黑色长发，白色上衣','person','低机位侧面中景',True)
        self.assertIn('低机位侧面中景',result);self.assertIn('不含背景',result);self.assertNotIn('沙发',result)

    def test_unrecognizable_person_is_rejected_instead_of_invented(self):
        with self.assertRaisesRegex(ValueError,'没有可识别'):l.parse_profile(learned(subject_present=False),'person',True)

    def test_registered_character_lora_and_trigger_do_not_leak_into_cake(self):
        self.c['image_loras']=[{'name':'Krea2/CML_character.safetensors','enabled':True,'weight':1}]
        card=self.c['reference_cards'][0]
        for noun,expect in [('美女',True),('蛋糕',False),('小院子',False)]:
            card.update(text=noun)
            graph=r.GraphBuilder();r.image_graph(graph,self.c,card,noun,'1');nodes=graph.finalize()
            self.assertEqual(any(x['class_type']=='LoraLoaderModelOnly' for x in nodes.values()),expect)
            self.assertEqual(bool(l.triggers_for(self.c,p.asset_kind(noun))),expect)
        self.assertEqual(l.character_trigger('Qwen/CML_v1.safetensors'),'CML')
        self.assertEqual(l.character_trigger('Qwen/XNJ-v2.safetensors'),'XNJ')

    def test_auto_person_prompt_cannot_override_lora_face(self):
        self.c['reference_cards'][0].update(text='美女',design=True)
        with patch.object(m,'design_completion',side_effect=['一位鹅蛋脸美女，细长鼻梁。','成年女性，淡粉色唇妆，白色上衣，侧面中景。']) as llm:
            self.run_aux('design')
        self.assertEqual(llm.call_count,2);self.assertNotIn('鹅蛋脸',r.read_job(self.id)['cards']['0']['prompt'])

    def test_scene_profile_is_valid_through_the_actual_sampler_graph(self):
        from PIL import Image
        source=__import__('pathlib').Path(self.tmp.name)/'source.png';Image.new('RGB',(32,64)).save(source)
        self.c['reference_cards'][0].update(text='小院子',design=True,draft='小院子中米白墙面与石板路，低角度全景。',reference_image=str(source),strict_reference=True,reference_description='米白墙面，石板路，陶盆绿植')
        with patch.object(m,'design_completion') as llm: graph=self.run_aux()['expand'];llm.assert_not_called()
        encoder=next(x for x in graph.values() if x['class_type']=='TextEncodeQwenImage21');self.assertIn('场景设定',encoder['inputs']['prompt']);self.assertNotIn('image_1',encoder['inputs'])
