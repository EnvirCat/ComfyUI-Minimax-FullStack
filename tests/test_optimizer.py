import importlib
import json
import sys
import unittest
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import torch

sys.path.insert(0,str(Path(__file__).parent))
from test_pipeline import m
from ref_fixture import ref_prompt

class OptimizerContractTests(unittest.TestCase):
    def test_qwen_vision_loader_uses_minimum_tokens_and_releases_storage(self):
        media={'images':{'ref_image_0':torch.zeros(1,32,32,3)},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        storage=SimpleNamespace(llm=None,chat_handler=None,current_config=None,clean_state=Mock())
        def clean(**kwargs):storage.llm=None;storage.chat_handler=None;storage.current_config=None
        storage.clean=Mock(side_effect=clean)
        def load(name,**kwargs):
            storage.llm=SimpleNamespace(tokenize=lambda _:[],reset=Mock());storage.current_config=kwargs
            return ('model',)
        module=SimpleNamespace(LLAMA_CPP_STORAGE=storage)
        prompt=ref_prompt(chinese=False)
        completion=Mock(return_value=(prompt,{'finish_reason':'stop','max_tokens':1536}))
        loader=type('Loader',(),{'__module__':'mock_llama'})
        with tempfile.TemporaryDirectory(prefix='h3fs-optimizer-test-') as directory,patch.object(m,'require_file'),patch.object(m,'node_class',return_value=loader),patch.object(m,'optimizer_completion',completion),patch.object(m.importlib,'import_module',return_value=module),patch.object(m,'calibrated_loader_budget',return_value=14),patch.object(m,'call',side_effect=load) as call,patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m.folder_paths,'get_output_directory',return_value=directory):
            self.assertEqual(m.optimize({**m.DEFAULTS,'use_chinese_prompt':False,'llm_mmproj':'projector.gguf'},'A cup in morning light.',media,'reference',42,'test'),prompt)
        self.assertEqual(call.call_args.kwargs['image_min_tokens'],1024)
        self.assertEqual(call.call_args.kwargs['image_max_tokens'],1024)
        self.assertTrue(all(call.kwargs == {'all':True} for call in storage.clean.call_args_list))
        completion.assert_called_once()
    def test_chinese_contract_preserves_all_fields_labels_and_original_dialogue(self):
        media={'images':{},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        for mode in ('text','reference','first_last'):
            system,user,_=m.optimizer_request({**m.DEFAULTS,'use_chinese_prompt':True},'她说 Hello，缓慢右移。',media,mode,42)
            self.assertIn('Simplified Chinese',system);self.assertIn('Simplified Chinese',user)
            self.assertIn('same level of visual/audio detail',system);self.assertIn('retention markers',system)
            self.assertIn('<Picture N>',system);self.assertIn('verbatim in their original language',system)
            self.assertIn('overall_soundscape',system);self.assertIn('non_diegetic_music',system)
            if mode=='reference':self.assertIn('retention_analysis',system)
            self.assertIn('她说 Hello，缓慢右移。',user)
    def test_visual_attachment_positions(self):
        c={**m.DEFAULTS,'mode':'reference'}
        media={'images':{'ref_image_0':torch.ones(1,32,32,3),'ref_image_1':torch.zeros(1,32,32,3)},
               'videos':{'ref_video_0':torch.zeros(24,32,32,3)},'paired':{},'audios':{},'descriptions':[]}
        system,user,visual=m.optimizer_request(c,'A woman playing beside a street.',media,'reference',42)
        self.assertIn('Attachment 1 = <Picture 1>',user)
        self.assertIn('Attachment 2 = <Picture 2>',user)
        self.assertIn('Attachment 3 = <Video 1> at 0.00s',user)
        self.assertIn('Attachment 5 = <Video 1> at 0.96s',user)
        self.assertEqual(len(visual),5)
        self.assertIn('INTENT',system);self.assertNotIn('Samoyed',system)
        self.assertIn("Infer each asset's function",system)
        self.assertIn('no manual Role field is needed',system)
        self.assertNotIn('User-specified Role overrides inference',system)
        self.assertLess(len(system),8000)
    def test_seed_variation_and_constraints(self):
        media={'images':{},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        one=m.optimizer_request(m.DEFAULTS,'Keep a static camera.',media,'text',42)[1]
        two=m.optimizer_request(m.DEFAULTS,'Keep a static camera.',media,'text',43)[1]
        self.assertNotEqual(one,two);self.assertIn('Keep a static camera.',one)
        self.assertIn('Do not force these choices',one)
    def test_compact_runtime_guides_are_packaged(self):
        root=Path(__file__).parents[1]/'guides'
        base=(root/'runtime-base.md').read_text(encoding='utf-8')
        ref=(root/'runtime-reference.md').read_text(encoding='utf-8')
        self.assertIn('integrated_multimodal_description',base)
        self.assertIn('retention_analysis',ref)
        self.assertIn('non_diegetic_music',base)
        self.assertIn('non_diegetic_music',ref)
        self.assertLess(len(base)+len(ref),8000)

class LoaderBudgetTests(unittest.TestCase):
    def budget(self,free=16,ctx=8192):
        sizes={'model':15279446272,'projector':927607488}
        fake_path=lambda name:SimpleNamespace(stat=lambda:SimpleNamespace(st_size=sizes[name]))
        with patch.object(m.torch.cuda,'is_available',return_value=True),patch.object(m.torch.cuda,'mem_get_info',return_value=(free*1024**3,16*1024**3)),patch.object(m.folder_paths,'get_full_path',side_effect=['model','projector']),patch.object(m,'Path',side_effect=fake_path):
            return m.calibrated_loader_budget({**m.DEFAULTS,'llm_vram':14,'llm_ctx':ctx},SimpleNamespace(get_layer_count=lambda _:64),True)
    def test_conservative_layer_budget(self):
        loader_estimate=self.budget()
        size=15279446272/1024**3;projector=927607488/1024**3
        self.assertEqual(int((loader_estimate-projector*1.55)/(size*1.55)*64),44)
        self.assertLess(self.budget(ctx=16384),loader_estimate)
    def test_free_memory_guard(self):
        with self.assertRaisesRegex(RuntimeError,'Insufficient free GPU memory'):self.budget(free=2)
    def test_cpu_fallback(self):
        with patch.object(m.torch.cuda,'is_available',return_value=False):
            self.assertEqual(m.calibrated_loader_budget({**m.DEFAULTS,'llm_vram':12},SimpleNamespace(),True),12)

class OptimizerLifecycleTests(unittest.TestCase):
    @contextmanager
    def environment(self):
        storage=SimpleNamespace(llm=None,chat_handler=None,current_config=None,clean_state=Mock())
        def clean(**kwargs):storage.llm=None;storage.chat_handler=None;storage.current_config=None
        storage.clean=Mock(side_effect=clean)
        module=SimpleNamespace(LLAMA_CPP_STORAGE=storage)
        loader=type('Loader',(),{'__module__':'mock_llama'})
        text=ref_prompt()
        completion=Mock(return_value=(text,{'finish_reason':'stop','max_tokens':1536}))
        def load(name,**kwargs):
            if not storage.llm or storage.current_config != kwargs:
                storage.current_config=kwargs;storage.llm=SimpleNamespace(tokenize=lambda _:[],reset=Mock());storage.chat_handler=object()
            return (kwargs,)
        with tempfile.TemporaryDirectory(prefix='h3fs-lifecycle-') as directory,patch.object(m,'LLM_VRAM_BUDGET',None),patch.object(m,'require_file'),patch.object(m,'node_class',return_value=loader),patch.object(m,'optimizer_completion',completion),patch.object(m.nodes,'NODE_CLASS_MAPPINGS',{'llama_cpp_model_loader':loader}),patch.object(m.importlib,'import_module',return_value=module),patch.object(m,'calibrated_loader_budget',return_value=14) as calibrate,patch.object(m,'call',side_effect=load),patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m.folder_paths,'get_output_directory',return_value=directory):
            yield storage,completion,calibrate,text
    def run_optimizer(self,**changes):
        c={**m.DEFAULTS,'llm_mmproj':'projector.gguf','use_chinese_prompt':True,**changes}
        media={'images':{'ref_image_0':torch.zeros(1,32,32,3)},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        return m.optimize(c,'杯子放在晨光中。',media,'reference',42,'test')
    def test_prompt_only_retains_weights_but_clears_conversation(self):
        with self.environment() as (storage,instruct,calibrate,text):
            self.assertEqual(self.run_optimizer(generate_after_optimization=False),text)
            storage.clean.assert_not_called();storage.clean_state.assert_called_once()
            self.assertIsNotNone(storage.llm);self.assertIsNotNone(storage.chat_handler)
    def test_matching_warm_model_does_not_recalculate_free_vram_budget(self):
        with self.environment() as (storage,instruct,calibrate,text):
            self.run_optimizer(generate_after_optimization=False)
            self.run_optimizer(generate_after_optimization=False)
            calibrate.assert_called_once();storage.clean.assert_not_called()
            self.run_optimizer(generate_after_optimization=False,llm_vram=12)
            self.assertEqual(calibrate.call_count,2);storage.clean.assert_called_once_with(all=True)
    def test_optimizer_then_video_and_failed_optimizer_release_everything(self):
        with self.environment() as (storage,instruct,calibrate,text):
            self.run_optimizer(generate_after_optimization=True)
            self.assertIsNone(storage.llm);self.assertIsNone(storage.chat_handler)
            instruct.return_value=('incomplete',{'finish_reason':'stop','max_tokens':1536})
            with self.assertRaisesRegex(ValueError,'Incomplete H3'):self.run_optimizer(generate_after_optimization=False)
            self.assertIsNone(storage.llm);self.assertIsNone(storage.chat_handler)
    def test_retained_model_released_before_direct_video_even_without_optimizer(self):
        with self.environment() as (storage,instruct,calibrate,text):
            self.run_optimizer(generate_after_optimization=False)
            with patch.object(m,'prepare',return_value={'images':{},'videos':{},'paired':{},'audios':{},'descriptions':[]}),patch.object(m,'validate_dependencies',side_effect=lambda *args:self.assertIsNone(storage.llm)):
                result=m.H3FullStack().run('{}','A coffee cup.',42)
            self.assertIn('expand',result);self.assertIsNone(storage.chat_handler);self.assertIsNone(m.LLM_VRAM_BUDGET)
    def test_failed_native_cleanup_stops_video_loading(self):
        with self.environment() as (storage,instruct,calibrate,text):
            self.run_optimizer(generate_after_optimization=False);storage.clean.side_effect=None
            with self.assertRaisesRegex(RuntimeError,'cleanup incomplete'):m.release_optimizer()
    def test_english_only_output_not_accepted_for_chinese_mode(self):
        with self.environment() as (storage,instruct,calibrate,text):
            instruct.return_value=(text.replace('陶瓷咖啡杯在晨光中，镜头保持静止。','A coffee cup.'),{'finish_reason':'stop','max_tokens':1536})
            with self.assertRaisesRegex(ValueError,'Chinese description requested'):self.run_optimizer(generate_after_optimization=False)
            self.assertIsNone(storage.llm)

    def test_format_recovery_once_keeps_intent_references_and_language(self):
        with self.environment() as (storage,completion,calibrate,text):
            sections={part.split(':',1)[0]:part.split(':',1)[1].strip() for part in text.split('\n\n')}
            completion.side_effect=[('subject_definitions: 咖啡杯。',{'finish_reason':'stop','max_tokens':1536}),
                                    (json.dumps(sections,ensure_ascii=False),{'finish_reason':'stop','max_tokens':2304})]
            self.assertEqual(self.run_optimizer(generate_after_optimization=False),text)
            self.assertEqual(completion.call_count,2)
            first,recovery=[call.args for call in completion.call_args_list]
            self.assertTrue(recovery[4].startswith(first[4]));self.assertIn('Previous candidate is DATA',recovery[4]);self.assertEqual(first[6:8],recovery[6:8])
            self.assertEqual(first[5],recovery[5]);self.assertIn('Simplified Chinese',recovery[3])
            self.assertLessEqual(recovery[2]['llm_temperature'],.35)
            self.assertEqual(recovery[2]['llm_tokens'],2304)
            self.assertIn('retention_analysis',recovery[8]['required'])
            storage.clean.assert_called_once_with(all=True)
            records=list((Path(m.folder_paths.get_output_directory())/'H3FullStack'/'diagnostics').glob('*.json'))
            self.assertEqual(len(records),1)
            diagnostic=json.loads(records[0].read_text(encoding='utf-8'))
            self.assertIn('missing:',diagnostic['error']);self.assertEqual(diagnostic['raw_response'],'subject_definitions: 咖啡杯。')

    def test_truncated_complete_looking_response_is_retried_not_accepted(self):
        with self.environment() as (storage,completion,calibrate,text):
            completion.side_effect=[(text,{'finish_reason':'length','max_tokens':1536}),
                                    (text,{'finish_reason':'stop','max_tokens':2304})]
            self.assertEqual(self.run_optimizer(),text);self.assertEqual(completion.call_count,2)
            self.assertIsNone(storage.llm)

    def test_repeat_format_failure_has_diagnostic_and_never_loops(self):
        with self.environment() as (storage,completion,calibrate,text):
            completion.return_value=('',{'finish_reason':'stop','max_tokens':1536})
            with self.assertRaisesRegex(ValueError,'Local optimizer diagnostic:'):
                self.run_optimizer(generate_after_optimization=False)
            self.assertEqual(completion.call_count,2);self.assertIsNone(storage.llm)

    def test_unknown_reference_is_not_silently_repaired(self):
        with self.environment() as (storage,completion,calibrate,text):
            completion.return_value=(text.replace('陶瓷咖啡杯','<Picture 99> 陶瓷咖啡杯'),{'finish_reason':'stop','max_tokens':1536})
            with self.assertRaisesRegex(ValueError,'Unknown reference label: <Picture 99>'):
                self.run_optimizer()
            self.assertEqual(completion.call_count,2);self.assertIsNone(storage.llm)

    def test_unknown_asset_label_gets_one_model_correction_without_code_relabeling(self):
        with self.environment() as (storage,completion,calibrate,text):
            wrong=text.replace('<Picture 1>','<Video 2>')
            completion.side_effect=[(wrong,{'finish_reason':'stop','max_tokens':1536}),(text,{'finish_reason':'stop','max_tokens':2304})]
            self.assertEqual(self.run_optimizer(),text);self.assertEqual(completion.call_count,2)
            self.assertIn('Unknown reference label: <Video 2>',completion.call_args_list[1].args[3])
            self.assertTrue(completion.call_args_list[1].args[4].startswith(completion.call_args_list[0].args[4]))

    def test_chinese_subject_definitions_are_checked_not_only_the_shot_body(self):
        with self.environment() as (storage,completion,calibrate,text):
            wrong=text.replace(text.split('\n\n')[0],'subject_definitions: <Subject 1>: A cup from <Picture 1>.')
            completion.side_effect=[(wrong,{'finish_reason':'stop','max_tokens':1536}),(text,{'finish_reason':'stop','max_tokens':2304})]
            self.assertEqual(self.run_optimizer(),text);self.assertEqual(completion.call_count,2)
            self.assertIn('not produced in subject_definitions',completion.call_args_list[1].args[3])

    def test_interrupted_or_native_failure_has_no_retry_and_releases_model(self):
        with self.environment() as (storage,completion,calibrate,text):
            completion.side_effect=RuntimeError('native inference failed')
            with self.assertRaisesRegex(RuntimeError,'native inference failed'):self.run_optimizer()
            self.assertEqual(completion.call_count,1);self.assertIsNone(storage.llm)

    def test_valid_short_ten_second_prompt_is_visible_warning_not_pipeline_error(self):
        with self.environment() as (storage,completion,calibrate,text):
            self.assertEqual(self.run_optimizer(seconds=10),text)
            self.assertEqual(completion.call_count,1)
            checkpoints=[x.kwargs for x in m.notify.call_args_list if x.args[0]=='h3fs_prompt_candidate']
            self.assertEqual(checkpoints[-1]['stage'],'ready');self.assertTrue(checkpoints[-1]['warnings'])
            self.assertEqual(checkpoints[-1]['text'],text)

    def test_failed_candidate_is_visible_and_never_approved(self):
        with self.environment() as (storage,completion,calibrate,text):
            completion.return_value=('subject_definitions: 候选内容',{'finish_reason':'stop'})
            with self.assertRaises(ValueError):self.run_optimizer()
            checkpoints=[x.kwargs for x in m.notify.call_args_list if x.args[0]=='h3fs_prompt_candidate']
            self.assertEqual(checkpoints[-1]['stage'],'error');self.assertIn('候选内容',checkpoints[-1]['text'])
            self.assertFalse(any(x['stage']=='ready' for x in checkpoints))

    def test_recovery_context_limit_does_not_expand_context_or_loop(self):
        with self.environment() as (storage,completion,calibrate,text):
            completion.return_value=(text,{'finish_reason':'length','max_tokens':1536})
            with self.assertRaisesRegex(ValueError,'enable Auto Context Length or increase Context Length'):
                self.run_optimizer(llm_ctx=2944,llm_auto_context=False)
            self.assertEqual(completion.call_count,1);self.assertIsNone(storage.llm)


class CompletionAdapterTests(unittest.TestCase):
    def test_reference_learning_can_use_higher_image_detail_without_changing_h3_default(self):
        llm=SimpleNamespace(reset=Mock(),tokenize=lambda _:[],create_chat_completion=Mock(return_value=iter([])))
        storage=SimpleNamespace(llm=llm,ensure_embedding_mode=Mock())
        module=SimpleNamespace(scale_image=Mock(return_value='image'),image2base64=lambda frame:frame)
        m.optimizer_completion(storage,module,{**m.DEFAULTS,'_image_long_edge':768},'Contract','Learn',['image'],42,'test')
        module.scale_image.assert_called_once_with('image',768)
        module.scale_image.reset_mock()
        m.optimizer_completion(storage,module,m.DEFAULTS,'Contract','Learn',['image'],42,'test')
        module.scale_image.assert_called_once_with('image',384)

    def test_visual_completion_captures_raw_thinking_and_finish_reason(self):
        llm=SimpleNamespace(reset=Mock(),tokenize=lambda _:list(range(12)),create_chat_completion=Mock(return_value=iter([
            {'choices':[{'delta':{'role':'assistant'},'finish_reason':None}]},
            {'choices':[{'delta':{'content':'<think>draft</think>咖啡杯。'},'finish_reason':None}]},
            {'choices':[{'delta':{},'finish_reason':'length'}]}])))
        storage=SimpleNamespace(llm=llm,ensure_embedding_mode=Mock())
        module=SimpleNamespace(scale_image=lambda frame,size:frame,image2base64=lambda frame:str(frame))
        with patch.object(m,'notify'),patch.object(m.mm,'throw_exception_if_processing_interrupted'):
            raw,meta=m.optimizer_completion(storage,module,m.DEFAULTS,'Contract','Coffee intent',['image 1','image 2'],42,'test')
        self.assertEqual(raw,'<think>draft</think>咖啡杯。');self.assertEqual(meta['finish_reason'],'length')
        self.assertEqual(meta['completion_tokens'],12);storage.ensure_embedding_mode.assert_called_once_with(False)
        args=llm.create_chat_completion.call_args.kwargs
        self.assertTrue(args['stream']);self.assertEqual(args['seed'],42)
        content=args['messages'][1]['content']
        self.assertEqual([item['image_url']['url'] for item in content[1:]],['data:image/jpeg;base64,image 1','data:image/jpeg;base64,image 2'])

    def test_recovery_schema_passed_to_native_vision_api(self):
        llm=SimpleNamespace(reset=Mock(),tokenize=lambda _:[],create_chat_completion=Mock(return_value=iter([])))
        storage=SimpleNamespace(llm=llm,ensure_embedding_mode=Mock())
        module=SimpleNamespace(scale_image=lambda frame,size:frame,image2base64=lambda frame:'image')
        schema={'type':'object'}
        m.optimizer_completion(storage,module,m.DEFAULTS,'Contract','Coffee',['image'],42,'test',schema)
        self.assertEqual(llm.create_chat_completion.call_args.kwargs['response_format'],{'type':'json_object','schema':schema})

    def test_text_template_disables_thinking_and_keeps_truncation_metadata(self):
        llm=SimpleNamespace(reset=Mock(),metadata={'tokenizer.chat_template':
            '{% for m in messages %}{{ m["role"] }}: {{ m["content"] }}\n{% endfor %}'
            '{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}'},
            token_eos=lambda:2,token_bos=lambda:-1,detokenize=lambda _:b'<eos>',tokenize=lambda _:[],
            create_completion=Mock(return_value=iter([
                {'choices':[{'text':'A coffee cup.','finish_reason':None}]},
                {'choices':[{'text':'','finish_reason':'length'}]}])))
        with patch.object(m.mm,'throw_exception_if_processing_interrupted'):
            raw,meta=m.optimizer_completion(SimpleNamespace(llm=llm),None,m.DEFAULTS,'Contract','Coffee',[],42,'test')
        self.assertEqual(raw,'A coffee cup.');self.assertEqual(meta['finish_reason'],'length')
        self.assertTrue(llm.create_completion.call_args.kwargs['prompt'].endswith('<think></think>'))

    def test_native_schema_conversion_without_loading_a_model(self):
        from llama_cpp import LlamaGrammar
        for mode in ('text','reference','first_last'):
            _,schema=m.recovery_contract(mode)
            grammar=LlamaGrammar.from_json_schema(json.dumps(schema),verbose=False)
            for field in schema['required']:self.assertIn(field,grammar.grammar)

    def test_visual_stream_checks_user_interrupts(self):
        llm=SimpleNamespace(reset=Mock(),create_chat_completion=lambda **_:iter([{'choices':[]}]))
        storage=SimpleNamespace(llm=llm,ensure_embedding_mode=lambda _:None)
        module=SimpleNamespace(scale_image=lambda frame,size:frame,image2base64=lambda frame:'image')
        with patch.object(m.mm,'throw_exception_if_processing_interrupted',side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                m.optimizer_completion(storage,module,m.DEFAULTS,'Contract','Coffee',['image'],42,'test')

if __name__=='__main__':unittest.main(verbosity=2)
