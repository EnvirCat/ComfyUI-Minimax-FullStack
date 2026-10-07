import importlib
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

class OptimizerContractTests(unittest.TestCase):
    def test_qwen_vision_loader_uses_minimum_tokens_and_releases_storage(self):
        media={'images':{'ref_image_0':torch.zeros(1,32,32,3)},'videos':{},'paired':{},'audios':{},'descriptions':[]}
        storage=SimpleNamespace(llm=SimpleNamespace(tokenize=lambda _:[]),chat_handler=None,current_config=None,clean=Mock(),clean_state=Mock())
        module=SimpleNamespace(LLAMA_CPP_STORAGE=storage)
        prompt='\n'.join(f'{name}: A ceramic cup.' for name in ('subject_definitions','summary','retention_analysis','detailed_description','overall_soundscape','non_diegetic_music'))
        instruct=Mock();instruct.return_value.process.return_value=(prompt,)
        loader=type('Loader',(),{'__module__':'mock_llama'})
        with tempfile.TemporaryDirectory(prefix='h3fs-optimizer-test-') as directory,patch.object(m,'require_file'),patch.object(m,'node_class',side_effect=lambda name: loader if name=='llama_cpp_model_loader' else instruct),patch.object(m.importlib,'import_module',return_value=module),patch.object(m,'calibrated_loader_budget',return_value=14),patch.object(m,'call',return_value=('model',)) as call,patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m.folder_paths,'get_output_directory',return_value=directory):
            self.assertEqual(m.optimize({**m.DEFAULTS,'llm_mmproj':'projector.gguf'},'A cup in morning light.',media,'reference',42,'test'),prompt)
        self.assertEqual(call.call_args.kwargs['image_min_tokens'],1024)
        self.assertEqual(call.call_args.kwargs['image_max_tokens'],1024)
        self.assertTrue(all(call.kwargs == {'all':True} for call in storage.clean.call_args_list))
        self.assertFalse(instruct.return_value.process.call_args.kwargs['force_offload'])
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
        self.assertLess(len(system),7000)
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
        self.assertLess(len(base)+len(ref),7000)

class LoaderBudgetTests(unittest.TestCase):
    def budget(self,free=16,ctx=8192):
        sizes={'model':15279446272,'projector':927607488}
        fake_path=lambda name:SimpleNamespace(stat=lambda:SimpleNamespace(st_size=sizes[name]))
        with patch.object(m.torch.cuda,'is_available',return_value=True),patch.object(m.torch.cuda,'mem_get_info',return_value=(free*1024**3,16*1024**3)),patch.object(m.folder_paths,'get_full_path',side_effect=['model','projector']),patch.object(m,'Path',side_effect=fake_path):
            return m.calibrated_loader_budget({**m.DEFAULTS,'llm_vram':14,'llm_ctx':ctx},SimpleNamespace(get_layer_count=lambda _:64),True)
    def test_conservative_layer_budget(self):
        loader_estimate=self.budget()
        size=15279446272/1024**3;projector=927607488/1024**3
        self.assertEqual(int((loader_estimate-projector*1.55)/(size*1.55)*64),47)
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
        instruct=Mock()
        text='\n'.join(f'{name}: 陶瓷咖啡杯在晨光中，镜头保持静止。' for name in ('subject_definitions','summary','retention_analysis','detailed_description','overall_soundscape','non_diegetic_music'))
        instruct.return_value.process.return_value=(text,)
        def load(name,**kwargs):
            storage.current_config=kwargs;storage.llm=SimpleNamespace(tokenize=lambda _:[]);storage.chat_handler=object()
            return (kwargs,)
        with tempfile.TemporaryDirectory(prefix='h3fs-lifecycle-') as directory,patch.object(m,'LLM_VRAM_BUDGET',None),patch.object(m,'require_file'),patch.object(m,'node_class',side_effect=lambda name:loader if name=='llama_cpp_model_loader' else instruct),patch.object(m.nodes,'NODE_CLASS_MAPPINGS',{'llama_cpp_model_loader':loader}),patch.object(m.importlib,'import_module',return_value=module),patch.object(m,'calibrated_loader_budget',return_value=14) as calibrate,patch.object(m,'call',side_effect=load),patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m.folder_paths,'get_output_directory',return_value=directory):
            yield storage,instruct,calibrate,text
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
            instruct.return_value.process.return_value=('incomplete',)
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
            instruct.return_value.process.return_value=(text.replace('陶瓷咖啡杯在晨光中，镜头保持静止。','A coffee cup.'),)
            with self.assertRaisesRegex(ValueError,'Chinese description requested'):self.run_optimizer(generate_after_optimization=False)
            self.assertIsNone(storage.llm)

if __name__=='__main__':unittest.main(verbosity=2)
