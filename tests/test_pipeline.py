"""Offline graph wiring tests, with model/file access mocked (no GPU generation)."""
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).parents[1]
COMFY=Path(os.environ.get('COMFYUI_ROOT', str(ROOT.parents[1])))
sys.path.insert(0,str(COMFY))
package=types.ModuleType('fullstack_test');package.__path__=[str(ROOT)];sys.modules[package.__name__]=package
m=importlib.import_module('fullstack_test.nodes')
# Existing native-owner unit tests deliberately exercise the in-process worker
# implementation with mocked loaders. Production always uses the isolated path.
m.ISOLATE_NATIVE=False

class GraphTests(unittest.TestCase):
    def run_graph(self, **changes):
        c={**m.DEFAULTS,**changes}
        with patch.object(m,'validate_dependencies'),patch.object(m,'notify'),patch.object(m,'prepare',return_value={
            'images':{'ref_image_0':'mock_image'} if changes.get('mode')=='first_frame' else {},
            'videos':{},'paired':{},'audios':{},'descriptions':[]}):
            return m.H3FullStack().run(json.dumps(c),'A quiet garden, no dialogue.',42,unique_id='test')
    def test_dmad_wiring(self):
        out=self.run_graph(loras=[{'name':'dmad.safetensors','weight':1,'enabled':True},{'name':'character.safetensors','weight':.4,'enabled':True}])
        graph=out['expand'];by=lambda name:[x['inputs'] for x in graph.values() if x['class_type']==name]
        self.assertEqual(len(by('LoraLoaderModelOnly')),2)
        self.assertEqual([x['strength_model'] for x in by('LoraLoaderModelOnly')],[1,.4])
        self.assertEqual(by('BasicScheduler')[0]['steps'],4)
        self.assertEqual(by('KSamplerSelect')[0]['sampler_name'],'euler')
        self.assertEqual(len(out['result']),3)
        self.assertTrue(all(x['override_display_id']=='test' for x in graph.values()))
    def test_exact_first_frame(self):
        graph=self.run_graph(mode='first_frame')['expand']
        cond=next(x['inputs'] for x in graph.values() if x['class_type']=='H3FSConditioning')
        self.assertEqual(cond['mode'],'first_frame');self.assertEqual(cond['length'],124)
    def test_attention_mutual_exclusion(self):
        kinds={'ModelAttentionBackend','BlockSparseAttention','PathchSageAttentionKJ','MiniMaxH3MemoryEfficientSageAttentionPatch'}
        for backend in ('pytorch attention','comfy kitchen attention','sage','memory_efficient_sage','sol-attn'):
            graph=self.run_graph(attention=backend)['expand']
            self.assertEqual(sum(x['class_type'] in kinds for x in graph.values()),1)
    def test_original_audio_and_double_output_dependency(self):
        graph=self.run_graph(upscale=True,rife=True)['expand']
        finish=next(x['inputs'] for x in graph.values() if x['class_type']=='H3FSFinish')
        audio_node=graph[finish['audio'][0]]
        self.assertEqual(audio_node['class_type'],'VAEDecodeAudio')
        self.assertEqual(json.loads(finish['settings'])['upscale'],True)
        self.assertEqual(json.loads(finish['settings'])['rife'],True)
    def test_preview_optional(self):
        graph=self.run_graph(preview=False,low_vram=False,chunk_ff=False,profile='standard')['expand']
        kinds=[x['class_type'] for x in graph.values()]
        for k in ('ModelPreviewOverrideKJ','MiniMaxLowVRAMAttention','MiniMaxChunkFeedForward','LoraLoaderModelOnly'):
            self.assertNotIn(k,kinds)
    def test_independent_steps(self):
        graph=self.run_graph(steps=6,sampler='er_sde',scheduler='beta',loras=[{'name':'dmad.safetensors','weight':.6,'enabled':True}])['expand']
        self.assertEqual(next(n['inputs']['steps'] for n in graph.values() if n['class_type']=='BasicScheduler'),6)
        self.assertEqual(next(n['inputs']['sampler_name'] for n in graph.values() if n['class_type']=='KSamplerSelect'),'er_sde')
    def test_optimizer_only_and_external_prompt(self):
        with patch.object(m,'optimize',return_value='optimized') as opt,patch.object(m,'validate_dependencies') as validate:
            out=self.run_graph(optimize_on_run=True,generate_after_optimization=False)
            self.assertEqual(out['ui']['text'],['optimized'])
            self.assertEqual(out['result'][2],'optimized')
            for value in out['result'][:2]:
                self.assertIsInstance(value,m.ExecutionBlocker);self.assertIsNone(value.message)
            validate.assert_not_called();opt.assert_called_once()
        with patch.object(m,'notify'),patch.object(m,'prepare',return_value={'images':{},'videos':{},'audios':{},'paired':{},'descriptions':[]}),patch.object(m,'optimize',return_value='external result') as opt:
            out=m.H3FullStack().run(json.dumps({**m.DEFAULTS,'optimize_on_run':True,'generate_after_optimization':False}),'',42,prompt_external='external input')
            self.assertEqual(opt.call_args.args[1],'external input');self.assertEqual(out['result'][2],'external result')
    def test_reviewed_prompt_skips_llm_and_preserves_string_output(self):
        with patch.object(m,'optimize') as opt:
            out=self.run_graph(use_reviewed_prompt=True,reviewed_prompt='Reviewed garden shot.')
        opt.assert_not_called()
        self.assertEqual(out['ui']['text'],['Reviewed garden shot.'])
        cond=next(n['inputs'] for n in out['expand'].values() if n['class_type']=='H3FSConditioning')
        self.assertEqual(cond['prompt'],'Reviewed garden shot.')
    def test_empty_reviewed_prompt_rejected(self):
        with self.assertRaisesRegex(ValueError,'No reviewed prompt saved'):
            self.run_graph(use_reviewed_prompt=True,reviewed_prompt=' ')
    def test_sol_dynamic_combo_accepted_by_core_execution(self):
        from execution import get_input_data
        from comfy_extras.nodes_sparse_attention import BlockSparseAttention
        graph=self.run_graph(attention='sol-attn',sol_tau=1.3)['expand']
        inputs=next(n['inputs'] for n in graph.values() if n['class_type']=='BlockSparseAttention')
        data,missing,v3=get_input_data(inputs,BlockSparseAttention,'sol-test')
        self.assertEqual(data['selection'],['sol-attn'])
        self.assertEqual(data['selection.tau'],[1.3])
        self.assertNotIn('selection',missing)
    def test_preview_has_no_audio(self):
        graph=self.run_graph(preview_audio=True)['expand']
        args=next(n['inputs'] for n in graph.values() if n['class_type']=='H3FSTimelinePreview')
        self.assertNotIn('audio_vae',args)
        self.assertEqual(args['frame_count'],124)

class MediaModeTests(unittest.TestCase):
    def test_disabled_media_preserved(self):
        import torch
        from fullstack_test.media import prepare
        from fullstack_test.config import parse_config
        stored=[{'kind':'image','path':f'{i}.png','slot':i} for i in range(9)]+[{'kind':'video','path':'v.mp4','slot':0},{'kind':'audio','path':'a.wav','slot':0}]
        c=parse_config({'version':2,'mode':'text','media':stored})
        self.assertFalse(prepare(c,124)['images']);self.assertEqual(len(c['media']),11)
        c['mode']='first_last'
        with patch('fullstack_test.media.image_tensor',return_value=torch.zeros(1,32,32,3)) as image:
            media=prepare(c,124);self.assertEqual(len(media['images']),2);self.assertFalse(media['videos']);self.assertFalse(media['audios']);self.assertEqual(image.call_count,2)
        self.assertEqual(len(c['media']),11)
    def test_image_socket_overrides_upload(self):
        import torch
        from fullstack_test.media import prepare
        from fullstack_test.config import parse_config
        c=parse_config({'version':2,'media':[{'kind':'image','path':'missing.png','slot':0}]})
        tensor=torch.ones(1,32,32,3)
        out=prepare(c,124,image_inputs={'image_1':tensor})
        self.assertTrue(torch.equal(out['images']['ref_image_0'],tensor))

class SaveTests(unittest.TestCase):
    def test_saved_videos_use_native_asset_entries_and_file_backed_outputs(self):
        import torch
        from comfy_api.latest import InputImpl
        with tempfile.TemporaryDirectory(prefix='h3fs-save-test-') as out:
            c={**m.DEFAULTS,'output_dir':out,'filename':'test','upscale':True,'upscale_method':'Lanczos'}
            with patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m,'preview_index_path',return_value=Path(out)/'index.json'),patch.object(m.folder_paths,'get_output_directory',return_value=out):
                finished=m.H3FSFinish().run(torch.zeros(5,64,96,3),None,json.dumps(c),'Test',42,'test')
            self.assertEqual(len(finished['ui']['images']),2)
            self.assertTrue(all(f['type']=='output' for f in finished['ui']['images']))
            self.assertIsInstance(finished['result'][0],InputImpl.VideoFromFile)
            self.assertIsInstance(finished['result'][1],InputImpl.VideoFromFile)
    def test_cancelled_writer_never_publishes_partial_mp4(self):
        import torch
        from fullstack_test.video_output import encode_mp4
        with tempfile.TemporaryDirectory(prefix='h3fs-save-test-') as out:
            path=Path(out)/'result.mp4'
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                encode_mp4(torch.zeros(5,64,96,3),None,24,path,18,cancel=lambda:(_ for _ in ()).throw(RuntimeError('cancelled')))
            self.assertFalse(path.exists());self.assertEqual(list(Path(out).iterdir()),[])
    def test_disabled_upscale_blocks_downstream_save(self):
        import torch
        with tempfile.TemporaryDirectory(prefix='h3fs-save-test-') as out:
            c={**m.DEFAULTS,'output_dir':out,'filename':'test','upscale':False}
            with patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m,'preview_index_path',return_value=Path(out)/'index.json'):
                finished=m.H3FSFinish().run(torch.zeros(24,64,96,3),None,json.dumps(c),'Test prompt',42,'test')
                result=finished['result'];self.assertEqual(len(finished['ui']['h3fs_files']),1)
                self.assertEqual(finished['ui']['h3fs_final_prompt'],['Test prompt'])
            self.assertIsNotNone(result[0]);self.assertIsInstance(result[1],m.ExecutionBlocker)
            self.assertEqual(result[3],'');self.assertEqual(result[4],'Test prompt')
    def test_real_mp4_two_versions_and_audio(self):
        import torch
        import av
        with tempfile.TemporaryDirectory(prefix='h3fs-save-test-') as out:
            c={**m.DEFAULTS,'output_dir':out,'filename':'test','upscale':True,'upscale_method':'Lanczos'}
            images=torch.rand(24,64,96,3)
            audio={'waveform':torch.zeros(1,2,48000),'sample_rate':48000}
            with patch.object(m,'notify'),patch.object(m.mm,'unload_all_models'),patch.object(m.mm,'soft_empty_cache'),patch.object(m,'preview_index_path',return_value=Path(out)/'preview-index.json'):
                finished=m.H3FSFinish().run(images,audio,json.dumps(c),'Test prompt',42,'test')
                result=finished['result'];self.assertEqual(len(finished['ui']['h3fs_files']),2)
                saved=dict(m.SAVED_FILES)
                m.SAVED_FILES.clear();m.restore_preview_index()
                self.assertTrue(all(Path(p).resolve() in m.SAVED_FILES.values() for p in result[2:4]))
                m.SAVED_FILES.clear();m.SAVED_FILES.update(saved)
            self.assertEqual(len(result),5)
            self.assertIsNotNone(result[0]);self.assertIsNotNone(result[1])
            dims=[]
            for path in result[2:4]:
                self.assertTrue(Path(path).is_file())
                with av.open(path) as container:
                    v=container.streams.video[0];dims.append((v.width,v.height))
                    self.assertEqual(v.average_rate,24);self.assertEqual(len(container.streams.audio),1)
                    self.assertAlmostEqual(float(container.duration/av.time_base),1,delta=.05)
            self.assertEqual(dims,[(96,64),(192,128)])

class LabelTests(unittest.TestCase):
    def test_roles_and_soundtrack_audio_ordinals(self):
        from fullstack_test.media import label_manifest
        media={'images':{'ref_image_0':None},'videos':{'ref_video_0':None},'paired':{'ref_video_audio_0':None},'audios':{'ref_audio_0':None},'descriptions':[
            {'kind':'image','name':'person.png','role':'identity'},
            {'kind':'video','name':'motion.mp4','role':'camera only'},
            {'kind':'audio','name':'room.wav','role':'sound ambience'}]}
        text=label_manifest(media)
        self.assertIn('<Picture 1>: image reference; file person.png; user-specified role: identity',text)
        self.assertIn('<Audio 1>: soundtrack paired with <Video 1>',text)
        self.assertIn('<Audio 2>: standalone audio reference; file room.wav; user-specified role: sound ambience',text)

if __name__=='__main__':unittest.main(verbosity=2)
