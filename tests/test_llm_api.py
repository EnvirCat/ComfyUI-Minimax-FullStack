import importlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from test_pipeline import m

a=importlib.import_module('fullstack_test.llm_api')
p=importlib.import_module('fullstack_test.api_prompts')

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        self.server.received=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.auth=self.headers.get('Authorization')
        status=getattr(self.server,'status',200);self.send_response(status)
        self.send_header('Content-Type',getattr(self.server,'mime','text/event-stream'));self.end_headers()
        if status!=200: self.wfile.write(b'{"error":"private echoed credential"}');return
        if getattr(self.server,'mime','')=='application/json': self.wfile.write(json.dumps({'choices':[{'message':{'content':'测试通过'},'finish_reason':'stop'}]}).encode());return
        events=[{'choices':[{'delta':{'reasoning_content':'hidden reasoning'},'finish_reason':None}]},
                {'choices':[{'delta':{'content':'测试通过'},'finish_reason':None}]}]
        if getattr(self.server,'finished',True): events.append({'choices':[{'delta':{},'finish_reason':getattr(self.server,'finish','stop')}]})
        for event in events: self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls): cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):
        self.c=a.settings({'provider':'lmstudio','base_url':f'http://127.0.0.1:{self.server.server_port}/v1','model':'test'})
        for k in ('status','mime','finished','finish'):
            if hasattr(self.server,k): delattr(self.server,k)
    def request(self,**kwargs):
        with patch.object(a,'key_for',return_value=''): return a.completion(self.c,'系统','用户',**kwargs)
    def test_removed_provider_is_not_registered(self):
        self.assertNotIn('deepseek',a.PRESETS)
        with self.assertRaises(a.APIError): a.settings({'provider':'deepseek'})
    def test_endpoint_and_no_secret_settings(self):
        self.assertEqual(a.endpoint('https://example.org/v1/'),'https://example.org/v1/chat/completions')
        self.assertEqual(a.endpoint('https://example.org/v1/chat/completions'),'https://example.org/v1/chat/completions')
        for url in ('http://example.org/v1','https://secret@example.org/v1','https://example.org/?key=secret','https://example.org/#key','https://example.org:999999/v1','https://example.org/../v1','https://example.org/\nv1'):
            with self.subTest(url=url),self.assertRaises(a.APIError): a.endpoint(url)
        with self.assertRaises(a.APIError): a.settings({'api_key':'private'})
        with self.assertRaises(a.APIError): a.settings({'timeout':True})
        with self.assertRaises(a.APIError): a.settings({'vision':'true'})
    def test_stream_is_real_http_and_ignores_reasoning(self):
        raw,meta=self.request();self.assertEqual(raw,'测试通过');self.assertEqual(meta['finish_reason'],'stop')
        self.assertEqual(self.server.received['messages'][1]['content'],'用户')
        self.assertNotIn('Authorization',self.server.received);self.assertIsNone(self.server.auth)
    def test_json_response_compatibility(self):
        self.server.mime='application/json';self.assertEqual(self.request()[0],'测试通过')
    def test_incomplete_stream_cannot_be_used(self):
        self.server.finished=False
        with self.assertRaisesRegex(a.APIError,'未完整结束'): self.request()
    def test_truncated_output_cannot_be_used(self):
        self.server.finish='length'
        with self.assertRaisesRegex(a.APIError,'截断'): self.request()
    def test_http_error_does_not_echo_server_body_or_retry(self):
        self.server.status=401
        with self.assertRaisesRegex(a.APIError,'HTTP 401') as error: self.request()
        self.assertNotIn('credential',str(error.exception))
    def test_visual_only_sent_after_explicit_opt_in(self):
        image=torch.zeros(1,16,16,3)
        with self.assertRaisesRegex(a.APIError,'未启用看图'): self.request(visual=[image])
        self.c['vision']=True;self.request(visual=[image])
        content=self.server.received['messages'][1]['content'];self.assertEqual(content[0]['text'],'用户')
        self.assertTrue(content[1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
    def test_cancel_before_network(self):
        with self.assertRaisesRegex(RuntimeError,'cancelled'): self.request(cancel=lambda:(_ for _ in ()).throw(RuntimeError('cancelled')))
    def test_openai_parameter_selection_and_custom_override(self):
        self.c.update(provider='openai',model='gpt-5-test')
        with patch.object(a,'key_for',return_value='test-only'): a.completion(self.c,'s','u')
        self.assertIn('max_completion_tokens',self.server.received);self.assertNotIn('temperature',self.server.received)
        self.c['token_field']='max_tokens'
        with patch.object(a,'key_for',return_value='test-only'): a.completion(self.c,'s','u')
        self.assertIn('max_tokens',self.server.received)
    @unittest.skipUnless(a.os.name=='nt','DPAPI only on Windows')
    def test_saved_key_encrypted_and_endpoint_scoped(self):
        with tempfile.TemporaryDirectory(prefix='h3fs-api-test-') as root,patch.object(a,'_root',return_value=Path(root)):
            a.save_key(self.c,'test-only-credential');self.assertTrue(a.key_status(self.c)['has_key'])
            self.assertNotIn(b'test-only-credential',next(Path(root).iterdir()).read_bytes())
            self.assertEqual(a.key_for(self.c),'test-only-credential')
            self.assertFalse(a.key_status({**self.c,'base_url':'http://127.0.0.1:12345/v1'})['has_key'])
            a.clear_key(self.c);self.assertFalse(a.key_status(self.c)['has_key'])

class RoutingTests(unittest.TestCase):
    def config(self): return m.parse_config({'llm_backend':'api','llm_api':{'provider':'lmstudio','base_url':'http://localhost:1234/v1','model':'test'}})
    def media(self): return {'images':{},'videos':{},'paired':{},'audios':{},'descriptions':[]}
    def test_api_text_does_not_require_local_weights(self):
        with patch.object(p,'completion',return_value=('中文提示词',{})),patch.object(m,'require_file') as require:
            self.assertEqual(m.design_completion(self.config(),'s','u',[],42,'1',768),'中文提示词');require.assert_not_called()
    def test_text_provider_visual_analysis_stays_local(self):
        c=self.config();visual=[torch.zeros(1,16,16,3)]
        with patch.object(m,'design_completion',return_value='中文观察') as local,patch.object(p,'completion') as api,patch.object(m,'notify'):
            self.assertEqual(p.design(m,c,'s','u',visual,42,'1',768),'中文观察')
            self.assertEqual(local.call_args.args[0]['llm_backend'],'local');api.assert_not_called()
    def test_text_provider_without_vision_never_invents_observations(self):
        c=self.config();c['llm_api']['local_vision']=False
        with patch.object(p,'completion') as api,self.assertRaisesRegex(ValueError,'需要识别素材'):
            p.optimize(m,c,'意图',{**self.media(),'images':{'ref_image_0':torch.zeros(1,16,16,3)}},'reference',42,'1')
        api.assert_not_called()
    def test_h3_api_output_validated_published_and_saved_without_key(self):
        text='integrated_multimodal_description: 阳光照亮一座安静的小院。\noverall_soundscape: 轻柔风声。\nnon_diegetic_music: N/A'
        with tempfile.TemporaryDirectory() as root,patch.object(m.folder_paths,'get_output_directory',return_value=root),patch.object(m,'notify') as notify,patch.object(m,'require_file') as require,patch.object(p,'completion',return_value=(text,{'finish_reason':'stop','backend':'api'})):
            self.assertEqual(m.optimize(self.config(),'安静的小院',self.media(),'text',42,'1'),p.normalize_prompt(text,'text'))
            self.assertEqual(notify.call_args.kwargs['stage'],'ready');require.assert_not_called()
            record=next(Path(root).rglob('prompt_*.json')).read_text(encoding='utf-8');self.assertNotIn('api_key',record)
    def test_bad_api_h3_format_gets_one_bounded_repair(self):
        with patch.object(p,'completion',return_value=('错误格式',{})) as api,patch.object(m,'notify'),self.assertRaises(Exception):
            m.optimize(self.config(),'小院',self.media(),'text',42,'1')
        self.assertEqual(api.call_count,2)

if __name__=='__main__': unittest.main()
