import importlib
import os
import subprocess
import sys
import time
import unittest
from test_pipeline import m

runner=importlib.import_module('fullstack_test.native_runner')


class IsolationTests(unittest.TestCase):
    def spawn(self,body):
        return subprocess.Popen([sys.executable,'-u','-c','import sys,json,time; json.loads(sys.stdin.readline()); '+body],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',env={**os.environ,'PYTHONUTF8':'1'})
    def test_success_waits_for_clean_exit_and_forwards_progress(self):
        body="print('H3FS_NATIVE:'+json.dumps({'type':'progress','event':'status','payload':{'text':'设计中'}})); print('H3FS_NATIVE:'+json.dumps({'type':'result','value':'中文提示词'}))"
        events=[];result=runner.watch_process(self.spawn(body),{},lambda *x:events.append(x),lambda:None,5)
        self.assertEqual(result,'中文提示词');self.assertEqual(events[0][1]['text'],'设计中')
    def test_native_crash_after_partial_result_cannot_escape_to_parent(self):
        process=self.spawn("print('H3FS_NATIVE:'+json.dumps({'type':'result','value':'partial'})); sys.exit(17)")
        with self.assertRaisesRegex(RuntimeError,'异常退出.*17'):runner.watch_process(process,{},lambda *x:None,lambda:None,5)
        self.assertEqual(process.poll(),17)
    def test_worker_python_error_is_actionable_without_killing_parent(self):
        process=self.spawn("print('H3FS_NATIVE:'+json.dumps({'type':'error','value':'素材需要重新设计'}))")
        with self.assertRaisesRegex(ValueError,'素材需要重新设计'):runner.watch_process(process,{},lambda *x:None,lambda:None,5)
    def test_cancel_and_timeout_stop_only_owned_child(self):
        process=self.spawn('time.sleep(30)')
        with self.assertRaisesRegex(RuntimeError,'超时'):runner.watch_process(process,{},lambda *x:None,lambda:None,.2)
        self.assertIsNotNone(process.poll())
        process=self.spawn('time.sleep(30)')
        def cancel():raise RuntimeError('用户取消')
        with self.assertRaisesRegex(RuntimeError,'用户取消'):runner.watch_process(process,{},lambda *x:None,cancel,5)
        self.assertIsNotNone(process.poll())
