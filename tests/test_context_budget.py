"""Offline context-size regression tests: no model load, inference, or queue."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import m


class ContextPlanTests(unittest.TestCase):
    def plan(self, initial=1379, recovery=1535, count=5, **changes):
        c = {**m.DEFAULTS, 'llm_ctx': 8192, 'llm_tokens': 1536, **changes}
        tokenize = Mock(side_effect=lambda value: range(recovery if value == b'SRU' else initial))
        with patch.object(m, 'recovery_contract', return_value=('R', {'type': 'object'})):
            result = m.optimizer_context_plan(c, tokenize, 'S', 'U', count, 'reference')
        return c, tokenize, result

    def test_exact_user_failure_reserves_every_image_and_full_recovery(self):
        c, tokenize, plan = self.plan()
        self.assertEqual(plan['initial_required'], 8419)
        self.assertEqual(plan['recovery_required'], 9343)
        self.assertEqual(plan['effective_context'], 10240)
        self.assertEqual(plan['reference_count'], 5)
        self.assertEqual(plan['reference_tokens'], 5120)
        self.assertEqual(plan['output_tokens'], 1536)
        self.assertEqual(plan['recovery_output_tokens'], 2304)
        self.assertEqual([call.args[0] for call in tokenize.call_args_list], [b'SU', b'SRU'])
        self.assertEqual(c['llm_ctx'], 8192)
        self.assertEqual(c['llm_tokens'], 1536)

    def test_manual_exact_user_failure_is_clear_and_never_expands(self):
        with self.assertRaisesRegex(ValueError, '8419.*configured 8192'):
            self.plan(llm_auto_context=False)

    def test_legacy_saved_workflow_without_option_defaults_to_auto(self):
        c = {**m.DEFAULTS, 'llm_ctx': 8192, 'llm_tokens': 1536}
        c.pop('llm_auto_context', None)
        with patch.object(m, 'recovery_contract', return_value=('R', {})):
            result = m.optimizer_context_plan(c, lambda _: range(1379), 'S', 'U', 5, 'reference')
        self.assertTrue(result['auto_context'])
        self.assertEqual(result['effective_context'], 10240)
        self.assertNotIn('llm_auto_context', c)

    def test_maximum_visual_attachments_are_not_counted_as_nine_images(self):
        _, _, plan = self.plan(count=18)
        self.assertEqual(plan['reference_count'], 18)
        self.assertEqual(plan['reference_tokens'], 18432)
        self.assertEqual(plan['initial_required'], 21731)
        self.assertEqual(plan['recovery_required'], 22655)
        self.assertEqual(plan['effective_context'], 23552)

    def test_rounding_boundary_includes_headroom(self):
        # Recovery + 768 headroom = 9216 exactly, then one token over.
        _, _, exact = self.plan(initial=0, recovery=5760, count=0)
        _, _, over = self.plan(initial=0, recovery=5761, count=0)
        self.assertEqual(exact['effective_context'], 9216)
        self.assertEqual(over['effective_context'], 10240)

    def test_requested_context_is_a_floor_not_a_target_to_shrink(self):
        _, _, plan = self.plan(initial=0, recovery=0, count=0, llm_ctx=24576)
        self.assertEqual(plan['effective_context'], 24576)

    def test_exact_cap_allowed_but_headroom_over_cap_rejected(self):
        _, _, plan = self.plan(initial=0, recovery=29312, count=0)
        self.assertEqual(plan['effective_context'], 32768)
        with self.assertRaisesRegex(ValueError, 'up to 32768.*no reference or output was silently removed'):
            self.plan(initial=0, recovery=29313, count=0)

    def test_manual_only_initial_budget_is_required_before_first_attempt(self):
        _, _, plan = self.plan(initial=1200, recovery=1800, count=4, llm_auto_context=False)
        self.assertEqual(plan['effective_context'], 8192)
        self.assertEqual(plan['initial_required'], 7216)
        self.assertEqual(plan['recovery_required'], 8584)
        self.assertEqual(plan['recovery_output_tokens'], 2304)

    def test_small_and_maximum_output_allowances_are_preserved(self):
        for tokens, retry in ((256, 768), (1536, 2304), (4096, 6144)):
            with self.subTest(tokens=tokens):
                _, _, plan = self.plan(llm_tokens=tokens)
                self.assertEqual(plan['output_tokens'], tokens)
                self.assertEqual(plan['recovery_output_tokens'], retry)
                self.assertGreaterEqual(plan['effective_context'], plan['recovery_required'] + 768)

    def test_unicode_and_saved_configuration_are_not_rewritten(self):
        c = copy.deepcopy(m.DEFAULTS)
        c.update(llm_ctx=8192, llm_tokens=1536)
        before = copy.deepcopy(c)
        tokenize = Mock(return_value=range(10))
        with patch.object(m, 'recovery_contract', return_value=('补救', {})):
            m.optimizer_context_plan(c, tokenize, '规则', '意图', 0, 'reference')
        self.assertEqual(c, before)
        self.assertEqual([call.args[0] for call in tokenize.call_args_list],
                         ['规则意图'.encode('utf-8'), '规则补救意图'.encode('utf-8')])

    def test_nine_images_and_three_video_samples_each_make_eighteen_ordered_attachments(self):
        images = {f'ref_image_{i}': torch.full((1, 4, 4, 3), float(i)) for i in range(9)}
        videos = {f'ref_video_{i}':
                  (torch.arange(24).reshape(24, 1, 1, 1).expand(24, 4, 4, 3) + 100 * (i + 1)).float()
                  for i in range(3)}
        media = {'images': images, 'videos': videos, 'paired': {}, 'audios': {}, 'descriptions': []}
        originals = [frame.clone() for frame in [*images.values(), *videos.values()]]
        _, user, visual = m.optimizer_request(m.DEFAULTS, 'Keep every ordered reference.', media, 'reference', 42)
        self.assertEqual(len(visual), 18)
        self.assertEqual([int(frame[0, 0, 0, 0]) for frame in visual],
                         list(range(9)) + [100, 111, 123, 200, 211, 223, 300, 311, 323])
        self.assertIn('Attachment 9 = <Picture 9>', user)
        self.assertIn('Attachment 10 = <Video 1> at 0.00s', user)
        self.assertIn('Attachment 18 = <Video 3> at 0.96s', user)
        self.assertTrue(all(torch.equal(value, original)
                            for value, original in zip([*images.values(), *videos.values()], originals)))

    def test_formatted_text_callback_replaces_raw_text_estimate_for_both_attempts(self):
        c = {**m.DEFAULTS, 'llm_ctx': 8192, 'llm_tokens': 1536}
        tokenize = Mock(return_value=[])
        actual_count = Mock(side_effect=[6500, 6800])
        with patch.object(m, 'recovery_contract', return_value=('R', {})):
            plan = m.optimizer_context_plan(c, tokenize, 'S', 'U', 0, 'text', actual_count)
        tokenize.assert_not_called()
        self.assertEqual([call.args for call in actual_count.call_args_list], [('S',), ('SR',)])
        self.assertEqual(plan['initial_required'], 8420)
        self.assertEqual(plan['recovery_required'], 9488)
        self.assertEqual(plan['effective_context'], 11264)
        self.assertEqual(plan['output_tokens'], 1536)
        self.assertEqual(plan['recovery_output_tokens'], 2304)

    def test_manual_cap_checks_formatted_text_even_when_raw_text_would_fit(self):
        c = {**m.DEFAULTS, 'llm_ctx': 8192, 'llm_tokens': 1536, 'llm_auto_context': False}
        tokenize = Mock(return_value=[])
        actual_count = Mock(side_effect=[7000, 9000])
        with patch.object(m, 'recovery_contract', return_value=('R', {})):
            with self.assertRaisesRegex(ValueError, '8920.*configured 8192'):
                m.optimizer_context_plan(c, tokenize, 'S', 'U', 0, 'text', actual_count)
        tokenize.assert_not_called()

    def test_actual_jinja_renderer_and_completion_use_identical_prompt_and_stop(self):
        template = ('{{ bos_token }}' + 'WRAPPER ' * 800 +
                    '{% for m in messages %}[{{ m["role"] }}]{{ m["content"] }}{{ eos_token }}{% endfor %}'
                    '{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}')
        llm = SimpleNamespace(
            metadata={'tokenizer.chat_template': template}, token_eos=lambda: 2, token_bos=lambda: 1,
            detokenize=lambda tokens: b'<bos>' if tokens == [1] else b'<eos>',
            tokenize=Mock(side_effect=lambda value: range(len(value))), reset=Mock(),
            create_completion=Mock(return_value=iter([
                {'choices': [{'text': 'A ceramic cup.', 'finish_reason': 'stop'}]}])))
        system, user = '使用完整的规则', '保留用户的全部意图'
        formatted = m.render_optimizer_text_prompt(llm, system, user)
        self.assertTrue(formatted.prompt.startswith('<bos>WRAPPER '))
        self.assertTrue(formatted.prompt.endswith('<think></think>'))
        self.assertIn(system, formatted.prompt)
        self.assertIn(user, formatted.prompt)
        raw_count = len(llm.tokenize((system + user).encode('utf-8')))
        formatted_count = len(llm.tokenize(formatted.prompt.encode('utf-8')))
        self.assertGreater(formatted_count, raw_count + 6000)
        c = {**m.DEFAULTS, 'llm_ctx': 8192, 'llm_tokens': 1536}
        with patch.object(m, 'recovery_contract', return_value=('R', {})):
            plan = m.optimizer_context_plan(c, llm.tokenize, system, user, 0, 'text',
                lambda value: len(llm.tokenize(m.render_optimizer_text_prompt(llm, value, user).prompt.encode('utf-8'))))
        self.assertEqual(plan['initial_required'], formatted_count + 384 + 1536)
        self.assertGreater(plan['effective_context'], c['llm_ctx'])
        with patch.object(m.mm, 'throw_exception_if_processing_interrupted'):
            raw, completion = m.optimizer_completion(SimpleNamespace(llm=llm), None,
                {**c, 'llm_ctx': plan['effective_context']}, system, user, [], 42, 'offline')
        self.assertEqual(raw, 'A ceramic cup.')
        self.assertEqual(completion['max_tokens'], 1536)
        self.assertEqual(llm.create_completion.call_args.kwargs['prompt'], formatted.prompt)
        self.assertEqual(llm.create_completion.call_args.kwargs['stop'], formatted.stop)


class ContextResizeLifecycleTests(unittest.TestCase):
    def media(self, count=5):
        return {'images': {f'ref_image_{i}': torch.full((1, 4, 4, 3), float(i)) for i in range(count)},
                'videos': {}, 'paired': {}, 'audios': {}, 'descriptions': []}

    def config(self, **changes):
        return {**m.DEFAULTS, 'llm_ctx': 8192, 'llm_tokens': 1536,
                'llm_model': 'fake.gguf', 'llm_mmproj': 'projector.gguf',
                'generate_after_optimization': False, **changes}

    @contextmanager
    def environment(self, initial=1379, recovery=1535):
        events = []
        storage = SimpleNamespace(llm=None, chat_handler=None, current_config=None, clean_state=Mock())

        def clean(**kwargs):
            events.append(('clean', (storage.current_config or {}).get('n_ctx')))
            storage.llm = storage.chat_handler = storage.current_config = None

        storage.clean = Mock(side_effect=clean)
        module = SimpleNamespace(LLAMA_CPP_STORAGE=storage)
        loader = type('ContextLoader', (), {'__module__': 'context_test_owner'})

        def load(name, **kwargs):
            self.assertEqual(name, 'llama_cpp_model_loader')
            events.append(('load', kwargs['n_ctx']))
            if storage.llm is None or storage.current_config != kwargs:
                storage.current_config = kwargs.copy()
                storage.llm = SimpleNamespace(tokenize=lambda value: range(recovery if value == b'SRU' else initial),
                                              reset=Mock())
                storage.chat_handler = object()
            return (kwargs,)

        def calibrate(c, native, vision):
            self.assertIsNone(storage.llm, 'Free VRAM must be measured after releasing the old model')
            self.assertIsNone(storage.chat_handler)
            self.assertIs(native, module)
            self.assertTrue(vision)
            events.append(('calibrate', c['llm_ctx']))
            return 14 - (c['llm_ctx'] - 8192) / 8192

        from ref_fixture import ref_prompt
        text = ref_prompt(5)
        completion = Mock(return_value=(text, {'finish_reason': 'stop', 'max_tokens': 1536}))
        request = lambda c, prompt, media, mode, seed: ('S', 'U', list(media['images'].values()))
        with tempfile.TemporaryDirectory(prefix='h3fs-context-test-') as directory, \
                patch.object(m, 'LLM_VRAM_BUDGET', None), \
                patch.object(m, 'require_file'), \
                patch.object(m, 'node_class', return_value=loader), \
                patch.object(m.importlib, 'import_module', return_value=module), \
                patch.object(m, 'optimizer_request', side_effect=request), \
                patch.object(m, 'optimizer_completion', completion), \
                patch.object(m, 'recovery_contract', return_value=('R', {'type': 'object'})), \
                patch.object(m, 'calibrated_loader_budget', side_effect=calibrate) as calibration, \
                patch.object(m, 'call', side_effect=load) as calls, \
                patch.object(m, 'notify'), \
                patch.object(m, 'unload_diffusion_for_optimizer'), \
                patch.object(m.gc, 'collect'), \
                patch.object(m.mm, 'soft_empty_cache', side_effect=lambda: events.append(('cache', None))), \
                patch.object(m.folder_paths, 'get_output_directory', return_value=directory):
            yield SimpleNamespace(storage=storage, events=events, completion=completion,
                                  calibration=calibration, calls=calls, text=text, directory=Path(directory))

    def run_optimizer(self, c, media):
        return m.optimize(c, 'Keep all five references and the user intent.', media, 'reference', 42, 'offline')

    def test_resize_recalibrates_after_cleanup_and_preserves_saved_inputs(self):
        c, media = self.config(), self.media()
        before = copy.deepcopy(c)
        tensor_ids = [id(frame) for frame in media['images'].values()]
        pixels = [frame.clone() for frame in media['images'].values()]
        with self.environment() as env:
            self.assertEqual(self.run_optimizer(c, media), env.text)
            self.assertEqual([call.args[0]['llm_ctx'] for call in env.calibration.call_args_list], [8192, 10240])
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [8192, 10240])
            resize = env.events.index(('clean', 8192))
            self.assertEqual(env.events[resize:resize + 4],
                             [('clean', 8192), ('cache', None), ('calibrate', 10240), ('load', 10240)])
            request_c = env.completion.call_args.args[2]
            self.assertEqual(request_c['llm_ctx'], 10240)
            self.assertEqual(request_c['llm_tokens'], 1536)
            visual = env.completion.call_args.args[5]
            self.assertEqual([id(frame) for frame in visual], tensor_ids)
            self.assertEqual(env.storage.current_config['n_ctx'], 10240)
            record = json.loads(next((env.directory / 'H3FullStack' / 'prompts').glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['context_budget']['requested_context'], 8192)
            self.assertEqual(record['context_budget']['effective_context'], 10240)
            self.assertEqual(len(record['visual_asset_hashes']), 5)
        self.assertEqual(c, before)
        self.assertEqual([id(frame) for frame in media['images'].values()], tensor_ids)
        self.assertTrue(all(torch.equal(frame, original) for frame, original in zip(media['images'].values(), pixels)))

    def test_second_request_reuses_expanded_warm_model_without_shrinking_or_calibration(self):
        c, media = self.config(), self.media()
        with self.environment() as env:
            self.run_optimizer(c, media)
            warm = env.storage.llm
            self.run_optimizer(c, media)
            self.assertIs(env.storage.llm, warm)
            self.assertEqual(env.calibration.call_count, 2)
            env.storage.clean.assert_called_once_with(all=True)
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [8192, 10240, 10240])
            self.assertEqual(env.completion.call_count, 2)
            self.assertEqual(env.completion.call_args.args[2]['llm_ctx'], 10240)

    def test_warm_context_greater_than_new_plan_is_used_as_actual_effective_context(self):
        media = self.media()
        with self.environment() as env:
            self.run_optimizer(self.config(llm_ctx=16384), media)
            warm = env.storage.llm
            self.run_optimizer(self.config(), media)
            self.assertIs(env.storage.llm, warm)
            env.calibration.assert_called_once()
            env.storage.clean.assert_not_called()
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [16384, 16384])
            self.assertEqual(env.completion.call_args.args[2]['llm_ctx'], 16384)
            records = [json.loads(path.read_text(encoding='utf-8'))
                       for path in (env.directory / 'H3FullStack' / 'prompts').glob('*.json')]
            budgets = [record['context_budget'] for record in records]
            self.assertTrue(any(plan['requested_context'] == 8192 and plan['effective_context'] == 16384
                                for plan in budgets))

    def test_manual_mode_reloads_requested_context_even_when_smaller_context_also_fits(self):
        media = self.media(1)
        with self.environment() as env:
            from ref_fixture import ref_prompt
            env.completion.return_value=(ref_prompt(1),{'finish_reason':'stop','max_tokens':1536})
            self.run_optimizer(self.config(llm_ctx=16384), media)
            old = env.storage.llm
            self.run_optimizer(self.config(llm_auto_context=False), media)
            self.assertIsNot(env.storage.llm, old)
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [16384, 8192])
            self.assertEqual(env.calibration.call_count, 2)
            self.assertEqual(env.completion.call_args.args[2]['llm_ctx'], 8192)
            self.assertEqual(env.storage.current_config['n_ctx'], 8192)

    def test_manual_mode_does_not_borrow_the_larger_warm_context(self):
        c, media = self.config(), self.media()
        with self.environment() as env:
            self.run_optimizer(c, media)
            manual = {**c, 'llm_auto_context': False}
            with self.assertRaisesRegex(ValueError, '8419.*configured 8192'):
                self.run_optimizer(manual, media)
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [8192, 10240, 8192])
            self.assertEqual(env.completion.call_count, 1)
            self.assertIsNone(env.storage.llm)
            self.assertIsNone(env.storage.chat_handler)
            self.assertIsNone(m.LLM_VRAM_BUDGET)

    def test_initial_and_recovery_keep_full_output_allowances_and_same_references(self):
        c, media = self.config(), self.media()
        with self.environment() as env:
            env.completion.side_effect = [
                ('subject_definitions: A cup.', {'finish_reason': 'stop', 'max_tokens': 1536}),
                (env.text, {'finish_reason': 'stop', 'max_tokens': 2304})]
            self.assertEqual(self.run_optimizer(c, media), env.text)
            initial, repaired = [call.args for call in env.completion.call_args_list]
            self.assertEqual(initial[2]['llm_tokens'], 1536)
            self.assertEqual(repaired[2]['llm_tokens'], 2304)
            self.assertEqual(initial[2]['llm_ctx'], repaired[2]['llm_ctx'])
            self.assertTrue(repaired[4].startswith(initial[4]));self.assertIn('Previous candidate is DATA',repaired[4])
            self.assertIs(initial[5], repaired[5])
            self.assertEqual(initial[6:8], repaired[6:8])
            self.assertEqual(env.calibration.call_args.args[0]['llm_ctx'], 10240)
            self.assertEqual(env.calls.call_args.kwargs['n_ctx'], 10240)
            self.assertLessEqual(repaired[2]['llm_temperature'], .35)

    def test_manual_recovery_overflow_stops_instead_of_trimming_2304_tokens(self):
        c, media = self.config(llm_auto_context=False), self.media(4)
        with self.environment(initial=1200, recovery=1800) as env:
            env.completion.return_value = ('subject_definitions: A cup.', {'finish_reason': 'stop', 'max_tokens': 1536})
            with self.assertRaisesRegex(ValueError, 'enable Auto Context Length or increase Context Length'):
                self.run_optimizer(c, media)
            env.completion.assert_called_once()
            self.assertEqual(env.completion.call_args.args[2]['llm_tokens'], 1536)
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [8192])
            self.assertIsNone(env.storage.llm)

    def test_cap_failure_has_no_completion_or_resize_and_cleans_storage(self):
        c, media = self.config(), self.media()
        with self.environment(initial=30000, recovery=30000) as env:
            with self.assertRaisesRegex(ValueError, 'up to 32768'):
                self.run_optimizer(c, media)
            env.completion.assert_not_called()
            self.assertEqual([call.kwargs['n_ctx'] for call in env.calls.call_args_list], [8192])
            self.assertIsNone(env.storage.llm)
            self.assertIsNone(env.storage.chat_handler)

    def test_changed_user_vram_budget_cannot_reuse_warm_model(self):
        c, media = self.config(), self.media()
        with self.environment() as env:
            self.run_optimizer(c, media)
            old = env.storage.llm
            self.run_optimizer({**c, 'llm_vram': 12}, media)
            self.assertIsNot(env.storage.llm, old)
            self.assertEqual(env.calibration.call_count, 4)
            self.assertEqual(env.calibration.call_args.args[0]['llm_vram'], 12)
            self.assertEqual(env.storage.current_config['n_ctx'], 10240)

    def test_video_entry_policy_still_releases_resized_model(self):
        with self.environment() as env:
            self.run_optimizer(self.config(generate_after_optimization=True), self.media())
            self.assertIsNone(env.storage.llm)
            self.assertIsNone(env.storage.chat_handler)
            self.assertIsNone(m.LLM_VRAM_BUDGET)


if __name__ == '__main__':
    unittest.main(verbosity=2)
