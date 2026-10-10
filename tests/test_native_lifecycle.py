"""Stateful native-owner regressions. No model weights or GPU jobs are loaded."""
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import test_optimizer as fixtures
from test_pipeline import m


class NativeContextTests(unittest.TestCase):
    def storage(self):
        state = {'dirty': True, 'resets': 0, 'requests': 0}
        def reset():
            state['dirty'] = False
            state['resets'] += 1
        def infer(**kwargs):
            if state['dirty']:
                raise RuntimeError('non-consecutive token position')
            state['dirty'] = True
            state['requests'] += 1
            yield {'choices': [{'delta': {'content': 'A quiet room.'}, 'finish_reason': None}]}
            yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        llm = SimpleNamespace(reset=reset, tokenize=lambda _: [], create_chat_completion=infer)
        storage = SimpleNamespace(llm=llm, ensure_embedding_mode=Mock())
        module = SimpleNamespace(scale_image=lambda frame, size: frame, image2base64=lambda frame: 'image')
        return state, storage, module

    def test_two_different_reference_requests_start_from_empty_native_memory(self):
        state, storage, module = self.storage()
        with patch.object(m.mm, 'throw_exception_if_processing_interrupted'):
            for seed, images in ((1, ['front', 'room']), (2, ['rear', 'street'])):
                text, metadata = m.optimizer_completion(storage, module, m.DEFAULTS, 'Contract', 'Intent', images, seed, 'test')
                self.assertEqual(text, 'A quiet room.')
                self.assertEqual(metadata['finish_reason'], 'stop')
        self.assertEqual(state['requests'], 2)
        self.assertEqual(state['resets'], 2)
        self.assertIsNotNone(storage.llm)

    def test_interruption_closes_generator_before_owner_can_be_released(self):
        events = []
        def response():
            try:
                yield {'choices': []}
            finally:
                events.append('stream closed')
        llm = SimpleNamespace(reset=Mock(), create_chat_completion=lambda **_: response())
        storage = SimpleNamespace(llm=llm, ensure_embedding_mode=Mock())
        module = SimpleNamespace(scale_image=lambda frame, size: frame, image2base64=lambda frame: 'image')
        with patch.object(m.mm, 'throw_exception_if_processing_interrupted', side_effect=m.mm.InterruptProcessingException()):
            with self.assertRaises(m.mm.InterruptProcessingException):
                m.optimizer_completion(storage, module, m.DEFAULTS, 'Contract', 'Intent', ['image'], 1, 'test')
        self.assertEqual(events, ['stream closed'])

    def test_completed_stream_is_closed_too(self):
        response = Mock()
        response.__iter__ = Mock(return_value=iter([]))
        storage = SimpleNamespace(llm=SimpleNamespace(reset=Mock(), tokenize=lambda _: [], create_chat_completion=lambda **_: response), ensure_embedding_mode=Mock())
        module = SimpleNamespace(scale_image=lambda frame, size: frame, image2base64=lambda frame: 'image')
        m.optimizer_completion(storage, module, m.DEFAULTS, 'Contract', 'Intent', ['image'], 1, 'test')
        response.close.assert_called_once_with()

    def test_unknown_native_reset_interface_is_rejected_not_silently_reused(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot reset its native context'):
            m.reset_optimizer_context(SimpleNamespace(llm=object()))

    def test_installed_llama_reset_clears_native_memory_and_python_cursor(self):
        from llama_cpp import Llama
        llm = object.__new__(Llama)
        llm._stack = ExitStack()
        llm._ctx = SimpleNamespace(memory_clear=Mock())
        llm.n_tokens = 1450
        llm.speculative = None
        llm._prefilled_prompt = ('old image tokens',)
        llm._restored_logits = 'old logits'
        m.reset_optimizer_context(SimpleNamespace(llm=llm))
        llm._ctx.memory_clear.assert_called_once_with(True)
        self.assertEqual(llm.n_tokens, 0)
        self.assertIsNone(llm._prefilled_prompt)
        self.assertIsNone(llm._restored_logits)
        llm.close()


class NativeReleaseTests(unittest.TestCase):
    def test_public_projector_close_precedes_native_model_close(self):
        events = []
        storage = SimpleNamespace(llm=object(), chat_handler=SimpleNamespace(close=lambda: events.append('projector')))
        def clean(**kwargs):
            events.append('model')
            storage.llm = None
            storage.chat_handler = None
        storage.clean = clean
        m.clean_optimizer_storage(storage)
        self.assertEqual(events, ['projector', 'model'])

    def test_actual_mtmd_public_close_frees_resources_not_in_exit_stack(self):
        from llama_cpp.llama_multimodal import MTMDChatHandler
        handler = object.__new__(MTMDChatHandler)
        handler._exit_stack = ExitStack()
        handler.mtmd_ctx = 'native visual context'
        handler.mctx_params = object()
        native = Mock()
        handler._mtmd_cpp = native
        storage = SimpleNamespace(llm=object(), chat_handler=handler)
        def clean(**kwargs):
            self.assertIsNone(handler.mtmd_ctx)
            self.assertIsNone(handler.mctx_params)
            storage.llm = None
            storage.chat_handler = None
        storage.clean = clean
        m.clean_optimizer_storage(storage)
        native.mtmd_free.assert_called_once_with('native visual context')
        handler.close()
        native.mtmd_free.assert_called_once()

    def test_model_release_attempted_even_if_projector_close_raises(self):
        storage = SimpleNamespace(llm=object(), chat_handler=SimpleNamespace(close=Mock(side_effect=RuntimeError('projector close failed'))), clean=Mock())
        with self.assertRaisesRegex(RuntimeError, 'projector close failed'):
            m.clean_optimizer_storage(storage)
        storage.clean.assert_called_once_with(all=True)

    def test_incomplete_owner_cleanup_is_rejected(self):
        storage = SimpleNamespace(llm=object(), chat_handler=None, clean=Mock())
        with self.assertRaisesRegex(RuntimeError, 'cleanup incomplete'):
            m.clean_optimizer_storage(storage)

    def test_projector_retaining_native_pointer_is_not_reported_as_unloaded(self):
        storage = SimpleNamespace(llm=object(), chat_handler=SimpleNamespace(close=Mock(), mtmd_ctx='still allocated'), clean=Mock())
        with self.assertRaisesRegex(RuntimeError, 'Visual projector cleanup incomplete'):
            m.clean_optimizer_storage(storage)
        storage.clean.assert_called_once_with(all=True)


class HookAndFailureTests(unittest.TestCase):
    def test_known_upstream_unload_hook_does_not_destroy_warm_optimizer(self):
        storage = SimpleNamespace(clean=Mock())
        namespace = {'LLAMA_CPP_STORAGE': storage, 'delegate': Mock()}
        exec('def unload():\n    LLAMA_CPP_STORAGE.clean(all=True)\n    delegate()\n', namespace)
        with patch.object(m.mm, 'unload_all_models', namespace['unload']), patch.object(m.mm, 'unload_all_models_backup', namespace['delegate'], create=True), patch.object(m.mm, 'soft_empty_cache'):
            m.unload_diffusion_for_optimizer(storage)
        namespace['delegate'].assert_called_once()
        storage.clean.assert_not_called()

    def test_unrelated_unload_hook_is_not_bypassed(self):
        unload, delegate = Mock(), Mock()
        with patch.object(m.mm, 'unload_all_models', unload), patch.object(m.mm, 'unload_all_models_backup', delegate, create=True), patch.object(m.mm, 'soft_empty_cache'):
            m.unload_diffusion_for_optimizer(SimpleNamespace())
        unload.assert_called_once()
        delegate.assert_not_called()

    def test_retained_success_resets_native_context_after_each_run(self):
        fixture = fixtures.OptimizerLifecycleTests()
        with fixture.environment() as (storage, completion, calibrate, text):
            fixture.run_optimizer(generate_after_optimization=False)
            first_model = storage.llm
            fixture.run_optimizer(generate_after_optimization=False)
            self.assertIs(storage.llm, first_model)
            first_model.reset.assert_has_calls([unittest.mock.call(), unittest.mock.call()])
            calibrate.assert_called_once()

    def test_inference_error_not_masked_by_secondary_cleanup_failure(self):
        fixture = fixtures.OptimizerLifecycleTests()
        with fixture.environment() as (storage, completion, calibrate, text):
            completion.side_effect = RuntimeError('original inference error')
            storage.clean.side_effect = RuntimeError('secondary cleanup error')
            with self.assertRaisesRegex(RuntimeError, 'original inference error'):
                fixture.run_optimizer()

    def test_comfy_cancellation_unloads_without_retry(self):
        fixture = fixtures.OptimizerLifecycleTests()
        with fixture.environment() as (storage, completion, calibrate, text):
            completion.side_effect = m.mm.InterruptProcessingException()
            with self.assertRaises(m.mm.InterruptProcessingException):
                fixture.run_optimizer(generate_after_optimization=False)
            completion.assert_called_once()
            self.assertIsNone(storage.llm)


if __name__ == '__main__':
    unittest.main()
