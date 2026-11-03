import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


class LauncherPolicyTests(unittest.TestCase):
    def test_kv_setting_round_trip_is_per_model(self):
        from strata_launcher import (StrataLauncher, THINKING_LABELS, RUNTIME_MODE_LABELS,
                                     THINKING_BUDGET_LABELS, ACCELERATION_LABELS, RUNTIME_KV_LABELS)
        class Var:
            def __init__(self, value): self.value = value
            def get(self): return self.value
            def set(self, value): self.value = value
        saved = {}
        app = SimpleNamespace(
            _selected_cfg_path=lambda: Path('model.json'),
            cfg=SimpleNamespace(get=lambda key: saved.get(key), set=lambda key, value: saved.update({key: value})),
            effort_var=Var(THINKING_LABELS['high']), optimization_mode_var=Var(RUNTIME_MODE_LABELS['single']),
            thinking_budget_var=Var(THINKING_BUDGET_LABELS[0]), spec_var=Var(ACCELERATION_LABELS['original']),
            profile_var=Var(False), kv_resident_var=Var('32K'), kv_precision_var=Var(RUNTIME_KV_LABELS['q4_0']))
        app._model_optimization = lambda: StrataLauncher._model_optimization(app)
        StrataLauncher._save_model_optimization(app)
        self.assertEqual(saved['model_optimizations'][str(Path('model.json').resolve())]['kv_resident'], 32768)
        app.kv_resident_var.set('全部驻留')
        app.kv_precision_var.set(RUNTIME_KV_LABELS['original'])
        StrataLauncher._load_model_optimization(app)
        self.assertEqual(app.kv_resident_var.get(), '32K')
        self.assertEqual(app.kv_precision_var.get(), RUNTIME_KV_LABELS['q4_0'])

    def test_stopped_policy_preserves_existing_config(self):
        from strata_launcher import StrataLauncher
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.json'
            original = {'exe': 'keep-engine', 'args': ['keep'], 'api_key': 'test-key'}
            path.write_text(json.dumps(original))
            StrataLauncher._save_context_policy(None, path,
                {'context_compress_enabled': True, 'fit_max_tokens': False})
            saved = json.loads(path.read_text())
            self.assertTrue(saved['context_compress_enabled'])
            self.assertFalse(saved['fit_max_tokens'])
            for key, value in original.items():
                self.assertEqual(saved[key], value)
            self.assertEqual(json.loads(Path(str(path) + '.context.bak').read_text()), original)

    def test_live_launcher_switch_controls_authenticated_server(self):
        from strata_launcher import StrataLauncher
        from strata_gateway import ContextGateway
        service = ContextGateway('127.0.0.1', 0, 1, {'context_compress_enabled': True, 'fit_max_tokens': False}, api_key='test-key')
        finished = threading.Event()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.json'
            path.write_text(json.dumps({'exe': 'keep-engine', 'api_key': 'test-key'}))
            service.cfg_path = path
            service.start()
            httpd = service.server
            app = SimpleNamespace(
                compress_var=SimpleNamespace(get=lambda: False),
                fit_var=SimpleNamespace(get=lambda: True),
                server=SimpleNamespace(cfg_path=path, host='127.0.0.1', port=httpd.server_port, running=lambda: True, set_context_policy=service.update_policy),
                cfg=SimpleNamespace(update=lambda **k: None), dash_log=SimpleNamespace(append=lambda s: None),
                _refresh_compress_status=lambda: finished.set() if hasattr(app, '_context_sync_generation') else None,
                ui=lambda fn, *args: fn(*args))
            try:
                StrataLauncher._on_compress_toggle(app)
                self.assertTrue(finished.wait(8), 'live update did not finish')
                self.assertFalse(service.policy["context_compress_enabled"])
                self.assertTrue(service.policy["fit_max_tokens"])
                self.assertIn('服务端已确认', app._context_sync_status)
                saved = json.loads(path.read_text())
                self.assertFalse(saved['context_compress_enabled'])
                self.assertEqual(saved['exe'], 'keep-engine')
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == '__main__':
    unittest.main()
