import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class LocalContextTests(unittest.TestCase):
    def test_native_inkling_one_million_context_does_not_add_qwen_flags(self):
        from strata_launch_support import save_local_context
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inkling.json'
            config = dict(arch='inkling2_loop', local_import={'family': 'inkling-2-mini'},
                          args=['--native', 'model.gguf', '--max-context', '262144', '--dspark', 'draft.gguf'])
            path.write_text(json.dumps(config), encoding='utf-8')
            result = save_local_context(path, 1048576)
            self.assertEqual(result['args'], ['--native', 'model.gguf', '--max-context', '1048576', '--dspark', 'draft.gguf'])

    def config(self):
        return {'exe': 'engine', 'args': ['--pack', 'private-pack', '--native', 'custom-model.gguf',
            '--ple-gguf', 'custom-shard2.gguf', '--max-context', '32768', '--kv', 'int8'],
            'model_name': 'full-custom-name', 'local_import': {'family': 'qwen', 'model': 'IQ3_XXS',
            'identity': 'keep', 'folder': 'custom-folder'}, 'api_key': 'keep-key',
            'sampling': {'temperature': 0.6}, 'context_compress_enabled': True}

    def test_llama_context_uses_ctx_size_and_preserves_draft_options(self):
        from strata_launch_support import local_context_choices, save_local_context
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'strata-inkling.json'
            config = {'engine': 'llama.cpp', 'local_import': {'family': 'inkling-2-mini'},
                      'args': ['--ctx-size', '4096', '--max-context', '8192', '--kv', 'int8',
                               '--spec-type', 'draft-dspark', '--cache-type-k', 'q8_0']}
            path.write_text(json.dumps(config), encoding='utf-8')
            self.assertEqual(local_context_choices(config)['context'], 4096)
            updated = save_local_context(path, 16384)
            self.assertEqual(updated['args'], ['--ctx-size', '16384', '--spec-type', 'draft-dspark',
                                                '--cache-type-k', 'q8_0'])

    def test_import_metadata_enables_context_selection(self):
        from strata_launcher import StrataEnv
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'strata-local-keep-iq3_xxs.json'
            path.write_text(json.dumps(self.config()))
            choices = StrataEnv().config_choices(path)
            self.assertEqual(choices['family'], 'qwen')
            self.assertEqual(choices['model'], 'IQ3_XXS')
            self.assertEqual(choices['context'], 32768)
            self.assertEqual(choices['kv'], 'int8')

    def test_context_update_preserves_import_paths_and_settings_and_backup(self):
        from strata_launch_support import save_local_context
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'local.json'
            original = self.config()
            path.write_text(json.dumps(original))
            save_local_context(path, 131072)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['args'][saved['args'].index('--max-context') + 1], '131072')
            for key in ('exe', 'model_name', 'local_import', 'api_key', 'sampling', 'context_compress_enabled'):
                self.assertEqual(saved[key], original[key])
            for key in ('--pack', '--native', '--ple-gguf', '--kv'):
                self.assertEqual(saved['args'][saved['args'].index(key)+1],original['args'][original['args'].index(key)+1])
            self.assertEqual(json.loads(Path(str(path)+'.context-size.bak').read_text()), original)

    def test_long_context_auto_rope_recalculates_and_is_removed_when_shrinking(self):
        from strata_launch_support import save_local_context
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'local.json'
            path.write_text(json.dumps(self.config()))
            save_local_context(path, 524288)
            cfg=json.loads(path.read_text());args=cfg['args']
            self.assertEqual(args[args.index('--rope-scaling')+1], 'yarn')
            self.assertEqual(args[args.index('--rope-scale')+1], '2')
            save_local_context(path, 32768)
            args=json.loads(path.read_text())['args']
            self.assertNotIn('--rope-scaling',args)
            self.assertNotIn('--rope-scale',args)

    def test_quick_context_updates_import_without_official_install_and_restarts_active_model(self):
        from strata_launcher import StrataLauncher
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'local.json';path.write_text(json.dumps(self.config()))
            ent={'family':'qwen','model':'IQ3_XXS','context':32768,'path':str(path)}
            events=[];widget=SimpleNamespace(configure=lambda **kw:None)
            app=SimpleNamespace(_quick_ent=ent,ctx_quick_menu=SimpleNamespace(get=lambda:'128K'),
                runner=SimpleNamespace(running=lambda:False),runtime={},
                server=SimpleNamespace(running=lambda:True,cfg_path=path),
                cfg=SimpleNamespace(set=lambda *a:None),ctx_quick_status=widget,
                env=SimpleNamespace(tables=lambda:({}, {}, [32768,131072]),_mod=None),
                _refresh_ctx_quick=lambda:None,model_cache=lambda:[ent],
                restart_model=lambda:events.append('restart'),toast=lambda *a:None,
                _apply_local_quick_context=lambda ent,ctx:StrataLauncher._apply_local_quick_context(app,ent,ctx))
            with patch('strata_launcher.messagebox.askyesno',return_value=True):
                StrataLauncher._apply_quick_context(app)
            self.assertEqual(json.loads(path.read_text())['args'][7],'131072')
            self.assertEqual(events,['restart'])
