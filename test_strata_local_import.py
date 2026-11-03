import tempfile
import unittest
from pathlib import Path


class LocalImportTests(unittest.TestCase):
    def test_import_q4_split_and_single_file(self):
        from strata_local_import import inspect_model
        for quant in ('UD-Q4_K_XL', 'UD-IQ4_XS'):
            for count in (1, 2):
                with tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    for i in range(1, count + 1):
                        suffix = '.gguf' if count == 1 else f'-{i:05d}-of-{count:05d}.gguf'
                        (root / ('Qwen3.8-Flash-Next-' + quant + suffix)).write_bytes(b'GGUF')
                    info = inspect_model(root)
                    self.assertEqual((info['family'], info['model']), ('unsloth', quant))
                    self.assertEqual(len(info['shards']), count)

    def test_unsupported_quant_has_engine_specific_error(self):
        from strata_local_import import inspect_model
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'Qwen3.8-Flash-Next-Q4_K_M.gguf').write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, 'Strata.*Q4_K_M'):
                inspect_model(d)

    def test_custom_projection_arguments_and_validation(self):
        from strata_local_import import projection_setup_args
        with tempfile.TemporaryDirectory() as d:
            vector = Path(d) / 'Huihui.gguf'
            vector.write_bytes(b'GGUF')
            self.assertEqual(projection_setup_args(True, str(vector)),
                ['--experimental-speed-projection', str(vector.resolve())])
            self.assertEqual(projection_setup_args(False, str(vector)),
                ['--experimental-speed-projection', 'off'])
            self.assertEqual(projection_setup_args(True, ''),
                ['--experimental-speed-projection', 'on'])
            with self.assertRaisesRegex(ValueError, '不存在'):
                projection_setup_args(True, str(Path(d) / 'missing.gguf'))
            vector.write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError, 'GGUF'):
                projection_setup_args(True, str(vector))

    def test_discovers_huihui_vector_next_to_coder_folder(self):
        from strata_local_import import find_huihui_vector
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / 'IQ1_M'
            folder.mkdir()
            self.assertIsNone(find_huihui_vector(folder))
            vector = root / 'Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf'
            vector.write_bytes(b'GGUF')
            self.assertEqual(find_huihui_vector(folder), vector)

    def test_runtime_vision_switch_enables_configured_encoder_and_engine_flag(self):
        from strata_local_import import apply_runtime_vision
        cfg = {'args': ['--native', 'model.gguf'],
               'vision': {'exe': 'strata-vision.exe', 'mmproj': 'mmproj.gguf'}}
        result = apply_runtime_vision(cfg, Path('.'), True)
        self.assertIn('--vision', result['args'])
        self.assertIn('vision', result)
        self.assertNotIn('--vision', cfg['args'])

    def test_runtime_vision_switch_disables_encoder_and_engine_flag(self):
        from strata_local_import import apply_runtime_vision
        cfg = {'args': ['--native', 'model.gguf', '--vision', '--vram-reserve-mib', '700', '--kv', 'int8'],
               'vision': {'exe': 'strata-vision.exe', 'mmproj': 'mmproj.gguf'}}
        result = apply_runtime_vision(cfg, Path('.'), False)
        self.assertEqual(result['args'], ['--native', 'model.gguf', '--kv', 'int8'])
        self.assertNotIn('vision', result)
        self.assertIn('--vision', cfg['args'])

    def test_runtime_vision_switch_does_not_duplicate_engine_flag(self):
        from strata_local_import import apply_runtime_vision
        cfg = {'args': ['--native', 'model.gguf', '--vision'],
               'vision': {'exe': 'strata-vision.exe'}}
        result = apply_runtime_vision(cfg, Path('.'), True)
        self.assertEqual(result['args'].count('--vision'), 1)

    def test_runtime_switch_overrides_old_local_text_only_preference(self):
        from strata_local_import import apply_runtime_vision
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            engine = root / 'engine'
            engine.mkdir()
            (engine / 'strata-vision.exe').write_bytes(b'exe')
            (engine / 'BUILD.json').write_text(json.dumps({'vision': 'gpu'}))
            shared = root / 'data'
            projector = shared / 'models' / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf'
            projector.parent.mkdir(parents=True)
            projector.write_bytes(b'GGUF')
            shard = root / 'model' / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
            shard.parent.mkdir()
            shard.write_bytes(b'GGUF')
            cfg = {'args': ['--native', str(shard)], 'local_import': {
                'family': 'qwen', 'shared_data': str(shared), 'vision_enabled': False}}
            result = apply_runtime_vision(cfg, root, True)
            self.assertIn('--vision', result['args'])
            self.assertIn('vision', result)
            self.assertFalse(cfg['local_import']['vision_enabled'])

    def test_fresh_launcher_defaults_runtime_vision_to_enabled(self):
        from strata_launcher import AppConfig
        self.assertIs(AppConfig.DEFAULTS['runtime_vision_enabled'], True)

    def test_existing_local_import_gains_vision_from_shared_files(self):
        from strata_local_import import ensure_local_vision
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            engine = root / 'engine'
            engine.mkdir()
            (engine / 'strata-vision.exe').write_bytes(b'exe')
            (engine / 'BUILD.json').write_text(json.dumps({'vision': 'gpu'}))
            shared = root / 'data'
            projector = shared / 'models' / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf'
            projector.parent.mkdir(parents=True)
            projector.write_bytes(b'GGUF')
            shard = root / '破限' / 'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
            shard.parent.mkdir()
            shard.write_bytes(b'GGUF')
            cfg = {'args': ['--native', str(shard)], 'local_import': {'family': 'qwen', 'shared_data': str(shared)}}
            result = ensure_local_vision(cfg, root)
            self.assertEqual(result['vision']['model'], str(shard))
            self.assertEqual(result['vision']['mmproj'], str(projector))
            self.assertTrue(result['vision']['gpu'])
            self.assertIn('--vision', result['args'])
            self.assertNotIn('--vision', cfg['args'])
            self.assertNotIn('vision', cfg)
            shard.unlink()
            self.assertNotIn('vision', ensure_local_vision(cfg, root))

    def test_existing_vision_section_repairs_missing_engine_flag_without_duplicate(self):
        from strata_local_import import ensure_local_vision
        cfg = {'args': ['--native', 'model.gguf'],
               'vision': {'exe': 'strata-vision.exe'},
               'local_import': {'family': 'qwen', 'vision_enabled': True}}
        repaired = ensure_local_vision(cfg, Path('.'))
        self.assertEqual(repaired['args'].count('--vision'), 1)
        self.assertEqual(ensure_local_vision(repaired, Path('.'))['args'].count('--vision'), 1)
        self.assertNotIn('--vision', cfg['args'])

    def test_new_install_defaults_to_gpu_vision(self):
        from strata_launcher import AppConfig
        self.assertEqual(AppConfig.DEFAULTS['install_vision'], 'gpu')

    def test_explicit_text_only_local_import_stays_text_only(self):
        from strata_local_import import ensure_local_vision
        cfg = {'local_import': {'family': 'qwen', 'vision_enabled': False}}
        self.assertIs(ensure_local_vision(cfg, Path('.')), cfg)

    def test_import_completion_selects_imported_model_before_start_offer(self):
        from strata_launcher import StrataLauncher
        from types import SimpleNamespace
        from unittest.mock import patch
        selected = []
        widget = SimpleNamespace(configure=lambda **kw: None)
        app = SimpleNamespace(local_import_btn=widget, install_btn=widget, cancel_btn=widget,
            runtime={}, _local_import_label='新模型', install_log=SimpleNamespace(append=lambda s: None),
            cfg=SimpleNamespace(set=lambda *a: None), rescan_models=lambda: None,
            model_var=SimpleNamespace(set=selected.append))
        def offer(*args):
            self.assertEqual(selected, ['新模型'])
            return False
        with patch('strata_launcher.messagebox.askyesno', offer):
            StrataLauncher._on_install_done(app, 0, None)

    def test_discovers_complete_abliterated_shards_and_isolates_variants(self):
        from strata_local_import import inspect_model
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'IQ3_XXS 破限'
            folder.mkdir()
            for i in (1, 2):
                (folder / f'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-{i:05}-of-00002.gguf').write_bytes(b'GGUF')
            result = inspect_model(folder)
            self.assertEqual(result['model'], 'IQ3_XXS')
            self.assertEqual(result['family'], 'qwen')
            self.assertEqual(len(result['shards']), 2)
            other = Path(d) / 'original'
            other.mkdir()
            for i in (1, 2):
                (other / f'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-{i:05}-of-00002.gguf').write_bytes(b'GGUF')
            self.assertNotEqual(result['identity'], inspect_model(other)['identity'])

    def test_coder_iq1_m_is_identified_and_prepared_as_coder(self):
        from strata_local_import import inspect_model, configure_setup
        from types import SimpleNamespace
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / 'Coder'
            folder.mkdir()
            for i in (1, 2):
                (folder / f'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-{i:05}-of-00002.gguf').write_bytes(b'GGUF')
            info = inspect_model(folder)
            self.assertEqual((info['model'], info['family']), ('IQ1_M', 'coder'))
            mod = SimpleNamespace(FAMILIES={'coder': {'tag': 'coder-', 'profile': 'expert-profile-coder.bin'}},
                data_folder=lambda _: None, find_in=lambda roots, rel: None,
                write_config=lambda p, c: p.write_text(json.dumps(c)))
            configure_setup(mod, root / 'shared', info, 'Coder 本地', vision_enabled=False)
            self.assertEqual(mod.FAMILIES['coder']['profile'], 'expert-profile-coder.bin')
            cfg_path = root / f"strata-local-{info['identity']}-iq1_m.json"
            mod.write_config(cfg_path, {'exe': 'engine', 'args': []})
            saved = json.loads(cfg_path.read_text())
            self.assertEqual(saved['local_import']['family'], 'coder')

    def test_coder_still_requires_all_shards(self):
        from strata_local_import import inspect_model
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf').write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, '缺少分片'):
                inspect_model(folder)

    def test_rejects_missing_shards_ambiguous_folder_and_invalid_header(self):
        from strata_local_import import inspect_model
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            first = folder / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
            first.write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, '缺少分片'):
                inspect_model(folder)
            second = folder / first.name.replace('00001-of', '00002-of')
            second.write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError, 'GGUF'):
                inspect_model(folder)
            second.write_bytes(b'GGUF')
            (folder / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf').write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, '一个模型'):
                inspect_model(folder)

    def test_preparation_does_not_migrate_or_reuse_original_pack(self):
        from strata_local_import import configure_setup
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shared = root / 'data'
            (shared / 'packs' / 'iq3_xxs').mkdir(parents=True)
            (shared / 'mtp').mkdir()
            mod = SimpleNamespace(FAMILIES={'qwen': {'tag': '', 'name': 'qwen'}},
                data_folder=lambda _: self.fail('must not migrate data'),
                find_in=lambda roots, rel: next((r / rel for r in roots if (r / rel).exists()), None),
                write_config=lambda p, c: p.write_text(__import__('json').dumps(c)))
            info = {'family': 'qwen', 'model': 'IQ3_XXS', 'identity': 'abc123', 'folder': str(root / 'gguf')}
            configure_setup(mod, shared, info, '破限版', vision_enabled=False)
            data, elsewhere = mod.data_folder('ignored')
            self.assertEqual(elsewhere, [])
            self.assertNotEqual(data, shared)
            # Current upstream passes migration/default flags positionally or by name.
            self.assertEqual(mod.data_folder('ignored', True, False, True), (data, []))
            self.assertEqual(mod.data_folder(requested='ignored', move=True,
                yes=False, set_default=True), (data, []))
            self.assertIsNone(mod.find_in([data], 'packs/iq3_xxs'))
            self.assertEqual(mod.find_in([data], 'mtp'), (shared / 'mtp').resolve())
            self.assertEqual(mod.FAMILIES['qwen']['tag'], 'local-abc123-')
            path = root / 'strata-local-abc123-iq3_xxs.json'
            mod.write_config(path, {'exe': 'engine', 'args': []})
            saved = __import__('json').loads(path.read_text())
            self.assertEqual(saved['model_name'], '破限版')
            self.assertEqual(saved['local_import']['folder'], info['folder'])
            self.assertFalse(saved['local_import']['vision_enabled'])

    def test_cli_generates_discoverable_config_and_preserves_original(self):
        import json
        import subprocess
        import sys
        import strata_local_import
        from strata_launcher import StrataEnv
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / '破限模型'
            folder.mkdir()
            for i in (1, 2):
                (folder / f'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-{i:05}-of-00002.gguf').write_bytes(b'GGUF')
            original = root / 'strata-iq3_s.json'
            original.write_text(json.dumps({'exe': 'original', 'args': [], 'model_name': '原版'}))
            before = original.read_bytes()
            # Stand in only for expensive upstream GPU checks/preparation. Run the real CLI adapter.
            (root / 'setup.py').write_text('''import json, sys
from pathlib import Path
ROOT = Path(__file__).parent
FAMILIES = {'qwen': {'tag': '', 'name': 'qwen'}}
def data_folder(requested):
    raise AssertionError('original migration must never run')
def find_in(roots, rel):
    return next((r / rel for r in roots if (r / rel).exists()), None)
def write_config(path, cfg):
    path.write_text(json.dumps(cfg))
def main():
    args = sys.argv
    data, roots = data_folder(args[args.index('--data-dir') + 1], False, True, False)
    assert roots == []
    assert '--no-start' in args
    assert args[args.index('--gguf-dir') + 1].endswith('破限模型')
    tag = FAMILIES['qwen']['tag'] + args[args.index('--model') + 1].lower()
    pack = data / 'packs' / tag
    pack.mkdir(parents=True)
    write_config(ROOT / ('strata-' + tag + '.json'), {'exe': 'local-engine', 'args': ['--pack', str(pack)]})
    return 0
''', encoding='utf-8')
            result = subprocess.run([sys.executable, strata_local_import.__file__, str(root), str(folder),
                str(root / 'shared'), '我的破限模型', '--yes', '--model', 'IQ3_XXS'],
                capture_output=True, encoding='utf-8', errors='replace', env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = StrataEnv()
            env.root = root
            configs = env.installed_configs()
            self.assertEqual(len(configs), 2)
            imported = next(p for p in configs if p != original)
            saved = json.loads(imported.read_text())
            self.assertEqual(saved['model_name'], '我的破限模型')
            self.assertIn('local-imports', saved['args'][1])
            self.assertEqual(saved['local_import']['model'], 'IQ3_S')
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(env.config_choices(imported)['model'], 'IQ3_S')
