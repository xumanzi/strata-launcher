import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import strata_launcher as launcher


def official_fixture():
    return SimpleNamespace(
        MODELS={'IQ2_XS': {'ram_gb': 48},
                'UD-Q4_K_XL': {'families': ('unsloth',), 'download_gb': 111.3}},
        FAMILIES={k: {'title': k, 'tag': k + '-'} for k in ('qwen', 'swift', 'coder', 'unsloth')},
        CONTEXTS=[8192, 32768, 262144], DSV41_CONTEXT=8192,
        OWN_PATH_FAMILIES={'qwen36': {'title': 'Qwen3.6', 'tag': 'qwen36-', 'file': '{q}.gguf',
            'vision': True, 'sha256': {'UD-Q4_K_XL.gguf': (22853663008, 'hash')},
            'models': {'UD-Q4_K_XL': {'ram_gb': 32, 'about': 'small'}}}},
        MULTI_FILE_FAMILIES={'deepseek41': {'title': 'DeepSeek', 'tag': 'deepseek-v4.1-flash-',
            'vision': False, 'arch': 'deepseek41', 'sha256': {'a': (138000000000, 'hash'),
                'shared': (119000000000, 'hash')}, 'models': {'IST-DAS-2.00bpw': {
                    'shards': ['a'], 'shared': ['shared'], 'ram_gb': 40, 'about': 'deepseek'}}}})


class CatalogTests(unittest.TestCase):
    def test_bundled_catalog_supports_new_install_without_existing_source(self):
        from strata_catalog import bundled_tables, family_models
        models, families, contexts = bundled_tables()
        self.assertEqual(len(families), 9)
        self.assertEqual(sum(len(family_models(models, f)) for f in families), 44)
        self.assertEqual(len(family_models(models, 'unsloth')), 12)
        self.assertIn('UD-IQ4_XS-v31', family_models(models, 'unsloth'))
        self.assertNotIn('UD-IQ1_S', family_models(models, 'unsloth'))
        self.assertEqual(set(family_models(models, 'hy4-flash')), {'SQ-64', 'SQ-96', 'SQ-128'})
        inkling = family_models(models, 'inkling-2-mini')
        self.assertTrue({'SQ-10', 'SQ-15', 'SQ-20', 'UD-IQ4_XS', 'UD-Q4_K_XL'}.issubset(inkling))
        self.assertEqual(len(inkling), 13)
        self.assertIn(262144, inkling['SQ-10']['contexts'])
        self.assertTrue(inkling['SQ-10']['vision'])
        self.assertIn(8192, contexts)

    def env(self):
        env = launcher.StrataEnv()
        env.root = Path('.')
        return env

    def test_all_three_official_tables_are_visible(self):
        with patch.object(launcher, 'load_official_setup', return_value=official_fixture()):
            env = self.env()
            self.assertIn('deepseek41', env.tables()[1])
            self.assertIn('qwen36', env.tables()[1])
            self.assertEqual(list(env.models_for('deepseek41')), ['IST-DAS-2.00bpw'])

    def test_same_quantization_uses_family_specific_metadata(self):
        with patch.object(launcher, 'load_official_setup', return_value=official_fixture()):
            env = self.env()
            self.assertAlmostEqual(env.models_for('qwen36')['UD-Q4_K_XL']['download_gb'], 22.9)
            self.assertEqual(env.models_for('unsloth')['UD-Q4_K_XL']['download_gb'], 111.3)

    def test_unrestricted_legacy_sizes_only_belong_to_qwen_and_swift(self):
        with patch.object(launcher, 'load_official_setup', return_value=official_fixture()):
            env = self.env()
            self.assertIn('IQ2_XS', env.models_for('swift'))
            self.assertNotIn('IQ2_XS', env.models_for('coder'))
            self.assertNotIn('IQ2_XS', env.models_for('deepseek41'))

    def test_deepseek_download_counts_shared_files(self):
        with patch.object(launcher, 'load_official_setup', return_value=official_fixture()):
            md = self.env().models_for('deepseek41')['IST-DAS-2.00bpw']
            self.assertEqual(md['download_gb'], 257.0)
            self.assertFalse(md['vision'])
            self.assertEqual(md['contexts'], [8192])

    def test_deepseek_install_drops_qwen_only_options(self):
        from strata_catalog import adapt_install_args
        args = ['--yes', '--family', 'deepseek41', '--model', 'IST-DAS-2.00bpw',
            '--context', '32768', '--vision', 'gpu', '--kv', 'int8',
            '--experimental-speed-projection', 'on', '--low-ram', 'on', '--draft-vocab', 'en',
            '--resident-budget-gib', '22', '--kv-streaming', 'on', '--parallel', '2',
            '--vram-reserve-mib', '700', '--gpus', '0,1', '--host', '127.0.0.1', '--no-start']
        out = adapt_install_args(args, {'multi_file': True, 'vision': False, 'contexts': [8192]})
        self.assertEqual(out[out.index('--context') + 1], '32768')
        self.assertEqual(out[out.index('--vision') + 1], 'no')
        self.assertEqual(out[out.index('--gpus') + 1], '0,1')
        for flag in ('--kv', '--experimental-speed-projection', '--low-ram', '--draft-vocab',
                     '--resident-budget-gib', '--kv-streaming', '--parallel', '--vram-reserve-mib'):
            self.assertNotIn(flag, out)

    def test_qwen36_does_not_receive_qwen38_projection_or_kv(self):
        from strata_catalog import adapt_install_args
        args = ['--family', 'qwen36', '--model', 'UD-Q4_K_XL', '--context', '32768',
                '--vision', 'gpu', '--kv', 'q4_0', '--experimental-speed-projection', 'off']
        out = adapt_install_args(args, {'own_path': True, 'vision': True})
        self.assertNotIn('--kv', out)
        self.assertNotIn('--experimental-speed-projection', out)
        self.assertIn('gpu', out)

    def test_deepseek_update_does_not_activate_qwen_component(self):
        import strata_update as update
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = root / 'strata-deepseek.json'
            cfg.write_text(json.dumps({'arch': 'deepseek41', 'exe': 'old-dsv41.exe', 'args': []}))
            with patch.object(update, 'load_component_api'), patch.object(update, 'run_signed_cli') as cli, \
                    patch.object(update, 'update_deepseek_components', create=True) as deep:
                update.update_signed_components(root, lambda x: None)
                cli.assert_not_called()
                deep.assert_called_once()
            self.assertEqual(json.loads(cfg.read_text())['exe'], 'old-dsv41.exe')

    def test_deepseek_runtime_preserves_its_native_flags(self):
        from strata_optimization import apply_runtime_optimization
        cfg = {'arch': 'deepseek41', 'args': ['--pack', 'model-pack', '--dspark', 'auto']}
        out, notes = apply_runtime_optimization(cfg, {'mode': 'concurrent', 'spec': '6', 'profile': True}, 'model.json', None)
        self.assertEqual(out['args'], cfg['args'] + ['--vram-reserve-mib', '500'])
        self.assertNotIn('expert_profile_save', out)
        self.assertEqual(out.get('parallel', 1), 1)
        self.assertTrue(notes)

    def test_activation_rejects_cross_architecture_engine(self):
        import strata_update as update
        from test_strata_signed_update import NativeUpdateTests
        with tempfile.TemporaryDirectory() as d:
            root, cfg, pkg, data = NativeUpdateTests().fixture(d)
            data['arch'] = 'deepseek41'
            cfg.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, '不匹配'):
                update.activate_components(root, {cfg: pkg})
            self.assertEqual(json.loads(cfg.read_text()), data)

    def test_installed_deepseek_variants_have_distinct_labels(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for size in ('IST-DAS-2.00bpw', 'IST-DAS-2.25bpw'):
                path = Path(d) / ('strata-' + size + '.json')
                path.write_text(json.dumps({'model_name': 'deepseek-v4.1-flash'}))
                paths.append(path)
            app = SimpleNamespace(env=SimpleNamespace(installed_configs=lambda: paths,
                config_choices=lambda p: {'family': 'deepseek41', 'model': p.stem[7:]}),
                cfg=SimpleNamespace(set=lambda *a: None), _data_dir_models=lambda: [])
            rows = launcher.StrataLauncher._scan_models(app)
            self.assertEqual(len({r['label'] for r in rows}), 2)
            self.assertIn('IST-DAS-2.00bpw', rows[0]['label'])

    def test_deepseek_signed_package_activation_preserves_model_and_network(self):
        import strata_update as update
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / 'strata-deepseek.json'
            original = {'arch': 'deepseek41', 'exe': 'strata-dsv41.exe', 'args': ['--pack', 'keep-pack'],
                        'port': 8080, 'api_key': 'keep', 'tokenizer': 'keep-tokenizer'}
            path.write_text(json.dumps(original))
            pkg = root / 'new-package'
            pkg.mkdir()
            exe = pkg / 'strata-dsv41.exe'
            exe.write_bytes(b'deepseek engine fixture')
            (pkg / 'engine-component.json').write_text(json.dumps({'version': '0.1.50',
                'entrypoint': exe.name, 'backend': 'cuda', 'files': [{'path': exe.name,
                'bytes': exe.stat().st_size, 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}]}))
            install = Mock(return_value=exe)
            ec = SimpleNamespace(read_manifest=Mock(), verify_signature=Mock())
            with patch.dict(sys.modules, {
                'strata_control': SimpleNamespace(dsv41_engine=SimpleNamespace(install=install)),
                'strata_cli': SimpleNamespace(update_cmd=SimpleNamespace(_feed_source=lambda *a: 'official-feed'))}), \
                    patch.object(update, 'load_component_api', return_value=ec):
                update.update_deepseek_components(root, {path: original}, lambda x: None)
            self.assertEqual(install.call_args.args[1], 'official-feed')
            ec.verify_signature.assert_called_once()
            saved = json.loads(path.read_text())
            self.assertEqual(Path(saved['exe']), exe.resolve())
            for key in ('args', 'port', 'api_key', 'tokenizer', 'arch'):
                self.assertEqual(saved[key], original[key])
            self.assertEqual(json.loads(Path(str(path) + '.update.bak').read_text()), original)


if __name__ == '__main__':
    unittest.main()
