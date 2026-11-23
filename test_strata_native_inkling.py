import unittest

from strata_native_inkling import INKLING_KV_FALLBACK, INKLING_WORK_RESERVE_MIB, inkling_kv, normalize_config


def inkling_config(**overrides):
    config = {
        'arch': 'inkling2_loop',
        'args': ['--native', 'M.gguf', '--max-context', '262144'],
        'user_args': [],
    }
    config.update(overrides)
    return config


class NativeInklingKVTest(unittest.TestCase):
    def test_aliases_map_to_engine_names(self):
        self.assertEqual(inkling_kv('int8'), 'k8v8')
        self.assertEqual(inkling_kv('k8v4'), 'k8v4')
        self.assertEqual(inkling_kv('FP16'), 'fp16')

    def test_unsupported_value_is_rejected(self):
        with self.assertRaises(ValueError):
            inkling_kv('q4_0')

    def test_selected_precision_replaces_existing_kv(self):
        config = inkling_config(args=['--native', 'M.gguf', '--kv', 'fp16',
                                      '--vram-reserve-mib', '120'])
        result = normalize_config(config, 'k8v4')
        self.assertEqual(result['args'][result['args'].index('--kv') + 1], 'k8v4')
        self.assertEqual(result['args'][result['args'].index('--vram-reserve-mib') + 1], str(INKLING_WORK_RESERVE_MIB))
        self.assertEqual(result['launcher_kv_precision'], 'k8v4')
        self.assertEqual(result['args'].count('--kv'), 1)

    def test_keep_larger_manual_reserve_and_stamp_generated_args(self):
        config = inkling_config(args=['--native', 'M.gguf', '--vram-reserve-mib', '2048'],
                                args_written=['--native', 'M.gguf'])
        result = normalize_config(config)
        self.assertEqual(result['args'][result['args'].index('--vram-reserve-mib') + 1], '2048')
        self.assertEqual(result['args_written'], result['args'])
        self.assertNotIn('--kv', config['args'])

    def test_original_keeps_saved_precision(self):
        result = normalize_config(inkling_config(launcher_kv_precision='fp8'), 'original')
        self.assertEqual(result['launcher_kv_precision'], 'fp8')

    def test_unsupported_precision_falls_back_to_int8(self):
        result = normalize_config(inkling_config(), 'q4_0')
        self.assertEqual(result['launcher_kv_precision'], INKLING_KV_FALLBACK)
        self.assertEqual(result['args'][result['args'].index('--kv') + 1], INKLING_KV_FALLBACK)

    def test_runtime_optimization_passes_precision_for_inkling(self):
        """Regression: the launcher used to drop the KV choice on the Inkling path."""
        from strata_optimization import apply_runtime_optimization
        result, notes = apply_runtime_optimization(inkling_config(), {'kv_precision': 'k8v4'},
                                                   'cfg.json', None)
        self.assertEqual(result['args'][result['args'].index('--kv') + 1], 'k8v4')
        self.assertTrue(any('k8v4' in note for note in notes))

    def test_launcher_kv_flag_follows_install_selection(self):
        import ast
        from pathlib import Path
        source = Path('strata_launcher.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        bodies = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                bodies[node.name] = node
        for name in ('_build_install_args', '_generate_config'):
            flags = [constant.value for constant in ast.walk(bodies[name])
                     if isinstance(constant, ast.Constant) and constant.value == '--launcher-inkling-kv']
            self.assertEqual(flags, ['--launcher-inkling-kv'], name)
        self.assertIn('_selected_inkling_kv', bodies)


if __name__ == '__main__':
    unittest.main()
