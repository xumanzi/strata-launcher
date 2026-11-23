import os
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from strata_optimization import apply_runtime_optimization, model_settings


class OptimizationTests(unittest.TestCase):
    def test_native_cuda_uses_official_work_ram_reserve(self):
        cfg = {'backend': 'cuda', 'model_name': 'Qwen3.8-Flash-Next-IQ3_S',
               'args': ['--native', 'model.gguf']}
        with patch.dict('os.environ', {}, clear=True):
            result, _ = apply_runtime_optimization(cfg, {}, 'model.json', None)
            manual, _ = apply_runtime_optimization(
                dict(cfg, env={'STRATA_RAM_KEEP_GIB': '12'}), {}, 'model.json', None)
        if os.name == 'nt':
            self.assertEqual(result['env']['STRATA_RAM_KEEP_GIB'], '8')
        self.assertEqual(manual['env']['STRATA_RAM_KEEP_GIB'], '12')
        self.assertNotIn('env', cfg)

    def test_default_preserves_thinking_and_single_request(self):
        with TemporaryDirectory() as tmp:
            original = {'args': ['--spec', '4'], 'parallel': 2}
            result, notes = apply_runtime_optimization(original, {}, Path(tmp) / 'model.json', Path(tmp))
            self.assertEqual(result['parallel'], 1)
            self.assertNotIn('reasoning_budget_tokens', result)
            self.assertEqual(original['parallel'], 2)

    def test_kv_resident_and_reserve_replace_existing_without_mutation(self):
        original = {'args': ['--kv-resident', '65536', '--vram-reserve-mib', '700', '--kv-grow'],
                    'env': {'STRATA_KV_GROW': '1'}}
        result, _ = apply_runtime_optimization(original, {'kv_resident': 32768}, 'model.json', None)
        self.assertEqual(result['args'].count('--kv-resident'), 1)
        self.assertEqual(result['args'][result['args'].index('--kv-resident') + 1], '32768')
        self.assertEqual(result['args'][result['args'].index('--vram-reserve-mib') + 1], '700')
        self.assertNotIn('--kv-grow', result['args'])
        self.assertEqual(result['env']['STRATA_KV_GROW'], '0')
        self.assertIn('--kv-grow', original['args'])

    def test_vram_floor_preserves_larger_user_reserve_and_disables_elastic_vision(self):
        original = {
            'vision': {'elastic': True},
            'vram_elastic': True,
            'args': ['--vram-elastic', '--vram-reserve-mib', '700'],
            'args_written': ['--vram-elastic', '--vram-reserve-mib', '700'],
            'user_args': ['--vram-reserve-mib', '2048', '--spec', '4', '--vram-elastic'],
        }
        result, _ = apply_runtime_optimization(original, {}, 'model.json', None)
        self.assertIs(result['vram_elastic'], False)
        self.assertEqual(result['vision'], original['vision'])
        self.assertEqual(result['args'], ['--vram-reserve-mib', '2048'])
        self.assertEqual(result['user_args'], ['--spec', '4', '--vram-reserve-mib', '2048'])
        self.assertEqual(original['user_args'][1], '2048')

    def test_full_kv_and_invalid_residency(self):
        result, _ = apply_runtime_optimization({'args': ['--kv-resident', '32768']},
                                              {'kv_resident': 0}, 'model.json', None)
        self.assertEqual(result['args'][result['args'].index('--kv-resident') + 1], '0')
        for value in (-1, 8192, 'bad', 32768.5, True):
            with self.assertRaises(ValueError):
                apply_runtime_optimization({}, {'kv_resident': value}, 'model.json', None)

    def test_kv_precision_combines_with_residency_without_mutating_model(self):
        original = {'args': ['--kv', 'int8', '--kv', 'int8']}
        result, _ = apply_runtime_optimization(original, {'kv_precision': 'q4_0', 'kv_resident': 32768}, 'model.json', None)
        self.assertEqual(result['args'].count('--kv'), 1)
        self.assertEqual(result['args'][result['args'].index('--kv') + 1], 'q4_0')
        self.assertIn('--kv-resident', result['args'])
        self.assertEqual(original['args'], ['--kv', 'int8', '--kv', 'int8'])

    def test_kv_precision_validation_original_and_architecture(self):
        for precision in ('int8', 'q4_0', 'k8v4', 'fp16'):
            result, _ = apply_runtime_optimization({}, {'kv_precision': precision}, 'model.json', None)
            self.assertEqual(result['args'][result['args'].index('--kv') + 1], precision)
        result, _ = apply_runtime_optimization({'args': ['--kv', 'int8']}, {'kv_precision': 'original'}, 'model.json', None)
        self.assertEqual(result['args'][:2], ['--kv', 'int8'])
        for precision in ('invalid', None, 4):
            with self.assertRaises(ValueError):
                apply_runtime_optimization({}, {'kv_precision': precision}, 'model.json', None)
        with self.assertRaises(ValueError):
            apply_runtime_optimization({'arch': 'deepseek41'}, {'kv_precision': 'q4_0'}, 'model.json', None)

    def test_concurrent_and_session_are_exclusive(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            concurrent, _ = apply_runtime_optimization({'slot_save_path': 'old'}, {'mode': 'concurrent'}, path, Path(tmp))
            self.assertEqual(concurrent['parallel'], 2)
            self.assertNotIn('slot_save_path', concurrent)
            session, _ = apply_runtime_optimization({'parallel': 2}, {'mode': 'session'}, path, Path(tmp))
            self.assertEqual(session['parallel'], 1)
            self.assertTrue(Path(session['slot_save_path']).is_dir())

    def test_reasoning_budget_and_effort(self):
        with TemporaryDirectory() as tmp:
            result, _ = apply_runtime_optimization({}, {'effort': 'high', 'budget': 8192}, Path(tmp) / 'model.json', Path(tmp))
            self.assertEqual(result['reasoning_budget_tokens'], 8192)
            self.assertEqual(model_settings({'effort': 'high'})['reasoning_effort'], 'high')

    def test_local_profile_keeps_legacy_history_across_engine_updates(self):
        from strata_optimization import _model_dir
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / '.strata-versions'
            old = store / 'release-0.2.15' / 'strata-local-abc-iq3_s.json'
            new = store / 'release-0.2.19' / 'terminal' / old.name
            old.parent.mkdir(parents=True)
            legacy = _model_dir(old, root) / 'expert-profile.bin'
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b'learned')
            cfg = {'local_import': {'identity': 'abc'}}
            newer, _ = apply_runtime_optimization(cfg, {'profile': True}, new, root)
            older, _ = apply_runtime_optimization(cfg, {'profile': True}, old, root)
            other, _ = apply_runtime_optimization(cfg, {'profile': True},
                                                   new.with_name('strata-local-abc-iq4_xs.json'), root)
            self.assertEqual(older['expert_profile_save'], newer['expert_profile_save'])
            self.assertEqual(Path(newer['expert_profile_save']).read_bytes(), b'learned')
            self.assertNotEqual(newer['expert_profile_save'], other['expert_profile_save'])
            self.assertEqual(legacy.read_bytes(), b'learned')

    def test_per_model_profile_paths_are_distinct(self):
        with TemporaryDirectory() as tmp:
            first, _ = apply_runtime_optimization({}, {'profile': True}, Path(tmp) / 'a.json', Path(tmp))
            second, _ = apply_runtime_optimization({}, {'profile': True}, Path(tmp) / 'b.json', Path(tmp))
            self.assertNotEqual(first['expert_profile_save'], second['expert_profile_save'])


if __name__ == '__main__':
    unittest.main()
