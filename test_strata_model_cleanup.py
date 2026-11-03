import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from strata_model_cleanup import build_cleanup_plan, execute_cleanup_plan


class ModelCleanupTests(unittest.TestCase):
    def _config(self, root, name, args, local=None):
        path = root / name
        data = {'model_name': name.removesuffix('.json'), 'args': [str(x) for x in args]}
        if local:
            data['local_import'] = local
        path.write_text(json.dumps(data), encoding='utf-8')
        return path, data

    def test_full_plan_includes_only_model_owned_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; configs = root / 'app'; configs.mkdir(); data.mkdir()
            pack = data / 'packs' / 'iq3'; pack.mkdir(parents=True); (pack / 'weights.bin').write_bytes(b'x' * 7)
            folder = data / 'models' / 'local'; folder.mkdir(parents=True)
            shard1 = folder / 'model-00001-of-00002.gguf'; shard1.write_bytes(b'a' * 11)
            shard2 = folder / 'model-00002-of-00002.gguf'; shard2.write_bytes(b'b' * 13)
            completed = Path(str(shard1) + '.done'); completed.write_bytes(b'ok')
            mtp = data / 'mtp' / 'rt'; mtp.mkdir(parents=True); (mtp / 'shared.bin').write_bytes(b'shared')
            mmproj = data / 'models' / 'mmproj.gguf'; mmproj.write_bytes(b'vision')
            cfg_path, cfg = self._config(configs, 'strata-local-test.json', [
                '--pack', pack, '--native', shard1, '--ple-gguf', shard2, '--mtp', mtp],
                {'folder': str(folder), 'identity': 'abc123'})
            cfg['vision'] = {'mmproj': str(mmproj), 'model': str(shard1)}

            plan = build_cleanup_plan(cfg_path, cfg, [cfg_path], data)
            deletable = {item.path for item in plan.items if item.deletable and item.scope == 'data'}

            self.assertIn(pack.resolve(), deletable)
            self.assertIn(shard1.resolve(), deletable)
            self.assertIn(shard2.resolve(), deletable)
            self.assertIn(completed.resolve(), deletable)
            self.assertNotIn(mtp.resolve(), deletable)
            self.assertNotIn(mmproj.resolve(), deletable)
            self.assertEqual(plan.reclaim_bytes, 33)

    def test_other_config_reference_protects_shared_model_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; configs = root / 'app'; configs.mkdir(); data.mkdir()
            model = data / 'models' / 'shared.gguf'; model.parent.mkdir(); model.write_bytes(b'model')
            selected, cfg = self._config(configs, 'strata-a.json', ['--native', model])
            other, _ = self._config(configs, 'strata-b.json', ['--native', model])

            plan = build_cleanup_plan(selected, cfg, [selected, other], data)
            item = next(item for item in plan.items if item.path == model.resolve())

            self.assertFalse(item.deletable)
            self.assertIn('其他模型', item.reason)

    def test_execute_full_removes_plan_items_but_keeps_shared_resources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; configs = root / 'app'; configs.mkdir(); data.mkdir()
            shard = data / 'models' / 'only.gguf'; shard.parent.mkdir(); shard.write_bytes(b'model')
            mtp = data / 'mtp' / 'rt'; mtp.mkdir(parents=True); shared = mtp / 'shared.bin'; shared.write_bytes(b'keep')
            cfg_path, cfg = self._config(configs, 'strata-only.json', ['--native', shard, '--mtp', mtp])
            settings = cfg_path.with_suffix('.shared-settings.json'); settings.write_text('{}')
            plan = build_cleanup_plan(cfg_path, cfg, [cfg_path], data)

            result = execute_cleanup_plan(plan, delete_data=True)

            self.assertFalse(cfg_path.exists())
            self.assertFalse(settings.exists())
            self.assertFalse(shard.exists())
            self.assertTrue(shared.exists())
            self.assertEqual(result.failed, ())

    def test_config_only_keeps_large_model_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; configs = root / 'app'; configs.mkdir(); data.mkdir()
            shard = data / 'models' / 'only.gguf'; shard.parent.mkdir(); shard.write_bytes(b'model')
            cfg_path, cfg = self._config(configs, 'strata-only.json', ['--native', shard])
            plan = build_cleanup_plan(cfg_path, cfg, [cfg_path], data)

            execute_cleanup_plan(plan, delete_data=False)

            self.assertFalse(cfg_path.exists())
            self.assertTrue(shard.exists())


if __name__ == '__main__':
    unittest.main()
