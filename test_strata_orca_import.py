import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from strata_orca_import import inspect_orca, prepare, PREFIX


class OrcaImportTests(unittest.TestCase):
    def test_two_shards_detected_and_missing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertIsNone(inspect_orca(root))
            (root / f'{PREFIX}-00001-of-00002.gguf').write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, '缺少'):
                inspect_orca(root)
            (root / f'{PREFIX}-00002-of-00002.gguf').write_bytes(b'GGUF')
            self.assertEqual(inspect_orca(root)['recipe'], 'orca')

    def test_conversion_preserves_native_ple_tokenizer_and_key(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            info={'identity':'abc','family':'qwen','model':'IQ3_XXS','shards':[str(root/'first.gguf'),str(root/'second.gguf')]}
            ec=types.SimpleNamespace(resolve=lambda *a,**k:types.SimpleNamespace(exe=root/'strata.exe'),verify_signature=lambda *a:None)
            preflight=types.ModuleType('strata_control.gguf_preflight')
            preflight.inspect_metadata=lambda p:{'architecture':'qwen4exp'}
            parent=types.ModuleType('strata_control');parent.gguf_preflight=preflight
            with patch.dict('sys.modules',{'strata_control':parent,'strata_control.gguf_preflight':preflight}), \
                 patch('strata_update.run_signed_cli'), patch('strata_update.load_component_api',return_value=ec), \
                 patch('strata_orca_import.run_command') as command:
                prepare(root,info,root/'data','Orca',['--api-key','secret','--context','65536'],'off')
            self.assertEqual(command.call_count,1)
            self.assertIn('--compat-bf16', command.call_args.args)
            cfg=json.loads((root/'strata-local-abc-iq3_xxs.json').read_text())
            self.assertEqual(cfg['api_key'],'secret')
            self.assertEqual(cfg['args'][cfg['args'].index('--ple-gguf')+1],info['shards'][0])
            self.assertTrue(cfg['tokenizer'].endswith('tokenizer'))
            self.assertNotIn('--mtp',cfg['args'])

    def test_dflash_rejected_before_download(self):
        with patch('strata_orca_import.run_command') as command:
            with self.assertRaisesRegex(ValueError,'DFlash2'):
                prepare(Path('.'),{},Path('.'),'Orca',[],'dflash2')
            command.assert_not_called()
