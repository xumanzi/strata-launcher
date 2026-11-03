import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class UnicodeAndConnectionTests(unittest.TestCase):
    def test_scaled_control_vector_path_keeps_strength_and_uses_path_adapter(self):
        from strata_launch_support import prepare_engine_paths
        cfg = {'args': ['--control-vector-scaled', 'C:\\中文\\direction.gguf:1.0']}
        with patch('strata_launch_support.windows_engine_path', return_value='C:/alias/direction.gguf') as adapter:
            result = prepare_engine_paths(cfg)
        adapter.assert_called_once_with('C:\\中文\\direction.gguf')
        self.assertEqual(result['args'][1], 'C:/alias/direction.gguf:1.0')
        self.assertEqual(cfg['args'][1], 'C:\\中文\\direction.gguf:1.0')

    @unittest.skipUnless(os.name == 'nt', 'Windows junction fallback')
    def test_junction_fallback_reads_same_file_when_short_names_are_unavailable(self):
        from strata_launch_support import windows_engine_path
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / '没有短名称的中文目录'
            folder.mkdir()
            path = folder / 'model.gguf'
            path.write_bytes(b'GGUF')
            with patch('strata_launch_support._short_windows_path', side_effect=str):
                alias = windows_engine_path(str(path))
            self.assertTrue(alias.isascii())
            self.assertTrue(os.path.samefile(alias, path))
            self.assertEqual(Path(alias).read_bytes(), b'GGUF')

    @unittest.skipUnless(os.name == 'nt', 'Windows engine path handling')
    def test_engine_paths_become_ascii_without_changing_original_config(self):
        from strata_launch_support import prepare_engine_paths
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'IQ3_XXS 破限'
            folder.mkdir()
            first = folder / 'Model-00001-of-00002.gguf'
            second = folder / 'Model-00002-of-00002.gguf'
            first.write_bytes(b'GGUF')
            second.write_bytes(b'GGUF')
            cfg = {'args': ['--native', str(first), '--ple-gguf', str(second)], 'model_name': '破限模型'}
            original = json.dumps(cfg)
            adapted = prepare_engine_paths(cfg)
            for index, source in ((1, first), (3, second)):
                self.assertTrue(adapted['args'][index].isascii())
                self.assertTrue(os.path.samefile(adapted['args'][index], source))
                self.assertEqual(Path(adapted['args'][index]).name, source.name)
            self.assertEqual(json.dumps(cfg), original)
            self.assertEqual(adapted['model_name'], '破限模型')

    def test_no_ascii_alias_fails_before_engine_launch_with_actionable_message(self):
        from strata_launch_support import windows_engine_path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '中文.gguf'
            path.touch()
            with patch('strata_launch_support._short_windows_path', return_value=str(path)), patch(
                'strata_launch_support._ascii_directory_alias', side_effect=OSError('unsupported')):
                with self.assertRaisesRegex(ValueError, '英文'):
                    windows_engine_path(str(path))

    def test_connections_use_full_api_model_name_and_do_not_fall_back_to_stale_key(self):
        from strata_launch_support import connection_fields, api_model_name
        cfg = {'model_name': 'IQ3_XXS 破限', 'local_import': {'folder': 'G:/models/破限'},
            'args': ['--native', 'G:/models/破限/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf'],
            'host': '0.0.0.0', 'port': 8080, 'api_key': ''}
        full = 'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS'
        self.assertEqual(api_model_name(cfg), full)
        fields = dict(connection_fields(cfg, actual_port=8082, ips=['192.168.1.85', '172.16.0.1'], api_key=''))
        self.assertEqual(fields['模型名称'], full)
        self.assertEqual(fields['本机访问地址'], 'http://127.0.0.1:8082')
        self.assertEqual(fields['局域网访问地址'], 'http://192.168.1.85:8082')
        self.assertEqual(fields['APIkey'], '')

    def test_lan_ip_uses_physical_adapter_instead_of_vpn_or_virtual_switch(self):
        from strata_launch_support import select_lan_ipv4
        adapters = [
            {'Physical': False, 'Up': True, 'IPs': ['198.18.0.1'], 'Gateway': True, 'Metric': 1},
            {'Physical': False, 'Up': True, 'IPs': ['172.20.112.1'], 'Gateway': False, 'Metric': 10},
            {'Physical': True, 'Up': True, 'IPs': ['192.168.1.85'], 'Gateway': True, 'Metric': 25}]
        self.assertEqual(select_lan_ipv4(adapters), ['192.168.1.85'])
        self.assertEqual(select_lan_ipv4(adapters[:2]), [])
