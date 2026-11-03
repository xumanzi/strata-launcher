"""Compatibility with the bundled upstream service; mock engine, no weights or GPU required."""
import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from strata_launcher import ServerProcess


class LatestServiceTests(unittest.TestCase):
    def test_deepseek_config_reaches_latest_service_through_gateway(self):
        root = Path('build/official-latest/Strata-0.2.6').resolve()
        self.assertTrue((root / 'serve/server.py').is_file())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'strata-deepseek-test.json'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            cfg = {'arch': 'deepseek41', 'exe': sys.executable,
                   'args': ['--pack', 'model-pack', '--max-context', '8192', '--dspark', 'auto'],
                   'model_name': 'deepseek-v4.1-flash', 'host': '127.0.0.1', 'port': port,
                   'api_key': 'audit-test', 'context_compress_enabled': False, 'fit_max_tokens': False}
            path.write_text(json.dumps(cfg))
            original = path.read_bytes()
            server = ServerProcess()
            native_popen = subprocess.Popen
            runtime_configs = []
            def mock_engine(cmd, **kwargs):
                runtime = Path(cmd[cmd.index('--config') + 1])
                runtime_configs.append(json.loads(runtime.read_text(encoding='utf-8')))
                cmd = list(cmd)
                cmd[cmd.index('--engine') + 1] = 'mock'
                return native_popen(cmd, **kwargs)
            try:
                with patch('subprocess.Popen', side_effect=mock_engine):
                    server.start(SimpleNamespace(root=root, python=lambda: sys.executable), path,
                                 open_browser=False, vision_enabled=False,
                                 optimization={'mode': 'concurrent', 'spec': '6', 'profile': True})
                for _ in range(120):
                    if not server.running():
                        self.fail('Upstream service stopped: ' + '\n'.join(list(server.log_queue.queue)))
                    try:
                        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        conn.request('GET', '/health')
                        response = conn.getresponse()
                        response.read()
                        if response.status == 200:
                            break
                    except OSError:
                        pass
                    finally:
                        conn.close()
                    time.sleep(.1)
                else:
                    self.fail('Upstream service did not become ready')
                self.assertEqual(runtime_configs[0]['args'], cfg['args'] + ['--vram-reserve-mib', '500'])
                self.assertEqual(runtime_configs[0]['arch'], 'deepseek41')
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
                conn.request('POST', '/v1/chat/completions', json.dumps({
                    'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': 'hello'}],
                    # Mock mode does not activate the native engine's family effort mapping.
                    'reasoning_effort': 'low', 'max_tokens': 64, 'stream': False}),
                    {'Content-Type': 'application/json', 'Authorization': 'Bearer audit-test'})
                response = conn.getresponse()
                data = response.read()
                conn.close()
                self.assertEqual(response.status, 200, data.decode()[:1500])
                self.assertTrue(json.loads(data)['choices'])
                self.assertEqual(path.read_bytes(), original)
            finally:
                server.stop()


if __name__ == '__main__':
    unittest.main()
