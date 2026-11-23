import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from strata_launcher import StrataLauncher


class OneClickUpdateTests(unittest.TestCase):
    def test_explicit_update_resolves_latest_in_worker_not_selected_release(self):
        app = SimpleNamespace(runtime={}, runner=Mock(), server=Mock(),
                              _version_log=Mock(), _install_version_worker=Mock())
        app.runner.running.return_value = False
        app.server.running.return_value = False
        with patch('strata_launcher.threading.Thread') as thread:
            StrataLauncher.do_update(app)
        self.assertEqual(thread.call_args.kwargs['args'], (None, True))
        self.assertTrue(app.runtime['updating'])

    def test_background_check_never_installs_or_switches(self):
        cfg = {'auto_update': True, 'version_pinned': False}
        app = SimpleNamespace(_remote_version='', cfg=SimpleNamespace(get=cfg.get, update=Mock()),
                              ver_label=Mock(), up_status=Mock(), _start_update=Mock())
        StrataLauncher._apply_update_info(app, '0.2.6', '0.2.11', '', '', '', True)
        app._start_update.assert_not_called()

    def test_failed_install_does_not_stop_running_model_or_activate(self):
        app = SimpleNamespace(runtime={'updating': True}, _update_resume={'root': Path('old')},
                              server=Mock(), _activate_version=Mock(),
                              _refresh_version_environments=Mock())
        StrataLauncher._finish_version_install(app, None, True)
        app.server.stop.assert_not_called()
        app._activate_version.assert_not_called()
        self.assertFalse(app.runtime['updating'])

    def test_success_stops_and_activates_before_resume(self):
        app = SimpleNamespace(runtime={'updating': True}, _update_resume={'root': Path('old')},
                              server=Mock(), _activate_version=Mock(), _version_log=Mock(),
                              _resume_after_update=Mock(), _refresh_version_environments=Mock())
        with patch('strata_launcher.threading.Thread') as thread:
            StrataLauncher._finish_version_install(app, Path('new'), True)
        app.server.stop.assert_called_once()
        app._activate_version.assert_called_once_with(Path('new'), confirm_differences=False)
        self.assertEqual(thread.call_args.kwargs['args'][0], Path('new'))

    def test_resume_failure_queues_rollback(self):
        app = SimpleNamespace(server=Mock(), env=Mock(), _version_log=Mock(),
                              _rollback_failed_update=Mock(), ui=Mock())
        app.server.start.side_effect = RuntimeError('load failed')
        resume = dict(config=Path('model.json'), vision=False, optimization={},
                      chat_template=None, data_dir=None, root=Path('old'))
        StrataLauncher._resume_after_update(app, Path('new'), resume)
        app.server.stop.assert_called_once()
        app.ui.assert_called_once_with(app._rollback_failed_update, resume)

    def test_rollback_restores_original_then_starts_it(self):
        app = SimpleNamespace(runtime={'updating': True}, _activate_version=Mock(),
                              _resume_after_update=Mock())
        resume = {'root': Path('old')}
        with patch('strata_launcher.threading.Thread') as thread:
            StrataLauncher._rollback_failed_update(app, resume)
        app._activate_version.assert_called_once_with(Path('old'), confirm_differences=False)
        self.assertEqual(thread.call_args.kwargs['args'], (Path('old'), resume, False))

    def test_reuse_sync_preserves_target_engine_and_model_paths(self):
        import json
        from strata_versions import sync_user_settings
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'old', Path(tmp) / 'new'
            old.mkdir(); new.mkdir()
            (old / 'strata-model.json').write_text(json.dumps(dict(
                exe='old-engine', args=['--native', 'old-pack', '--max-context', '65536'],
                port=9999, drafting='off')))
            target = new / 'strata-model.json'
            target.write_text(json.dumps(dict(exe='new-engine', lib_dirs=['new-libs'],
                args=['--native', 'new-pack', '--max-context', '8192'], port=8080)))
            sync_user_settings(old, new)
            result = json.loads(target.read_text())
            self.assertEqual(result['exe'], 'new-engine')
            self.assertEqual(result['lib_dirs'], ['new-libs'])
            self.assertEqual(result['args'], ['--native', 'new-pack', '--max-context', '65536'])
            self.assertEqual(result['port'], 9999)
            self.assertEqual(result['drafting'], 'off')

    def test_reuse_sync_restores_exact_original_bytes_after_partial_write(self):
        import json
        import strata_versions
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'old', Path(tmp) / 'new'
            old.mkdir(); new.mkdir()
            originals = {}
            for name in ('a', 'b'):
                filename = f'strata-{name}.json'
                (old / filename).write_text(json.dumps(dict(exe='old', args=[], port=9999)))
                path = new / filename
                originals[path] = b'\xef\xbb\xbf{ "exe": "new", "args": [], "port": 8080 }\r\n'
                path.write_bytes(originals[path])
            real_write = strata_versions.atomic_json
            count = 0
            def fail_second(path, data):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError('fixture disk failure')
                real_write(path, data)
            with patch('strata_versions.atomic_json', side_effect=fail_second):
                with self.assertRaises(OSError):
                    strata_versions.sync_user_settings(old, new)
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_rollback_resume_uses_exact_original_config_path(self):
        app = SimpleNamespace(server=Mock(), env=Mock(), _version_log=Mock(),
                              _finish_update_resume=Mock(), ui=Mock(), _drain_server_log=Mock())
        app.server.running.return_value = True
        app.server.local_url.return_value = 'http://127.0.0.1:8080'
        resume = dict(config=Path('external-configs/model.json'), vision=False, optimization={},
                      chat_template=None, data_dir=None, root=Path('old'))
        with patch('strata_launcher.strata_health_ready', return_value=True), \
             patch('strata_launcher.http_probe', return_value=(200, {})), \
             patch('strata_launcher.threading.Thread'):
            StrataLauncher._resume_after_update(app, Path('old'), resume, False)
        self.assertEqual(app.server.start.call_args.args[1], resume['config'])
        app.ui.assert_called_once_with(app._finish_update_resume, False)


if __name__ == '__main__':
    unittest.main()
