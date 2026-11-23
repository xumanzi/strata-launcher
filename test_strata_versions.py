import unittest
from unittest.mock import patch
from strata_versions import normalize_release, release_catalog
import tempfile
import json
import zipfile
from pathlib import Path
from strata_versions import (VersionStore, copy_configs, extract_source, check_exact_engines,
                             pinned_feed, validate_config_flags, release_engine)


class VersionsTest(unittest.TestCase):
    def test_exact_release_validation(self):
        self.assertEqual(normalize_release('v0.2.10'), '0.2.10')
        for value in ('../0.2.10', 'latest', '0.2', '0.2.10;exit'):
            with self.assertRaises(ValueError):
                normalize_release(value)

    @patch('strata_versions.urllib.request.urlopen')
    def test_excludes_unpublished_and_preview(self, opened):
        import io
        opened.return_value.__enter__.return_value = io.StringIO(
            '[{"tag_name":"v0.2.10","html_url":"https://github.com/test"},'
            '{"tag_name":"v0.2.11","prerelease":true},'
            '{"tag_name":"v0.2.12","draft":true}]')
        self.assertEqual([r['version'] for r in release_catalog()], ['0.2.10'])

    def test_rate_limited_api_falls_back_to_paginated_release_pages(self):
        import io
        import urllib.error
        pages = [
            '<a href="/Niko1221/Strata/releases/tag/v0.2.10">Release</a>'
            '<a href="/Niko1221/Strata/releases/tag/v0.2.9">Release</a>'
            '<a rel="next" href="/Niko1221/Strata/releases?page=2">Next</a>',
            '<a href="/Niko1221/Strata/releases/tag/v0.2.8">Release</a>'
            '<a href="/Niko1221/Strata/releases/tag/v0.2.6">Release</a>']
        replies = [urllib.error.HTTPError('https://api.github.com', 403, 'rate limit', {}, None)]
        replies += [io.BytesIO(page.encode()) for page in pages]
        with patch('strata_versions.urllib.request.urlopen', side_effect=replies) as opened:
            result = release_catalog()
        self.assertEqual([r['version'] for r in result], ['0.2.10', '0.2.9', '0.2.8', '0.2.6'])
        self.assertEqual(opened.call_count, 3)
        self.assertEqual(result[0]['source'], '官方发布页')

    def test_api_reads_next_page(self):
        import io
        batch = [{'tag_name': f'v0.1.{i}', 'html_url': 'https://github.com/test'} for i in range(100)]
        replies = [io.StringIO(json.dumps(batch)), io.StringIO(json.dumps([
            {'tag_name': 'v0.2.9', 'html_url': 'https://github.com/test'}]))]
        with patch('strata_versions.urllib.request.urlopen', side_effect=replies) as opened:
            result = release_catalog()
        self.assertEqual(len(result), 101)
        self.assertIn('page=2', opened.call_args.args[0].full_url)

    def test_no_cache_does_not_repeat_network_request(self):
        with tempfile.TemporaryDirectory() as tmp, patch('strata_versions.release_catalog') as fetch:
            self.assertEqual(VersionStore(tmp).catalog(), [])
            fetch.assert_not_called()

    def test_corrupt_cache_does_not_prevent_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionStore(tmp)
            (store.root / 'catalog.json').write_text('invalid json')
            self.assertEqual(store.catalog(), [])
            with patch('strata_versions.release_catalog', return_value=[{'version': '0.2.9'}]):
                self.assertEqual(store.catalog(refresh=True)[0]['version'], '0.2.9')

    def test_refresh_keeps_user_selected_version(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from strata_launcher import StrataLauncher
        selection = SimpleNamespace(get=lambda: '0.2.8 · 引擎 0.1.54', set=Mock())
        app = SimpleNamespace(version_choice=selection,
                              _release_labels={'0.2.8 · 引擎 0.1.54': '0.2.8'},
                              version_menu=Mock(), version_list_status=Mock())
        StrataLauncher._apply_release_list(app, [{'version': '0.2.10'}, {'version': '0.2.8'}], 'fresh')
        self.assertTrue(selection.set.call_args.args[0].startswith('0.2.8 ·'))

    def tree(self, root, name):
        path = root / name
        path.mkdir()
        (path / 'setup.py').write_text('MIN_ENGINE = (0, 1, 46)')
        (path / 'CMakeLists.txt').write_text('project(strata VERSION 0.1.56)')
        return path

    def test_switch_and_rollback_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = VersionStore(root)
            old = self.tree(root, 'original')
            new = self.tree(store.root, 'release-0.2.10-test')
            store.register(old, 'original', original=True)
            store.activate(old)
            store.register(new, '0.2.10', verified=True)
            store.activate(new)
            self.assertEqual(store.state()['previous'], str(old.resolve()))
            with self.assertRaises(RuntimeError):
                store.remove(new)
            store.activate(old)
            self.assertEqual(store.state()['previous'], str(new.resolve()))

    def test_configuration_copy_does_not_rewrite_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = self.tree(root, 'old'), self.tree(root, 'new')
            (old / 'model.gguf').touch()
            config = {'exe': 'engine/strata.exe', 'args': ['--model', 'model.gguf'], 'cwd': str(old)}
            path = old / 'strata-model.json'
            path.write_text(json.dumps(config))
            before = path.read_bytes()
            copy_configs(old, new)
            copied = json.loads((new / path.name).read_text())
            self.assertEqual(copied['args'][1], str((old / 'model.gguf').resolve()))
            self.assertEqual(path.read_bytes(), before)

    def test_archive_traversal_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr('../escape.py', 'bad')
            with self.assertRaises(RuntimeError):
                extract_source(archive, root / 'source')
            self.assertFalse((root / 'escape.py').exists())

    def test_engine_version_is_independent_of_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.tree(Path(tmp), 'source')
            exe = root / 'strata.exe'
            exe.touch()
            with patch('strata_versions.engine_targets', return_value={exe: 'cuda'}), \
                 patch('strata_versions.component_metadata', return_value={'version': '0.1.57'}):
                self.assertEqual(check_exact_engines(root), ['strata.exe: 0.1.57'])

    def test_release_feed_never_points_to_latest(self):
        self.assertIn('/download/v0.2.6/', pinned_feed('0.2.6'))
        self.assertNotIn('latest', pinned_feed('0.2.6'))

    def test_failed_environment_cannot_be_activated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionStore(tmp)
            target = self.tree(store.root, 'release-failed')
            store.register(target, '0.2.10')
            with self.assertRaises(RuntimeError):
                store.activate(target)
            self.assertEqual(store.state()['current'], '')

    def test_downgrade_rejects_newer_cli_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = root / 'src/program/generate.cpp'
            parser.parent.mkdir(parents=True)
            parser.write_text('if (arg == "--model") {}')
            with self.assertRaises(RuntimeError):
                validate_config_flags(root, {Path('model.json'): {'args': ['--draft', 'dflash2']}})
            validate_config_flags(root, {Path('model.json'): {'args': ['--model', 'weights.gguf']}})

    def test_release_body_engine_mapping(self):
        self.assertEqual(release_engine('The app and Terminal are 0.2.10. Control and every engine package are 0.1.56.'), '0.1.56')
        self.assertEqual(release_engine('Release 0.2.10'), '')

    def test_pinned_cli_receives_explicit_release_feed(self):
        import os
        import strata_update
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'strata.py').write_text(
                "import sys,json\nfrom pathlib import Path\n"
                "Path(__file__).with_name('argv.json').write_text(json.dumps(sys.argv))\n"
                "print(json.dumps({'ok':True,'results':[{'component':'engine','status':'current'}]}))\n")
            with patch.dict(os.environ, {'STRATA_LAUNCHER_RELEASE': '0.2.6'}):
                strata_update.run_signed_cli(root, root / 'components', log=lambda _: None)
            argv = json.loads((root / 'argv.json').read_text())
            self.assertEqual(argv[argv.index('--feed') + 1], pinned_feed('0.2.6'))

    def test_failed_install_keeps_original_environment_active(self):
        import io
        import sys
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import strata_launcher as launcher
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = VersionStore(root)
            original = self.tree(root, 'original')
            store.register(original, 'original', original=True)
            store.activate(original)
            logs = []
            app = SimpleNamespace(_version_store=lambda: store,
                                  env=SimpleNamespace(ensure_root=lambda log: original),
                                  runtime={'updating': True}, _version_log=logs.append,
                                  ui=lambda fn: fn(), _refresh_version_environments=lambda: None)
            def archive_download(url, path, log):
                with zipfile.ZipFile(path, 'w') as package:
                    package.writestr('source/setup.py', 'MIN_ENGINE = (0, 1, 46)')
                    package.writestr('source/CMakeLists.txt', 'project(strata VERSION 0.1.56)')
                    package.writestr('source/strata_cli/update_cmd.py', '')
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = ['fixture install failure']
            process.wait.return_value = 1
            with patch('strata_launcher.urllib.request.urlopen', return_value=io.BytesIO(b'{}')), \
                 patch('strata_launcher.download_source', side_effect=archive_download), \
                 patch.object(launcher.StrataEnv, 'ensure_venv', return_value=sys.executable), \
                 patch('strata_launcher.subprocess.Popen', return_value=process):
                launcher.StrataLauncher._install_version_worker(app, '0.2.10')
            self.assertEqual(store.state()['current'], str(original.resolve()))
            self.assertFalse(app.runtime['updating'])
            self.assertTrue(any('[安装失败]' in line for line in logs))
            self.assertTrue((original / 'setup.py').is_file())


def test_0215_control_version_does_not_reject_signed_release_engine(tmp_path, monkeypatch):
    import strata_versions as versions
    (tmp_path / 'CMakeLists.txt').write_text('project(strata VERSION 0.1.61)')
    exe = tmp_path / 'strata-engine.exe'
    exe.touch()
    monkeypatch.setattr(versions, 'engine_targets', lambda root: [exe])
    monkeypatch.setattr(versions, 'component_metadata', lambda path: {'version': '0.1.60'})
    assert versions.check_exact_engines(tmp_path, '0.2.15') == ['strata-engine.exe: 0.1.60']


def test_prune_keeps_current_and_model_data(tmp_path):
    from strata_versions import VersionStore
    store = VersionStore(tmp_path)
    active, old, with_models = [store.root / name for name in
                                ('release-active', 'release-old', 'release-models')]
    for env in (active, old, with_models):
        env.mkdir()
        (env / 'setup.py').touch()
        (env / 'CMakeLists.txt').write_text('project(strata VERSION 0.1.60)')
        store.register(env, '0.2.14', verified=True)
    (with_models / 'models').mkdir()
    store.activate(old)
    store.activate(active)
    store.prune()
    assert active.exists() and with_models.exists() and not old.exists()
    assert not store.state()['previous']

def test_launch_rehomes_deleted_release_log(tmp_path):
    from strata_launch_support import resolve_launch_config
    previous = tmp_path / '.strata-versions' / 'release-0.2.11-old'
    current = tmp_path / '.strata-versions' / 'release-0.2.15-new'
    current.mkdir(parents=True)
    config = dict(exe='component:strata-engine', args=[], cwd=str(current),
                  log=str(previous / 'strata-model.log'))
    resolved = resolve_launch_config(config, current)
    assert resolved['log'] == str(current / 'strata-model.log')
    with open(resolved['log'], 'a', encoding='utf-8') as stream:
        stream.write('vision ready')
    assert config['log'] == str(previous / 'strata-model.log')

def test_deleted_release_profile_is_removed_and_bundled_resources_rebased(tmp_path):
    from strata_launch_support import resolve_launch_config
    root = tmp_path / '.strata-versions' / 'release-new'
    root.mkdir(parents=True)
    (root / 'data').mkdir()
    (root / 'data' / 'tokenizer.json').write_text('{}')
    old = tmp_path / '.strata-versions' / 'release-old'
    config = dict(exe='strata.exe', cwd=str(root), args=[
        '--expert-profile', str(old / 'data' / 'expert-profile.bin'),
        '--tokenizer', str(old / 'data' / 'tokenizer.json')])
    result = resolve_launch_config(config, root)
    assert '--expert-profile' not in result['args']
    assert result['args'] == ['--tokenizer', str(root / 'data' / 'tokenizer.json')]
