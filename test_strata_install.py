import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from strata_install import config_snapshot, finish_resources, prepare_setup_args, take_drafting


class AuxiliaryInstallTests(unittest.TestCase):
    def test_main_install_button_uses_selected_community_source(self):
        from strata_launcher import StrataLauncher
        calls = []
        screen = types.SimpleNamespace(
            community_link=types.SimpleNamespace(get=lambda: 'https://huggingface.co/example/model'),
            import_local_model=lambda **kwargs: calls.append(kwargs))
        StrataLauncher.run_install(screen)
        self.assertEqual(calls, [{'community': True}])

    def test_manual_projection_overrides_official_resolution_by_content(self):
        from strata_install import configure_auxiliaries, take_auxiliaries

        with tempfile.TemporaryDirectory() as folder:
            projection = Path(folder) / "renamed.gguf"
            projection.write_bytes(b"GGUF" + bytes(60))
            args, selected = take_auxiliaries(
                ["--family", "qwen", "--vision", "gpu", "--launcher-mmproj", str(projection)])
            self.assertNotIn("--launcher-mmproj", args)
            mod = types.SimpleNamespace(resolve_reader=lambda *_: ("fetch", None),
                                       _hash_file=lambda *_: "official",
                                       accepted_alt_reader=lambda *_, **kw: None)
            configure_auxiliaries(mod, args, selected)
            self.assertEqual(mod.resolve_reader("pinned", None, None, [], None, 64, "official"),
                             ("renamed", projection.resolve()))
            with self.assertRaises(ValueError):
                mod.resolve_reader("pinned", None, None, [], None, 64, "different")

    def test_empty_selection_preserves_official_vision_and_native_draft(self):
        from strata_install import auxiliary_roles, configure_auxiliaries

        mod = types.SimpleNamespace(resolve_reader=lambda *_: ("fetch", None))
        original = mod.resolve_reader
        configure_auxiliaries(mod, ["--family", "qwen", "--vision", "gpu"], {})
        self.assertIs(mod.resolve_reader, original)
        self.assertEqual(auxiliary_roles("inkling-2-mini", "no"), ("dspark",))
        self.assertEqual(auxiliary_roles("hy4-flash", "gpu"), ("vision", "engram"))

    def test_community_manifest_pins_every_shard_and_rejects_missing_shard(self):
        from io import BytesIO
        from strata_install import community_manifest

        repo = "https://huggingface.co/community/model"
        prefix = "IQ3_S/flash-abliterated-IQ3_S"
        names = [f"{prefix}-{i:05d}-of-00002.gguf" for i in (1, 2)]
        digest = "b" * 64
        data = {"sha": "a" * 40, "siblings": [
            {"rfilename": name, "lfs": {"size": 100, "sha256": digest}} for name in names]}
        with patch("urllib.request.urlopen", side_effect=lambda *_a, **_k: BytesIO(json.dumps(data).encode())):
            rows = community_manifest(f"{repo}/blob/main/{names[1]}")
            self.assertEqual([row["name"] for row in rows], [Path(name).name for name in names])
            self.assertTrue(all("/resolve/" + "a" * 40 + "/" in row["url"] for row in rows))
            data["siblings"].pop()
            with self.assertRaises(ValueError):
                community_manifest(repo)


class InstallResourceTests(unittest.TestCase):
    def test_wrapper_option_removed_and_validated(self):
        self.assertEqual(take_drafting(['--yes', '--launcher-drafting', 'mtp']), (['--yes'], 'mtp'))
        with self.assertRaises(ValueError):
            take_drafting(['--launcher-drafting', 'both'])

    def test_old_release_refuses_explicit_unsupported_dflash(self):
        with tempfile.TemporaryDirectory() as d:
            setup = Path(d) / 'setup.py'
            setup.write_text('# old setup', encoding='utf-8')
            mod = types.SimpleNamespace(__file__=str(setup))
            with self.assertRaisesRegex(ValueError, 'DFlash2'):
                prepare_setup_args(mod, [], 'dflash2')
            setup.write_text('# --drafter --mtp', encoding='utf-8')
            self.assertEqual(prepare_setup_args(mod, ['--drafter', 'mtp'], 'dflash2'), ['--drafter', 'dflash2'])
            self.assertEqual(prepare_setup_args(mod, [], 'off'), ['--mtp', 'off'])

    def test_only_current_install_configs_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = root / 'strata-old.json'
            old.write_text(json.dumps({'exe': 'engine', 'args': [], 'drafting': 'auto'}))
            original = old.read_bytes()
            before = config_snapshot(root)
            new = root / 'strata-new.json'
            new.write_text(json.dumps({'exe': 'engine', 'args': []}))
            finish_resources(root, 'off', before)
            self.assertEqual(old.read_bytes(), original)
            self.assertEqual(json.loads(new.read_text())['drafting'], 'off')
            self.assertFalse(json.loads(new.read_text())['mtp'])

    def test_absent_setup_result_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                finish_resources(d, 'mtp', {})


class AllDraftersTests(unittest.TestCase):
    def test_flag_removed_without_changing_runtime_mode(self):
        from strata_install import take_all_drafters
        self.assertEqual(take_all_drafters(['--yes', '--launcher-all-drafters']), (['--yes'], True))
        self.assertEqual(take_all_drafters(['--yes']), (['--yes'], False))

    def test_all_downloads_dflash_but_keeps_mtp_runtime_choice(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            before=config_snapshot(root)
            path=root/'strata-new.json'
            path.write_text(json.dumps({'exe':'engine','args':['--pack',str(root/'packs'/'qwen')], 'model_name':'qwen3.8-flash-next'}))
            ensure=Mock()
            drafting=types.ModuleType('strata_control.drafting')
            drafting.data_dir_of=lambda args:root
            drafting.supported_model=lambda *args:True
            drafting.needs_pack=lambda **kwargs:True
            drafting.pack_ready=lambda data:False
            fetch=types.ModuleType('strata_control.drafting_fetch');fetch.ensure=ensure
            control=types.ModuleType('strata_control');control.drafting=drafting;control.drafting_fetch=fetch
            command=types.ModuleType('strata_cli.drafting_cmd');command._inputs=lambda args:{'mode':args.mode}
            cli=types.ModuleType('strata_cli');cli.drafting_cmd=command
            with patch.dict('sys.modules',{'strata_control':control,'strata_control.drafting':drafting,
                 'strata_control.drafting_fetch':fetch,'strata_cli':cli,'strata_cli.drafting_cmd':command}):
                finish_resources(root,'mtp',before,True)
            ensure.assert_called_once()
            self.assertEqual(json.loads(path.read_text())['drafting'],'mtp')
            self.assertTrue(json.loads(path.read_text())['launcher_all_drafters'])
