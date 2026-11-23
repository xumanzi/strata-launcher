import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strata_launcher_release import parse_release, stage_launcher, check_launcher_release


class LauncherUpdateTests(unittest.TestCase):
    def setUp(self):
        self.data = b"MZ" * 1024
        self.asset = {
            "name": "AI-Manzi-Strata-Launcher-v3.7.8-windows-x64.exe",
            "size": len(self.data), "state": "uploaded",
            "digest": "sha256:" + hashlib.sha256(self.data).hexdigest(),
        }
        self.release = {"tag_name": "v3.7.8", "assets": [self.asset], "body": "Release notes"}

    def test_stable_version_uses_the_exact_asset_and_rejects_missing_digest(self):
        offer = parse_release(self.release, "3.7.7")
        self.assertEqual(offer.name, self.asset["name"])
        self.assertIsNone(parse_release(self.release, "3.7.8"))
        self.assertIsNone(parse_release(dict(self.release, prerelease=True), "3.7.7"))
        del self.asset["digest"]
        with self.assertRaises(ValueError):
            parse_release(self.release, "3.7.7")

    def test_update_is_staged_without_overwriting_previous_launcher(self):
        offer = parse_release(self.release, "3.7.7")
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "original.exe"
            old.write_bytes(b"old")
            with patch("urllib.request.urlopen", return_value=io.BytesIO(self.data)):
                new = stage_launcher(offer, directory)
            self.assertEqual(new.read_bytes(), self.data)
            self.assertEqual(old.read_bytes(), b"old")
            with patch("urllib.request.urlopen", side_effect=AssertionError("must reuse validated download")):
                self.assertEqual(stage_launcher(offer, directory), new)

    def test_hash_failure_does_not_leave_a_runnable_file(self):
        offer = parse_release(self.release, "3.7.7")
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"NO" * 1024)):
                with self.assertRaises(ValueError):
                    stage_launcher(offer, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
