import unittest
from types import SimpleNamespace
from unittest.mock import patch

from strata_launcher import StrataLauncher, QUARK_MODEL_URL, QUARK_MODEL_CODE


class DownloadLinkTests(unittest.TestCase):
    def test_open_download_uses_the_given_share_link(self):
        notices = []
        app = SimpleNamespace(toast=lambda *args: notices.append(args))
        with patch('webbrowser.open', return_value=True) as browser:
            StrataLauncher.open_quark_model_download(app)
        browser.assert_called_once_with(QUARK_MODEL_URL)
        self.assertEqual(QUARK_MODEL_URL, 'https://pan.quark.cn/s/51ce260152fa')
        self.assertEqual(notices, [])

    def test_copy_extraction_code_puts_exact_code_on_clipboard(self):
        events = []
        app = SimpleNamespace(clipboard_clear=lambda: events.append('clear'),
            clipboard_append=lambda value: events.append(value),
            toast=lambda *args: events.append('toast'))
        StrataLauncher.copy_download_info(app, QUARK_MODEL_CODE, '提取码')
        self.assertEqual(QUARK_MODEL_CODE, 'aB6u')
        self.assertEqual(events, ['clear', 'aB6u', 'toast'])


if __name__ == '__main__':
    unittest.main()
