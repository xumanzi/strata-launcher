import tempfile
import unittest
from pathlib import Path
from strata_chat_template import apply_chat_template, validate_template, bundled_template


class ChatTemplateTests(unittest.TestCase):
    def test_disabled_does_not_touch_missing_paths(self):
        cfg = {'tokenizer': 'missing'}
        self.assertIs(apply_chat_template(cfg, {'enabled': False, 'path': 'missing'}, 'missing'), cfg)

    def test_overlay_preserves_original_and_other_tokenizer_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = root / 'tokenizer'
            original.mkdir()
            (original / 'chat_template.jinja').write_text('original')
            (original / 'tokenizer.json').write_text('{}')
            cfg = {'tokenizer': str(original)}
            result = apply_chat_template(cfg, {'enabled': True}, root)
            self.assertEqual((original / 'chat_template.jinja').read_text(), 'original')
            self.assertEqual(cfg['tokenizer'], str(original))
            overlay = Path(result['tokenizer'])
            self.assertEqual((overlay / 'tokenizer.json').read_text(), '{}')
            self.assertEqual((overlay / 'chat_template.jinja').read_text(encoding='utf-8'), validate_template(bundled_template()))

    def test_invalid_custom_template_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.jinja'
            path.write_text('{% if %}')
            with self.assertRaisesRegex(ValueError, '语法'):
                validate_template(path)

    def test_bundled_template_thinking_switch(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=['jinja2.ext.loopcontrols']).from_string(validate_template(bundled_template()))
        prompt = dict(messages=[{'role': 'user', 'content': '你好'}], add_generation_prompt=True)
        self.assertTrue(template.render(**prompt, enable_thinking=True).endswith('<think>\n'))
        self.assertIn('</think>', template.render(**prompt, enable_thinking=False))

    def test_selection_is_saved_per_model(self):
        from types import SimpleNamespace
        from strata_launcher import StrataLauncher
        class Variable:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value
        settings = {}
        cfg = SimpleNamespace(get=lambda key: settings.get(key),
                              set=lambda key, value: settings.__setitem__(key, value))
        with tempfile.TemporaryDirectory() as d:
            current = [Path(d) / 'a.json']
            app = SimpleNamespace(cfg=cfg, _selected_cfg_path=lambda: current[0],
                chat_template_var=Variable(True), chat_template_path_var=Variable('custom.jinja'),
                _refresh_chat_template=lambda: None)
            StrataLauncher._save_chat_template(app)
            self.assertEqual(StrataLauncher._model_chat_template(app)['path'], 'custom.jinja')
            current[0] = Path(d) / 'b.json'
            self.assertEqual(StrataLauncher._model_chat_template(app), {})
            app.chat_template_var.set(False)
            StrataLauncher._save_chat_template(app)
            current[0] = Path(d) / 'a.json'
            self.assertTrue(StrataLauncher._model_chat_template(app)['enabled'])
