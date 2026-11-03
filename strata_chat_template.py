"""Optional runtime template overlay. Never change the installed tokenizer."""
import shutil
import sys
from pathlib import Path


def bundled_template():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'bundled_chat_template.jinja'


def validate_template(path):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    path = Path(path)
    if not path.is_file():
        raise ValueError('聊天模板文件不存在：' + str(path))
    source = path.read_text(encoding='utf-8-sig')
    if not source.strip():
        raise ValueError('聊天模板为空')
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
    try:
        environment.parse(source)
    except Exception as error:
        raise ValueError('聊天模板语法错误：' + str(error)) from error
    return source


def apply_chat_template(config, options, runtime_dir):
    if not options or not options.get('enabled'):
        return config
    selected = options.get('path') or bundled_template()
    source = validate_template(selected)
    tokenizer = Path(config.get('tokenizer') or '')
    if not config.get('tokenizer') or not tokenizer.is_dir():
        raise ValueError('无法加载聊天模板：tokenizer 目录不存在')
    overlay = Path(runtime_dir) / 'template-tokenizer'
    shutil.copytree(tokenizer, overlay)
    (overlay / 'chat_template.jinja').write_text(source, encoding='utf-8')
    return dict(config, tokenizer=str(overlay))
