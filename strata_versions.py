"""Release catalogue, isolated environment registry and compatibility checks."""
import json
import os
import tempfile
import re
import urllib.request
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from functools import wraps
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from strata_update import atomic_json, model_configs, component_metadata, engine_targets, launcher_update_lock, installed_release, is_shape_config, is_external_config
from strata_launch_support import normalize_llamacpp_config, resolve_launch_config


def source_engine_version(root):
    root = Path(root)
    candidates = (root / 'CMakeLists.txt', root / 'engine/CMakeLists.txt',
                  root.parent / 'engine/CMakeLists.txt')
    cmake = next((path for path in candidates if path.is_file()), None)
    if cmake is None:
        raise RuntimeError('目标源码缺少引擎 CMakeLists.txt。')
    text = cmake.read_text(encoding='utf-8-sig')
    match = re.search(r'project\(\s*strata\s+VERSION\s+([\d.]+)', text, re.I)
    if not match:
        raise RuntimeError('目标源码缺少引擎版本。')
    return match.group(1)


def pinned_feed(version):
    return f'https://github.com/Niko1221/Strata/releases/download/v{normalize_release(version)}/strata-feed.json'


def check_exact_engines(root, release=None):
    results = []
    for exe in engine_targets(root):
        metadata = component_metadata(exe.parent)
        if not exe.is_file():
            raise RuntimeError(f'引擎入口不存在：{exe}')
        results.append(f'{exe.name}: {metadata.get("version")}')
    return results


def validate_config_flags(root, configs=None):
    """Check Qwen's saved flags against the selected release's own CLI source."""
    root = Path(root)
    configs = configs if configs is not None else model_configs(root)
    for path, config in configs.items():
        if is_shape_config(config):
            inkling = config.get('arch') == 'inkling2_loop' or config.get('exe') == 'component:strata-inkling'
            kind = 'inkling_kind.py' if inkling else 'hy4_kind.py'
            if not (root / 'serve' / kind).is_file():
                raise RuntimeError(f'{Path(path).name} 需要支持该模型的 Strata 版本。')
    parser = root / 'src/program/generate.cpp'
    if not parser.is_file():
        parser = root.parent / 'engine/src/program/generate.cpp'
    if not parser.is_file():
        return
    supported = set(re.findall(r'"(--[\w-]+)"', parser.read_text(encoding='utf-8-sig')))
    if not supported:
        raise RuntimeError('无法识别此版本的参数接口，需更新启动器适配规则。')
    for path, config in configs.items():
        if is_shape_config(config) or is_external_config(config):
            continue
        if config.get('arch') in ('deepseek41', 'qwen35moe'):
            continue
        unknown = [arg.split('=', 1)[0] for arg in config.get('args', [])
                   if isinstance(arg, str) and arg.startswith('--') and arg.split('=', 1)[0] not in supported]
        if unknown:
            raise RuntimeError(f'{Path(path).name} 含此版本不支持的参数：{", ".join(unknown)}；请调整配置或选择新版。')


def extract_source(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        for member in members:
            parts = PurePosixPath(member.filename.replace('\\', '/')).parts
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination) or ':' in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('源码压缩包含有不安全路径。')
        package.extractall(destination)
    children = list(destination.iterdir())
    source = children[0] if len(children) == 1 and children[0].is_dir() else destination
    if (source / 'terminal/setup.py').is_file():
        runtime = source / 'terminal'
    else:
        runtime = source
    if not (runtime / 'setup.py').is_file():
        raise RuntimeError('源码压缩包缺少 setup.py。')
    source_engine_version(runtime)
    return source


def copy_configs(source, destination):
    """Resolve relative data paths against their original working directory."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    for path, config in model_configs(source).items():
        config = normalize_llamacpp_config(resolve_launch_config(config, source))
        cwd = Path(config.get('cwd') or source)
        if not cwd.is_absolute():
            cwd = source / cwd
        def absolute(value):
            if isinstance(value, str) and value and not value.startswith('-'):
                candidate = cwd / value
                if candidate.exists():
                    return str(candidate.resolve())
            return value
        config['args'] = [absolute(value) for value in config['args']]
        for key in ('exe', 'tokenizer', 'log'):
            if key in config:
                config[key] = absolute(config[key])
        config['lib_dirs'] = [absolute(value) for value in config.get('lib_dirs', [])]
        if isinstance(config.get('vision'), dict):
            config['vision'] = {k: absolute(v) if k in ('exe', 'cwd', 'model', 'mmproj') else v
                                for k, v in config['vision'].items()}
        if config.get('log'):
            from pathlib import PureWindowsPath
            config['log'] = str(destination / PureWindowsPath(config['log']).name)
        config['cwd'] = str(destination)
        atomic_json(destination / path.name, config)


def sync_user_settings(source, destination):
    """Refresh saved settings without replacing the target's engine or prepared model paths."""
    settings = ('host', 'port', 'api_key', 'parallel', 'drafting', 'draft_vocab', 'mtp',
                'split_engine', 'user_args', 'env', 'model_name', 'launcher_context_auto_rope')
    flags = ('--max-context', '--kv', '--kv-resident', '--spec', '--temperature', '--top-p',
             '--top-k', '--min-p', '--repeat-penalty', '--rope-scaling', '--rope-scale')
    targets = model_configs(destination)
    for path, current in model_configs(source).items():
        target_path = Path(destination) / path.name
        if target_path not in targets:
            raise RuntimeError(f'目标环境缺少 {path.name}，请先安装或导入该模型。')
        current = normalize_llamacpp_config(current)
        merged = dict(normalize_llamacpp_config(targets[target_path]))
        for key in settings:
            if key in current:
                merged[key] = current[key]
            else:
                merged.pop(key, None)
        args = list(merged.get('args', []))
        old_args = current.get('args', [])
        if is_external_config(current) != is_external_config(merged):
            sync_flags = ()
        elif is_external_config(merged):
            sync_flags = ('--ctx-size', '--cache-type-k', '--cache-type-v')
        else:
            sync_flags = flags
        for flag in sync_flags:
            while flag in args:
                index = args.index(flag)
                del args[index:index + 2]
            if flag in old_args:
                index = old_args.index(flag)
                if index + 1 < len(old_args):
                    args.extend((flag, old_args[index + 1]))
        merged['args'] = args
        targets[target_path] = merged
    validate_config_flags(destination, targets)
    originals = {path: path.read_bytes() for path in targets}
    try:
        for path, config in targets.items():
            atomic_json(path, config)
    except BaseException:
        failures = []
        for path, original in originals.items():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(original)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            except OSError as error:
                failures.append(f'{path}: {error}')
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if failures:
            raise RuntimeError('配置同步失败，部分原始配置无法恢复：' + '; '.join(failures))
        raise


def locked_state(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with launcher_update_lock(self.root):
            return method(self, *args, **kwargs)
    return locked


class VersionStore:
    def __init__(self, app_dir):
        self.root = Path(app_dir).resolve() / '.strata-versions'
        self.root.mkdir(exist_ok=True)
        self.state_path = self.root / 'state.json'

    def state(self):
        if not self.state_path.exists():
            return {'environments': {}, 'current': '', 'previous': ''}
        return json.loads(self.state_path.read_text(encoding='utf-8'))

    @locked_state
    def register(self, path, release, verified=False, original=False):
        path = Path(path).resolve()
        state = self.state()
        state['environments'][str(path)] = {'release': release, 'engine': source_engine_version(path),
                                            'verified': verified, 'original': original, 'installed_at': time.time()}
        atomic_json(self.state_path, state)

    @locked_state
    def activate(self, path):
        path = str(Path(path).resolve())
        state = self.state()
        if path not in state['environments'] or not (Path(path) / 'setup.py').is_file():
            raise RuntimeError('所选环境未登记或已丢失。')
        entry = state['environments'][path]
        if not entry.get('verified') and not entry.get('original'):
            raise RuntimeError('环境安装尚未完成，不能切换。')
        if state.get('current') != path:
            state['previous'] = state.get('current', '')
            state['current'] = path
            atomic_json(self.state_path, state)
        return path

    @locked_state
    def remove(self, path):
        path = Path(path).resolve()
        state = self.state()
        if str(path) in (state.get('current'), state.get('previous')):
            raise RuntimeError('当前环境和回退环境不能清理。')
        self._remove_environment(path, state)

    @locked_state
    def prune(self):
        state = self.state()
        current = state.get('current')
        if not current or not (Path(current) / 'setup.py').is_file():
            return []
        state['previous'] = ''
        atomic_json(self.state_path, state)
        messages = []
        for path, entry in list(state['environments'].items()):
            if path == current or entry.get('original'):
                continue
            try:
                self._remove_environment(Path(path), state)
                messages.append(f'已清理旧版：{path}')
            except (OSError, RuntimeError) as exc:
                messages.append(f'旧版清理待完成：{path}：{exc}')
        return messages

    def _remove_environment(self, path, state):
        if path.resolve() != path or path.is_symlink() or path.is_junction():
            raise RuntimeError('版本目录包含重解析路径。')
        directory = path.parent if path.name == 'terminal' else path
        if str(path) not in state['environments'] or directory.parent != self.root or not directory.name.startswith('release-'):
            raise RuntimeError('仅允许清理版本管理器创建的环境。')
        # Model data stays outside managed environments. Refuse an unexpected local data directory.
        if any((path / name).exists() for name in ('models', 'packs', 'mtp', 'Strata-data')):
            raise RuntimeError('此环境含模型数据，请先移出模型后再清理。')
        if next(directory.rglob('*.gguf'), None) is not None:
            raise RuntimeError('此环境包含 GGUF 权重，未执行清理。')
        current_configs = model_configs(Path(state['current'])) if state.get('current') else {}
        serialized = json.dumps(list(current_configs.values()), ensure_ascii=False)
        config_text = serialized.replace(chr(92) * 2, '/').casefold()
        if directory.as_posix().casefold() in config_text:
            raise RuntimeError('当前配置仍引用此目录。')
        if path.exists():
            shutil.rmtree(directory)
        del state['environments'][str(path)]
        atomic_json(self.state_path, state)

    def catalog(self, refresh=False):
        cache = self.root / 'catalog.json'
        if not refresh:
            try:
                cached = json.loads(cache.read_text(encoding='utf-8'))
                return cached if isinstance(cached, list) else []
            except (OSError, ValueError):
                return []
        result = release_catalog()
        atomic_json(cache, result)
        return result


def normalize_release(value):
    value = str(value).strip().removeprefix('v')
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:\.\d+)?', value):
        raise ValueError('请输入发布版本，例如 0.2.10；不能填写引擎内部版本。')
    return value


def release_catalog(timeout=15):
    releases = []
    try:
        for page in range(1, 101):
            request = urllib.request.Request(
                f'https://api.github.com/repos/Niko1221/Strata/releases?per_page=100&page={page}',
                headers={'User-Agent': 'StrataLauncher', 'Accept': 'application/vnd.github+json'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                batch = json.load(response)
            if not isinstance(batch, list):
                raise ValueError('官方版本接口未返回列表')
            releases.extend(batch)
            if len(batch) < 100:
                break
        if not releases:
            raise ValueError('官方版本接口返回空列表')
    except (OSError, ValueError) as api_error:
        try:
            return webpage_catalog(timeout)
        except (OSError, ValueError) as page_error:
            raise RuntimeError(f'官方接口与发布页均读取失败：{api_error}；{page_error}') from page_error
    result = []
    for item in releases:
        if item.get('draft') or item.get('prerelease'):
            continue
        try:
            version = normalize_release(item['tag_name'])
        except (ValueError, KeyError):
            continue
        result.append({'version': version, 'url': item['html_url'],
                       'published_at': item.get('published_at', ''),
                       'engine': release_engine(item.get('body', ''))})
    return result


class ReleasePage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.versions = []
        self.next_url = ''

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        attrs = dict(attrs)
        href = attrs.get('href', '')
        match = re.fullmatch(r'/Niko1221/Strata/releases/tag/(v?\d+(?:\.\d+){2,3})', href)
        if match:
            version = normalize_release(match.group(1))
            if version not in self.versions:
                self.versions.append(version)
        if attrs.get('rel') == 'next' or 'next_page' in attrs.get('class', '').split():
            candidate = urljoin('https://github.com', href)
            parsed = urlparse(candidate)
            if parsed.scheme == 'https' and parsed.netloc == 'github.com' and parsed.path == '/Niko1221/Strata/releases':
                self.next_url = candidate


def webpage_catalog(timeout=15):
    url = 'https://github.com/Niko1221/Strata/releases'
    visited, versions = set(), {}
    while url and url not in visited:
        if len(visited) >= 100:
            raise ValueError('发布页分页异常，请稍后重试')
        visited.add(url)
        request = urllib.request.Request(url, headers={'User-Agent': 'StrataLauncher'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html = response.read().decode('utf-8')
        parser = ReleasePage()
        parser.feed(html)
        if not parser.versions:
            raise ValueError('发布页没有可识别的版本，未覆盖已缓存列表')
        for version in parser.versions:
            versions.setdefault(version, {'version': version,
                'url': f'https://github.com/Niko1221/Strata/releases/tag/v{version}',
                'published_at': '', 'engine': '', 'source': '官方发布页'})
        url = parser.next_url
    return sorted(versions.values(), key=lambda item: tuple(map(int, item['version'].split('.'))), reverse=True)


def release_engine(body):
    match = re.search(r'(?:every engine package (?:are|is)|Control and (?:every|all) engine packages? (?:are|is))\s*[*`]*(\d+\.\d+\.\d+(?:\.\d+)?)', body, re.I)
    return match.group(1) if match else ''
