"""Release catalogue, isolated environment registry and compatibility checks."""
import json
import re
import urllib.request
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from functools import wraps
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from strata_update import atomic_json, model_configs, component_metadata, engine_targets, launcher_update_lock
from strata_launch_support import resolve_launch_config


def source_engine_version(root):
    text = (Path(root) / 'CMakeLists.txt').read_text(encoding='utf-8-sig')
    match = re.search(r'project\(\s*strata\s+VERSION\s+([\d.]+)', text, re.I)
    if not match:
        raise RuntimeError('目标源码缺少引擎版本。')
    return match.group(1)


def pinned_feed(version):
    return f'https://github.com/Niko1221/Strata/releases/download/v{normalize_release(version)}/strata-feed.json'


def check_exact_engines(root):
    expected = source_engine_version(root)
    results = []
    for exe in engine_targets(root):
        metadata = component_metadata(exe.parent)
        if metadata.get('version') != expected:
            raise RuntimeError(f'所选发布需要引擎 {expected}，实际为 {metadata.get("version")}；未切换。')
        results.append(f'{exe.name}: {expected}')
    return results


def validate_config_flags(root, configs=None):
    """Check Qwen's saved flags against the selected release's own CLI source."""
    parser = Path(root) / 'src/program/generate.cpp'
    if not parser.is_file():
        return
    supported = set(re.findall(r'"(--[\w-]+)"', parser.read_text(encoding='utf-8-sig')))
    if not supported:
        raise RuntimeError('无法识别此版本的参数接口，需更新启动器适配规则。')
    for path, config in (configs if configs is not None else model_configs(root)).items():
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
    if not (source / 'setup.py').is_file():
        raise RuntimeError('源码压缩包缺少 setup.py。')
    source_engine_version(source)
    return source


def copy_configs(source, destination):
    """Resolve relative data paths against their original working directory."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    for path, config in model_configs(source).items():
        config = resolve_launch_config(config, source)
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
        config['cwd'] = str(destination)
        atomic_json(destination / path.name, config)


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
        if str(path) not in state['environments'] or path.parent != self.root or not path.name.startswith('release-'):
            raise RuntimeError('仅允许清理版本管理器创建的环境。')
        # Model data stays outside managed environments. Refuse an unexpected local data directory.
        if any((path / name).exists() for name in ('models', 'packs', 'mtp', 'Strata-data')):
            raise RuntimeError('此环境含模型数据，请先移出模型后再清理。')
        if next(path.rglob('*.gguf'), None) is not None:
            raise RuntimeError('此环境包含 GGUF 权重，未执行清理。')
        shutil.rmtree(path)
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
