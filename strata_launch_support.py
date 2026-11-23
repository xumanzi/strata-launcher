"""Shared launcher checks, independent of the context service and GUI."""
import csv
import ctypes
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PureWindowsPath



def cuda_headroom():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=3,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) == 1:
            free, total = map(int, rows[0])
            return free, total
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def check_launch_headroom(config, log):
    """Apply the llama.cpp fit floor and preflight CUDA when VRAM is visible."""
    args = config.get("args", [])
    external = config.get("engine") == "llama.cpp"

    def value(names, default):
        for i in range(len(args) - 1, -1, -1):
            key, sep, inline = args[i].partition("=")
            if key in names:
                return inline if sep else args[i + 1]
        return default

    def set_flag(names, setting):
        for i in range(len(args) - 1, -1, -1):
            if args[i] in names:
                del args[i:i + 2]
            elif args[i].split("=", 1)[0] in names:
                del args[i]
        args.extend((names[0], setting))

    target = ("--fit-target", "-fitt") if external else ("--vram-reserve-mib", "--reserve-mib")
    reserve = [max(500, int(n)) for n in str(value(target, "500")).split(",")]
    if external:
        set_flag(("--fit", "-fit"), "on")
        # DSpark fitting can omit the draft context. Budget its weights and
        # a work-buffer margin separately from the user's free-VRAM floor.
        draft = value(("--model-draft", "-md"), None)
        spec = value(("--spec-type",), "")
        extra = 0
        if draft and "draft-dspark" in spec.split(",") and Path(draft).is_file():
            mib = 1024**2
            extra = (Path(draft).stat().st_size + mib - 1) // mib + 256
        set_flag(target, ",".join(str(n + extra) for n in reserve))
        context = int(value(("--ctx-size", "-c"), 0))
        if context > 0:
            set_flag(("--fit-ctx", "-fitc"), str(context))
    if not external and config.get("backend") not in ("cuda", None):
        return
    if external and value(("--n-gpu-layers", "-ngl", "--gpu-layers"), None) == "0":
        return
    memory = cuda_headroom()
    if memory is None:
        log("[memory] CUDA 空闲显存不可读或有多张显卡，采用引擎的预留设置。")
        return None
    free, total = memory
    if free <= reserve[0]:
        raise ValueError(f"当前 CUDA 空闲显存 {free} MiB，无法预留 {reserve[0]} MiB 后加载模型。")
    log(f"[memory] CUDA 空闲显存 {free} MiB，预留目标 {reserve[0]} MiB。")
    if (external and config.get("launcher_managed_dspark") and extra and context >= 262144
            and 15000 <= total <= 16384 and value(("--n-gpu-layers", "-ngl", "--gpu-layers"), None) is None):
        set_flag(("--n-gpu-layers", "-ngl", "--gpu-layers"), "12")
        if value(("--n-gpu-layers-draft", "-ngld", "--gpu-layers-draft", "--spec-draft-ngl"), None) is None:
            set_flag(("--n-gpu-layers-draft",), "0")
        if not any(flag in args for flag in ("--no-kv-offload", "-nkvo")):
            args.append("--no-kv-offload")
        log("[memory] 16 GiB 显卡运行 262K DSpark：部分模型层、KV 和草稿权重使用系统内存。")
    return (500 if config.get('launcher_speed_priority') else reserve[0]) if (external or config.get('launcher_speed_priority')) else None


def normalize_llamacpp_config(config):
    """Drop flags written by old launcher versions' Strata context editor."""
    if config.get('engine') != 'llama.cpp':
        return config
    result = dict(config)
    args = list(config.get('args', []))
    obsolete = ['--max-context', '--kv', '--kv-resident', '--rope-scaling']
    if result.pop('launcher_context_auto_rope', None):
        obsolete.append('--rope-scale')
    legacy_context = None
    for flag in obsolete:
        while flag in args:
            index = args.index(flag)
            if flag == '--max-context' and index + 1 < len(args):
                legacy_context = args[index + 1]
            del args[index:index + 2]
    if '--ctx-size' not in args and legacy_context and str(legacy_context).isdigit():
        args.extend(('--ctx-size', str(legacy_context)))
    result['args'] = args
    return result


def local_context_choices(config):
    config = normalize_llamacpp_config(config)
    local = config.get('local_import')
    if not local:
        return None
    args = config.get('args', [])
    def value(flag):
        return args[args.index(flag) + 1] if flag in args else None
    context_flag = '--ctx-size' if config.get('engine') == 'llama.cpp' else '--max-context'
    return dict(family=local.get('family'), model=local.get('model'),
                context=int(value(context_flag)) if value(context_flag) else None,
                kv=value('--kv') or value('--cache-type-k'), local_import=True)


def save_local_context(path, context):
    """Update imported model context without regenerating its model configuration."""
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError('上下文必须是正整数')
    path = Path(path)
    original = path.read_bytes()
    config = normalize_llamacpp_config(json.loads(original.decode('utf-8-sig')))
    if not config.get('local_import'):
        raise ValueError('该配置不是本地导入模型')
    args = list(config.get('args', []))
    def value(flag):
        return args[args.index(flag) + 1] if flag in args else None
    def remove(flag):
        while flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    def set_value(flag, val):
        if flag in args:
            args[args.index(flag) + 1] = str(val)
        else:
            args.extend([flag, str(val)])
    if config.get('engine') == 'llama.cpp':
        set_value('--ctx-size', context)
    elif config.get('arch') in ('inkling2_loop', 'hy4flash'):
        remove('--rope-scaling')
        remove('--rope-scale')
        remove('--kv-resident')
        config.pop('launcher_context_auto_rope', None)
        set_value('--max-context', context)
    else:
        old_context = int(value('--max-context') or 32768)
        scaling, scale = value('--rope-scaling'), value('--rope-scale')
        trained = 262144
        automatic = config.get('launcher_context_auto_rope', False) or (
            scaling == 'yarn' and old_context > trained and scale is not None
            and abs(float(scale) - old_context / trained) < 1e-6)
        if automatic:
            remove('--rope-scaling'); remove('--rope-scale')
            scaling = None
        if context > trained and scaling in (None, 'none'):
            set_value('--rope-scaling', 'yarn')
            set_value('--rope-scale', format(context / trained, 'g'))
            config['launcher_context_auto_rope'] = True
        else:
            config.pop('launcher_context_auto_rope', None)
        if context < 65536:
            remove('--kv-resident')
        if context > 8192 and '--kv' not in args:
            set_value('--kv', 'int8')
        set_value('--max-context', context)
    config['args'] = args
    Path(str(path) + '.context-size.bak').write_bytes(original)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if temp and temp.exists():
            temp.unlink()
    return config


def _short_windows_path(path):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_short = kernel.GetShortPathNameW
    get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    get_short.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_short(str(path), buffer, len(buffer))
    return buffer.value if 0 < length < len(buffer) else str(path)


def _ascii_directory_alias(directory):
    """An NTFS junction is an alternate path; no model files are copied or moved."""
    import _winapi
    target = Path(directory).resolve()
    base = Path(_short_windows_path(tempfile.gettempdir()))
    if not str(base).isascii():
        base = Path(os.environ.get('ProgramData', 'C:/ProgramData'))
    if not str(base).isascii():
        raise OSError('没有可用的英文路径别名目录')
    aliases = base / 'strata-path-aliases'
    aliases.mkdir(parents=True, exist_ok=True)
    alias = aliases / hashlib.sha256(str(target).casefold().encode('utf-8')).hexdigest()[:24]
    if not alias.exists():
        try:
            _winapi.CreateJunction(str(target), str(alias))
        except FileExistsError:
            pass
    if not os.path.samefile(alias, target):
        raise OSError('路径别名指向其他目录')
    return str(alias)


def windows_engine_path(value):
    """Keep narrow Windows argv compatible with the engine's UTF-8 file reader."""
    if os.name != 'nt' or not isinstance(value, str) or value.isascii():
        return value
    path = Path(value)
    # GGUF filenames encode the shard pattern; shortening them loses that information.
    if path.is_dir():
        short = _short_windows_path(path)
        if short.isascii() and os.path.samefile(short, path):
            return short
    parent = _short_windows_path(path.parent)
    candidate = str(Path(parent) / path.name)
    if candidate.isascii() and (not path.exists() or os.path.samefile(candidate, path)):
        return candidate
    try:
        candidate = str(Path(_ascii_directory_alias(path.parent)) / path.name)
        if candidate.isascii() and (not path.exists() or os.path.samefile(candidate, path)):
            return candidate
    except OSError:
        pass
    raise ValueError(f'引擎无法安全读取中文路径：{value}。无法建立英文路径别名，请将模型放入全英文路径后重新导入。原文件未移动。')


def prepare_engine_paths(config):
    adapted = dict(config)
    for key in ('exe', 'cwd', 'tokenizer', 'log'):
        if isinstance(adapted.get(key), str):
            adapted[key] = windows_engine_path(adapted[key])
    if isinstance(adapted.get('args'), list):
        converted = []
        original_args = adapted['args']
        for index, arg in enumerate(original_args):
            if index and original_args[index - 1] == '--control-vector-scaled' and isinstance(arg, str):
                path, separator, strength = arg.rpartition(':')
                try:
                    float(strength)
                    scaled = bool(separator and path)
                except ValueError:
                    scaled = False
                if scaled:
                    converted.append(windows_engine_path(path) + ':' + strength)
                    continue
            converted.append(windows_engine_path(arg) if isinstance(arg, str) and (
                Path(arg).is_absolute() or PureWindowsPath(arg).is_absolute()) else arg)
        adapted['args'] = converted
    if isinstance(adapted.get('lib_dirs'), list):
        adapted['lib_dirs'] = [windows_engine_path(path) for path in adapted['lib_dirs']]
    if isinstance(adapted.get('vision'), dict):
        adapted['vision'] = {key: windows_engine_path(value) if key in ('exe', 'cwd', 'model', 'mmproj')
                             else value for key, value in adapted['vision'].items()}
    for key in ('args', 'user_args'):
        if isinstance(adapted.get(key), list):
            values = list(adapted[key])
            while '--experimental-speed-projection' in values:
                index = values.index('--experimental-speed-projection')
                del values[index:index + 2]
            adapted[key] = values
    return adapted


def api_model_name(config):
    if config.get('local_import'):
        args = config.get('args') or []
        if '--native' in args and args.index('--native') + 1 < len(args):
            filename = PureWindowsPath(str(args[args.index('--native') + 1])).name
            return re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', filename, flags=re.I)
    return str(config.get('model_name') or '')


def select_lan_ipv4(adapters):
    candidates = []
    for adapter in adapters:
        if not adapter.get('Physical') or not adapter.get('Up'):
            continue
        ips = adapter.get('IPs') or []
        for value in ([ips] if isinstance(ips, str) else ips):
            try:
                address = ipaddress.ip_address(value)
            except ValueError:
                continue
            if address.version != 4 or address.is_loopback or address.is_link_local or address.is_unspecified:
                continue
            if not address.is_private:
                continue
            candidates.append((not adapter.get('Gateway', False), int(adapter.get('Metric') or 9999), str(address)))
    candidates.sort()
    return [candidates[0][2]] if candidates else []


def configured_lan_ipv4():
    if os.name != 'nt':
        return []
    script = '''$ErrorActionPreference='Stop'
$items=@(Get-NetAdapter -Physical | Where-Object Status -eq 'Up' | ForEach-Object {
 $idx=$_.ifIndex
 $ips=@(Get-NetIPAddress -InterfaceIndex $idx -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object AddressState -eq 'Preferred' | Select-Object -ExpandProperty IPAddress)
 $metric=(Get-NetIPInterface -InterfaceIndex $idx -AddressFamily IPv4 -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty InterfaceMetric)
 $gw=@(Get-NetRoute -InterfaceIndex $idx -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue)
 [pscustomobject]@{Physical=$true;Up=$true;IPs=$ips;Metric=$metric;Gateway=($gw.Count -gt 0)}
})
ConvertTo-Json -InputObject $items -Compress -Depth 4'''
    powershell = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    try:
        result = _capture([str(powershell), '-NoProfile', '-NonInteractive', '-Command', script], timeout=10)
        if result.returncode == 0:
            adapters = json.loads(result.stdout or '[]')
            return select_lan_ipv4([adapters] if isinstance(adapters, dict) else adapters)
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        pass
    return []


def connection_fields(config, actual_port=None, ips=(), api_key=None):
    port = int(actual_port or config.get('port') or 8080)
    local = f'http://127.0.0.1:{port}'
    host = config.get('host') or '127.0.0.1'
    lan = '未开启'
    if host in ('0.0.0.0', '::'):
        ip = next(iter(ips), None)
        lan = (f'http://[{ip}]:{port}' if ':' in ip else f'http://{ip}:{port}') if ip else '未检测到局域网地址'
    if host not in ('0.0.0.0', '::', '127.0.0.1', 'localhost'):
        lan = f'http://[{host}]:{port}' if ':' in host else f'http://{host}:{port}'
    return [('模型名称', api_model_name(config)), ('本机访问地址', local), ('局域网访问地址', lan),
            ('APIkey', str(config.get('api_key') or '') if api_key is None else api_key)]


def http_probe(url, timeout=2.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.status, response.read(1024 * 1024).decode('utf-8', 'replace')
    except urllib.error.HTTPError as error:
        return error.code, error.read(8192).decode('utf-8', 'replace')
    except (OSError, urllib.error.URLError) as error:
        return 0, str(error)


def strata_health_ready(status, body, *, llama_backend=False):
    if status != 200:
        return False
    try:
        health = json.loads(body)
    except (TypeError, ValueError):
        return False
    return (isinstance(health, dict) and health.get('status') == 'ok'
            and (llama_backend or health.get('service') == 'strata'))


def port_in_use(port):
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError('端口必须在 1 至 65535 之间')
    for family, address in ((socket.AF_INET, '0.0.0.0'), (socket.AF_INET6, '::')):
        try:
            with socket.socket(family) as sock:
                if os.name == 'nt':
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind((address, port))
        except OSError as error:
            if family == socket.AF_INET6 and error.errno in (97, 10047, 10049):
                continue
            return True
    return False


def _capture(command, timeout=4):
    return subprocess.run(command, capture_output=True, text=True, errors='replace',
        timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def port_owner(port):
    if os.name != 'nt':
        return 0, ''
    try:
        result = _capture(['netstat', '-ano', '-p', 'tcp'])
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0].upper() != 'TCP':
                continue
            if fields[1].rsplit(':', 1)[-1] != str(int(port)):
                continue
            if fields[-2].upper() != 'LISTENING' and fields[-2] != '侦听':
                continue
            pid = int(fields[-1])
            names = _capture(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'])
            for row in csv.reader(io.StringIO(names.stdout)):
                if len(row) >= 2 and row[1] == str(pid):
                    return pid, row[0]
            return pid, ''
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return 0, ''


def is_protected_process(name):
    return str(name or '').lower() in {'system', 'system idle process', 'registry',
        'smss.exe', 'csrss.exe', 'wininit.exe', 'winlogon.exe', 'services.exe',
        'lsass.exe', 'svchost.exe', 'dwm.exe', 'fontdrvhost.exe', 'secure system'}


def kill_pid_force(pid):
    try:
        pid = int(pid)
        if pid <= 4 or pid == os.getpid():
            return False, '不能结束系统进程或启动器自身'
        if os.name != 'nt':
            return False, '该端口修复功能仅支持 Windows'
        result = _capture(['taskkill', '/F', '/T', '/PID', str(pid)])
        return result.returncode == 0, (result.stdout or result.stderr).strip()
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def resolve_launch_config(config, root):
    """Adapt existing local-drive paths to their verified shared-directory equivalents."""
    root = Path(root).resolve()
    old_cwd = PureWindowsPath(str(config.get('cwd') or ''))
    def resolve(value):
        if not isinstance(value, str) or not value or Path(value).exists():
            return value
        old = PureWindowsPath(value)
        if not old.is_absolute():
            return value
        candidates = []
        parts = old.parts
        if '.strata-versions' in parts:
            index = parts.index('.strata-versions') + 1
            if index < len(parts) and parts[index].startswith('release-'):
                candidates.append(root.joinpath(*parts[index + 1:]))
        # Release-owned resources follow the active release even after cwd was migrated.
        if '.strata-versions' in old.parts:
            index = old.parts.index('.strata-versions')
            if len(old.parts) > index + 2 and old.parts[index + 1].startswith('release-'):
                candidates.append(root.joinpath(*old.parts[index + 2:]))
        if old_cwd.is_absolute():
            for previous, current in ((old_cwd, root), (old_cwd.parent, root.parent)):
                try:
                    candidates.append(current / old.relative_to(previous))
                except ValueError:
                    pass
        if root.drive.startswith('\\\\') and root.drive.rsplit('\\', 1)[-1].lower() == old.drive.rstrip(':').lower():
            candidates.append(Path(root.anchor) / old.relative_to(old.anchor))
        return str(next((candidate for candidate in candidates if candidate.exists()), value))
    adapted = dict(config)
    for key in ('exe', 'cwd', 'tokenizer', 'log'):
        if key in adapted:
            adapted[key] = resolve(adapted[key])
    if isinstance(adapted.get('args'), list):
        adapted['args'] = [resolve(arg) for arg in adapted['args']]
        args = adapted['args']
        # Old local imports can point to a profile from a removed release.
        for index in range(len(args) - 2, -1, -1):
            if args[index] == '--expert-profile' and not Path(args[index + 1]).is_file():
                del args[index:index + 2]
        local = adapted.get('local_import') or {}
        family = local.get('family')
        if (adapted.get('backend') == 'cuda' and family in ('qwen', 'coder')
                and 'flash-next' in str(adapted.get('model_name', '')).lower()
                and '--native' in args and '--expert-profile' not in args
                and '--expert-profile' not in (adapted.get('user_args') or [])):
            # Native IQ packs require speculative verification. Use the same
            # shipped profile as Strata setup to seed the GPU expert cache.
            default_name = 'expert-profile-coder.bin' if family == 'coder' else 'expert-profile.bin'
            default_profile = root / 'data' / default_name
            if default_profile.is_file():
                args.extend(('--expert-profile', str(default_profile)))
                if '--expert-cache' not in args:
                    args.extend(('--expert-cache', 'auto'))
        profile = adapted.get('expert_profile')
        if profile:
            profile = resolve(profile)
            if Path(profile).is_file():
                adapted['expert_profile'] = profile
            else:
                adapted.pop('expert_profile', None)
    if isinstance(adapted.get('lib_dirs'), list):
        adapted['lib_dirs'] = [resolve(path) for path in adapted['lib_dirs']]
    if isinstance(adapted.get('vision'), dict):
        adapted['vision'] = {k: resolve(v) if k in ('exe', 'cwd', 'model', 'mmproj') else v
                             for k, v in adapted['vision'].items()}
    log = adapted.get('log')
    if isinstance(log, str) and log:
        log_path = Path(log)
        # Managed release directories are disposable. Logs belong to the active release.
        windows_log = PureWindowsPath(log)
        stale_release = any(part.startswith('release-') for part in windows_log.parts) and '.strata-versions' in windows_log.parts
        if stale_release or not log_path.is_absolute():
            log_path = root / windows_log.name
        elif not log_path.parent.is_dir():
            log_path = root / windows_log.name
        adapted['log'] = str(log_path)
    exe = adapted.get('exe')
    if isinstance(exe, str) and not PureWindowsPath(exe).is_absolute():
        cwd = Path(adapted.get('cwd') or root)
        if not cwd.is_absolute():
            cwd = root / cwd
        candidate = cwd / exe
        if candidate.is_file():
            adapted['exe'] = str(candidate.resolve())
    return adapted
