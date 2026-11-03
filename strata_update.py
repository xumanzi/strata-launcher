"""Release identity and verified updates, also runnable inside Strata's own venv."""
import hashlib
import contextlib
import json
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path
from strata_launch_support import resolve_launch_config

MARKER = '.strata-launcher-release.json'
SOURCE_MARKER = '.strata-launcher-source.json'


def numbers(value):
    return tuple(int(x) for x in re.findall(r'\d+', str(value))[:4])


def is_deepseek_config(config):
    return config.get('arch') == 'deepseek41' or Path(str(config.get('exe', ''))).name.lower() in ('strata-dsv41', 'strata-dsv41.exe')


def fingerprint(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in ('setup.py', 'CMakeLists.txt')}


def record_release(root, version, marker=MARKER):
    path = Path(root) / marker
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'version': version, 'files': fingerprint(root)}), encoding='utf-8')
    tmp.replace(path)


def installed_release(root, marker=MARKER):
    try:
        data = json.loads((Path(root) / marker).read_text(encoding='utf-8'))
        return str(data['version']) if data['files'] == fingerprint(root) else ''
    except (OSError, ValueError, KeyError, TypeError):
        return ''


def model_configs(root):
    root = Path(root)
    configs = {}
    for path in root.glob('strata-*.json'):
        try:
            cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and cfg.get('exe') and isinstance(cfg.get('args'), list):
            configs[path] = cfg
    return configs


def engine_targets(root):
    root = Path(root)
    targets = {}
    for cfg in model_configs(root).values():
        if cfg['exe'] == 'component:strata-engine':
            raise RuntimeError('组件引用尚未解析，请先完成启动器的组件更新。')
        if cfg.get('exe'):
            cfg = resolve_launch_config(cfg, root)
            exe = Path(cfg['exe'])
            if not exe.is_absolute():
                exe = root / exe
            targets[exe.resolve()] = cfg.get('backend', '')
    for folder in (() if targets else ('engine', 'engine-cuda12')):
        exe = root / folder / 'strata.exe'
        if exe.exists():
            targets.setdefault(exe.resolve(), '')
    return targets


def minimum_engine(root):
    text = (Path(root) / 'setup.py').read_text(encoding='utf-8-sig')
    match = re.search(r'^MIN_ENGINE\s*=\s*\(([\d,\s]+)\)', text, re.M)
    if not match:
        raise RuntimeError('无法识别官方 setup.py 的 MIN_ENGINE，不能确认引擎更新结果。')
    return numbers(match.group(1))


def component_metadata(folder, full=True):
    folder = Path(folder).resolve()
    data = json.loads((folder / 'engine-component.json').read_text(encoding='utf-8-sig'))
    files = data.get('files')
    if not isinstance(files, list) or not files:
        raise RuntimeError(f'组件文件清单为空：{folder}')
    entry_found = False
    for item in files:
        path = (folder / item['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise RuntimeError(f'组件文件无效：{path}')
        if path.stat().st_size != item['bytes']:
            raise RuntimeError(f'组件文件大小不符：{path}')
        if full:
            with path.open('rb') as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                digest = h.hexdigest()
            if digest != item['sha256']:
                raise RuntimeError(f'组件文件 SHA256 不符：{path}')
        entry_found |= item['path'] == data['entrypoint']
    if not entry_found:
        raise RuntimeError(f'组件入口不在文件清单中：{folder}')
    return data


def verify_engines(root, full=True):
    minimum = minimum_engine(root)
    deepseek = {Path(resolve_launch_config(c, root)['exe']).resolve()
                for c in model_configs(root).values() if is_deepseek_config(c)}
    report = []
    for exe in engine_targets(root):
        if not exe.is_file():
            raise RuntimeError(f'配置中的引擎不存在：{exe}')
        if (exe.parent / 'engine-component.json').exists():
            try:
                meta = component_metadata(exe.parent, full=full)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f'组件清单损坏：{exe.parent}: {exc}') from exc
            version = numbers(meta.get('version', ''))
            if (exe.parent / meta['entrypoint']).resolve() != exe.resolve():
                raise RuntimeError(f'配置入口与组件清单不符：{exe}')
        else:
            try:
                meta = json.loads((exe.parent / 'BUILD.json').read_text(encoding='utf-8-sig'))
                version = numbers(meta.get('version', ''))
            except (OSError, ValueError, AttributeError):
                raise RuntimeError(f'引擎缺少或损坏 BUILD.json：{exe.parent}') from None
        required = (0, 1, 50) if exe.resolve() in deepseek else minimum
        if not version or version < required:
            raise RuntimeError(f'引擎仍为 {meta.get("version", "未知")}，需要至少 {".".join(map(str, required))}：{exe}')
        report.append(f'{exe}: {meta["version"]}')
    return report or ['尚未安装引擎，源码已就绪。']


def repair_missing_metadata(root, mod, log=print):
    """Official update silently skips binaries without BUILD.json. Fetch via its own downloader."""
    root = Path(root).resolve()
    for exe, backend in engine_targets(root).items():
        if not exe.is_file():
            raise RuntimeError(f'配置中的引擎不存在：{exe}')
        try:
            meta = json.loads((exe.parent / 'BUILD.json').read_text(encoding='utf-8-sig'))
            if isinstance(meta, dict) and numbers(meta.get('version', '')):
                continue
        except (OSError, ValueError):
            pass
        if exe.parent.resolve() not in (root / 'engine', root / 'engine-cuda12'):
            raise RuntimeError(f'外部引擎缺失 BUILD.json，请在其所属 Strata 目录更新：{exe}')
        log(f'[更新] {exe.parent} 缺失 BUILD.json，调用官方引擎下载器修复。')
        # Fail before the downloader touches files if another process has the engine open.
        with exe.open('r+b'):
            pass
        base = os.environ.get('STRATA_PREBUILT_URL') or mod['PREBUILT_URL']
        if backend == 'hip':
            gpu = next((g for g in mod['amd_gpus']() if mod['amd_problem'](g) is None), None)
            result = mod['get_prebuilt_hip'](base, gpu, updating=True) if gpu else None
        elif backend in ('sycl', 'vulkan'):
            raise RuntimeError(f'{backend} 引擎缺少元数据，请使用对应官方安装流程修复：{exe}')
        else:
            toolkit = 12 if exe.parent.name == 'engine-cuda12' else 13
            gpu = mod.get('update_engine_gpu', lambda t: mod['gpu_info']())(toolkit)
            result = mod['get_prebuilt'](base, gpu, 'gpu', updating=True, toolkit=toolkit) if gpu else None
            if result:
                mod['pip_cuda_libs'](toolkit)
        if not result:
            raise RuntimeError('官方引擎下载未成功，请检查 GPU、网络及上方输出；本次不会标记为更新成功。')


@contextlib.contextmanager
def launcher_update_lock(root):
    """A real OS lock, released even on process death; never delete another updater's lock."""
    path = Path(root) / '.strata-launcher-update.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'此安装目录已有启动器更新任务，或无法取得写锁：{path} ({exc})') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.launcher-tmp')
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def activate_components(root, choices):
    """Switch model configs only after every package has passed; restore all on write failure."""
    prepared, originals = {}, {}
    for path, folder in choices.items():
        path, folder = Path(path), Path(folder).resolve()
        original = path.read_bytes()
        data = json.loads(original.decode('utf-8-sig'))
        adapted = resolve_launch_config(data, root)
        old = Path(adapted['exe']).parent if adapted['exe'] != 'component:strata-engine' else None
        meta = component_metadata(folder)
        expected = ('strata-dsv41.exe', 'strata-dsv41') if is_deepseek_config(data) else ('strata.exe', 'strata')
        if Path(meta['entrypoint']).name not in expected:
            raise RuntimeError(f'新组件与当前模型架构不匹配：{meta["entrypoint"]}；未切换配置。')
        data['exe'] = str(folder / meta['entrypoint'])
        data['backend'] = meta.get('backend', data.get('backend', ''))
        if old and Path(str(adapted.get('cwd', ''))).resolve() == old.resolve():
            data['cwd'] = str(folder)
        lib_dirs = [str(folder)]
        for f in meta['files']:
            if f['path'].lower().endswith(('.dll', '.so')):
                parent = str((folder / f['path']).parent)
                if parent not in lib_dirs:
                    lib_dirs.append(parent)
        for value in adapted.get('lib_dirs') or []:
            if old and not Path(value).resolve().is_relative_to(old.resolve()) and value not in lib_dirs:
                lib_dirs.append(value)
        data['lib_dirs'] = lib_dirs
        if isinstance(data.get('vision'), dict) and data['vision'].get('exe'):
            vision_name = Path(data['vision']['exe']).name
            vexe = folder / vision_name
            if not vexe.is_file():
                raise RuntimeError(f'新组件没有当前配置需要的图像引擎：{vexe}；未切换配置。')
            data['vision']['exe'] = str(vexe)
            data['vision']['lib_dirs'] = lib_dirs
            if old and Path(str(data['vision'].get('cwd', ''))).resolve() == old.resolve():
                data['vision']['cwd'] = str(folder)
        originals[path] = original
        prepared[path] = data
    for path, original in originals.items():
        backup = Path(str(path) + '.update.bak')
        tmp = backup.with_suffix('.bak.tmp')
        tmp.write_bytes(original)
        tmp.replace(backup)
    changed = []
    try:
        for path, data in prepared.items():
            atomic_json(path, data)
            changed.append(path)
    except Exception:
        for path in reversed(changed):
            tmp = path.with_name(path.name + '.restore-tmp')
            tmp.write_bytes(originals[path])
            tmp.replace(path)
        raise


def require_engine_result(result):
    engines = [r for r in result.get('results', []) if r.get('component') == 'engine']
    if not result.get('ok') or len(engines) != 1 or engines[0].get('status') not in ('updated', 'current'):
        raise RuntimeError('官方更新器未确认引擎更新成功：' + json.dumps(result, ensure_ascii=False))


def run_signed_cli(root, store, backend=None, config=None, log=print):
    root, store = Path(root), Path(store)
    entry = [str(root / 'strata.py')]
    if (root / 'strata_control/transfer.py').is_file():
        entry = [str(Path(__file__).with_name('strata_download.py')), str(root)]
    cmd = [sys.executable, '-u', *entry, 'update', '--no-control',
           '--components-dir', str(store), '--output', 'jsonl']
    if os.environ.get('STRATA_LAUNCHER_RELEASE'):
        from strata_versions import pinned_feed
        cmd += ['--feed', pinned_feed(os.environ['STRATA_LAUNCHER_RELEASE'])]
    if backend:
        cmd += ['--backend', backend]
    if config:
        cmd += ['--model-config', str(config)]
    env = dict(os.environ, STRATA_COMPONENTS_DIR=str(store), PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               STRATA_NO_HANDOVER='1', STRATA_REQUIRE_SIGNED_COMPONENTS='1')
    env.pop('STRATA_NO_ACQUIRE', None)  # this explicit update is authorized to download
    log('[更新] 使用官方签名组件更新器：' + str(store))
    for attempt in range(3):
        result, last, percent = None, '', -1
        last_progress = 0.0
        with (root / '.strata-launcher-update.log').open('a', encoding='utf-8') as saved:
            saved.write('\nCOMMAND ' + repr(cmd) + '\n')
            with subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                  text=True, encoding='utf-8', errors='replace') as proc:
                for line in proc.stdout:
                    saved.write(line)
                    saved.flush()
                    last = line.strip()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        if last:
                            log(last)
                        continue
                    if not isinstance(event, dict):
                        continue
                    if 'ok' in event:
                        result = event
                    if event.get('event') == 'progress':
                        p = event.get('percent', 0)
                        now = time.monotonic()
                        if p > percent or now-last_progress >= 10 or p == 100:
                            done = event.get('done', 0)/1048576
                            total = event.get('total', 0)/1048576
                            speed = event.get('bytes_per_second', 0)/1048576
                            log(f'[下载引擎] {p}%（{done:.1f}/{total:.1f} MB，{speed:.2f} MB/s）')
                            percent = p
                            last_progress = now
                    elif event.get('message'):
                        log('[官方更新器] ' + str(event['message']))
                    elif event.get('error'):
                        log('[官方更新器] ' + json.dumps(event['error'], ensure_ascii=False))
                rc = proc.wait()
        code = (result or {}).get('error', {}).get('code', rc)
        if code == 7 and attempt < 2:
            log('[更新] 组件目录正忙，等待 2 秒后重试；不删除锁。')
            time.sleep(2)
            continue
        if rc or result is None:
            raise RuntimeError(f'签名组件更新失败（退出码 {rc}）：{json.dumps(result, ensure_ascii=False) if result else last}。完整日志：{root / ".strata-launcher-update.log"}')
        require_engine_result(result)
        return result


def load_component_api(root):
    sys.path.insert(0, str(root))
    from strata_control import engine_component as ec, signing
    signing.ensure_component_verifier()
    return ec


def update_signed_components(root, log=print):
    root = Path(root).resolve()
    ec = load_component_api(root)
    configs = model_configs(root)
    deepseek = {p: c for p, c in configs.items() if is_deepseek_config(c)}
    if deepseek:
        update_deepseek_components(root, deepseek, log)
    groups = {}
    for path, cfg in configs.items():
        if is_deepseek_config(cfg):
            continue
        backend = cfg.get('backend') or None
        if backend and backend not in ('cuda', 'hip', 'sycl', 'metal', 'cpu', 'vulkan'):
            raise RuntimeError(f'未知引擎后端 {backend}：{path}')
        slot = backend or 'auto'
        adapted = resolve_launch_config(cfg, root)
        try:
            relative = Path(adapted['exe']).resolve().relative_to(root / '.strata-launcher-components')
            if relative.parts[0] in ('auto', 'cuda', 'hip', 'sycl', 'metal', 'cpu', 'vulkan'):
                slot = relative.parts[0]
        except ValueError:
            pass
        groups.setdefault((slot, backend), []).append(path)
    if not groups:
        log('[更新] DeepSeek 独立引擎已处理，无需更新 Qwen 引擎。' if deepseek else
            '[更新] 尚未安装模型，仅更新源码；模型安装时再选择引擎。')
        return
    choices = {}
    for (slot, backend), paths in groups.items():
        store = root / '.strata-launcher-components' / slot
        result = run_signed_cli(root, store, backend, paths[0], log)
        require_engine_result(result)
        log('[更新校验] 正在检查签名组件、驱动兼容性和所有文件摘要…')
        component = ec.resolve(store, backend=backend, full=True)
        # Signature verification stays with Strata's own trusted-key implementation.
        ec.verify_signature(component, {**os.environ, 'STRATA_REQUIRE_SIGNED_COMPONENTS': '1'})
        actual_version = component_metadata(component.root)['version']
        result_engine = next(r for r in result['results'] if r.get('component') == 'engine')
        confirmed = result_engine.get('installed_after') or result_engine.get('installed_before') or {}
        if confirmed.get('version') and numbers(actual_version) != numbers(confirmed['version']):
            raise RuntimeError('官方更新结果与实际解析的组件版本不一致，未切换配置。')
        if numbers(actual_version) < minimum_engine(root):
            raise RuntimeError('签名源提供的引擎低于当前源码最低要求，未切换配置。')
        choices.update({path: component.root for path in paths})
    activate_components(root, choices)
    log('[更新] 已切换到签名组件；原 engine 目录及模型文件保持原样，配置备份为 .update.bak。')


def update_deepseek_components(root, configs, log=print):
    """Acquire DeepSeek's own signed feed member; never activate a Qwen engine for it."""
    from strata_control import dsv41_engine
    from strata_cli import update_cmd
    from types import SimpleNamespace
    ec = load_component_api(root)
    source = update_cmd._feed_source(SimpleNamespace(feed=None), Path(root))
    if os.environ.get('STRATA_LAUNCHER_RELEASE'):
        from strata_versions import pinned_feed
        source = pinned_feed(os.environ['STRATA_LAUNCHER_RELEASE'])
    # A separate generation leaves the current engine untouched until every check and config write succeeds.
    stage = Path(root) / '.strata-launcher-components' / 'deepseek41' / str(time.time_ns())
    stage.mkdir(parents=True)
    exe = dsv41_engine.install(stage / 'engine-dsv41', source, emit=log)
    if exe is None:
        raise RuntimeError('官方源未能提供此电脑适用的 DeepSeek 独立引擎，原配置保留。')
    meta = component_metadata(exe.parent)
    if numbers(meta.get('version')) < (0, 1, 50):
        raise RuntimeError('DeepSeek 引擎低于最低要求 0.1.50，未切换配置。')
    component = ec.read_manifest(exe.parent)
    ec.verify_signature(component, {**os.environ, 'STRATA_REQUIRE_SIGNED_COMPONENTS': '1'})
    activate_components(root, {p: exe.parent for p in configs})
    log('[更新] DeepSeek 独立签名引擎已更新，模型与原配置备份保留。')


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    with launcher_update_lock(root):
        native = (root / 'strata.py').is_file() and (root / 'strata_cli/update_cmd.py').is_file()
        if native:
            # Do not call setup's legacy get_prebuilt: 0.2 removed its ZIP assets and its shared feed cache may be busy.
            # In the official update branch --build skips legacy engine acquisition; it does not compile.
            # Dependencies and config-schema migrations still use the official setup command.
            env = dict(os.environ)
            env.pop('STRATA_NO_ACQUIRE', None)
            print('[更新] 先执行官方依赖和配置迁移（--update --build 跳过旧式引擎获取，不触发编译）。', flush=True)
            rc = subprocess.call([sys.executable, str(root / 'setup.py'), '--update', '--build', '--yes'], cwd=root, env=env,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if rc:
                raise RuntimeError(f'官方配置/依赖更新失败（{rc}），未切换引擎。')
            update_signed_components(root)
        else:
            mod = runpy.run_path(str(root / 'setup.py'), run_name='strata_update_setup')
            repair_missing_metadata(root, mod)
            rc = subprocess.call([sys.executable, str(root / 'setup.py'), '--update', '--yes'], cwd=root)
            if rc:
                return rc
        for line in verify_engines(root):
            print('[更新校验] ' + line, flush=True)
        if os.environ.get('STRATA_LAUNCHER_RELEASE'):
            from strata_versions import check_exact_engines
            for line in check_exact_engines(root):
                print('[目标版本校验] ' + line, flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('[更新失败] ' + str(exc), flush=True)
        sys.exit(4)
