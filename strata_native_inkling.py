"""Install the launcher-built Strata Inkling runner with pinned artifact checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def inkling_kv(value):
    aliases = {'int8': 'k8v8', 'q8_0': 'k8v8', 'k8v8': 'k8v8',
               'k8v4': 'k8v4', 'fp16': 'fp16', 'fp8': 'fp8', 'auto': 'auto'}
    try:
        return aliases[str(value).lower()]
    except KeyError:
        raise ValueError('Inkling 的 KV 精度支持 INT8、k8v4、fp16、fp8 或 auto；当前选择不受此引擎支持。') from None


# Strata's Inkling runner has no Q4 kernel; the closest supported setting is INT8.
INKLING_KV_FALLBACK = 'k8v8'
# The runner allocates decode/vision workspaces after placing the experts.
# A 500 MiB placement reserve alone left <500 MiB during real requests.
INKLING_WORK_RESERVE_MIB = 1536


def normalize_config(config, precision='original'):
    result = dict(config)
    args = list(config.get('args') or [])
    user = list(config.get('user_args') or [])
    def saved_value(values):
        for index in range(len(values) - 1, -1, -1):
            if values[index] == '--kv' and index + 1 < len(values):
                return values[index + 1]
        return None
    selected = precision if precision != 'original' else (
        saved_value(user) or saved_value(args) or config.get('launcher_kv_precision') or 'int8')
    try:
        selected = inkling_kv(selected)
    except ValueError:
        selected = INKLING_KV_FALLBACK
    reserve = INKLING_WORK_RESERVE_MIB
    for values in (args, user):
        for index, flag in enumerate(values[:-1]):
            if flag == '--vram-reserve-mib':
                reserve = max(reserve, int(values[index + 1]))
    for values in (args, user):
        for flag in ('--kv', '--vram-reserve-mib'):
            while flag in values:
                index = values.index(flag)
                del values[index:index + 2]
        values += ['--kv', selected, '--vram-reserve-mib', str(reserve)]
    result.update(args=args, user_args=user, launcher_kv_precision=selected)
    if 'args_written' in result:
        result['args_written'] = list(args)
    return result


def configure_installer(mod, args, shared, kv='int8'):
    """Use Strata's local shape-runner support for the selected Inkling install."""
    if '--family' not in args or args[args.index('--family') + 1] != 'inkling-2-mini':
        return
    original = mod.shape_engine_step
    def native_step(options, gpu, vision, **kwargs):
        if kwargs.get('hip') or os.name != 'nt':
            return original(options, gpu, vision, **kwargs)
        from strata_control import engine_component
        component = mod.shape_engine_fetch(options)
        if component is None or component.backend != 'cuda':
            return original(options, gpu, vision, **kwargs)
        engine_component.verify_signature(component)
        engine = managed_engine(shared)
        mod.SHAPE_ENGINE['local_exe'] = str(engine)
        libraries = [str(component.root), *engine_component.package_lib_dirs(component.root)]
        metadata = dict(version='0.1.66', backend='cuda', source='local', shape='inkling2')
        return mod.llama_cpp_if_here(), engine.parent, metadata, libraries, gpu, 'none'
    mod.shape_engine_step = native_step
    write = mod.write_config
    def write_native(path, config):
        if '/launcher-components/strata-inkling-' in str(config.get('exe', '')).replace('\\', '/'):
            config = dict(config, launcher_native_build=True, launcher_component_data=str(shared))
            config = normalize_config(config, kv)
        return write(path, config)
    mod.write_config = write_native


def migrate_config(path, root, python):
    import subprocess
    path, root = Path(path), Path(root)
    original = path.read_bytes()
    config = json.loads(original.decode('utf-8-sig'))
    if config.get('engine') != 'llama.cpp':
        return
    local = config.get('local_import') or {}
    if local.get('family') != 'inkling-2-mini':
        raise ValueError('此旧配置需要重新通过 Strata 导入模型。')
    args = config.get('args', [])
    def value(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default
    model = value('--model')
    shared = local.get('shared_data')
    if not model or not shared:
        raise ValueError('旧配置缺少主模型或数据目录，请重新导入。')
    source = Path(__file__).parent
    for name in ('strata_install.py', 'strata_local_import.py', 'strata_native_inkling.py', 'strata_inkling_vision.py',
                 'strata_orca_import.py', 'strata_update.py', 'strata_versions.py', 'strata_launch_support.py', 'strata_download.py'):
        if (source / name).resolve() != (root / name).resolve():
            shutil.copy2(source / name, root / name)
    if (source / 'strata_native_inkling').resolve() != (root / 'strata_native_inkling').resolve():
        shutil.copytree(source / 'strata_native_inkling', root / 'strata_native_inkling', dirs_exist_ok=True)
    command = [str(python), str(root / 'strata_local_import.py'), str(root), model, shared,
               local.get('display_name') or config.get('model_name') or Path(model).stem,
               '--family', 'inkling-2-mini', '--model', 'SQ-10',
               '--context', value('--ctx-size', '262144'), '--parallel', value('--parallel', '1'),
               '--vram-reserve-mib', '500', '--vision', 'gpu' if value('--mmproj') else 'no',
               '--host', config.get('host', '127.0.0.1'), '--port', str(config.get('port', 8080)),
               '--api-key', config.get('api_key', ''), '--launcher-drafting', 'auto' if value('--model-draft') else 'off']
    if value('--mmproj'):
        command += ['--launcher-vision', value('--mmproj')]
    if value('--model-draft'):
        # Use the author's original file, before the old llama.cpp-only conversion.
        candidates = [Path(shared) / 'models' / 'dspark-Inkling-2-Mini-BF16.gguf',
                      Path(model).parent / 'dspark-Inkling-2-Mini-BF16.gguf']
        draft = next((p for p in candidates if p.is_file()), None)
        if draft:
            command += ['--launcher-dspark', str(draft)]
    backup = path.with_suffix('.pre-strata-native.bak')
    if not backup.exists():
        backup.write_bytes(original)
    completed = subprocess.run(command, cwd=root, capture_output=True, encoding='utf-8', errors='replace',
                               env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}, timeout=900,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if completed.returncode:
        path.write_bytes(original)
        raise ValueError('Strata 原生迁移失败：' + (completed.stderr or completed.stdout)[-1800:])
    updated = json.loads(path.read_text(encoding='utf-8-sig'))
    if updated.get('engine') == 'llama.cpp' or updated.get('arch') != 'inkling2_loop':
        path.write_bytes(original)
        raise ValueError('迁移没有生成对应的 Strata 原生配置。')
    for key in ('context_compress_enabled', 'fit_max_tokens'):
        if key in config:
            updated[key] = config[key]
    temporary = path.with_suffix('.native.tmp')
    temporary.write_text(json.dumps(updated, ensure_ascii=True, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def managed_engine(shared):
    source = Path(__file__).with_name('strata_native_inkling')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    target = Path(shared) / 'launcher-components' / ('strata-inkling-' + manifest['build'])
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('原生引擎组件文件名无效。')
        original, installed = source / name, target / name
        with original.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('内置 Strata 引擎组件校验失败：' + name)
        if installed.is_file():
            with installed.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == digest:
                    continue
        partial = installed.with_name(installed.name + '.' + uuid.uuid4().hex + '.tmp')
        shutil.copyfile(original, partial)
        os.replace(partial, installed)
    return target / 'strata-inkling.exe'
