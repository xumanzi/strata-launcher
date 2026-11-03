"""Local GGUF import adapter; runs with Strata's own Python environment."""
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def projection_setup_args(enabled, filename=''):
    """Select the bundled or custom runtime projection without modifying weights."""
    if not enabled:
        return ['--experimental-speed-projection', 'off']
    filename = filename.strip()
    if not filename:
        return ['--experimental-speed-projection', 'on']
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise ValueError('拒答方向文件不存在：' + str(path))
    with path.open('rb') as stream:
        if stream.read(4) != b'GGUF':
            raise ValueError('拒答方向文件不是有效的 GGUF：' + str(path))
    return ['--experimental-speed-projection', str(path)]


def find_huihui_vector(folder):
    folder = Path(folder)
    name = 'Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf'
    return next((p / name for p in (folder, folder.parent) if (p / name).is_file()), None)


def apply_runtime_vision(config, root, enabled):
    """Return a runtime copy with vision consistently enabled or disabled."""
    if not enabled:
        result = dict(config)
        result.pop('vision', None)
        args = [arg for arg in (config.get('args') or []) if arg != '--vision']
        if config.get('vision') and '--vram-reserve-mib' in args:
            index = args.index('--vram-reserve-mib')
            if index + 1 < len(args) and str(args[index + 1]) in ('500', '700'):
                del args[index:index + 2]
        result['args'] = args
        return result
    if config.get('vision'):
        args = list(config.get('args') or [])
        if '--vision' in args:
            return config
        result = dict(config)
        result['args'] = [*args, '--vision']
        return result
    candidate = config
    local = config.get('local_import')
    if isinstance(local, dict) and local.get('vision_enabled') is False:
        candidate = dict(config)
        candidate['local_import'] = dict(local, vision_enabled=True)
    result = ensure_local_vision(candidate, root)
    if result is candidate and candidate is not config:
        return config
    if result is not config and isinstance(result.get('local_import'), dict):
        result = dict(result)
        result['local_import'] = local
    return result


def ensure_local_vision(config, root):
    """Add a runtime-only vision section for older Qwen local imports when components exist."""
    local = config.get('local_import') or {}
    if not local or local.get('family') not in ('qwen', 'coder') or local.get('vision_enabled') is False:
        return config
    args = config.get('args') or []
    if config.get('vision'):
        if '--vision' in args:
            return config
        result = dict(config)
        result['args'] = [*args, '--vision']
        if '--vram-reserve-mib' not in args:
            result['args'] += ['--vram-reserve-mib', '500']
        return result
    if '--native' not in args or args.index('--native') + 1 >= len(args):
        return config
    model = Path(args[args.index('--native') + 1])
    root = Path(root)
    exe = root / 'engine' / ('strata-vision.exe' if sys.platform == 'win32' else 'strata-vision')
    candidates = [Path(local['shared_data'])] if local.get('shared_data') else []
    candidates.append(root.parent / 'Strata-data')
    mmproj = next((base / 'models' / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf' for base in candidates
                   if (base / 'models' / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf').is_file()), None)
    if not model.is_file() or not exe.is_file() or mmproj is None:
        return config
    try:
        build = json.loads((exe.parent / 'BUILD.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        build = {}
    result = dict(config)
    gpu = build.get('vision') == 'gpu'
    result['vision'] = {'exe': str(exe), 'mmproj': str(mmproj), 'model': str(model),
                        'gpu': gpu, 'max_tokens': 1024 if gpu else 300}
    result['args'] = [*args]
    if '--vision' not in args:
        result['args'].append('--vision')
    if '--vram-reserve-mib' not in args:
        result['args'] += ['--vram-reserve-mib', '500']
    if not gpu:
        result['vision']['threads'] = max(1, (os.cpu_count() or 8) // 2)
    return result


def inspect_model(folder):
    folder = Path(folder).resolve()
    files = sorted(folder.glob('*.gguf'))
    firsts = [p for p in files if not p.name.lower().startswith('mmproj')
              and 'refusal-direction' not in p.name.lower()
              and (re.search(r'-00001-of-\d{5}\.gguf$', p.name, re.I)
                   or not re.search(r'-\d{5}-of-\d{5}\.gguf$', p.name, re.I))]
    if len(firsts) != 1:
        raise ValueError('请选择只含一个模型全部分片的文件夹，不要选择 models 总目录。')
    first = firsts[0]
    match = re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', first.name, re.I)
    stem, _, total = match.groups() if match else (first.stem, '00001', '1')
    quant = re.search(r'-((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))$', stem, re.I)
    model = quant.group(1).upper() if quant else '未知量化'
    unsloth = model in ('UD-Q4_K_XL', 'UD-IQ4_XS')
    gsq = model in ('Q2_0', 'IQ2_XS', 'IQ3_XXS', 'IQ3_S', 'IQ1_M') and 'GSQ-RCO' in stem.upper()
    if not (unsloth or gsq) or 'qwen3.8-flash-next' not in stem.lower():
        raise ValueError('当前 Strata 引擎不支持此导入格式：' + model + '。支持 GSQ-RCO Q2_0/IQ2_XS/IQ3_XXS/IQ3_S、Coder IQ1_M，以及 UD-IQ4_XS/UD-Q4_K_XL。其他 GGUF 请使用支持对应量化的 llama.cpp。')
    count = int(total)
    if not 1 <= count <= 16:
        raise ValueError('模型分片数量不正确。')
    shards = [folder / f'{stem}-{i:05d}-of-{count:05d}.gguf' for i in range(1, count + 1)] if match else [first]
    for shard in shards:
        if not shard.is_file():
            raise ValueError('缺少分片：' + shard.name + '。请下载完整后再导入。')
        with shard.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError('文件不是有效的 GGUF：' + shard.name)
    fingerprint = '\n'.join(f'{str(p).casefold()}|{p.stat().st_size}|{p.stat().st_mtime_ns}' for p in shards)
    identity = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:16]
    return {'folder': str(folder), 'model': quant.group(1).upper(),
            'family': ('unsloth' if unsloth else 'coder' if quant.group(1).upper() == 'IQ1_M'
                       else 'swift' if stem.lower().startswith('swift-') else 'qwen'),
            'shards': [str(p) for p in shards], 'identity': identity}


def configure_setup(mod, shared, info, label, vision_enabled=None):
    """Adapt only this setup process; never edit upstream setup or migrate data."""
    shared = Path(shared).resolve()
    isolated = shared / 'local-imports' / info['identity']
    isolated.mkdir(parents=True, exist_ok=True)
    mod.data_folder = lambda requested, move=False, yes=True, set_default=False: (isolated, [])
    find = mod.find_in

    def find_local(roots, rel):
        # Only auxiliary resources may be shared. Expert packs always stay isolated.
        allowed = rel == 'mtp' or rel.startswith('mtp/') or rel.startswith('models/mmproj-')
        return find([*roots, shared] if allowed else roots, rel)

    mod.find_in = find_local
    family = dict(mod.FAMILIES[info['family']])
    # Published hashes apply only to upstream downloads, never third-party local files.
    family.pop('sha256', None)
    family['tag'] = 'local-' + info['identity'] + '-'
    mod.FAMILIES[info['family']] = family
    mod.gguf_dir_shards = lambda folder, fam, model: [Path(p) for p in info['shards']]
    write = mod.write_config
    expected = f"strata-local-{info['identity']}-{info['model'].lower()}.json"

    def write_local(path, cfg):
        if Path(path).name == expected:
            cfg = dict(cfg, model_name=label, local_import={
                'folder': info['folder'], 'identity': info['identity'],
                'family': info['family'], 'model': info['model'], 'shared_data': str(shared),
                'display_name': label, 'vision_enabled': vision_enabled})
            args = cfg.get('args') or []
            if '--native' in args and args.index('--native') + 1 < len(args):
                filename = Path(args[args.index('--native') + 1]).name
                cfg['model_name'] = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', filename, flags=re.I)
        return write(path, cfg)

    mod.write_config = write_local
    return isolated


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root, folder, shared, label, *args = argv
    root = Path(root).resolve()
    info = inspect_model(folder)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('strata_import_setup', root / 'setup.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not all(hasattr(mod, name) for name in ('data_folder', 'find_in', 'write_config', 'FAMILIES', 'main')):
        raise ValueError('当前 Strata 源码不支持本地导入，请更新 Strata 后再试。')
    vision_enabled = args[args.index('--vision') + 1] != 'no' if '--vision' in args and args.index('--vision') + 1 < len(args) else None
    isolated = configure_setup(mod, shared, info, label, vision_enabled=vision_enabled)
    print(f'本地模型：{label}\n文件目录：{folder}\n独立准备目录：{isolated}', flush=True)
    # Disallow caller overrides that could accidentally select another model/data directory.
    for option in ('--family', '--model', '--data-dir', '--models-dir', '--gguf-dir'):
        while option in args:
            index = args.index(option)
            del args[index:index + 2]
    sys.argv = [str(root / 'setup.py'), *args, '--family', info['family'], '--model', info['model'],
                '--data-dir', str(isolated), '--gguf-dir', info['folder'], '--no-start']
    rc = mod.main()
    if rc not in (None, 0):
        return rc
    print('本地模型准备完成。返回概览页，选择模型名称，再点击启动模型。', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('本地导入失败：' + str(error), flush=True)
        sys.exit(1)
