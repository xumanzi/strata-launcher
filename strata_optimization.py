"""Non-destructive, per-model launch presets for Strata."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


EFFORTS = {'none', 'high', 'medium', 'low'}
MODES = {'speed', 'single', 'concurrent', 'session'}
VRAM_RESERVE_MIB = 500


def model_settings(options):
    """Only settings understood by Strata's shared API defaults."""
    effort = options.get('effort', 'high')
    if effort not in EFFORTS:
        raise ValueError('思考强度必须为 none、high、medium 或 low')
    return {'reasoning_effort': effort}


def apply_runtime_optimization(config, options, cfg_path, data_dir):
    """Return a runtime copy; never write to the installed model configuration."""
    options = dict(options)
    if options.get('mode', 'speed' if config.get('launcher_speed_priority') else 'single') == 'speed':
        return speed_priority_config(config, options, cfg_path, data_dir)
    if config.get('arch') == 'inkling2_loop':
        from strata_native_inkling import normalize_config
        result = normalize_config(config, options.get('kv_precision', 'original'))
        reserve = result['args'][result['args'].index('--vram-reserve-mib') + 1]
        notes = [f"Inkling 原生 KV 精度 {result['launcher_kv_precision']}；工作区预留 {reserve} MiB，目标运行余量至少 500 MiB；KV 驻留位置由引擎自动分配"]
        if options.get('kv_precision') == 'q4_0':
            notes.append('Inkling 没有 Q4 KV 内核，已使用支持的 INT8 KV')
        return result, notes
    if config.get('arch') == 'hy4flash':
        return dict(config), ['沿用官方独立引擎的显存、KV 和草稿设置']
    compatibility_notes = []
    if config.get('arch') in ('deepseek41', 'qwen35moe'):
        options.update(spec='original', profile=False)
        compatibility_notes.append('使用此模型的官方原生加速配置')
    if config.get('arch') == 'deepseek41':
        options['mode'] = 'single'
        compatibility_notes.append('DeepSeek 使用单请求模式；保留 DSpark 草稿参数')
    out = dict(config)
    out['args'] = list(config.get('args') or [])
    out['env'] = dict(config.get('env') or {})
    # The upstream automatic full arena otherwise leaves no work-memory target.
    # Older engines ignore this variable. Explicit environment settings still win.
    if (os.name == 'nt' and config.get('backend') == 'cuda'
            and '--native' in out['args']
            and 'flash-next' in str(config.get('model_name', '')).lower()
            and not os.environ.get('STRATA_RAM_KEEP_GIB')
            and 'STRATA_RAM_KEEP_GIB' not in out['env']):
        out['env']['STRATA_RAM_KEEP_GIB'] = '8'
        compatibility_notes.append('系统内存预留目标 8 GiB；其余参数沿用模型设置')
    # Runtime overrides are applied after vision adds its fallback reserve.
    def replace_arg(flag, value):
        args = out['args']
        while flag in args:
            index = args.index(flag)
            del args[index:index + 2]
        args.extend((flag, str(value)))

    # The engine raises the reserve for vision + elastic VRAM in resident RAM mode.
    # Declare false explicitly so the server cannot turn elastic VRAM on by default.
    out['vram_elastic'] = False
    flag = '--vram-reserve-mib'
    reserve = VRAM_RESERVE_MIB
    for values in (out['args'], config.get('user_args') or []):
        for i, arg in enumerate(values[:-1]):
            if arg == flag:
                reserve = max(reserve, int(values[i + 1]))
    replace_arg(flag, reserve)
    out['args'] = [arg for arg in out['args'] if arg != '--vram-elastic']
    # Strata merges user_args over args and uses it to mark the reserve as fixed.
    user_args = list(config.get('user_args') or [])
    for index in range(len(user_args) - 1, -1, -1):
        if user_args[index] == '--vram-elastic':
            del user_args[index]
        elif user_args[index] == flag:
            has_value = index + 1 < len(user_args) and not user_args[index + 1].startswith('--')
            del user_args[index:index + 1 + has_value]
    out['user_args'] = [*user_args, flag, str(reserve)]
    compatibility_notes.append(f'显存工作区预留 {reserve} MiB；运行空闲目标至少 500 MiB')
    if 'kv_resident' in options:
        resident = options['kv_resident']
        if isinstance(resident, bool) or not isinstance(resident, int) or (resident != 0 and resident < 20480):
            raise ValueError('KV 显存驻留上限必须为 0（全部）或不小于 20480 的整数')
        if config.get('arch') in ('deepseek41', 'qwen35moe'):
            if resident:
                raise ValueError('此模型引擎暂不支持 KV 显存驻留限制，请选择全部驻留')
        else:
            replace_arg('--kv-resident', resident)
            out['args'] = [arg for arg in out['args'] if arg != '--kv-grow']
            out['env']['STRATA_KV_GROW'] = '0'
            compatibility_notes.append('KV 全部驻留显存' if resident == 0 else
                                       f'KV 显存驻留上限 {resident}；其余使用系统内存')
    precision = options.get('kv_precision', 'original')
    if precision not in ('original', 'int8', 'q4_0', 'k8v4', 'fp16'):
        raise ValueError('KV 精度必须为原配置、int8、q4_0、k8v4 或 fp16')
    if precision != 'original':
        if config.get('arch') in ('deepseek41', 'qwen35moe'):
            raise ValueError('此模型引擎暂不支持切换 KV 精度，请选择沿用模型配置')
        replace_arg('--kv', precision)
        compatibility_notes.append(f'KV 精度 {precision}')
    mode = options.get('mode', 'single')
    if mode not in MODES:
        raise ValueError('运行模式无效')
    notes = compatibility_notes
    if mode == 'concurrent':
        out['parallel'] = 2
        out.pop('slot_save_path', None)
        # Batch MTP is an opt-in engine experiment. Preserve normal MTP behavior.
        out['env'].pop('STRATA_BATCH_MTP', None)
        notes.append('双请求并发；会话快照不可用')
    else:
        out['parallel'] = 1
        out['env'].pop('STRATA_BATCH_MTP', None)
        if mode == 'session':
            location = _model_dir(cfg_path, data_dir) / 'sessions'
            location.mkdir(parents=True, exist_ok=True)
            out['slot_save_path'] = str(location)
            notes.append('会话 KV 保存/恢复已启用；需由客户端调用 /slots/0')
        else:
            out.pop('slot_save_path', None)
            notes.append('单请求模式')

    budget = options.get('budget', 0)
    try:
        budget = int(budget or 0)
    except (TypeError, ValueError):
        raise ValueError('思考上限必须是非负整数') from None
    if budget < 0:
        raise ValueError('思考上限必须是非负整数')
    if budget:
        out['reasoning_budget_tokens'] = budget
        notes.append(f'思考上限 {budget} tokens；不关闭思考')
    else:
        out.pop('reasoning_budget_tokens', None)
        notes.append('思考 token 不设上限')

    spec = options.get('spec', 'original')
    if spec not in ('original', '4', '6'):
        raise ValueError('投机步数仅支持原配置、4 或 6')
    if spec != 'original':
        args = out['args']
        if '--spec' in args:
            index = args.index('--spec')
            if index + 1 < len(args):
                args[index + 1] = spec
        else:
            args.extend(('--spec', spec))
        notes.append(f'投机步数 {spec}；效果需实测')

    if options.get('profile'):
        target = _expert_profile_target(config, cfg_path, data_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        out['expert_profile_save'] = str(target)
        out['expert_profile_save_every'] = 10
        notes.append('专家自适应配置按模型独立保存')
    return out, notes


def speed_priority_config(config, options, cfg_path, data_dir):
    """Measured single-request preset. Keep weights/context and original config intact."""
    inkling = config.get('arch') == 'inkling2_loop'
    qwen = (config.get('backend') == 'cuda' and
            'flash-next' in str(config.get('model_name', '')).lower())
    if not inkling and not qwen:
        raise ValueError('速度优先预设目前仅适用于已实测的 Inkling 和 Qwen Flash-Next CUDA 模型')
    out = dict(config)
    out['args'] = list(config.get('args') or [])
    out['user_args'] = list(config.get('user_args') or [])
    out['env'] = dict(config.get('env') or {})
    def remove(flag):
        for key in ('args', 'user_args'):
            values = out[key]
            i = 0
            while i < len(values):
                if values[i].split('=', 1)[0] == flag:
                    count = 1 if '=' in values[i] else 1 + (i+1 < len(values) and not values[i+1].startswith('--'))
                    del values[i:i+count]
                else:
                    i += 1
    def setflag(flag, value):
        remove(flag)
        for key in ('args', 'user_args'):
            out[key] += [flag, str(value)]
    for flag in ('--adapt-every','--adapt-swaps','--adapt-decay','--vram-elastic','--parallel'):
        remove(flag)
    out.update(parallel=1, vram_elastic=False, launcher_speed_priority=True)
    out.pop('slot_save_path', None)
    out['env'].pop('STRATA_BATCH_MTP', None)
    reserve = 2048 if inkling else 1024
    for key in ('args', 'user_args'):
        values = out[key]
        for i, arg in enumerate(values):
            key, sep, inline = arg.partition('=')
            if key == '--vram-reserve-mib':
                reserve = max(reserve, int(inline if sep else values[i+1]))
    setflag('--vram-reserve-mib', reserve)
    if inkling:
        from strata_native_inkling import normalize_config
        setflag('--parallel', 1)
        for flag in ('--cpu-threads','--vram-experts-gb','--ram-experts-gb','--chunk'):
            remove(flag)
        setflag('--kv-offload', 'auto')
        out = normalize_config(out, options.get('kv_precision', 'int8'))
        note = 'Inkling 速度优先：INT8 KV、自动 KV 分配、自动专家预算、单请求；保留已配置的 DSpark 和读图'
    else:
        # A CLI flag wins over old STRATA_RAM_KEEP_GIB=8 in process/user environments.
        out['env'].pop('STRATA_RAM_KEEP_GIB', None)
        remove('--kv-grow')
        setflag('--ram-keep-gib', 2)
        setflag('--prefill', 'auto')
        if options.get('kv_precision', 'int8') != 'original':
            setflag('--kv', options.get('kv_precision', 'int8'))
        setflag('--kv-resident', options.get('kv_resident', 65536))
        setflag('--spec', options.get('spec') if options.get('spec') in ('4','6') else '4')
        out['env']['STRATA_KV_GROW'] = '0'
        note = 'Qwen 速度优先：系统内存预留 2 GiB、自动分块、默认专家交换策略、单请求；内存足够时完整驻留'
    out.pop('reasoning_budget_tokens', None)
    if options.get('budget'):
        out['reasoning_budget_tokens'] = int(options['budget'])
    out.pop('expert_profile_save', None)
    out.pop('expert_profile_save_every', None)
    if not options.get('profile'):
        for flag in ('--expert-profile-save', '--expert-profile-save-every'):
            remove(flag)
    if options.get('profile') and qwen:
        target = _expert_profile_target(config, cfg_path, data_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.update(expert_profile_save=str(target), expert_profile_save_every=10)
    out['args_written'] = list(out['args'])
    return out, [note, f'工作区预留 {reserve} MiB；运行空闲显存目标至少 500 MiB']


def _model_dir(cfg_path, data_dir):
    path = Path(cfg_path).resolve()
    digest = hashlib.sha256(str(path).casefold().encode('utf-8')).hexdigest()[:16]
    base = Path(data_dir) if data_dir else path.parent
    return base / 'launcher-optimizations' / digest


def _expert_profile_target(config, cfg_path, data_dir):
    path = Path(cfg_path).resolve()
    legacy = _model_dir(path, data_dir) / 'expert-profile.bin'
    local = config.get('local_import')
    identity = local.get('identity') if isinstance(local, dict) else None
    if not data_dir or not identity or not path.name.startswith(f'strata-local-{identity}-'):
        return legacy

    # The importer encodes model identity and quantization in the config name.
    # Keep the same profile across managed release directories.
    digest = hashlib.sha256(path.name.casefold().encode('utf-8')).hexdigest()[:16]
    target = legacy.parent.parent / f'local-{digest}' / legacy.name
    if target.exists():
        return target

    candidates = [legacy]
    release = next((p for p in path.parents if p.parent.name == '.strata-versions'), None)
    if release is not None:
        for sibling in release.parent.glob('release-*'):
            for directory in (sibling, sibling / 'terminal'):
                candidates.append(_model_dir(directory / path.name, data_dir) / legacy.name)
    candidates = [source for source in candidates if source.is_file()]
    if candidates:
        source = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            # The server validates the migrated profile against the model's base profile.
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target
