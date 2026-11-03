"""Non-destructive, per-model launch presets for Strata."""

import hashlib
from pathlib import Path


EFFORTS = {'high', 'medium', 'low'}
MODES = {'single', 'concurrent', 'session'}


def model_settings(options):
    """Only settings understood by Strata's shared API defaults."""
    effort = options.get('effort', 'high')
    if effort not in EFFORTS:
        raise ValueError('思考强度必须为 high、medium 或 low')
    return {'reasoning_effort': effort}


def apply_runtime_optimization(config, options, cfg_path, data_dir):
    """Return a runtime copy; never write to the installed model configuration."""
    options = dict(options)
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
    # Runtime overrides are applied after vision adds its fallback reserve.
    def replace_arg(flag, value):
        args = out['args']
        while flag in args:
            index = args.index(flag)
            del args[index:index + 2]
        args.extend((flag, str(value)))

    replace_arg('--vram-reserve-mib', 500)
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
        target = _model_dir(cfg_path, data_dir) / 'expert-profile.bin'
        target.parent.mkdir(parents=True, exist_ok=True)
        out['expert_profile_save'] = str(target)
        out['expert_profile_save_every'] = 10
        notes.append('专家自适应配置按模型独立保存')
    return out, notes


def _model_dir(cfg_path, data_dir):
    path = Path(cfg_path).resolve()
    digest = hashlib.sha256(str(path).casefold().encode('utf-8')).hexdigest()[:16]
    base = Path(data_dir) if data_dir else path.parent
    return base / 'launcher-optimizations' / digest
