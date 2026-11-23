"""Adapt the official installer's three model tables without changing its install paths."""
import json
import sys
import importlib.util
from pathlib import Path


def official_tables(mod):
    models = {name: dict(row) for name, row in mod.MODELS.items()}
    families = {name: dict(row) for name, row in mod.FAMILIES.items()}
    contexts = list(mod.CONTEXTS)
    for table in ('OWN_PATH_FAMILIES', 'MULTI_FILE_FAMILIES'):
        multi = table == 'MULTI_FILE_FAMILIES'
        for family, definition in (getattr(mod, table, {}) or {}).items():
            families[family] = dict(definition, own_path=True, multi_file=multi)
            for name, original in definition.get('models', {}).items():
                row = dict(original, families=[family], official_model=name, own_path=True,
                           multi_file=multi, arch=definition.get('arch'),
                           vision=bool(definition.get('vision', not multi)))
                files = ([*original['shards'], *original.get('shared', [])] if multi else
                         [definition['file'].format(q=name)])
                row['download_gb'] = round(sum(definition['sha256'][f][0] for f in dict.fromkeys(files)) / 1e9, 1)
                if multi:
                    row['contexts'] = [getattr(mod, 'DSV41_CONTEXT', 8192)]
                    # Use the selected release's rules; old releases keep their original limit.
                    if (Path(getattr(mod, '__file__', '')).parent / 'strata_control/dsv41_context.py').is_file():
                        spec = importlib.util.spec_from_file_location('launcher_dsv41_context',
                            Path(mod.__file__).parent / 'strata_control/dsv41_context.py')
                        context_api = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(context_api)
                        row['contexts'] = context_api.choices()
                        row['context_tested_max'] = context_api.TESTED
                        row['default_context'] = getattr(mod, 'DSV41_CONTEXT', 131072)
                models[family + '::' + name] = row
    for table, api_name in (('HY4_FAMILIES', 'hy4_family'), ('INKLING_FAMILIES', 'inkling_family')):
        for family, definition in getattr(mod, table, {}).items():
            api = getattr(mod, api_name)
            families[family] = dict(definition, own_path=True, shape_engine=True)
            for name, original in definition['models'].items():
                files = api.files_of(name, vision=False)
                models[family + '::' + name] = dict(
                    original, families=[family], official_model=name, shape_engine=True,
                    arch=definition['arch'], vision=True, contexts=list(api.CONTEXTS),
                    default_context=api.DEFAULT_CONTEXT,
                    download_gb=round(sum(api.FILES[file][0] for file in files) / 1e9, 1))
    return models, families, contexts


def family_models(models, family):
    return {row.get('official_model', name): row for name, row in models.items()
            if family in row.get('families', ('qwen', 'swift'))}


def bundled_tables():
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
    data = json.loads((base / 'strata_model_catalog.json').read_text(encoding='utf-8'))
    return data['models'], data['families'], data['contexts']


def adapt_install_args(args, metadata):
    """Do not pass Flash-Next-only tuning into a different engine architecture."""
    args = list(args)
    family = args[args.index('--family') + 1] if '--family' in args else ''
    remove = set()
    if family not in ('qwen', 'coder'):
        remove.add('--experimental-speed-projection')
    if metadata.get('own_path') or metadata.get('multi_file') or metadata.get('shape_engine'):
        remove.update(('--kv', '--low-ram', '--resident-budget-gib', '--draft-vocab', '--kv-streaming'))
    if metadata.get('shape_engine'):
        remove.add('--vram-reserve-mib')
    if metadata.get('multi_file'):
        remove.update(('--parallel', '--vram-reserve-mib'))
    out, i = [], 0
    while i < len(args):
        flag = args[i]
        if flag in remove:
            i += 2
            continue
        if flag == '--vision' and metadata.get('vision') is False:
            out.extend((flag, 'no'))
            i += 2
            continue
        out.append(flag)
        i += 1
    return out
