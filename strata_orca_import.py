"""The explicitly documented OrcaRouter IQ3_XXS conversion, with original weights kept in place."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PREFIX = 'Qwen3.8-Flash-Next-Uncensored-IQ3_XXS'


def inspect_orca(folder):
    folder = Path(folder).resolve()
    if folder.is_file():
        folder = folder.parent
    shards = [folder / f'{PREFIX}-{i:05d}-of-00002.gguf' for i in (1, 2)]
    if not any(p.exists() for p in shards):
        return None
    for path in shards:
        if not path.is_file():
            raise ValueError('OrcaRouter 缺少分片：' + path.name)
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError('OrcaRouter 文件不是 GGUF：' + path.name)
    identity = hashlib.sha256('\n'.join(f'{p}|{p.stat().st_size}|{p.stat().st_mtime_ns}' for p in shards).encode()).hexdigest()[:16]
    return dict(folder=str(folder), family='qwen', model='IQ3_XXS', recipe='orca', shards=[str(p) for p in shards], identity=identity)


def run_command(root, *args):
    result = subprocess.call([sys.executable, *map(str, args)], cwd=root,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result:
        raise RuntimeError(f'OrcaRouter 准备步骤失败（{result}）：{args[0]}')


def prepare(root, info, shared, label, args, mode, all_drafters=False):
    root, shared = Path(root), Path(shared)
    if mode == 'dflash2':
        raise ValueError('OrcaRouter 不支持此 DFlash2 转换，特殊转换仅有官方 MTP 路线，请选择自动、MTP 或关闭。')
    from strata_control.gguf_preflight import inspect_metadata
    for shard in info['shards']:
        metadata = inspect_metadata(Path(shard))
        if metadata.get('architecture') != 'qwen4exp':
            raise ValueError('OrcaRouter 模型架构不兼容。')
    from strata_update import run_signed_cli, load_component_api
    store = root / '.strata-launcher-components' / 'auto'
    run_signed_cli(root, store)
    ec = load_component_api(root)
    component = ec.resolve(store, full=True)
    ec.verify_signature(component, {**os.environ, 'STRATA_REQUIRE_SIGNED_COMPONENTS': '1'})
    exe = component.exe
    isolated = shared / 'local-imports' / info['identity']
    pack = isolated / 'packs' / 'orca-iq3_xxs'
    run_command(root, root / 'tools/iq_pack.py', '--gguf', info['shards'][0], '--out', pack, '--compat-bf16')
    def option(flag, default):
        return args[args.index(flag) + 1] if flag in args else default
    context = int(option('--context', '32768'))
    engine_args = ['--pack', str(pack), '--native', info['shards'][0], '--ple-gguf', info['shards'][0],
                   '--expert-profile', str(root / 'data/expert-profile.bin'), '--expert-cache', 'auto',
                   '--prefill', '512', '--spec', '4', '--spec-min-p', '0.5', '--max-context', str(context), '--kv', 'int8']
    if mode != 'off' or all_drafters:
        mtp = shared / 'mtp'
        run_command(root, root / 'tools/mtp_fetch.py', 'fetch', '--out', mtp)
        run_command(root, root / 'tools/mtp_pack.py', '--src', mtp, '--experts', 'q2_0', '--out', mtp / 'mtp-q2_0.gguf')
        run_command(root, root / 'tools/mtp_rt.py', '--gguf', mtp / 'mtp-q2_0.gguf', '--out', mtp / 'rt')
        import shutil
        shutil.copy2(root / 'data/draft_vocab.bin', mtp / 'rt/draft_vocab.bin')
        engine_args += ['--mtp', str(mtp / 'rt')]
    if mode == 'dflash2':
        raise ValueError('OrcaRouter 不支持此 DFlash2 转换，特殊转换仅有官方 MTP 路线，请选择自动、MTP 或关闭。')
    if all_drafters:
        print('[辅助资源] OrcaRouter 官方兼容路线仅准备 MTP，DFlash2 已跳过。', flush=True)
    cfg = dict(exe=str(exe), args=engine_args, cwd=str(root), tokenizer=str(pack / 'tokenizer'),
               model_name=label, log=str(root / ('strata-local-' + info['identity'] + '.log')),
               host=option('--host', '127.0.0.1'), port=int(option('--port', '8080')), api_key=option('--api-key', ''),
               drafting='mtp' if mode != 'off' else 'off', mtp=mode != 'off',
               local_import=dict(info, shared_data=str(shared), display_name=label, vision_enabled=False))
    path = root / f"strata-local-{info['identity']}-iq3_xxs.json"
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding='utf-8')
    os.replace(tmp, path)
    print('OrcaRouter 兼容转换完成（文字模式）；使用模型自身 tokenizer，PLE 保持首分片路径。', flush=True)
    return 0
