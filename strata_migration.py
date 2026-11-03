"""One-time restoration of our known old patches; new Strata versions need no patch."""
import hashlib
import json
import os
import sys
from pathlib import Path


def restore_legacy_patch(root, log=None):
    root = Path(root)
    server = root / 'serve/server.py'
    source = server.read_text(encoding='utf-8-sig')
    if 'from serve.context_guard import ' not in source:
        return False
    resource = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)) / 'strata_legacy_restore.json'
    manifest = json.loads(resource.read_text(encoding='utf-8'))
    digest = lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest()
    matched = next((v for v in manifest['variants'] if digest(source) == v['server_hash']), None)
    if matched is None:
        if log:
            log('[context] 检测到其他来源的上下文补丁，未改动源码；内部运行配置会关闭旧压缩。')
        return False
    changes = {}
    for name, allowed in matched['files'].items():
        path = root / 'serve' / name
        current = path.read_text(encoding='utf-8-sig')
        backup = path.with_name(path.name + '.context-original.bak')
        if digest(current) not in allowed['patched_hashes'] or not backup.is_file() or digest(backup.read_text(encoding='utf-8-sig')) != allowed['original_hash']:
            if log:
                log('[context] 旧补丁或备份包含其他修改，未自动恢复；内部运行配置会关闭旧压缩。')
            return False
        changes[path] = backup.read_bytes()
    originals = {p: p.read_bytes() for p in changes}
    try:
        for path, data in originals.items():
            archive = path.with_name(path.name + '.context-layer-previous.bak')
            if not archive.exists():
                archive.write_bytes(data)
        for path, data in changes.items():
            temp = path.with_name(path.name + '.context-restore.tmp')
            temp.write_bytes(data)
            os.replace(temp, path)
    except Exception:
        for path, data in originals.items():
            path.write_bytes(data)
        raise
    if log:
        log('[context] 已恢复原生 Strata 服务端；摘要压缩改由独立服务处理。')
    return True
