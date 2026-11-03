"""Build and execute conservative cleanup plans for one Strata model."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil


MODEL_FILE_FLAGS = ('--native', '--ple-gguf')
MODEL_DIR_FLAGS = ('--pack',)
SHARED_FLAGS = ('--mtp', '--expert-profile')


@dataclass(frozen=True)
class CleanupItem:
    path: Path
    scope: str                 # config / data
    size: int
    deletable: bool = True
    reason: str = ''


@dataclass(frozen=True)
class CleanupPlan:
    config_path: Path
    items: tuple

    @property
    def reclaim_bytes(self):
        return sum(item.size for item in self.items if item.deletable and item.scope == 'data')


@dataclass(frozen=True)
class CleanupResult:
    deleted: tuple
    failed: tuple


def build_cleanup_plan(config_path, config, all_configs, data_root):
    config_path = Path(config_path).resolve()
    data_root = Path(data_root).resolve()
    selected_refs = _model_references(config)
    other_refs = set()
    for entry in all_configs:
        path, candidate = _config_entry(entry)
        if path.resolve() == config_path:
            continue
        other_refs.update(_model_references(candidate))

    candidates = []
    _append_if_exists(candidates, config_path, 'config')
    _append_if_exists(candidates, config_path.with_suffix('.shared-settings.json'), 'config')

    local = config.get('local_import') if isinstance(config.get('local_import'), dict) else {}
    local_folder = _resolved_path(local.get('folder'))
    for flag, path in selected_refs:
        if flag in MODEL_FILE_FLAGS and _safe_model_file(path, data_root, local_folder):
            candidates.append((path, 'data'))
            marker = Path(str(path) + '.done')
            if marker.is_file():
                candidates.append((marker.resolve(), 'data'))
        elif flag in MODEL_DIR_FLAGS and _safe_model_directory(path, data_root):
            candidates.append((path, 'data'))

    identity = str(local.get('identity') or '').strip()
    if identity and identity.replace('-', '').isalnum():
        isolated = (data_root / 'local-imports' / identity).resolve()
        if _is_within(isolated, data_root / 'local-imports') and isolated.exists():
            candidates.append((isolated, 'data'))

    digest = hashlib.sha256(str(config_path).casefold().encode('utf-8')).hexdigest()[:16]
    optimization = (data_root / 'launcher-optimizations' / digest).resolve()
    if optimization.exists():
        candidates.append((optimization, 'data'))

    candidates = _deduplicate_candidates(candidates)
    items = []
    for path, scope in candidates:
        referenced_path = Path(str(path)[:-5]) if str(path).lower().endswith('.done') else path
        shared = scope == 'data' and any(_paths_overlap(referenced_path, ref) for _, ref in other_refs)
        items.append(CleanupItem(path=path, scope=scope, size=_path_size(path),
                                 deletable=not shared,
                                 reason='其他模型仍在使用，已保护' if shared else ''))
    return CleanupPlan(config_path=config_path, items=tuple(items))


def execute_cleanup_plan(plan, delete_data):
    selected = [item for item in plan.items if item.deletable and
                (delete_data or item.scope == 'config')]
    deleted, failed = [], []
    for item in sorted(selected, key=lambda value: len(value.path.parts), reverse=True):
        path = item.path
        try:
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif _is_reparse_point(path):
                raise OSError('拒绝删除目录链接或重解析点')
            else:
                shutil.rmtree(path)
            deleted.append(path)
        except OSError as error:
            failed.append((path, str(error)))
    return CleanupResult(tuple(deleted), tuple(failed))


def _config_entry(entry):
    if isinstance(entry, tuple) and len(entry) == 2:
        return Path(entry[0]), entry[1]
    path = Path(entry)
    try:
        return path, json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return path, {}


def _model_references(config):
    args = config.get('args') if isinstance(config.get('args'), list) else []
    found = []
    for flag in (*MODEL_FILE_FLAGS, *MODEL_DIR_FLAGS, *SHARED_FLAGS):
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args):
                path = _resolved_path(args[index + 1])
                if path:
                    found.append((flag, path))
    return found


def _resolved_path(value):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        return None
    return Path(value).resolve()


def _safe_model_file(path, data_root, local_folder):
    if not path.is_file():
        return False
    if _is_within(path, data_root / 'models'):
        return True
    return bool(local_folder and _is_within(path, local_folder) and path.suffix.lower() == '.gguf')


def _safe_model_directory(path, data_root):
    return path.is_dir() and (_is_within(path, data_root / 'packs') or
                              _is_within(path, data_root / 'local-imports'))


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except (OSError, ValueError):
        return False


def _paths_overlap(candidate, reference):
    candidate, reference = Path(candidate).resolve(), Path(reference).resolve()
    return candidate == reference or _is_within(reference, candidate)


def _append_if_exists(items, path, scope):
    path = Path(path).resolve()
    if path.exists():
        items.append((path, scope))


def _deduplicate_candidates(candidates):
    unique = []
    for path, scope in sorted(set(candidates), key=lambda item: len(item[0].parts)):
        if any(existing_scope == scope and _is_within(path, existing)
               for existing, existing_scope in unique):
            continue
        unique.append((path, scope))
    return unique


def _path_size(path):
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for base, directories, files in os.walk(path, followlinks=False):
            directories[:] = [name for name in directories
                              if not (Path(base) / name).is_symlink()]
            for name in files:
                file = Path(base) / name
                if not file.is_symlink():
                    try:
                        total += file.stat().st_size
                    except OSError:
                        pass
        return total
    except OSError:
        return 0


def _is_reparse_point(path):
    try:
        return bool(path.stat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False
