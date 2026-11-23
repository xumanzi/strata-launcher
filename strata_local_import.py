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
    if config.get('arch') in ('hy4flash', 'inkling2_loop'):
        if enabled:
            return config
        result = dict(config)
        for key in ('args', 'user_args'):
            if isinstance(config.get(key), list):
                args = list(config[key])
                while '--vision' in args:
                    index = args.index('--vision')
                    del args[index:index + 2]
                result[key] = args
        return result
    if not enabled:
        result = dict(config)
        result.pop('vision', None)
        args = [arg for arg in (config.get('args') or []) if arg != '--vision']
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
    if folder.is_file():
        selected = folder
        folder = folder.parent
    else:
        selected = None
    files = sorted(folder.glob('*.gguf'))
    firsts = [p for p in files if not p.name.lower().startswith('mmproj')
              and not any(token in p.name.lower() for token in
                          ('refusal-direction', 'engram', 'per_layer_token_embd'))
              and not p.name.lower().startswith(('mtp-', 'mtp_', 'dflash', 'draft-', 'draft_'))
              and (selected is None or p == selected or
                   re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', p.name, flags=re.I) ==
                   re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', selected.name, flags=re.I))
              and (re.search(r'-00001-of-\d{5}\.gguf$', p.name, re.I)
                   or not re.search(r'-\d{5}-of-\d{5}\.gguf$', p.name, re.I))]
    if len(firsts) != 1:
        raise ValueError('请选择只含一个模型全部分片的文件夹，不要选择 models 总目录。')
    first = firsts[0]
    match = re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', first.name, re.I)
    stem, _, total = match.groups() if match else (first.stem, '00001', '1')
    quant = re.search(r'-((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))$', stem, re.I)
    model = quant.group(1).upper() if quant else '未知量化'
    unsloth = model in ('UD-Q4_K_XL', 'UD-IQ4_XS', 'UD-Q5_K_XL', 'UD-Q6_K_XL', 'Q8_0',
                        'UD-IQ1_M', 'UD-Q2_0', 'UD-IQ2_XS', 'UD-Q2_K_XL', 'UD-IQ3_XXS', 'UD-Q3_K_XL')
    if model == 'UD-IQ1_S' and 'qwen3.8-flash-next' in stem.lower():
        raise ValueError('Strata 当前不支持 Qwen3.8 UD-IQ1_S 的混合专家格式，请选择 UD-IQ1_M。')
    gsq = model in ('Q2_0', 'IQ2_XS', 'IQ3_XXS', 'IQ3_S', 'IQ1_M') and 'GSQ-RCO' in stem.upper()
    if not (unsloth or gsq) or 'qwen3.8-flash-next' not in stem.lower():
        raise ValueError('当前 Strata 引擎尚无此格式的可用加载或转换规则：' + model + '。请更新 Strata 后重新检查，原模型文件已保留。')
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
    if unsloth and model == 'UD-IQ4_XS' and count == 4:
        model = 'UD-IQ4_XS-v31'
    return {'folder': str(folder), 'model': model,
            'family': ('unsloth' if unsloth else 'coder' if quant.group(1).upper() == 'IQ1_M'
                       else 'swift' if stem.lower().startswith('swift-') else 'qwen'),
            'shards': [str(p) for p in shards], 'identity': identity}


def info_from_plan(plan):
    if plan.get('issues'):
        raise ValueError('；'.join(plan.get('plain_issues') or plan['issues']))
    selection = plan.get('candidate_selection') or {}
    if not selection.get('family') or not selection.get('model'):
        raise ValueError('官方检查未识别可准备的模型规格。')
    model_dir = Path(plan['model_dir'])
    shards = [str(row.get('path') or model_dir / row['name']) for row in plan.get('files', [])]
    if not shards:
        raise ValueError('官方检查未返回可准备的模型文件。')
    identity = hashlib.sha256(json.dumps(plan['files'], sort_keys=True).encode()).hexdigest()[:16]
    return dict(folder=str(model_dir), model=selection['model'], family=selection['family'],
                shards=shards, identity=identity)


def inspect_inkling_gguf(folder):
    """Recognize Inkling by GGUF architecture before the generic Qwen importer."""
    from strata_control.gguf_preflight import inspect_metadata, ProfileError

    picked = Path(folder).resolve()
    if picked.is_file():
        files = [picked]
    else:
        files = sorted(picked.glob("*.gguf"))
    candidates = [file for file in files if file.suffix.lower() == ".gguf"
                  and not file.name.lower().startswith(("mmproj", "dspark", "mtp", "draft"))]
    if len(candidates) != 1:
        return None
    model = candidates[0]
    try:
        meta = inspect_metadata(model)
    except (ValueError, OSError, RuntimeError, ProfileError):
        return None
    if meta.get("architecture") not in ("inkling2", "inkling2_loop"):
        return None
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", model.name, re.I):
        raise ValueError("Inkling 本地导入当前需要单文件 GGUF。")
    stat = model.stat()
    key = f"{model}|{stat.st_size}|{stat.st_mtime_ns}"
    return dict(folder=str(model.parent), family="inkling-2-mini",
                model=model.stem, shards=[str(model)], format="inkling-gguf",
                identity=hashlib.sha256(key.encode()).hexdigest()[:16])


def inkling_dspark_layout_error(path):
    """Detect the published legacy DSpark layout rejected by current llama.cpp."""
    from strata_control.gguf_preflight import _Reader, ProfileError

    try:
        with path.open("rb") as stream:
            reader = _Reader(stream, path.stat().st_size)
            if reader.take(4) != b"GGUF" or reader.u32() != 3:
                return None
            count, keys = reader.u64(), reader.u64()
            vocab = None
            for _ in range(keys):
                key = reader.string(capture=True)
                value = reader.value(reader.u32(), capture=key == "tokenizer.ggml.tokens")
                if key == "tokenizer.ggml.tokens" and isinstance(value, tuple):
                    vocab = value[1]
            markov_rows = None
            has_w2 = False
            for _ in range(count):
                name = reader.string(capture=True)
                dims = [reader.u64() for _ in range(reader.u32())]
                reader.skip(12)
                if name == "markov_w1.weight" and len(dims) == 2:
                    markov_rows = dims[1]
                elif name == "markov_w2.weight":
                    has_w2 = True
    except (OSError, ProfileError):
        return None
    if markov_rows == 201024 and vocab == 200058 and not has_w2:
        return "markov_w1 为 201024 行，词表实际为 200058，且缺少当前后端要求的 markov_w2"
    return None


def inkling_file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fetch_inkling_auxiliary(mod, shared, role, validate=lambda path: True):
    from strata_install import INKLING_GGUF_REPO, INKLING_GGUF_AUX, community_manifest, download_manifest

    name = INKLING_GGUF_AUX[role]
    cache = Path(shared) / "launcher-auxiliaries"
    source = f"{INKLING_GGUF_REPO}/resolve/main/{name}"
    try:
        for path in cache.glob(f"*/{name}"):
            if re.fullmatch(r"[a-f0-9]{64}", path.parent.name):
                if inkling_file_digest(path) == path.parent.name and validate(path):
                    print(f"[辅助模型] 复用已校验文件：{path}", flush=True)
                    return path
        rows = community_manifest(source, auxiliary=True)
        row = rows[0]
        folder = cache / row["sha"]
        print(f"[辅助模型] 自动匹配 {name} · {source}", flush=True)
        download_manifest(mod, rows, folder)
        candidate = folder / row["name"]
        if validate(candidate):
            return candidate
        print(f"[辅助模型] 引擎未确认配套文件：{name}", flush=True)
    except (ValueError, OSError, SystemExit) as error:
        print(f"[辅助模型] 自动准备 {name} 失败：{error}", flush=True)
    return None


def managed_inkling_server(shared):
    """Keep the verified executable and its adjacent DLLs together."""
    import shutil
    import tempfile
    from strata_install import INKLING_DSPARK_COMPAT as compat

    files = compat["engine_files"]
    target = Path(shared) / "launcher-components" / "inkling-dspark-32218"

    def verified(folder):
        return all((folder / name).is_file() and inkling_file_digest(folder / name) == digest
                   for name, digest in files.items())

    if verified(target):
        return target / "llama-server.exe"
    override = os.environ.get("STRATA_INKLING_DSPARK_SERVER")
    legacy = Path(compat["legacy_server"])
    source = Path(override).parent if override else legacy.parent
    if target.exists():
        raise ValueError(f"DSpark 组件校验失败：{target}")
    if not verified(source):
        if override:
            raise ValueError(f"DSpark component does not match its pinned build: {source}")
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        staging = Path(work) / "engine"
        staging.mkdir()
        for name in files:
            shutil.copyfile(source / name, staging / name)
        if not verified(staging):
            raise ValueError("DSpark 组件复制后校验失败。")
        try:
            staging.rename(target)
        except FileExistsError:
            if not verified(target):
                raise
    return target / "llama-server.exe"


def prepare_inkling_llamacpp(mod, info, shared, label, args, selected, drafting):
    import shutil

    def option(name, default):
        return args[args.index(name) + 1] if name in args else default

    import subprocess

    paths = [os.environ.get("STRATA_LLAMA_SERVER"), shutil.which("llama-server"),
             managed_inkling_server(shared)]
    roots = [Path(mod.ROOT).parent / "llamacpp", Path("E:/llamacpp")]
    for directory in roots:
        paths.extend(directory.glob("llama-*-bin-win-cuda-*/llama-server.exe"))
    exe = None
    for path in dict.fromkeys(Path(item).resolve() for item in paths if item and Path(item).is_file()):
        try:
            version = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                                     errors="replace", timeout=10,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except (OSError, subprocess.SubprocessError):
            continue
        build = re.search(r"\b(?:version:|build:?)\s*(\d+)\b(?!\.)", version.stdout + version.stderr, re.I)
        if version.returncode == 0 and ((build and int(build[1]) >= 12638)
                                       or path == Path(shared).resolve() / "launcher-components" /
                                       "inkling-dspark-32218" / "llama-server.exe"):
            exe = path
            break
    if not exe:
        raise ValueError("此模型需要支持 inkling2_loop 的 llama.cpp b12638 或更新版。"
                         "请将 STRATA_LLAMA_SERVER 指向对应的 llama-server.exe。")
    model = Path(info["shards"][0])
    shared = Path(shared).resolve()
    port = int(option("--port", 8080))
    reserve = max(500, int(option("--vram-reserve-mib", 500)))
    engine_args = ["--model", str(model), "--jinja", "--fit", "on", "--fit-target", str(reserve),
                   "--ctx-size", str(option("--context", 32768)), "--parallel", str(option("--parallel", 1)),
                   "--ubatch-size", "96", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                   "--flash-attn", "on", "--no-warmup"]

    def auxiliary(role, names):
        if role in selected:
            return selected[role]
        return next((folder / name for folder in (model.parent, shared / "models")
                     for name in names if (folder / name).is_file()), None)

    vision_mode = option("--vision", "no")
    vision = (auxiliary("vision", ("mmproj-Inkling-2-Mini-BF16.gguf", "mmproj-BF16.gguf"))
              if vision_mode not in ("no", "none") else None)
    draft = (auxiliary("dspark", ("dspark-Inkling-2-Mini-Q8_0.gguf", "dspark-Inkling-2-Mini-BF16.gguf"))
             if drafting != "off" or "dspark" in selected else None)
    if vision_mode not in ("no", "none") and vision is None:
        vision = fetch_inkling_auxiliary(mod, shared, "vision")
    if drafting != "off" and draft is None:
        draft = fetch_inkling_auxiliary(mod, shared, "dspark")
    if vision:
        engine_args += ["--mmproj", str(vision)]
        if vision_mode == "cpu":
            engine_args.append("--no-mmproj-offload")
    managed_draft = False
    if draft:
        from strata_install import INKLING_DSPARK_COMPAT as compat

        mismatch = inkling_dspark_layout_error(draft)
        converted = shared / "models" / "inkling-2-mini" / compat["converted"]
        if mismatch or draft.resolve() == converted.resolve():
            known_digest = compat["originals"].get(draft.name.lower())
            compat_exe = managed_inkling_server(shared) if converted.is_file() else None
            if converted.is_file() and compat_exe is not None and (known_digest or not mismatch):
                if mismatch and inkling_file_digest(draft) != known_digest and "dspark" not in selected:
                    print("[DSpark] Cached draft is not the pinned conversion input; DSpark skipped.", flush=True)
                    draft = None
                if draft is not None and mismatch and inkling_file_digest(draft) != known_digest:
                    raise ValueError("所选旧版 DSpark 文件与已适配的发布文件校验值不同。")
                if draft is not None and inkling_file_digest(model) != compat["main_sha256"]:
                    if "dspark" in selected:
                        raise ValueError("此 DSpark 转换配置尚未验证你选择的主模型版本。")
                    print("[DSpark] 转换权重未验证用于此主模型，本次未启用。", flush=True)
                    draft = None
                if inkling_file_digest(converted) != compat["converted_sha256"]:
                    raise ValueError("转换后的 DSpark 文件校验失败。")
                print(f"[DSpark] 为当前主模型启用 Inkling 专用后端和转换权重：{converted.name}", flush=True)
                if draft is not None:
                    exe, draft = compat_exe, converted
                    managed_draft = True
            elif "dspark" in selected:
                raise ValueError(f"此 DSpark 需要转换和 Inkling 专用运行版：{mismatch or '缺少运行版'}。")
            else:
                print(f"[DSpark] 已跳过 {draft.name}：{mismatch}。主模型和读图继续配置。", flush=True)
                draft = None
    if draft:
        capabilities = subprocess.run([str(exe), "--help"], capture_output=True, text=True,
                                      errors="replace", timeout=15,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if capabilities.returncode or "draft-dspark" not in capabilities.stdout + capabilities.stderr:
            if "dspark" in selected:
                raise ValueError("所选 llama.cpp 引擎不支持 draft-dspark。")
            print("[DSpark] 当前引擎不支持 draft-dspark，本次未启用。", flush=True)
            draft = None
    print(f"[辅助模型] mmproj：{vision or '未启用'}；DSpark：{draft or '未启用'}", flush=True)
    if draft:
        engine_args += ["--model-draft", str(draft), "--spec-type", "draft-dspark", "--spec-draft-n-max", "3"]
    path = Path(mod.home()) / f"strata-local-{info['identity']}-inkling-gguf.json"
    mod.write_config(path, dict(exe=str(exe), args=engine_args, cwd=str(mod.ROOT),
        engine="llama.cpp", launcher_managed_dspark=managed_draft, model_name=model.stem, port=port, host=option("--host", "127.0.0.1"),
        api_key=option("--api-key", ""), log=str(path.with_suffix(".log")),
        local_import=dict(info, shared_data=str(shared), display_name=label, vision_enabled=bool(vision))))
    print(f"Inkling GGUF 已生成 llama.cpp 启动配置：{path}", flush=True)
    return 0


def prepare_inkling_gguf(mod, info, shared, label, args, selected, drafting):
    import subprocess
    from types import SimpleNamespace
    from strata_control import engine_component, inkling_family, shape_engine

    option = lambda name, default=None: args[args.index(name) + 1] if name in args else default
    os.environ["STRATA_REQUIRE_SIGNED_COMPONENTS"] = "1"
    try:
        component = shape_engine.find(inkling_family.SHAPE, full=True)
    except shape_engine.ShapeEngineMissing:
        mod.SHAPE_ENGINE.update(shape=inkling_family.SHAPE, title="Inkling 2 Mini")
        component = mod.shape_engine_fetch(SimpleNamespace(feed=option("--feed")))
    if component is None:
        raise ValueError("未安装支持 Inkling 2 Mini 的签名引擎。")
    engine_component.verify_signature(component)
    engine = Path(component.exe)
    managed_native = False
    lib_dirs = engine_component.package_lib_dirs(component.root)
    env = dict(os.environ)
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[variable] = os.pathsep.join([str(engine.parent), *lib_dirs, env.get(variable, "")])
    model = Path(info["shards"][0])
    shared = Path(shared).resolve()
    home = Path(mod.home())
    isolated = shared / "local-imports" / info["identity"]

    def probe(*flags):
        flags = list(flags)
        if '--vision' in flags:
            index = flags.index('--vision') + 1
            if Path(flags[index]).suffix.lower() == '.gguf':
                from strata_inkling_vision import convert_projector
                flags[index] = convert_projector(flags[index], shared)
        done = subprocess.run([str(engine), "--info", str(model), *map(str, flags)],
                              cwd=engine.parent, env=env, timeout=60, capture_output=True,
                              text=True, errors="replace")
        if done.returncode:
            raise ValueError("Inkling 引擎拒绝模型或配套文件：" + (done.stderr or done.stdout).strip()[-1200:])
        for line in done.stdout.splitlines():
            if line.startswith("arch="):
                return dict(field.split("=", 1) for field in line.split() if "=" in field)
        raise ValueError("Inkling 引擎未返回模型架构。")

    try:
        details = probe()
    except ValueError as error:
        if "missing tensor loop_res_query.weight" not in str(error):
            raise
        from strata_native_inkling import managed_engine
        engine = managed_engine(shared)
        managed_native = True
        env[variable] = os.pathsep.join([str(engine.parent), *lib_dirs, env.get(variable, "")])
        details = probe()
        print("[引擎] 使用内置编译的 Strata Inkling 原生 GGUF 加载器。", flush=True)
    if details.get("arch") not in ("inkling2", "inkling2_loop"):
        raise ValueError("Inkling 引擎未确认此 GGUF 的模型架构。")
    vision = selected.get("vision")
    dspark = selected.get("dspark")
    if vision and Path(vision).suffix.lower() == '.gguf':
        from strata_inkling_vision import convert_projector
        vision = convert_projector(vision, shared)
    vision_requested = option("--vision", "no") not in ("none", "no")
    if vision_requested and not vision:
        for name in ("mmproj-Inkling-2-Mini-BF16.gguf", "mmproj-BF16.gguf"):
            for folder in (model.parent, shared / "models"):
                candidate = folder / name
                if candidate.is_file():
                    try:
                        readable = probe("--vision", candidate).get("images") == "1"
                    except ValueError:
                        continue
                    if readable:
                        vision = candidate
                        break
            if vision:
                break
    if not dspark and drafting != "off":
        for folder in (model.parent, shared / "models"):
            for name in ("dspark-Inkling-2-Mini-Q8_0.gguf", "dspark-Inkling-2-Mini-BF16.gguf"):
                candidate = folder / name
                if candidate.is_file():
                    try:
                        supported = probe("--dspark", candidate).get("drafter") == "dspark"
                    except ValueError:
                        continue
                    if supported:
                        dspark = candidate
                        break
            if dspark:
                break
    def fetch_matching(role, flag, capability, expected):
        def validate(path):
            try:
                return probe(flag, path).get(capability) == expected
            except (ValueError, OSError, subprocess.SubprocessError):
                return False

        return fetch_inkling_auxiliary(mod, shared, role, validate)

    if vision_requested and vision is None:
        vision = fetch_matching("vision", "--vision", "images", "1")
    if drafting != "off" and dspark is None:
        dspark = fetch_matching("dspark", "--dspark", "drafter", "dspark")

    if vision and Path(vision).suffix.lower() == '.gguf':
        from strata_inkling_vision import convert_projector
        vision = convert_projector(vision, shared)

    flags = []
    if vision:
        flags += ["--vision", vision]
    if dspark:
        flags += ["--dspark", dspark]
    if flags:
        details = probe(*flags)
    if vision and details.get("images") != "1":
        raise ValueError("Inkling 引擎未确认所选视觉权重。")
    if dspark and details.get("drafter") != "dspark":
        raise ValueError("Inkling 引擎未确认所选 DSpark 权重。")
    if vision_requested and not vision:
        print("未找到经引擎验证的视觉文件，已准备文字模式。可重新导入并手选视觉权重。", flush=True)
    tokenizer = mod.inkling_tokenizer_dir(SimpleNamespace(), model, model.parent,
                                          isolated / "tokenizer", engine, True)
    if not tokenizer:
        raise ValueError("无法从此 GGUF 导出 tokenizer；请在模型旁提供配套 tokenizer.json 和 chat_template.jinja。")
    parallel = int(option("--parallel", 1))
    from strata_native_inkling import inkling_kv
    kv = inkling_kv(option('--launcher-inkling-kv', option('--kv', 'int8')))
    engine_args = mod.build_inkling_import_args(
        model, max_context=int(option("--context", 32768)), parallel=parallel,
        kv=kv,
        vram_reserve_mib=max(500, int(option("--vram-reserve-mib", 500))),
        dspark_path=dspark, vision_path=vision)
    if drafting == 'off':
        engine_args += ['--draft', 'off']
    path = home / f"strata-local-{info['identity']}-inkling-gguf.json"
    cfg = dict(exe=str(engine) if managed_native else shape_engine.reference(inkling_family.SHAPE), args=engine_args,
               launcher_native_build=managed_native,
               launcher_kv_precision=kv,
               launcher_component_data=str(shared) if managed_native else None,
               cwd=str(mod.ROOT), tokenizer=str(tokenizer), model_name=model.stem,
               arch=inkling_family.ARCH, log=str(path.with_suffix(".log")), lib_dirs=lib_dirs,
               parallel=parallel, port=int(option("--port", 8080)),
               host=option("--host", "127.0.0.1"), api_key=option("--api-key", ""),
               drafting="auto" if dspark or drafting != "off" else "off",
               local_import=dict(info, shared_data=str(shared), display_name=label,
                                 vision_enabled=bool(vision)))
    from strata_native_inkling import normalize_config
    cfg = normalize_config(cfg, kv)
    mod.write_config(path, cfg)
    print(f"Inkling GGUF 已通过引擎检查并生成启动配置：{path}", flush=True)
    return 0


def preview_model(root, folder, python=None):
    """Use the selected release's interpreter, avoiding module pollution in the GUI."""
    import subprocess
    root = Path(root).resolve()
    from strata_orca_import import inspect_orca
    special = inspect_orca(folder)
    if special is not None:
        return special
    models_source = root / 'strata_control' / 'models.py'
    if not models_source.is_file() or 'def plan_import(' not in models_source.read_text(encoding='utf-8-sig'):
        return inspect_model(folder)
    python = python or root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    code = ('import json,sys; from strata_control.models import plan_import; '
            'sys.path.insert(0, sys.argv[2]); from strata_local_import import inspect_inkling_gguf; '
            'info=inspect_inkling_gguf(sys.argv[1]); '
            'print(json.dumps(info or plan_import(sys.argv[1]), ensure_ascii=True))')
    result = subprocess.run([str(python), '-c', code, str(Path(folder).resolve()), str(Path(__file__).parent)], cwd=root,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120,
                            env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise ValueError('官方模型检查失败：' + result.stderr[-1600:])
    parsed = json.loads(result.stdout)
    return parsed if parsed.get('format') == 'inkling-gguf' else info_from_plan(parsed)


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
    table = next((getattr(mod, name, {}) for name in
                  ('FAMILIES', 'OWN_PATH_FAMILIES', 'MULTI_FILE_FAMILIES')
                  if info['family'] in getattr(mod, name, {})), None)
    if table is None:
        raise ValueError('目标 Strata 版本没有此模型家族：' + info['family'])
    family = dict(table[info['family']])
    # Published hashes apply only to upstream downloads, never third-party local files.
    # Keep official pins for recognised published files. Custom GSQ variants retain
    # the historical compatibility path; their provenance is not official.
    if table is getattr(mod, 'FAMILIES', None) and any(
            'abliterated' in Path(p).name.lower() for p in info.get('shards', [])):
        family.pop('sha256', None)
    family['tag'] = 'local-' + info['identity'] + '-'
    table[info['family']] = family
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
    from strata_install import (take_drafting, prepare_setup_args, config_snapshot, finish_resources,
        take_all_drafters, take_auxiliaries, configure_auxiliaries, take_remote_auxiliaries, download_manifest)
    args, drafting_mode = take_drafting(args)
    args, all_drafters = take_all_drafters(args)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('strata_import_setup', root / 'setup.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sources = []
    if '--launcher-community' in args:
        sources = [row['url'] for row in json.loads(args[args.index('--launcher-community') + 1])]
    write_config = mod.write_config
    def write_labelled(path, config):
        metadata = ' '.join((label, folder, *sources, json.dumps(config.get('local_import') or {})))
        if re.search(r'abliterat|uncensored|unrestricted|无限制|破限', metadata, re.I):
            config = dict(config, launcher_unrestricted=True)
        if sources:
            config = dict(config, launcher_model_sources=sources)
        return write_config(path, config)
    mod.write_config = write_labelled
    args = take_remote_auxiliaries(mod, args, shared)
    args, selected = take_auxiliaries(args)
    community = "--launcher-community" in args
    if community:
        position = args.index("--launcher-community")
        rows = json.loads(args[position + 1])
        del args[position:position + 2]
        download_manifest(mod, rows, Path(folder))
    if hasattr(mod, "build_inkling_import_args"):
        inkling = inspect_inkling_gguf(folder)
        if inkling is not None:
            return prepare_inkling_gguf(mod, inkling, shared, label, args, selected, drafting_mode)
    from strata_orca_import import inspect_orca, prepare as prepare_orca
    special = inspect_orca(folder)
    if special is not None:
        if selected:
            raise ValueError("此模型家族暂不支持手选辅助权重。")
        return prepare_orca(root, special, shared, label, args, drafting_mode, all_drafters=all_drafters)
    # New releases own architecture/tensor/layout recognition. Never silently fall
    # back when their preflight reports an invalid candidate.
    if (root / 'strata_control' / 'models.py').is_file():
        from strata_control import models as official_models
        plan_import = getattr(official_models, 'plan_import', None)
        plan = plan_import(str(Path(folder).resolve())) if callable(plan_import) else None
        if plan is None:
            info = inspect_model(folder)
        else:
            info = info_from_plan(plan)
    else:
        info = inspect_model(folder)
    if community:
        family = args[args.index('--family') + 1]
        model = args[args.index('--model') + 1]
        flash_next = {'qwen', 'swift'}
        compatible_family = (family == info['family'] or
                             {family, info['family']} == flash_next)
        if not compatible_family or model != info['model']:
            raise ValueError(f"实际下载的是 {info['family']} / {info['model']}。请在安装向导选择同一家族和规格后重试。")
    if not all(hasattr(mod, name) for name in ('data_folder', 'find_in', 'write_config', 'FAMILIES', 'main')):
        raise ValueError('当前 Strata 源码不支持本地导入，请更新 Strata 后再试。')
    tables = [getattr(mod, name, {}) for name in
              ('FAMILIES', 'OWN_PATH_FAMILIES', 'MULTI_FILE_FAMILIES', 'HY4_FAMILIES', 'INKLING_FAMILIES')]
    definition = next((table[info['family']] for table in tables if info['family'] in table), None)
    models = definition.get('models', getattr(mod, 'MODELS', {info['model']: {}})) if definition else {}
    if definition is None or info['model'] not in models:
        raise ValueError('所选 Strata 版本不支持此模型规格：' + info['model'] + '，请先一键更新最新版。')
    vision_enabled = args[args.index('--vision') + 1] != 'no' if '--vision' in args and args.index('--vision') + 1 < len(args) else None
    shape = info['family'] in {**getattr(mod, 'HY4_FAMILIES', {}), **getattr(mod, 'INKLING_FAMILIES', {})}
    if shape:
        os.environ['STRATA_REQUIRE_SIGNED_COMPONENTS'] = '1'
        isolated = Path(shared).resolve() / 'local-imports' / info['identity']
        isolated.mkdir(parents=True, exist_ok=True)
        mod.data_folder = lambda requested, move=False, yes=True, set_default=False: (isolated, [])
        write = mod.write_setup_config
        def write_shape(path, cfg, *rest, **kwargs):
            if cfg.get('arch') == definition['arch']:
                cfg = dict(cfg, local_import={
                    'folder': info['folder'], 'identity': info['identity'],
                    'family': info['family'], 'model': info['model'],
                    'shared_data': str(shared), 'display_name': label,
                    'vision_enabled': vision_enabled})
                path = Path(path).with_name(f"strata-local-{info['identity']}-{info['model'].lower()}.json")
            return write(path, cfg, *rest, **kwargs)
        mod.write_setup_config = write_shape
        script = mod.write_run_script
        def write_script(tag, path, *rest, **kwargs):
            name = f"strata-local-{info['identity']}-{info['model'].lower()}.json"
            return script(f"local-{info['identity']}", Path(path).with_name(name), *rest, **kwargs)
        mod.write_run_script = write_script
    else:
        isolated = configure_setup(mod, shared, info, label, vision_enabled=vision_enabled)
    print(f'本地模型：{label}\n文件目录：{folder}\n独立准备目录：{isolated}', flush=True)
    # Disallow caller overrides that could accidentally select another model/data directory.
    for option in ('--family', '--model', '--data-dir', '--models-dir', '--gguf-dir'):
        while option in args:
            index = args.index(option)
            del args[index:index + 2]
    args += ['--family', info['family'], '--model', info['model']]
    configure_auxiliaries(mod, args, selected, shared, drafting=drafting_mode)
    args = prepare_setup_args(mod, args, 'mtp' if all_drafters else drafting_mode)
    before = config_snapshot(root)
    sys.argv = [str(root / 'setup.py'), *args,
                '--data-dir', str(isolated), '--gguf-dir', info['folder'], '--no-start']
    rc = mod.main()
    if rc not in (None, 0):
        return rc
    if not shape:
        finish_resources(root, drafting_mode, before, all_drafters)
    print('本地模型准备完成。返回概览页，选择模型名称，再点击启动模型。', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('本地导入失败：' + str(error), flush=True)
        sys.exit(1)
