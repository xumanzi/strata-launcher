"""Explicit installation wrapper: prepare the model and its selected auxiliary resources."""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

INKLING_GGUF_REPO = "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF"
INKLING_GGUF_AUX = {"vision": "mmproj-BF16.gguf", "dspark": "dspark-Inkling-2-Mini-Q8_0.gguf"}
INKLING_DSPARK_COMPAT = {
    "legacy_server": "E:/llamacpp/inkling-dspark-e4b1cb5/out/bin/llama-server.exe",
    "engine_files": {
        "ggml-base.dll": "873595c901d7cd8f3733754c314c9707b558c54de5a23992d24ac7a6f0f41ba5",
        "ggml-cpu.dll": "25435e565233e2522b56af103fde86360c4aa81b2ccb2167f36cd6ae23747668",
        "ggml-cuda.dll": "cdef16739e344d49e6aa4504757dd2c4134387de31248a124fba7927eb1ed4f6",
        "ggml.dll": "b4b90eaf8a02b33406fbc433d6f74dd9a904faf2509517aa69c94c995bdeb67c",
        "llama-common.dll": "26408a368c43914918f2fcb74e305624a305670f538173151e0c7832df16076b",
        "llama-server-impl.dll": "8a7f8f4b1d92ac0412e69220106e308f47468fd0fe200f560b7ce77e7f1bba08",
        "llama-server.exe": "6e1a7a7ca9daa988385b7d1ed2154cddf0fbe0aa0c3d8f1f4cc13bc21b0d6b0b",
        "llama.dll": "ada071efdb9bbd5e59f0a05e3bbd6c4c953c906ea8e98baaccbef034c19d20b2",
        "mtmd.dll": "cc574f8fde0cbaac727da6cccd91a3d34f2b39231b66505c7c12315dcdd68960",
    },
    "converted": "dspark-Inkling-2-Mini-BF16-llamacpp-32218.gguf",
    "main_sha256": "aabee5f8a2e8efe6272dc1f56322e9ad49f26c35b39e8744601e9eab73cdc487",
    "converted_sha256": "2da8a6bf40340f64575b925594907ce8914703e3f4a8c66bf433782758cc22f2",
    "originals": {
        "dspark-inkling-2-mini-bf16.gguf": "3efe9eb782f3910bb2d604526f093af501c09c54e4912273b6ba2de842962c4c",
        "dspark-inkling-2-mini-q8_0.gguf": "334cbc2e08c061081c59ebb3bc17e915a77cc89dd2d893027af70daa4b69c99e",
    },
}
HUIHUI_REPO = "https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF"
HUIHUI_FILE = "Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf"

MODES = ('auto', 'mtp', 'dflash2', 'off')


AUXILIARY_LABELS = {
    "mmproj": "视觉投影 mmproj",
    "vision": "视觉权重 Vision",
    "dspark": "DSpark 解码权重",
    "engram": "Engram 权重",
}


def auxiliary_roles(family, vision):
    kinds = {
        "inkling-2-mini": ("vision", "dspark"),
        "hy4-flash": ("vision", "engram"),
    }
    if family in kinds:
        image, other = kinds[family]
        return (image, other) if vision not in ("no", "none") else (other,)
    return ("mmproj",) if vision not in ("no", "none") else ()


def take_auxiliaries(args):
    args = list(args)
    selected = {}
    for role in AUXILIARY_LABELS:
        flag = "--launcher-" + role
        if flag not in args:
            continue
        index = args.index(flag)
        if index + 1 == len(args):
            raise ValueError(f"{flag} 缺少文件路径")
        path = Path(args[index + 1]).resolve(strict=True)
        with path.open("rb") as stream:
            header = stream.read(4)
        if role == "mmproj" and header != b"GGUF":
            raise ValueError(f"视觉投影文件不是 GGUF：{path}")
        if path.stat().st_size < 32:
            raise ValueError(f"辅助文件不完整：{path}")
        selected[role] = path
        del args[index:index + 2]
    return args, selected


def configure_auxiliaries(mod, args, selected, shared=None, *, drafting=None):
    family = args[args.index("--family") + 1] if "--family" in args else ""
    vision = args[args.index("--vision") + 1] if "--vision" in args else "no"
    if set(selected) - set(auxiliary_roles(family, vision)):
        raise ValueError("当前模型或读图设置不支持所选辅助文件。")
    own_family = getattr(mod, "OWN_PATH_FAMILIES", {}).get(family)
    if own_family and own_family.get("mmproj") and vision not in ("no", "none"):
        original = own_family["mmproj"]
        size, sha = own_family["sha256"][original]
        chosen = selected.get("mmproj")
        if chosen is None and shared is not None:
            chosen = Path(shared).resolve() / "models" / family / original
            engine_step = mod.engine_step

            def prepare_projector(*values, **kwargs):
                result = engine_step(*values, **kwargs)
                if result[-1] != "none":
                    mod.download(own_family["mmproj_hf"] + original, chosen, "vision encoder")
                return result

            mod.engine_step = prepare_projector
        if chosen is not None:
            family_table = mod.OWN_PATH_FAMILIES
            family_table[family] = dict(own_family, mmproj=str(chosen),
                                        sha256={**own_family["sha256"], str(chosen): (size, sha)})
            done, verify = mod.done, mod.verify_sha256
            if "mmproj" in selected:
                mod.done = lambda path: Path(path) == chosen or done(path)

            def verify_file(path, expected_size, expected_sha, *values, **kwargs):
                if Path(path) != chosen:
                    return verify(path, expected_size, expected_sha, *values, **kwargs)
                if chosen.stat().st_size != expected_size or mod._hash_file(chosen, expected_size) != expected_sha:
                    raise ValueError(f"所选视觉投影与当前模型的权重不匹配：{chosen}")

            if "mmproj" in selected:
                mod.verify_sha256 = verify_file
            selected = {role: path for role, path in selected.items() if role != "mmproj"}
    if "mmproj" in selected:
        if not hasattr(mod, "resolve_reader"):
            raise ValueError("当前引擎不支持手选视觉投影。")

        def resolve_reader(pinned, path, models_dir, roots, gguf_dir, size, sha=None):
            chosen = selected["mmproj"]
            if sha and chosen.stat().st_size == size and mod._hash_file(chosen, size) == sha:
                return "renamed", chosen
            accepted = mod.accepted_alt_reader(pinned, [chosen], start=True)
            if accepted is not None and Path(accepted).resolve() == chosen:
                return "alt", chosen
            raise ValueError(f"所选视觉投影与已验证的配套权重不匹配：{chosen}")

        mod.resolve_reader = resolve_reader

    prefixes = {"inkling-2-mini": "inkling", "hy4-flash": "hy4"}
    if family not in prefixes:
        return
    prefix = prefixes[family]
    resources = getattr(mod, prefix + "_family", None)
    if resources is None:
        return
    install_name = "install_inkling2" if prefix == "inkling" else "install_hy4_flash"
    if hasattr(mod, install_name):
        install = getattr(mod, install_name)

        def reserved_gpu(gpu):
            if not gpu or not gpu.get("vram_gb"):
                return gpu
            return dict(gpu, vram_gb=max(0, float(gpu["vram_gb"]) - 500 / 1024))

        def install_shape(options, family, gpu, hip, amd, cuda_tk, multi, *rest):
            return install(options, family, reserved_gpu(gpu), hip, amd, cuda_tk,
                           [reserved_gpu(card) for card in multi] if multi else multi, *rest)

        setattr(mod, install_name, install_shape)
    files = {getattr(resources, "VISION_FILE", None): "vision",
             getattr(resources, "ENGRAM_FILE", None): "engram"}
    files.pop(None, None)
    fetch = getattr(mod, prefix + "_fetch")
    have = getattr(mod, prefix + "_have")

    def custom_file(name):
        return selected.get(files.get(name))

    def fetch_file(options, name, folder, own):
        chosen = custom_file(name)
        if chosen is not None:
            if prefix == 'inkling' and files.get(name) == 'vision':
                return chosen  # The native runner validates this pairing before the config is written.
            size, sha = resources.FILES[name]
            if not sha or chosen.stat().st_size != size or mod._hash_file(chosen, size) != sha:
                raise ValueError(f"所选辅助文件不匹配：{name} · {chosen}")
            return chosen
        if own and shared is not None and name in files and not have(options, folder / name):
            return fetch(options, name, Path(shared) / "models" / family, False)
        return fetch(options, name, folder, own)

    setattr(mod, prefix + "_fetch", fetch_file)
    setattr(mod, prefix + "_have", lambda options, dst: custom_file(dst.name) is not None or have(options, dst))
    if prefix == "inkling" and drafting == "off" and "dspark" not in selected:
        save_config = mod.write_setup_config

        def write_without_dspark(path, cfg, *values, **kwargs):
            if cfg.get("arch") == resources.ARCH:
                cfg["drafting"] = "off"
                # Apply after setup carries forward prior launch settings.
                previous = kwargs.get("after_carry")
                def disable(config):
                    if previous:
                        previous(config)
                    config["drafting"] = "off"
                kwargs["after_carry"] = disable
            return save_config(path, cfg, *values, **kwargs)

        mod.write_setup_config = write_without_dspark
    if "dspark" in selected or (prefix == 'inkling' and 'vision' in selected):
        from subprocess import run, SubprocessError

        draft = selected.get("dspark")
        if draft and draft.suffix.lower() not in (".gguf", ".strata", ".safetensors"):
            raise ValueError("DSpark 文件须为 GGUF、Strata 或 safetensors。")
        located = None
        engine_step = mod.engine_step

        def prepare_engine(*values, **kwargs):
            nonlocal located
            result = engine_step(*values, **kwargs)
            located = Path(result[1]) / getattr(resources, "EXE")
            return result

        mod.engine_step = prepare_engine
        write = mod.write_config

        def write_config(path, cfg):
            if cfg.get("arch") == resources.ARCH:
                if located is None or not located.is_file():
                    raise ValueError("当前没有可验证 DSpark 配对的本地 Inkling 引擎。")
                args = cfg["args"]
                model = args[args.index("--native") + 1]
                env = dict(os.environ)
                env["PATH"] = os.pathsep.join([*cfg.get("lib_dirs", []), env.get("PATH", "")])
                flags = []
                if draft:
                    flags += ['--dspark', str(draft)]
                if selected.get('vision'):
                    flags += ['--vision', str(selected['vision'])]
                try:
                    result = run([str(located), "--info", model, *flags],
                                 capture_output=True, text=True, errors="replace",
                                 env=env, cwd=located.parent, timeout=45)
                except (OSError, SubprocessError) as error:
                    raise ValueError(f"DSpark 配对检查失败：{error}") from error
                confirmed = not result.returncode and re.search(r'(?m)^arch=', result.stdout)
                if draft:
                    confirmed = confirmed and re.search(r'\bdrafter=dspark\b', result.stdout)
                if selected.get('vision'):
                    confirmed = confirmed and re.search(r'\bimages=1\b', result.stdout)
                if not confirmed:
                    detail = (result.stderr or result.stdout).strip()[-700:]
                    raise ValueError(f"Inkling 引擎未确认 DSpark 配对：{detail}")
                for flag in ('--vision', '--dspark'):
                    if flag in flags:
                        while flag in args:
                            index = args.index(flag)
                            del args[index:index + 2]
                args += flags
                if draft:
                    while '--draft' in args:
                        index = args.index('--draft')
                        del args[index:index + 2]
                    args += ['--draft', 'dspark']
                    cfg["drafting"] = "auto"
            return write(path, cfg)

        mod.write_config = write_config


def community_manifest(link, *, auxiliary=False, model=None):
    """Resolve a public HF repository or file to immutable LFS entries."""
    from urllib.parse import quote, unquote, urlsplit
    from urllib.request import Request, urlopen

    address = urlsplit(link.strip())
    parts = unquote(address.path).strip("/").split("/")
    if (address.scheme, address.netloc) != ("https", "huggingface.co") or len(parts) < 2:
        raise ValueError("请输入 Hugging Face 公开仓库或文件的 HTTPS 链接。")
    repo = "/".join(parts[:2])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo, re.ASCII):
        raise ValueError("Hugging Face 仓库名无效。")
    revision = "main"
    chosen = None
    if len(parts) > 2:
        if len(parts) < 5 or parts[2] not in ("resolve", "blob"):
            raise ValueError("请使用仓库首页、blob 或 resolve 文件链接。")
        revision = parts[3]
        chosen = "/".join(parts[4:])
    url = f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}?blobs=true"
    with urlopen(Request(url, headers={"User-Agent": "StrataLauncher"}), timeout=15) as response:
        metadata = json.loads(response.read(8 * 1024 * 1024))
    commit = metadata.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("无法确定仓库的固定提交版本。")
    suffixes = (".gguf", ".strata", ".safetensors") if auxiliary else (".gguf",)
    rows = {row["rfilename"]: row for row in metadata.get("siblings", [])
            if row["rfilename"].lower().endswith(suffixes)
            and (auxiliary or not re.search(r"(^|[/_-])(mmproj|dspark|engram|mtp|dflash|esp|refusal-direction)([-_.]|$)",
                                            row["rfilename"], re.I))}
    if chosen is None:
        firsts = [name for name in rows if not re.search(r"-(?!00001)\d{5}-of-\d{5}\.", name)]
        if model and len(firsts) > 1:
            tier = re.compile(r"(?<![A-Za-z0-9])" + re.escape(model) + r"(?![A-Za-z0-9])", re.I)
            matching = [name for name in firsts if tier.search(name)]
            if matching:
                firsts = matching
        if len(firsts) != 1:
            raise ValueError("请使用具体文件的链接。仓库候选文件：" + chr(10).join(["", *firsts[:12]]))
        chosen = firsts[0]
    if chosen not in rows:
        raise ValueError("仓库里没有对应的模型文件。")
    shard = re.fullmatch(r"(.+)-\d{5}-of-(\d{5})\.gguf", chosen, re.I)
    names = [chosen]
    if shard and not auxiliary:
        stem, total = shard.groups()
        if not 1 <= int(total) <= 128:
            raise ValueError("模型分片数量超出支持范围。")
        names = [f"{stem}-{index:05d}-of-{total}.gguf" for index in range(1, int(total) + 1)]
    result = []
    for name in names:
        row = rows.get(name) or {}
        lfs = row.get("lfs") or {}
        sha, size = lfs.get("sha256", ""), lfs.get("size")
        filename = name.rsplit("/", 1)[-1]
        if (not re.fullmatch(r"[a-f0-9]{64}", sha) or not isinstance(size, int) or size <= 0
                or re.search(r'[<>:"\\|?*\x00-\x1f]', filename) or filename.startswith(".")):
            raise ValueError(f"缺少分片、校验信息或文件名无效：{name}")
        result.append({"url": f"https://huggingface.co/{repo}/resolve/{commit}/{quote(name)}",
                       "name": filename, "sha": sha, "size": size})
    return result


def download_manifest(mod, rows, folder):
    pins = {row["url"]: (row["size"], row["sha"]) for row in rows}
    original = mod.file_pin
    mod.file_pin = lambda url: pins.get(url) or original(url)
    try:
        for row in rows:
            dst = folder / row["name"]
            mod.download(row["url"], dst, row["name"])
            mod.verify_sha256(dst, row["size"], row["sha"])
    finally:
        mod.file_pin = original


def take_remote_auxiliaries(mod, args, shared):
    args = list(args)
    if "--launcher-remote-aux" not in args:
        return args
    index = args.index("--launcher-remote-aux")
    remote = json.loads(args[index + 1])
    del args[index:index + 2]
    for role, rows in remote.items():
        if role not in (*AUXILIARY_LABELS, "projection") or len(rows) != 1:
            raise ValueError("辅助模型下载清单无效。")
        row = rows[0]
        folder = Path(shared) / "launcher-auxiliaries" / row["sha"]
        download_manifest(mod, rows, folder)
        path = str(folder / row["name"])
        if role == "projection":
            flag = "--experimental-speed-projection"
            if flag in args:
                args[args.index(flag) + 1] = path
            else:
                args += [flag, path]
        else:
            args += ["--launcher-" + role, path]
    return args


def take_drafting(args):
    args = list(args)
    mode = 'auto'
    while '--launcher-drafting' in args:
        i = args.index('--launcher-drafting')
        if i + 1 >= len(args) or args[i + 1] not in MODES:
            raise ValueError('草稿模式必须为 auto、mtp、dflash2 或 off')
        mode = args[i + 1]
        del args[i:i + 2]
    return args, mode


def take_all_drafters(args):
    args = list(args)
    enabled = '--launcher-all-drafters' in args
    return [arg for arg in args if arg != '--launcher-all-drafters'], enabled


def prepare_setup_args(mod, args, mode):
    """Keep setup as acquisition authority; unsupported old releases fail explicitly."""
    args = list(args)
    for flag in ('--drafter', '--mtp'):
        while flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    if shape_family(mod, args):
        return args
    source = Path(mod.__file__).read_text(encoding='utf-8-sig')
    if mode == 'off':
        if '--mtp' not in source:
            raise ValueError('此旧版本不能保存关闭草稿设置，请更新 Strata。')
        return args + ['--mtp', 'off']
    if mode == 'dflash2':
        if '--drafter' not in source:
            raise ValueError('此版本不支持 DFlash2，请更新 Strata 或选择 MTP。')
        return args + ['--drafter', 'dflash2']
    if mode == 'auto' and '--drafter' in source:
        try:
            from strata_control import drafting
            from strata_cli.drafting_cmd import _hardware
            backend, archs = _hardware()
            family = args[args.index('--family') + 1] if '--family' in args else 'qwen'
            parallel = int(args[args.index('--parallel') + 1]) if '--parallel' in args else 1
            chosen = drafting.choice_of('auto', family=family, backend=backend, archs=archs)
            if parallel > 1:
                chosen = 'mtp'
            return args + ['--drafter', chosen]
        except (ImportError, AttributeError):
            pass
    # Auto leaves an older official setup's default intact. It supplies MTP as a
    # fallback; the post-install official drafting decision prepares DFlash2
    # only when supported, and one drafter is selected at runtime.
    if mode == 'mtp' and '--drafter' in source:
        return args + ['--drafter', 'mtp']
    return args


def shape_family(mod, args):
    family = args[args.index('--family') + 1] if '--family' in args else ''
    return (getattr(mod, 'HY4_FAMILIES', {}).get(family)
            or getattr(mod, 'INKLING_FAMILIES', {}).get(family))


def config_snapshot(root):
    return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in Path(root).glob('strata-*.json') if p.is_file()}


def finish_resources(root, mode, before, all_drafters=False):
    """Only changed setup configs belong to this install; never rewrite other models."""
    root = Path(root)
    changed = [p for p in root.glob('strata-*.json')
               if before.get(str(p)) != (p.stat().st_mtime_ns, p.read_bytes())]
    if not changed:
        raise RuntimeError('安装程序未生成或更新模型配置，无法确认辅助资源已准备。')
    for path in changed:
        cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(cfg, dict) or not isinstance(cfg.get('args'), list) or not cfg.get('exe'):
            continue
        cfg['drafting'] = mode
        if mode == 'off':
            cfg['mtp'] = False
        elif cfg.get('mtp') is False:
            cfg.pop('mtp')
        if mode in ('auto', 'dflash2') or all_drafters:
            try:
                from strata_control import drafting, drafting_fetch
            except ImportError:
                if mode == 'dflash2':
                    raise RuntimeError('此版本缺少 DFlash2 下载接口。')
            else:
                family = (cfg.get('local_import') or {}).get('family')
                data = drafting.data_dir_of(cfg['args'])
                if all_drafters and (data is None or not drafting.supported_model(cfg.get('model_name'), family)):
                    print('[辅助资源] 此模型不适用独立 DFlash2，保留其官方内置草稿资源。', flush=True)
                if data is not None and drafting.supported_model(cfg.get('model_name'), family):
                    # The official resolver owns hardware and concurrency rules.
                    from types import SimpleNamespace
                    from strata_cli.drafting_cmd import _inputs
                    wanted = _inputs(SimpleNamespace(config=str(path), data_dir=str(data), mode='dflash2' if all_drafters else mode,
                                                      family=family, no_split=False, engines_unloading=False))
                    if drafting.needs_pack(**wanted):
                        print('[辅助资源] 准备 DFlash2（固定 revision、校验并转换）', flush=True)
                        drafting_fetch.ensure(data, progress=lambda step, done, total:
                            print(f'[辅助资源] {step} {done}/{total}', flush=True))
                    elif all_drafters and not drafting.pack_ready(data):
                        print('[辅助资源] 此模型或硬件不适用 DFlash2，已跳过；运行草稿选择保持不变。', flush=True)
                    elif mode == 'dflash2' and not drafting.pack_ready(data):
                        raise RuntimeError('此模型或引擎不支持 DFlash2，请选择自动或 MTP。')
        if all_drafters:
            cfg['launcher_all_drafters'] = True
        tmp = path.with_name(path.name + '.drafting.tmp')
        tmp.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding='utf-8')
        os.replace(tmp, path)
        print(f'[辅助资源] 已保存草稿模式 {mode}：{path.name}', flush=True)


def main(argv=None):
    root, *args = list(sys.argv[1:] if argv is None else argv)
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    args, mode = take_drafting(args)
    args, all_drafters = take_all_drafters(args)
    spec = importlib.util.spec_from_file_location('strata_install_setup', root / 'setup.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if "--launcher-remote-aux" in args:
        shared = args[args.index("--data-dir") + 1]
        args = take_remote_auxiliaries(mod, args, shared)
    args, selected = take_auxiliaries(args)
    kv = 'int8'
    if '--launcher-inkling-kv' in args:
        index = args.index('--launcher-inkling-kv')
        kv = args[index + 1]
        del args[index:index + 2]
    shared = args[args.index('--data-dir') + 1] if '--data-dir' in args else None
    if shared and '--family' in args and args[args.index('--family') + 1] == 'inkling-2-mini':
        from strata_native_inkling import configure_installer
        configure_installer(mod, args, shared, kv)
        if selected.get('vision') and selected['vision'].suffix.lower() == '.gguf':
            from strata_inkling_vision import convert_projector
            selected['vision'] = convert_projector(selected['vision'], shared)
    configure_auxiliaries(mod, args, selected, drafting=mode)
    native = shape_family(mod, args)
    if native:
        os.environ['STRATA_REQUIRE_SIGNED_COMPONENTS'] = '1'
    args = prepare_setup_args(mod, args, 'mtp' if all_drafters else mode)
    before = config_snapshot(root)
    sys.argv = [str(root / 'setup.py'), *args]
    rc = mod.main()
    if rc not in (None, 0):
        return rc
    if native:
        changed = config_snapshot(root)
        expected = native['arch']
        if not any(before.get(path) != entry and json.loads(entry[1].decode('utf-8-sig')).get('arch') == expected
                   for path, entry in changed.items()):
            raise RuntimeError('官方安装程序未生成对应架构的模型配置。')
    else:
        finish_resources(root, mode, before, all_drafters)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('模型与辅助资源准备失败：' + str(exc), flush=True)
        sys.exit(1)

