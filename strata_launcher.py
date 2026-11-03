# -*- coding: utf-8 -*-
"""
Strata 中文图形启动器
=====================

把 Strata（Qwen3.8-Flash-Next 推理引擎，仓库 Niko1221/Strata）的 setup.py 安装与启动流程
完整集成进一个图形界面，用户不需要单独安装 Python 环境。

两种模式（启动时自动判断，也可手动切换）：
  · 已有模型  —— 只做启动 / 停止 / 监控 / 配置 / 维护，界面上不出现任何下载或提问。
  · 全新安装  —— 分步向导：环境准备 → 硬件体检 → 选家族 → 选规格 → 选上下文 → 图片/KV/实验
                → 端口与局域网 → 下载确认 → 实时进度 → 安装完成后可选直接启动。

Strata 运行环境自持（默认全程不询问目录，也可手动指定）：
  1) 用户手动指定的 Strata 目录（在“网络与配置 → 目录设置”里选择）→ 优先使用
  2) 启动器所在目录（或同级 Strata\\ 子目录）里已有 setup.py → 直接使用
  3) 开发调试目录 .strata-src → 直接使用
  4) 打包内置的 Strata-main.zip → 自动释放到 启动器目录\\Strata
  5) 都没有 → 联网下载 main.zip 解压到 启动器目录\\Strata

版本更新：可从 GitHub 检查最新版本，一键更新源码（git pull 或重下 main.zip）并执行 setup.py --update；
也可勾选“启动时自动检查并更新”。

用户可手动指定的是“Strata 目录”与“模型数据目录”（默认均为自动）。
"""

import os
import re
import sys
import json
import time
import queue
import shutil
import socket
import zipfile
import ctypes
import threading
import tempfile
import subprocess
import urllib.request
import importlib.util
from pathlib import Path
from strata_download import download_source
from strata_versions import (VersionStore, normalize_release, pinned_feed, extract_source,
                             copy_configs, check_exact_engines, source_engine_version, validate_config_flags)
from strata_catalog import official_tables, family_models, bundled_tables, adapt_install_args
from strata_update import installed_release, record_release, verify_engines, SOURCE_MARKER, launcher_update_lock
from strata_chat_template import apply_chat_template, validate_template
from strata_local_import import inspect_model, apply_runtime_vision, projection_setup_args, find_huihui_vector
from strata_optimization import apply_runtime_optimization, model_settings
from strata_model_cleanup import build_cleanup_plan, execute_cleanup_plan
from strata_responsive import initial_geometry, layout_mode, monitor_workareas, restore_geometry
from strata_launch_support import (http_probe, port_in_use, port_owner,
    kill_pid_force, is_protected_process, resolve_launch_config, prepare_engine_paths,
    api_model_name, connection_fields, configured_lan_ipv4, local_context_choices, save_local_context,
    strata_health_ready)
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# --------------------------------------------------------------------------- 依赖自举
REQUIRED_PACKAGES = {"customtkinter": "customtkinter", "jinja2": "jinja2"}


def _auto_install_dependencies():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            if importlib.util.find_spec(module_name) is None:
                missing.append(package_name)
        except (ImportError, ValueError):
            missing.append(package_name)
    if not missing:
        return
    print("=" * 56)
    print("检测到缺少以下依赖，正在自动安装：")
    for pkg in missing:
        print("   -> " + pkg)
    print("=" * 56)
    pip_cmd = None
    for cmd in ([sys.executable, "-m", "pip", "install"], ["pip", "install"], ["pip3", "install"]):
        try:
            if subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=10).returncode == 0:
                pip_cmd = cmd
                break
        except Exception:
            continue
    if pip_cmd is None:
        print("未找到 pip，请手动安装：pip install " + " ".join(missing))
        sys.exit(1)
    for pkg in missing:
        print(f"\n正在安装 {pkg} ...")
        subprocess.run(pip_cmd + [pkg], timeout=300)
    import site
    site.main()


_auto_install_dependencies()

import customtkinter as ctk  # noqa: E402

# --------------------------------------------------------------------------- 常量
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
LAUNCHER_CONFIG = Path.home() / ".strata_launcher_config.json"
APP_NAME = "AI蛮子 Strata 全模型启动器 V3.5.1"
GITHUB_REPO = "Niko1221/Strata"
GITHUB_LATEST_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_MAIN_ZIP = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"
STRATA_ZIP_URL = GITHUB_MAIN_ZIP
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases"
KNOWN_LATEST = "0.2.6"          # 内置官方发布版本
SCAN_CACHE_VERSION = 5           # 模型清单缓存格式版本：升级后首次启动会自动后台刷新一次
HF_MIRROR = "https://hf-mirror.com"

FONT = "Microsoft YaHei UI"
FONT_MONO = "Consolas"

COLORS = {
    "bg": "#f4f6fb",
    "card": "#ffffff",
    "border": "#dfe3ec",
    "text": "#000000",           # 功能文本：纯黑
    "text_dim": "#5c6270",
    "placeholder": "#a6abb8",    # 占位符：灰色
    "accent": "#2f6fed",
    "accent_hover": "#1d59d0",
    "success": "#12a05a",
    "warning": "#d99100",
    "danger": "#d93b3b",
    "chip": "#eef2fb",
}

# setup.py 的真实常量（导入失败时用静态兜底）
FALLBACK_FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen；GSQ-RCO 量化为 ISTA-DASLab", "about": "原始模型"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI 基于 Qwen3.8-Flash-Next 的微调", "about": "思考更短，回答更快"},
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab 的代码版本", "about": "保留一半专家，约 32GB 内存即可"},
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth 的约 4bit 量化", "about": "质量最高，体积与内存要求也最高"},
}
FALLBACK_MODELS = {
    "Q2_0": {"about": "2bit，最快", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0, "families": ("qwen",)},
    "IQ2_XS": {"about": "2bit i-quant，质量略好、速度接近", "download_gb": 68.0, "ram_gb": 48, "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3bit i-quant，质量更好、更慢", "download_gb": 75.8, "ram_gb": 60, "arena_gb": 42.9},
    "IQ3_S": {"about": "3.5bit i-quant，质量最好、最慢", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen",)},
    "IQ1_M": {"about": "Coder 专用：保留一半专家，适合 32GB 内存；中文及通用能力较弱", "download_gb": 58.4, "ram_gb": 32, "arena_gb": 23.4,
              "families": ("coder",)},
    "UD-Q4_K_XL": {"about": "4bit（Unsloth Dynamic），实验性：质量最好，多数专家从 SSD 读取",
                   "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0, "families": ("unsloth",), "budget": True,
                   "nvidia_only": True, "experimental": True},
    "UD-IQ4_XS": {"about": "约 4bit i-quant，质量介于 IQ3_S 与 UD-Q4_K_XL 之间",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "vision": True},
}
FALLBACK_CONTEXTS = [8192, 32768, 65536, 131072, 262144, 393216, 524288]

VISION_CHOICES = [("关闭（不读图）", "no"), ("GPU 上读图（推荐）", "gpu"), ("CPU 上读图（省显存）", "cpu")]
QUARK_MODEL_URL = "https://pan.quark.cn/s/51ce260152fa"
QUARK_MODEL_CODE = "aB6u"
IQ4_MODEL_URL = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ4_XS"
RUNTIME_KV_LABELS = {'original': '沿用模型配置', 'int8': 'INT8（推荐）', 'q4_0': 'Q4（省内存）',
                     'k8v4': 'K8V4（混合）', 'fp16': 'FP16（高精度）'}
KV_CHOICES = [("int8（默认，推荐）", "int8"), ("q4_0（省一半显存）", "q4_0"), ("k8v4（混合）", "k8v4")]
LOWRAM_CHOICES = [("auto（自动判断）", "auto"), ("on（低内存模式）", "on"), ("off（关闭）", "off"),
                  ("resident（常驻内存）", "resident"), ("mmap（文件映射）", "mmap")]
DRAFT_CHOICES = [("cjk（中/日/韩，默认）", "cjk"), ("en（仅英文与代码，省显存）", "en"),
                 ("cyrillic（西里尔字母）", "cyrillic"), ("fr（法语）", "fr")]
KVSTREAM_CHOICES = [("auto（自动判断）", "auto"), ("on（开启）", "on"), ("off（关闭）", "off")]
ROPE_CHOICES = [("yarn（官方默认）", "yarn"), ("linear（位置插值）", "linear")]
THINKING_LABELS = {
    "high": "深入思考",
    "medium": "均衡思考",
    "low": "快速回答",
}
RUNTIME_MODE_LABELS = {
    "single": "单任务优先（推荐）",
    "concurrent": "多任务同时运行",
    "session": "长对话快速恢复（高级）",
}
ACCELERATION_LABELS = {
    "original": "自动（推荐）",
    "4": "标准加速",
    "6": "增强加速（实验）",
}
THINKING_BUDGET_LABELS = {
    0: "由模型自动决定（推荐）",
    4096: "较短 · 4096",
    8192: "标准 · 8192",
    16384: "较长 · 16384",
}


def optimization_from_labels(effort_label, mode_label, budget_label, acceleration_label, profile):
    """Translate beginner-facing labels to stable Strata configuration values."""
    effort_by_label = {label: key for key, label in THINKING_LABELS.items()}
    mode_by_label = {label: key for key, label in RUNTIME_MODE_LABELS.items()}
    budget_by_label = {label: value for value, label in THINKING_BUDGET_LABELS.items()}
    acceleration_by_label = {label: key for key, label in ACCELERATION_LABELS.items()}
    if budget_label in budget_by_label:
        budget = budget_by_label[budget_label]
    else:
        budget = int(budget_label)
    return {
        'effort': effort_by_label[effort_label],
        'mode': mode_by_label[mode_label],
        'budget': budget,
        'spec': acceleration_by_label[acceleration_label],
        'profile': bool(profile),
    }

STEP1_DONE = "环境准备"


# --------------------------------------------------------------------------- 工具
def log_line(text):
    try:
        if sys.stdout is not None:
            print(text, flush=True)
    except (OSError, ValueError):
        pass
    try:
        path = Path.home() / '.strata_launcher.log'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(time.strftime('%Y-%m-%d %H:%M:%S ') + str(text) + '\n')
    except OSError:
        pass


def format_file_size(size):
    value = float(max(0, size or 0))
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f'{value:.0f} {unit}' if unit in ('B', 'KB') else f'{value:.2f} {unit}'
        value /= 1024


class AppConfig:
    """启动器自己的持久化配置（与 Strata 的 settings.json 相互独立）。"""

    DEFAULTS = {
        "mode": "auto",                 # auto / installed / fresh
        "strata_dir": "",               # 手动指定的 Strata 目录（空 = 自动定位）
        "data_dir": "",                 # 模型数据目录（空 = 不指定；安装时必填）
        "auto_update": False,           # 启动时自动检查并更新 Strata
        "version_pinned": True,
        "preferred_release": "",
        "managed_version_active": False,
        "last_remote_version": "",      # 上次检查到的最新版本
        "last_auto_updated_version": "",  # 上次已成功更新到的版本（避免重复自动下载）
        "last_update_check": "",        # 上次检查时间
        "use_hf_mirror": False,         # 是否使用 hf-mirror.com 镜像
        "open_browser": True,           # 启动模型后自动打开浏览器
        "last_family": "qwen",
        "last_model": "IQ2_XS",
        "last_context": 32768,
        "cached_models": None,          # 已安装模型清单缓存（None = 从未扫描过）
        "cached_scan_version": 0,       # 缓存对应的扫描版本（低于 SCAN_CACHE_VERSION 时启动后自动刷新）
        "last_host": "127.0.0.1",
        "last_api_key": "",
        "last_port": 8080,
        "last_lan": False,
        "install_vision": "gpu",
        "runtime_vision_enabled": True,
        "model_optimizations": {},
        "install_kv": "int8",
        "install_esp": False,
        "install_esp_path": "",
        "install_hf_mirror": False,
        "install_gpu": "",
        "install_lowram": "auto",
        "install_ram_budget": "",
        "install_vram_reserve": "500",
        "install_parallel": "",
        "install_draft_vocab": "cjk",
        "install_kv_streaming": "auto",
        "install_advanced_open": False,
        "context_compress_enabled": True,   # 自动接入独立摘要服务；保留用户关闭选择
        "fit_max_tokens": True,             # 输出预算超限时适配剩余额度，独立于摘要压缩
        "window_geometry": "",
        "window_maximized": False,
    }

    def __init__(self):
        self.data = dict(self.DEFAULTS)
        try:
            if LAUNCHER_CONFIG.is_file():
                self.data.update(json.loads(LAUNCHER_CONFIG.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        self.data.pop("compress_port", None)

    def get(self, key, default=None):
        return self.data.get(key, self.DEFAULTS.get(key, default))

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def update(self, **kw):
        self.data.update(kw)
        self.save()

    def save(self):
        try:
            LAUNCHER_CONFIG.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except OSError:
            pass


def find_system_python():
    """找一个可用的系统 Python（3.10+，64 位）。exe 模式下 sys.executable 是启动器本身，不可用。"""
    if not getattr(sys, "frozen", False):
        return sys.executable
    candidates = []
    for launcher in ("py", "python"):
        try:
            r = subprocess.run([launcher, "-c", "import sys;print(sys.executable)"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                candidates.append(r.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            continue
    for ver in ("313", "312", "311", "310"):
        p = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / f"Python{ver}" / "python.exe"
        if p.is_file():
            candidates.append(str(p))
    for c in candidates:
        try:
            r = subprocess.run([c, "-c", "import sys;print(sys.version_info >= (3,10) and sys.maxsize > 2**32)"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and "True" in r.stdout:
                return c
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def dir_free_gb(path):
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return 0.0


def pick_largest_free_drive():
    """剩余空间最大的盘上创建 Strata-data。"""
    best, best_free = None, -1.0
    for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(drive + ":\\")
        if root.is_dir():
            free = dir_free_gb(root)
            if free > best_free:
                best, best_free = root / "Strata-data", free
    return best or (APP_DIR / "Strata-data")


def scan_disk_roots(max_depth=2):
    """扫描本地磁盘，返回 (源码目录候选, 模型数据目录候选)。

    源码目录：含 setup.py，且同时含 serve/ 或 tools/ 或 CMakeLists.txt。
    数据目录：名字为 Strata-data，或内部含 models/packs/mtp 子目录。
    只下探 max_depth 层，跳过系统目录，避免全盘扫描过慢。
    """
    sources, datas = [], []
    seen = set()
    skip = {"windows", "$recycle.bin", "system volume information",
            "program files", "program files (x86)", "programdata", "perflogs",
            "node_modules", ".git", "__pycache__", "venv", ".venv", "site-packages"}

    def look(path, depth):
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        try:
            if not path.is_dir():
                return
        except OSError:
            return
        try:
            if (path / "setup.py").is_file() and (
                    (path / "serve").is_dir() or (path / "tools").is_dir()
                    or (path / "CMakeLists.txt").is_file()):
                sources.append(str(path))
                return
        except OSError:
            pass
        try:
            name = path.name.lower()
            is_data = name == "strata-data" or (
                (path / "packs").is_dir()
                and ((path / "mtp").is_dir() or (path / "models").is_dir()))
            if is_data:
                datas.append(str(path))
                return
        except OSError:
            pass
        if depth <= 0:
            return
        try:
            children = [c for c in path.iterdir() if c.is_dir()]
        except OSError:
            return
        for child in children:
            if child.name.lower() in skip:
                continue
            look(child, depth - 1)

    for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(drive + ":\\")
        try:
            if root.is_dir():
                look(root, max_depth)
        except OSError:
            continue
    return sources, datas


def choose_from_list(parent, title, prompt, items):
    """弹出候选列表窗口，返回用户选中项；取消返回 None。"""
    if not items:
        return None
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry(restore_geometry('760x440+99999+99999', monitor_workareas(parent)))
    win.transient(parent)
    win.grab_set()
    tk.Label(win, text=prompt, justify="left", wraplength=720,
             font=(FONT, 11), fg=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 8))
    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=16)
    bar = tk.Scrollbar(frame, orient="vertical")
    box = tk.Listbox(frame, font=(FONT, 11), activestyle="none", yscrollcommand=bar.set)
    bar.configure(command=box.yview)
    bar.pack(side="right", fill="y")
    box.pack(side="left", fill="both", expand=True)
    for it in items:
        box.insert("end", it)
    box.selection_set(0)
    out = {"value": None}

    def ok(_=None):
        sel = box.curselection()
        if sel:
            out["value"] = box.get(sel[0])
        win.destroy()

    box.bind("<Double-Button-1>", ok)
    brow = tk.Frame(win)
    brow.pack(fill="x", padx=16, pady=12)
    tk.Button(brow, text="使用选中项", width=12, command=ok).pack(side="left")
    tk.Button(brow, text="取消", width=8, command=win.destroy).pack(side="left", padx=8)
    parent.wait_window(win)
    return out["value"]


_LAN_IP_CACHE = {'ips': [], 'updated': 0.0, 'busy': False}


def local_ip_addresses():
    if not _LAN_IP_CACHE['busy'] and time.monotonic() - _LAN_IP_CACHE['updated'] > 30:
        _LAN_IP_CACHE['busy'] = True
        def refresh():
            try:
                _LAN_IP_CACHE['ips'] = configured_lan_ipv4()
            finally:
                _LAN_IP_CACHE['updated'] = time.monotonic()
                _LAN_IP_CACHE['busy'] = False
        threading.Thread(target=refresh, daemon=True, name='strata-lan-ip').start()
    return list(_LAN_IP_CACHE['ips'])


def generate_api_key():
    import secrets
    return "sk-strata-" + secrets.token_hex(16)


def ctx_label(n):
    """上下文数值转可读标签：8192 -> '8K'，131072 -> '128K'。"""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def ctx_value(text):
    """上下文标签转数值：'8K' -> 8192；纯数字原样返回；无法解析返回 None。"""
    t = str(text).strip().upper()
    try:
        if t.endswith("M"):
            return int(round(float(t[:-1]) * 1024 * 1024))
        if t.endswith("K"):
            return int(round(float(t[:-1]) * 1024))
        return int(float(t))
    except ValueError:
        return None


_SYSINFO_CACHE = {}          # 显卡/内存等开机后不变的信息，本次运行内只探测一次


def query_nvidia_smi():
    """返回 [(index, name, vram_gb)]，失败返回 []。结果缓存，避免切换模式时重复调用 nvidia-smi。"""
    if "gpus" not in _SYSINFO_CACHE:
        _SYSINFO_CACHE["gpus"] = _detect_gpus()
    return _SYSINFO_CACHE["gpus"]


def _detect_gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    out.append((int(parts[0]), parts[1], float(parts[2]) / 1024.0))
                except ValueError:
                    continue
        return out
    except (OSError, subprocess.TimeoutExpired):
        return []


def total_ram_gb():
    try:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / (1024 ** 3)
    except Exception:
        return 0.0


def version_tuple(text):
    """'v0.1.39' / '0.1.39' → (0, 1, 39)；无法解析时返回 ()。"""
    nums = re.findall(r"\d+", str(text or ""))
    return tuple(int(n) for n in nums[:4]) if nums else ()


def read_local_version(root):
    """从 Strata 源码的 CMakeLists.txt 读取版本号：project(strata VERSION x.y.z)。"""
    if not root:
        return ""
    try:
        txt = (Path(root) / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"project\(\s*strata\s+VERSION\s+([\d.]+)", txt, re.IGNORECASE)
    return m.group(1) if m else ""


def remote_latest_release(timeout=12):
    """查询 GitHub 最新 release；API 限流时使用发布页的官方重定向。"""
    req = urllib.request.Request(GITHUB_LATEST_API, headers={
        "User-Agent": "StrataLauncher", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        fallback = urllib.request.Request(GITHUB_RELEASES_PAGE + "/latest", headers={"User-Agent": "StrataLauncher"})
        with urllib.request.urlopen(fallback, timeout=timeout) as resp:
            url = resp.geturl()
        match = re.search(r"/releases/tag/(v?\d+(?:\.\d+){2,3})$", url)
        if not match:
            raise RuntimeError("无法解析 GitHub 最新发布标签。")
        data = {"tag_name": match.group(1), "html_url": url}
    return (str(data.get("tag_name") or "").lstrip("vV"),
            data.get("html_url") or GITHUB_RELEASES_PAGE, data.get("published_at") or "")


# --------------------------------------------------------------------------- Strata 运行环境
_setup_import_lock = threading.RLock()


def load_official_setup(root):
    """Load a checkout with its sibling packages, refreshing them after a source update."""
    with _setup_import_lock:
        for name in list(sys.modules):
            if name in ('strata_setup', 'strata_control', 'strata_cli') or name.startswith(('strata_control.', 'strata_cli.')):
                del sys.modules[name]
        previous_path = sys.path[:]
        try:
            sys.path.insert(0, str(Path(root).resolve()))
            importlib.invalidate_caches()
            spec = importlib.util.spec_from_file_location('strata_setup', str(Path(root) / 'setup.py'))
            mod = importlib.util.module_from_spec(spec)
            sys.modules['strata_setup'] = mod
            spec.loader.exec_module(mod)
            return mod
        finally:
            sys.path[:] = previous_path


class StrataEnv:
    """负责 Strata 根目录自定位、内置源码释放、Python 虚拟环境准备。"""

    def __init__(self):
        self.root = None
        self.manual_root = None      # 用户手动指定的 Strata 目录
        self.hint_root = None        # 自动定位线索：如数据目录旁检测到的 Strata 安装
        self._tables = None
        self._python = None
        self._mod = None             # 已加载的官方 setup.py 模块（复用其函数）

    def set_manual_root(self, path):
        """设置手动 Strata 目录；传空则恢复自动定位。"""
        self.manual_root = Path(path) if path else None
        self.root = None
        self._tables = None
        self._python = None
        self._mod = None

    # ---- 根目录定位
    def descend_root(self, path):
        """候选目录本身不含 setup.py 时，向下一层查找源码根。

        常见情况：解压后多套了一层目录，例如
        R:\\Strata-main\\Strata-main\\setup.py（而 R:\\Strata-main 只是外层）。
        """
        if path is None:
            return None
        try:
            base = Path(path)
            if not base.is_dir():
                return None
            children = [c for c in base.iterdir() if c.is_dir()]
        except OSError:
            return None
        pref = {"strata-main": 0, "strata": 1, "strata-src": 2}
        children.sort(key=lambda p: (pref.get(p.name.lower(), 9), p.name.lower()))
        for child in children:
            try:
                if (child / "setup.py").is_file():
                    return child.resolve()
            except OSError:
                continue
        return None

    def find_root(self):
        cands = []
        if self.manual_root:
            cands.append(self.manual_root)
        elif self.hint_root:
            cands.append(self.hint_root)   # 自动定位：优先数据目录旁已有的 Strata 安装
        cands += [APP_DIR, APP_DIR / "Strata", APP_DIR / ".strata-src", APP_DIR / "Strata-main"]
        for cand in cands:
            if cand is None:
                continue
            try:
                if (cand / "setup.py").is_file():
                    return cand.resolve()
            except OSError:
                pass
            sub = self.descend_root(cand)      # 多套一层目录时向下探测
            if sub is not None:
                return sub
        return None

    def bundled_zip(self):
        bases = [APP_DIR]
        meipass = getattr(sys, "_MEIPASS", None)   # PyInstaller 单文件模式：内置资源在临时目录
        if meipass:
            bases.append(Path(meipass))
        for base in bases:
            for name in ("Strata-main.zip", "main.zip", "strata-src.zip"):
                p = base / name
                if p.is_file():
                    return p
        return None

    def extract_zip(self, zip_path, dest, on_line=lambda s: None):
        tmp = dest.parent / (dest.name + ".extracting")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            on_line(f"解压 {zip_path.name}（{len(names)} 个文件）...")
            zf.extractall(tmp)
        entries = [p for p in tmp.iterdir()]
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(src), str(dest))
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        return dest

    def download(self, url, dest_file, on_line=lambda s: None):
        return download_source(url, dest_file, on_line)

    def ensure_root(self, on_line=lambda s: None):
        """返回可用的 Strata 根目录，全程不询问用户。"""
        if self.root and (self.root / "setup.py").is_file():
            return self.root
        found = self.find_root()
        if found:
            self.root = found
            if self.manual_root and found != self.manual_root.resolve():
                on_line(f"[提示] 手动指定的 Strata 目录不可用，已改用：{found}")
            return self.root
        if self.manual_root:
            on_line(f"[提示] 手动指定的 Strata 目录无效（缺少 setup.py）：{self.manual_root}")
        dest = APP_DIR / "Strata"
        z = self.bundled_zip()
        if z:
            on_line(f"未发现 Strata 环境，正在从内置包释放到 {dest} ...")
            self.root = self.extract_zip(z, dest, on_line)
            return self.root
        on_line("未发现 Strata 环境，正在联网获取官方源码（main.zip）...")
        tmp_zip = APP_DIR / "Strata-main.zip.download"
        try:
            self.download(STRATA_ZIP_URL, tmp_zip, on_line)
            self.root = self.extract_zip(tmp_zip, dest, on_line)
        finally:
            try:
                tmp_zip.unlink()
            except OSError:
                pass
        return self.root

    def source_version(self):
        """本地源码版本号（读 CMakeLists.txt，读不到返回空串）。"""
        return read_local_version(self.root or self.find_root())

    def update_from_zip(self, on_line=lambda s: None, version=None):
        """下载指定发布标签，发布版本与 CMake 内部引擎版本独立。"""
        if not self.root or not (self.root / "setup.py").is_file():
            raise RuntimeError("Strata 目录尚未就绪，无法更新。")
        version = version or remote_latest_release()[0]
        if not re.fullmatch(r"\d+(?:\.\d+){2,3}", version):
            raise RuntimeError(f"无效发布版本：{version}")
        if installed_release(self.root, SOURCE_MARKER) == version:
            on_line(f"发布 v{version} 源码已就绪且摘要一致，直接重试引擎更新。")
            return self.root
        archive_url = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/v{version}.zip"
        work = Path(tempfile.mkdtemp(prefix=".strata-update-", dir=APP_DIR)).resolve()
        tmp_zip = work / "source.zip"
        tmp_dir = work / "source"
        try:
            self.download(archive_url, tmp_zip, on_line)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_zip) as zf:
                # Harness end-to-end scripts are development tools, not runtime dependencies.
                # They can be unavailable in Windows temporary folders; do not let them block engine updates.
                members = [name for name in zf.namelist()
                           if not ("/integrations/harness/" in "/" + name and name.endswith("_e2e.py"))]
                skipped = len(zf.namelist()) - len(members)
                if skipped:
                    on_line(f"跳过 {skipped} 个 Harness 开发端到端测试脚本，保留全部运行代码。")
                zf.extractall(tmp_dir, members=members)
            entries = [p for p in tmp_dir.iterdir()]
            src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_dir
            if not (src / "setup.py").is_file() or not read_local_version(src):
                raise RuntimeError("发布压缩包缺少 setup.py 或有效 CMake 版本，已中止覆盖。")
            with launcher_update_lock(self.root):
                on_line(f"正在覆盖发布 v{version} 源码到 {self.root} ...")
                shutil.copytree(str(src), str(self.root), dirs_exist_ok=True)
                (self.root / ".strata-launcher-release.json").unlink(missing_ok=True)
                record_release(self.root, version, SOURCE_MARKER)
        finally:
            if work.parent == APP_DIR.resolve() and work.name.startswith(".strata-update-"):
                shutil.rmtree(work, ignore_errors=True)
        self._tables = None
        self._mod = None
        on_line(f"发布 v{version} 源码已更新，等待引擎校验。")
        return self.root

    # ---- 读取 setup.py 的模型表
    def tables(self):
        if self._tables is not None:
            return self._tables
        models, families, contexts = dict(FALLBACK_MODELS), dict(FALLBACK_FAMILIES), list(FALLBACK_CONTEXTS)
        try:
            models, families, contexts = bundled_tables()
        except (OSError, ValueError, KeyError):
            pass
        root = self.root or self.find_root()
        if root is not None:
            self.root = root
            self._mod = None
            try:
                mod = load_official_setup(root)    # setup.py 仅在 __main__ 下执行 main()
                models, families, contexts = official_tables(mod)
                self._mod = mod
            except Exception as e:                 # noqa: BLE001
                log_line(f"[warn] 读取 setup.py 常量失败，使用内置表：{e}")
        self._tables = (models, families, contexts)
        return self._tables

    def config_choices(self, path):
        """复用官方 setup.py 的 choices_from_config，读出某个已安装配置的家族/规格/上下文/KV。"""
        try:
            local = json.loads(Path(path).read_text(encoding="utf-8-sig")).get("local_import")
            if local:
                return local_context_choices(json.loads(Path(path).read_text(encoding="utf-8-sig")))
            self.tables()                          # 确保官方模块已装载
            if self._mod is None:
                return None
            return self._mod.choices_from_config(Path(path))
        except Exception as e:                     # noqa: BLE001
            log_line(f"[warn] 读取配置失败：{e}")
            return None

    def models_for(self, family):
        models, _, _ = self.tables()
        return family_models(models, family)

    # ---- Python 环境
    def venv_python(self):
        return self.root / ".venv" / "Scripts" / "python.exe"

    def python(self):
        if self._python:
            return self._python
        vp = self.venv_python()
        if vp.is_file():
            self._python = str(vp)
        else:
            self._python = find_system_python()
        return self._python

    def ensure_venv(self, on_line=lambda s: None):
        """创建 Strata 私有虚拟环境（若已存在则直接使用）。"""
        if self.venv_python().is_file():
            self._python = str(self.venv_python())
            return self._python
        base = find_system_python()
        if not base:
            raise RuntimeError("未找到 Python 3.10+（64 位）。请先安装 Python 3.12：https://www.python.org/downloads/ "
                               "（安装时勾选 Add python.exe to PATH），然后重新打开本启动器。")
        on_line(f"正在创建 Strata 独立运行环境（{self.venv_python()}）...")
        r = subprocess.run([base, "-m", "venv", str(self.root / ".venv")],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not self.venv_python().is_file():
            raise RuntimeError("创建虚拟环境失败：\n" + (r.stderr or r.stdout or "")[-1500:])
        self._python = str(self.venv_python())
        on_line("运行环境创建完成。")
        return self._python

    # ---- 已安装模型检测
    def installed_configs(self):
        out = []
        if self.root is None:
            return out
        try:
            for p in sorted(self.root.glob("strata-*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    cfg = json.loads(p.read_text(encoding="utf-8-sig"))
                    if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
                        out.append(p)
                except (OSError, ValueError):
                    continue
        except OSError:
            pass
        return out

    def data_dir(self):
        """模型数据目录：优先 Strata 官方记录，其次源码根及其父目录下的 Strata-data。

        官方记录位置：%APPDATA%/Strata/settings.json。
        未记录时探测 R:\\Strata-main\\Strata-data 这类与源码根同级的目录。
        """
        p = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Strata" / "settings.json"
        try:
            return Path(json.loads(p.read_text(encoding="utf-8"))["data_dir"])
        except (OSError, ValueError, KeyError):
            pass
        bases = []
        if self.root is not None:
            bases += [self.root, self.root.parent]
        for base in bases:
            cand = base / "Strata-data"
            try:
                if cand.is_dir():
                    return cand
            except OSError:
                continue
        return None

    def models_present(self, data_dir):
        for base in [d for d in (data_dir, self.root / "Strata-data", self.root) if d]:
            for sub in ("models", "packs", "mtp"):
                d = base / sub
                try:
                    if d.is_dir() and any(d.iterdir()):
                        return True
                except OSError:
                    continue
        return False

    def has_model(self):
        if self.installed_configs():
            return True
        return self.models_present(self.data_dir())


# --------------------------------------------------------------------------- 进程/服务工具
class ProcRunner:
    """后台跑一条命令，逐行回调；可取消。"""

    def __init__(self):
        self.proc = None

    def run(self, cmd, cwd, on_line, on_done, env=None):
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        creation = 0
        if os.name == "nt":
            creation = subprocess.CREATE_NO_WINDOW if getattr(sys, "frozen", False) else 0

        def worker():
            try:
                self.proc = subprocess.Popen(
                    [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=full_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, creationflags=creation)
            except Exception as e:                 # noqa: BLE001
                on_done(-1, str(e))
                return
            try:
                for line in self.proc.stdout:
                    on_line(line.rstrip("\r\n"))
            except Exception:                      # noqa: BLE001
                pass
            rc = self.proc.wait()
            on_done(rc, None)

        threading.Thread(target=worker, daemon=True).start()

    def cancel(self):
        p = self.proc
        if p and p.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    p.terminate()
            except Exception:                      # noqa: BLE001
                pass

    def running(self):
        return self.proc is not None and self.proc.poll() is None


class ServerProcess:
    """Native Strata backend behind the independent context service."""
    def __init__(self):
        self.proc = None
        self.log_queue = queue.Queue()
        self.port = None
        self.host = "127.0.0.1"
        self.cfg_path = None
        self.gateway = None

    def local_url(self):
        host = '127.0.0.1' if self.host in ('0.0.0.0', '::', '', 'localhost') else self.host
        if ':' in host and not host.startswith('['):
            host = '[' + host + ']'
        return f'http://{host}:{self.port or 8080}'

    def set_context_policy(self, policy):
        if not self.running() or self.gateway is None:
            raise RuntimeError('独立摘要服务未运行')
        return self.gateway.update_policy(policy)

    def start(self, env: StrataEnv, cfg_path, open_browser=True, vision_enabled=True,
              optimization=None, data_dir=None, chat_template=None):
        import tempfile
        from strata_gateway import ContextGateway
        from strata_migration import restore_legacy_patch
        self.log_queue = queue.Queue()
        logs = self.log_queue
        restore_legacy_patch(env.root, logs.put)
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8-sig"))
        self.cfg_path = Path(cfg_path).resolve()
        self.port = int(cfg.get('port') or 8080)
        self.host = cfg.get('host') or '127.0.0.1'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            backend_port = sock.getsockname()[1]
        temp = tempfile.TemporaryDirectory(prefix='strata-service-')
        runtime = Path(temp.name) / self.cfg_path.name
        resolved = resolve_launch_config(cfg, env.root)
        resolved = apply_runtime_vision(resolved, env.root, bool(vision_enabled))
        try:
            resolved = apply_chat_template(resolved, chat_template, temp.name)
        except Exception:
            temp.cleanup()
            raise
        if chat_template and chat_template.get('enabled'):
            logs.put('[template] 已启用自定义聊天模板；不保证消除低比特量化导致的重复。')
        optimization = optimization or {}
        resolved, optimization_notes = apply_runtime_optimization(
            resolved, optimization, self.cfg_path, data_dir)
        if (env.root / '.strata-launcher-release.json').is_file() and env.root.parent.name == '.strata-versions':
            try:
                validate_config_flags(env.root, {self.cfg_path: resolved})
            except Exception:
                temp.cleanup()
                raise
        logs.put('[optimization] ' + '；'.join(optimization_notes))
        if vision_enabled and not resolved.get('vision'):
            logs.put('[vision] 已开启视觉模型，但当前配置缺少视觉引擎或 mmproj；本次以文字模式启动。')
        elif vision_enabled:
            logs.put('[vision] 视觉模型已开启，引擎将使用 --vision 启动。')
        else:
            logs.put('[vision] 视觉模型已关闭，本次以纯文字模式启动。')
        native = prepare_engine_paths(resolved)
        if cfg.get('local_import'):
            native['model_name'] = api_model_name(cfg)
        self.active_config = dict(native)
        if native.get('args') != cfg.get('args'):
            logs.put('[path] 已转换为引擎兼容路径；原模型文件和配置保留。')
        native.update(host='127.0.0.1', port=backend_port, open_browser=False,
                      context_compress_enabled=False, fit_max_tokens=False)
        runtime.write_text(json.dumps(native, ensure_ascii=False, indent=1), encoding='utf-8')
        settings = self.cfg_path.with_suffix('.shared-settings.json')
        shared = {}
        if settings.exists():
            try:
                shared = json.loads(settings.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                logs.put('[optimization] 原共享设置无法解析，本次使用思考强度预设。')
        shared.update(model_settings(optimization))
        runtime.with_suffix('.shared-settings.json').write_text(
            json.dumps(shared, ensure_ascii=False, indent=1), encoding='utf-8')
        full_env = os.environ.copy()
        for k, v in (native.get('env') or {}).items():
            full_env[str(k)] = str(v)
        self.active_api_key = full_env.get('STRATA_API_KEY') or cfg.get('api_key', '')
        gateway = None
        proc = None
        try:
            gateway = ContextGateway(self.host, self.port, backend_port,
                {k: bool(cfg.get(k)) for k in ('context_compress_enabled', 'fit_max_tokens')},
                api_key=full_env.get('STRATA_API_KEY') or cfg.get('api_key', ''),
                cfg_path=self.cfg_path, runtime_config=runtime, log=logs.put,
                allowed_hosts=cfg.get('allowed_hosts') if isinstance(cfg.get('allowed_hosts'), list) else [],
                trusted_origins=cfg.get('trusted_origins') if isinstance(cfg.get('trusted_origins'), list) else [],
                summary_timeout=cfg.get('context_summary_timeout_seconds', 600),
                 web_root=env.root / 'serve' / 'web',
                 thinking_default=model_settings(optimization)['reasoning_effort'])
            cmd = [env.python(), str(env.root / 'serve/server.py'), '--engine', 'strata',
                   '--config', str(runtime), '--host', '127.0.0.1', '--port', str(backend_port)]
            creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(env.root), env=full_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', bufsize=1, creationflags=creation)
            self.proc, self.gateway = proc, gateway
            gateway.start()
        except Exception:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
                proc.stdout.close()
            if gateway:
                gateway.stop()
            temp.cleanup()
            raise
        logs.put(f'[context] 独立摘要服务 {self.host}:{self.port} → 本机 Strata:{backend_port}')
        def pump():
            try:
                for line in proc.stdout:
                    logs.put(line.rstrip('\r\n'))
            finally:
                exit_code = proc.wait()
                logs.put(f'[server] 模型服务已退出，返回码 {exit_code}')
                proc.stdout.close()
                gateway.stop()
                temp.cleanup()
                logs.put('__EXIT__')
        self._pump_thread = threading.Thread(target=pump, daemon=True)
        self._pump_thread.start()
        if open_browser and cfg.get('open_browser') is not False:
            url = self.local_url()
            def browser_ready():
                wrong_service = 0
                for _ in range(300):
                    if proc.poll() is not None:
                        return
                    status, body = http_probe(url + '/health')
                    if strata_health_ready(status, body):
                        import webbrowser
                        webbrowser.open(url + '/?strata-ui=' + str(time.time_ns()))
                        return
                    if status == 200:
                        wrong_service += 1
                        if wrong_service >= 3:
                            logs.put('[web] 本机地址返回的不是 Strata 服务，已停止自动打开浏览器；请检查端口占用。')
                            return
                    threading.Event().wait(1)
            threading.Thread(target=browser_ready, daemon=True).start()
        return cmd

    def stop(self):
        p = self.proc
        if p and p.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                p.terminate()
        self.proc = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None


# --------------------------------------------------------------------------- 界面辅助
def make_card(parent, title=None):
    card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12,
                        border_width=1, border_color=COLORS["border"])
    if title:
        ctk.CTkLabel(card, text=title, text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=14, weight="bold")).pack(anchor="w", padx=16, pady=(12, 4))
    return card


def make_field(parent, label, hint=None):
    """返回 (行容器, 内容容器)。"""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    ctk.CTkLabel(row, text=label, text_color=COLORS["text"],
                 font=ctk.CTkFont(family=FONT, size=12, weight="bold")).pack(anchor="w")
    if hint:
        ctk.CTkLabel(row, text=hint, text_color=COLORS["text_dim"], justify="left",
                     font=ctk.CTkFont(family=FONT, size=11), wraplength=760).pack(anchor="w", pady=(1, 4))
    inner = ctk.CTkFrame(row, fg_color="transparent")
    inner.pack(fill="x")
    return row, inner


class LogPane(ctk.CTkFrame):
    """带滚动条的日志显示区。"""

    def __init__(self, parent, height=220):
        super().__init__(parent, fg_color=COLORS["card"], corner_radius=12, border_width=1,
                         border_color=COLORS["border"])
        self.box = ctk.CTkTextbox(self, height=height, fg_color="#0f172a", text_color="#e6edf6",
                                  font=ctk.CTkFont(family=FONT_MONO, size=11), wrap="word")
        self.box.pack(fill="both", expand=True, padx=8, pady=8)
        self.box.configure(state="disabled")

    def append(self, text):
        self.box.configure(state="normal")
        self.box.insert("end", text + "\n")
        self.box.see("end")
        self.box.configure(state="disabled")

    def clear(self):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.configure(state="disabled")


# --------------------------------------------------------------------------- 主窗口
class StrataLauncher(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.cfg = AppConfig()
        self.env = StrataEnv()
        self.env.set_manual_root(self.cfg.get("strata_dir"))
        self.server = ServerProcess()
        self.runner = ProcRunner()
        self.ui_queue = queue.Queue()
        self.runtime = {"running": False, "health": None, "last_log": ""}
        self._remote_version = self.cfg.get("last_remote_version") or ""
        self._remote_url = GITHUB_RELEASES_PAGE

        self.title(APP_NAME)
        workareas = monitor_workareas(self)
        self.geometry(restore_geometry(self.cfg.get("window_geometry"), workareas))
        primary = workareas[0]
        self.minsize(min(760, primary[2] - primary[0]), min(560, primary[3] - primary[1]))
        self.configure(fg_color=COLORS["bg"])
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._layout_mode = None
        self._resize_after = None
        self._normal_geometry = self.geometry()

        self._build_header()
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._build_sidebar(self.body)
        self.content = ctk.CTkFrame(self.body, fg_color="transparent")
        self.content.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.pages = {}
        self._build_pages()
        self.bind("<Configure>", self._schedule_responsive_layout, add="+")
        self.after(80, self._apply_initial_window_state)
        self.after(120, self.show_page, "dashboard")
        self.after(150, self._pump_ui_queue)
        self.after(200, self.bootstrap)
        self.after(1500, self._tick)

    # ---------------- 顶部
    def _build_header(self):
        self.header = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=0, height=64)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        self.header_left = ctk.CTkFrame(self.header, fg_color="transparent")
        self.header_left.pack(side="left", padx=18)
        ctk.CTkLabel(self.header_left, text=APP_NAME, text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=19, weight="bold")).pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(self.header_left, text="DeepSeek / Qwen 本地推理 · 一键安装与启动", text_color=COLORS["text_dim"],
                     font=ctk.CTkFont(family=FONT, size=11)).pack(anchor="w")

        self.header_right = ctk.CTkFrame(self.header, fg_color="transparent")
        self.header_right.pack(side="right", padx=18)
        self.mode_label = ctk.CTkLabel(self.header_right, text="模式：检测中…", text_color=COLORS["text"],
                                       font=ctk.CTkFont(family=FONT, size=12, weight="bold"))
        self.mode_label.pack(side="left", padx=(0, 10))
        self.mode_btn = ctk.CTkButton(self.header_right, text="切换模式", width=92, height=30, corner_radius=8,
                                      fg_color=COLORS["chip"], text_color=COLORS["text"],
                                      hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                                      command=self.toggle_mode)
        self.mode_btn.pack(side="left", padx=6)
        self.status_chip = ctk.CTkLabel(self.header_right, text="● 未启动", text_color=COLORS["text_dim"],
                                        font=ctk.CTkFont(family=FONT, size=12, weight="bold"))
        self.status_chip.pack(side="left", padx=(10, 0))

    # ---------------- 侧栏
    def _build_sidebar(self, parent):
        self.sidebar = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12, width=170,
                           border_width=1, border_color=COLORS["border"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.nav_buttons = {}
        items = [("dashboard", "概览"), ("install", "安装 / 模型"), ("network", "网络与配置"),
                 ("tools", "维护工具"), ("help", "使用帮助")]
        for key, text in items:
            btn = ctk.CTkButton(self.sidebar, text=text, anchor="w", height=38, corner_radius=8,
                                fg_color="transparent", text_color=COLORS["text"], hover_color=COLORS["chip"],
                                font=ctk.CTkFont(family=FONT, size=13),
                                command=lambda k=key: self.show_page(k))
            btn.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = btn

    def _apply_initial_window_state(self):
        if self.cfg.get("window_maximized"):
            try:
                self.state("zoomed")
            except tk.TclError:
                pass
        self._apply_responsive_layout(layout_mode(self.winfo_width()))

    def _schedule_responsive_layout(self, _event=None):
        if self.state() == "normal":
            self._normal_geometry = self.geometry()
        if self._resize_after:
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(150, self._responsive_layout_now)

    def _responsive_layout_now(self):
        self._resize_after = None
        self._apply_responsive_layout(layout_mode(self.winfo_width()))

    def _apply_responsive_layout(self, mode):
        if mode == self._layout_mode:
            self._update_wraplengths()
            return
        self._layout_mode = mode
        self.sidebar.pack_forget()
        self.content.pack_forget()
        for button in self.nav_buttons.values():
            button.pack_forget()
        self.header_left.pack_forget()
        self.header_right.pack_forget()
        if mode == "compact":
            self.header.configure(height=104)
            self.header_left.pack(side="top", fill="x", padx=14)
            self.header_right.pack(side="top", fill="x", padx=14, pady=(4, 8))
            self.sidebar.configure(width=1)
            self.sidebar.pack(side="top", fill="x", pady=(0, 8))
            self.sidebar.pack_propagate(True)
            for button in self.nav_buttons.values():
                button.pack(side="left", fill="x", expand=True, padx=2, pady=4)
            self.content.pack(side="top", fill="both", expand=True)
        else:
            self.header.configure(height=64)
            self.header_left.pack(side="left", padx=18)
            self.header_right.pack(side="right", padx=18)
            self.sidebar.configure(width=170)
            self.sidebar.pack(side="left", fill="y")
            self.sidebar.pack_propagate(False)
            for button in self.nav_buttons.values():
                button.pack(fill="x", padx=10, pady=3)
            self.content.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.after_idle(self._update_wraplengths)

    def _update_wraplengths(self):
        available = max(320, self.content.winfo_width() - 64)
        def visit(widget):
            for child in widget.winfo_children():
                try:
                    current = int(float(child.cget("wraplength")))
                    if current > 0:
                        child.configure(wraplength=available)
                except (AttributeError, TypeError, ValueError, tk.TclError):
                    pass
                visit(child)
        visit(self.content)

    def _build_pages(self):
        self.pages["dashboard"] = self._page_dashboard()
        self.pages["install"] = self._page_install()
        self.pages["network"] = self._page_network()
        self.pages["tools"] = self._page_tools()
        self.pages["help"] = self._page_help()

    def show_page(self, key):
        for k, frame in self.pages.items():
            if k == key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        for k, btn in self.nav_buttons.items():
            btn.configure(fg_color=COLORS["accent"] if k == key else "transparent",
                          text_color="#ffffff" if k == key else COLORS["text"])

    # ---------------- 工具方法
    def report_callback_exception(self, exc_type, exc_value, exc_traceback):
        import traceback
        log_line(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        self.toast('操作失败', f'{exc_value}\n详情已保存到用户目录的 .strata_launcher.log', 'error')

    def ui(self, fn, *args):
        self.ui_queue.put((fn, args))

    def _pump_ui_queue(self):
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                try:
                    fn(*args)
                except Exception as e:             # noqa: BLE001
                    self.report_callback_exception(type(e), e, e.__traceback__)
        except queue.Empty:
            pass
        self.after(120, self._pump_ui_queue)

    def toast(self, title, msg, kind="info"):
        if kind == "error":
            messagebox.showerror(title, msg)
        elif kind == "warn":
            messagebox.showwarning(title, msg)
        else:
            messagebox.showinfo(title, msg)

    def set_status(self, text, color):
        self.status_chip.configure(text="● " + text, text_color=color)

    def current_mode(self):
        m = self.cfg.get("mode", "auto")
        if m in ("installed", "fresh"):
            return m
        return "installed" if self.model_cache() else "fresh"      # auto 用缓存判断，不扫描磁盘

    def toggle_mode(self):
        target = "fresh" if self.current_mode() == "installed" else "installed"
        self.cfg.set("mode", target)
        self.apply_mode()

    def apply_mode(self):
        mode = self.current_mode()
        self.mode_label.configure(text="模式：" + ("已有模型" if mode == "installed" else "全新安装"))
        self._refresh_install_page()
        self._refresh_dashboard()

    # ---------------- 已安装模型缓存（切换模式时不做任何磁盘扫描）
    def model_cache(self):
        v = self.cfg.get("cached_models")
        return v if isinstance(v, list) else []

    def _data_dir_models(self):
        """从「模型数据目录」识别已下载的模型（依据 packs/<规格> 目录，忽略大小写）。

        返回 [{family, model, dir, label}]，用于让概览页也能列出「已下载但未配置」的模型。
        """
        out = []
        dd = self.effective_data_dir()
        if not dd:
            return out
        models, families, _ctx = self.env.tables()
        packs = Path(dd) / "packs"
        try:
            dirs = [p for p in packs.iterdir() if p.is_dir()] if packs.is_dir() else []
        except OSError:
            return out
        for p in sorted(dirs, key=lambda x: x.name.lower()):
            own = next(((family, name) for family, fd in families.items() if fd.get('own_path')
                        for name in self.env.models_for(family)
                        if p.name.lower() == (fd.get('tag', '') + name).lower()), None)
            if own:
                fam, spec = own
                out.append({'family': fam, 'model': spec, 'dir': str(p), 'label': p.name})
                continue
            spec = next((k for k in models if k.lower() == p.name.lower()), None)
            if spec is None:
                continue
            fams = models.get(spec, {}).get("families") or ('qwen', 'swift')
            fam = "qwen" if "qwen" in fams else (fams[0] if fams else "qwen")
            fname = families.get(fam, {}).get("name") or "strata"
            out.append({"family": fam, "model": spec, "dir": str(p),
                        "label": f"{fname}-{spec.lower()}"})
        return out

    def _scan_models(self):
        """真正扫描磁盘：已安装配置 + 数据目录中已下载但未配置的模型。仅在用户显式操作时调用。"""
        out = []
        seen = set()
        for c in self.env.installed_configs():
            label = None
            scanned = {}
            try:
                scanned = json.loads(c.read_text(encoding="utf-8-sig"))
                label = (scanned.get("local_import") or {}).get("display_name") or scanned.get("model_name")
            except (OSError, ValueError):
                pass
            ch = self.env.config_choices(c) or {}
            if ch.get('model') and not scanned.get('local_import'):
                label = f"{label or c.stem} · {ch['model']}"
            seen.add((ch.get("family"), ch.get("model")))
            out.append({"label": label or c.stem, "path": str(c),
                        "family": ch.get("family"), "model": ch.get("model"),
                        "context": ch.get("context"), "kv": ch.get("kv")})
        for ent in self._data_dir_models():
            if (ent["family"], ent["model"]) in seen:
                continue                      # 已有配置，不重复列出
            out.append({"label": ent["label"] + "（未配置）", "path": "",
                        "family": ent["family"], "model": ent["model"],
                        "context": None, "kv": None})
        self.cfg.set("cached_models", out)
        self.cfg.set("cached_scan_version", SCAN_CACHE_VERSION)
        return out

    def rescan_models(self, note=None):
        """用户显式要求重新扫描（点「使用自动」「保存」「扫描硬盘」或「刷新」时调用）。"""
        try:
            self._scan_models()
        except Exception as e:                     # noqa: BLE001
            self.toast("扫描失败", str(e), "error")
            return
        self.apply_mode()
        self._refresh_dir_fields()
        n = len(self.model_cache())
        if note and hasattr(self, "dir_status"):
            self.dir_status.configure(text=f"{note}　已扫描到 {n} 个模型。", text_color=COLORS["success"])
        self.dash_log.append(f"已扫描模型：{n} 个。")

    # ---------------- 启动引导
    def bootstrap(self):
        self.dash_log.append("正在定位 Strata 运行环境…")
        adopted = self._relocate_root_from_data()  # 自动使用数据目录旁的已有安装
        try:
            root = self.env.ensure_root(lambda s: self.ui(self.dash_log.append, s))
            self.dash_log.append(f"Strata 环境：{root}")
            dd = self.env.data_dir()
            if dd:
                self.dash_log.append(f"模型数据目录（Strata 记录）：{dd}")
        except Exception as e:                     # noqa: BLE001
            self.dash_log.append("环境准备失败：" + str(e))
            self.toast("环境准备失败",
                       f"无法准备 Strata 运行环境。\n\n{e}\n\n"
                       f"可在「网络与配置 → 目录设置」中手动指定 Strata 目录，"
                       f"或把源码放到：{APP_DIR / 'Strata'}", "error")
        self.env._tables = None                    # 已定位到真实源码，重新读取官方模型表
        try:
            _models, _families, contexts = self.env.tables()
            labels = [ctx_label(x) for x in contexts]
            self.ctx_menu.configure(values=labels)
            if self.ctx_var.get() not in labels:
                self.ctx_var.set(labels[0])
            self._rebuild_family_radios()
        except Exception as e:                     # noqa: BLE001
            self.dash_log.append(f"读取模型表失败：{e}")
        self.apply_mode()
        self._fill_env_info()
        self._refresh_dir_fields()
        self._refresh_ctx_quick()
        self._load_model_optimization()
        self._refresh_cleanup_models()
        if (adopted or self.cfg.get("cached_models") is None
                or self.cfg.get("cached_scan_version") != SCAN_CACHE_VERSION):
            self.dash_log.append("正在后台刷新模型清单（不阻塞界面，稍后自动显示）…")
            threading.Thread(target=self._first_scan, daemon=True).start()
        self.after(400, lambda: self.check_update(auto=True))

    def _first_scan(self):
        try:
            self._scan_models()
        except Exception as e:                     # noqa: BLE001
            log_line(f"[warn] 首次扫描失败：{e}")
        self.ui(self._after_first_scan)

    def _after_first_scan(self):
        self.apply_mode()
        self._refresh_dir_fields()
        self.dash_log.append(f"已扫描已安装模型：{len(self.model_cache())} 个。")

    # =================================================================== 概览页
    def _page_dashboard(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)

        # 环境信息
        card = make_card(page, "运行环境")
        card.pack(fill="x", pady=(0, 12))
        self.env_info = ctk.CTkLabel(card, text="检测中…", text_color=COLORS["text"], justify="left",
                                     font=ctk.CTkFont(family=FONT, size=12), wraplength=820)
        self.env_info.pack(anchor="w", padx=16, pady=(0, 8))
        self.env_paths = ctk.CTkLabel(card, text="", text_color=COLORS["text_dim"], justify="left",
                                      font=ctk.CTkFont(family=FONT_MONO, size=11), wraplength=820)
        self.env_paths.pack(anchor="w", padx=16, pady=(0, 12))

        # 控制卡片
        ctrl = make_card(page, "模型控制")
        ctrl.pack(fill="x", pady=(0, 12))
        row = ctk.CTkFrame(ctrl, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(4, 8))
        self.model_var = ctk.StringVar(value="")
        self.model_menu = ctk.CTkOptionMenu(row, variable=self.model_var, values=["（暂无已安装模型）"],
                                            width=360, height=34, corner_radius=8,
                                            fg_color=COLORS["chip"], text_color=COLORS["text"],
                                            button_color="#dde5f7", button_hover_color="#cfdaf5",
                                            dropdown_text_color=COLORS["text"],
                                             command=self._on_dashboard_model_change,
                                            font=ctk.CTkFont(family=FONT, size=12))
        self.model_menu.pack(side="left")
        ctk.CTkButton(row, text="刷新", width=68, height=34, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7",
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self.rescan_models).pack(side="left", padx=8)

        # 上下文（启动前随时可调；全新模式下调的是安装时要用的值）
        ctxrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        ctxrow.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkLabel(ctxrow, text="上下文", width=52, anchor="w", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.ctx_quick_menu = ctk.CTkOptionMenu(ctxrow, values=["32768"], width=140, height=34, corner_radius=8,
                                                fg_color=COLORS["chip"], button_color="#dde5f7",
                                                button_hover_color="#cfdaf5", text_color=COLORS["text"],
                                                dropdown_text_color=COLORS["text"],
                                                font=ctk.CTkFont(family=FONT, size=12))
        self.ctx_quick_menu.pack(side="left", padx=8)
        ctk.CTkButton(ctxrow, text="应用上下文", width=110, height=34, corner_radius=8, fg_color=COLORS["accent"],
                      hover_color=COLORS["accent_hover"], font=ctk.CTkFont(family=FONT, size=12),
                      command=self._apply_quick_context).pack(side="left")
        ctk.CTkButton(ctxrow, text="打开配置文件", width=110, height=34, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._open_config_file).pack(side="left", padx=8)

        self.ctx_quick_info = ctk.CTkLabel(ctrl, text="", text_color=COLORS["text_dim"], justify="left",
                                           font=ctk.CTkFont(family=FONT, size=11), wraplength=820)
        self.ctx_quick_info.pack(anchor="w", padx=16, pady=(0, 2))
        self.ctx_quick_status = ctk.CTkLabel(ctrl, text="", text_color=COLORS["text_dim"], justify="left",
                                             font=ctk.CTkFont(family=FONT, size=11), wraplength=820)
        self.ctx_quick_status.pack(anchor="w", padx=16, pady=(0, 10))

        # 服务端摘要压缩；运行中切换从下一次请求生效。
        comprow = ctk.CTkFrame(ctrl, fg_color="transparent")
        comprow.pack(fill="x", padx=16, pady=(0, 2))
        ctk.CTkLabel(comprow, text="上下文压缩（生成摘要）", width=52, anchor="w", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.compress_var = ctk.BooleanVar(value=bool(self.cfg.get("context_compress_enabled")))
        self.compress_switch = ctk.CTkSwitch(comprow, text="", variable=self.compress_var,
                                             width=40, height=26, corner_radius=8,
                                             font=ctk.CTkFont(family=FONT, size=12),
                                             command=self._on_compress_toggle)
        self.compress_switch.pack(side="left", padx=8)
        self.compress_status = ctk.CTkLabel(comprow, text="", text_color=COLORS["text_dim"], anchor="w",
                                            font=ctk.CTkFont(family=FONT, size=11), wraplength=720)
        self.compress_status.pack(side="left", padx=(8, 0))
        self._refresh_compress_status()

        fitrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        fitrow.pack(fill="x", padx=16, pady=(0, 8))
        self.fit_var = ctk.BooleanVar(value=bool(self.cfg.get("fit_max_tokens")))
        ctk.CTkSwitch(fitrow, text="自动适配输出额度（不删除历史）", variable=self.fit_var,
                      command=self._on_compress_toggle,
                      font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")

        visionrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        visionrow.pack(fill="x", padx=16, pady=(0, 8))
        self.runtime_vision_var = ctk.BooleanVar(value=bool(self.cfg.get("runtime_vision_enabled")))
        ctk.CTkSwitch(visionrow, text="视觉模型（图片 / 视频）", variable=self.runtime_vision_var,
                      command=self._on_runtime_vision_toggle,
                      font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.runtime_vision_status = ctk.CTkLabel(visionrow, text="", text_color=COLORS["text_dim"],
                                                  font=ctk.CTkFont(family=FONT, size=11))
        self.runtime_vision_status.pack(side="left", padx=(12, 0))
        self._refresh_runtime_vision_status()
        templaterow = ctk.CTkFrame(ctrl, fg_color='transparent')
        templaterow.pack(fill='x', padx=16, pady=4)
        self.chat_template_var = ctk.BooleanVar(value=False)
        self.chat_template_path_var = ctk.StringVar(value='')
        ctk.CTkSwitch(templaterow, text='自定义聊天模板', variable=self.chat_template_var,
                      command=self._save_chat_template).pack(side='left', padx=(0, 12))
        self.chat_template_menu = ctk.CTkOptionMenu(templaterow,
            values=['内置 Froggeric v21.3', '自选文件'], command=self._save_chat_template)
        self.chat_template_menu.pack(side='left', padx=4)
        self.chat_template_choose = ctk.CTkButton(templaterow, text='选择模板文件', width=110,
                                                command=self._choose_chat_template)
        self.chat_template_choose.pack(side='left', padx=4)
        self.chat_template_status = ctk.CTkLabel(ctrl, text='', anchor='w')
        self.chat_template_status.pack(fill='x', padx=16)
        self._load_model_optimization()

        optrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        optrow.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(optrow, text="思考方式", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.effort_var = ctk.StringVar(value=THINKING_LABELS["high"])
        ctk.CTkOptionMenu(optrow, variable=self.effort_var, values=list(THINKING_LABELS.values()), width=110,
                          command=self._save_model_optimization).pack(side="left", padx=(8, 16))
        ctk.CTkLabel(optrow, text="运行方式", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.optimization_mode_var = ctk.StringVar(value=RUNTIME_MODE_LABELS["single"])
        ctk.CTkOptionMenu(optrow, variable=self.optimization_mode_var,
                          values=list(RUNTIME_MODE_LABELS.values()), width=175,
                          command=self._save_model_optimization).pack(side="left", padx=8)

        kvrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        kvrow.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(kvrow, text="KV 显存驻留上限", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.kv_resident_var = ctk.StringVar(value="全部驻留")
        ctk.CTkOptionMenu(kvrow, variable=self.kv_resident_var,
                          values=["全部驻留", "32K", "64K", "128K"], width=145,
                          command=self._save_model_optimization).pack(side="left", padx=8)
        ctk.CTkLabel(kvrow, text="其余 KV 使用内存；不改变上下文上限。显存预留 500 MiB，重启后生效。",
                     text_color=COLORS["text_dim"], font=ctk.CTkFont(family=FONT, size=11),
                     wraplength=450).pack(side="left", padx=8)

        precisionrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        precisionrow.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(precisionrow, text="KV 缓存精度", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.kv_precision_var = ctk.StringVar(value=RUNTIME_KV_LABELS['original'])
        ctk.CTkOptionMenu(precisionrow, variable=self.kv_precision_var,
                          values=list(RUNTIME_KV_LABELS.values()), width=170,
                          command=self._save_model_optimization).pack(side="left", padx=8)
        ctk.CTkLabel(precisionrow, text="Q4 的 KV 占用约为 INT8 一半，精度较低；只改变缓存，不改模型量化。重启后生效。",
                     text_color=COLORS["text_dim"], font=ctk.CTkFont(family=FONT, size=11),
                     wraplength=430).pack(side="left", padx=8)

        advanced = ctk.CTkFrame(ctrl, fg_color="transparent")
        advanced.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(advanced, text="单次思考长度", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.thinking_budget_var = ctk.StringVar(value=THINKING_BUDGET_LABELS[0])
        ctk.CTkOptionMenu(advanced, variable=self.thinking_budget_var,
                          values=list(THINKING_BUDGET_LABELS.values()), width=190,
                          command=self._save_model_optimization).pack(side="left", padx=(8, 16))
        ctk.CTkLabel(advanced, text="生成加速强度", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.spec_var = ctk.StringVar(value=ACCELERATION_LABELS["original"])
        ctk.CTkOptionMenu(advanced, variable=self.spec_var, values=list(ACCELERATION_LABELS.values()), width=150,
                          command=self._save_model_optimization).pack(side="left", padx=8)
        self.profile_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(advanced, text="记住模型加速结果", variable=self.profile_var,
                        command=self._save_model_optimization, text_color=COLORS["text"],
                        font=ctk.CTkFont(family=FONT, size=12)).pack(side="left", padx=8)
        ctk.CTkLabel(ctrl, text="运行方式和生成加速强度重启后生效。长对话快速恢复需要客户端支持；单次思考长度不能超过输出额度。",
                     text_color=COLORS["text_dim"], font=ctk.CTkFont(family=FONT, size=11),
                     wraplength=820).pack(anchor="w", padx=16, pady=(0, 8))
        ctk.CTkButton(ctrl, text="查看这些设置的详细说明", width=185, height=28,
                      fg_color=COLORS["chip"], text_color=COLORS["text"],
                      command=self.show_optimization_help).pack(anchor="w", padx=16, pady=(0, 10))

        btnrow = ctk.CTkFrame(ctrl, fg_color="transparent")
        btnrow.pack(fill="x", padx=16, pady=(0, 12))
        self.start_btn = ctk.CTkButton(btnrow, text="启动模型", width=130, height=38, corner_radius=8,
                                       fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                       font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
                                       command=self.start_model)
        self.start_btn.pack(side="left")
        self.stop_btn = ctk.CTkButton(btnrow, text="停止", width=92, height=38, corner_radius=8,
                                      fg_color=COLORS["danger"], hover_color="#c22f2f", state="disabled",
                                      font=ctk.CTkFont(family=FONT, size=13),
                                      command=self.stop_model)
        self.stop_btn.pack(side="left", padx=8)
        ctk.CTkButton(btnrow, text="重启", width=92, height=38, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=13),
                      command=self.restart_model).pack(side="left")
        ctk.CTkButton(btnrow, text="打开聊天页面", width=130, height=38, corner_radius=8,
                      fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                      font=ctk.CTkFont(family=FONT, size=13),
                      command=self.open_chat).pack(side="left", padx=8)

        self.run_hint = ctk.CTkLabel(ctrl, text="", text_color=COLORS["warning"], justify="left",
                                     font=ctk.CTkFont(family=FONT, size=11), wraplength=820)
        self.run_hint.pack(anchor="w", padx=16, pady=(0, 10))

        # 运行状态与连接信息（合并展示：启动后直接能看到访问地址与密钥）
        metric = make_card(page, "运行状态与连接信息")
        metric.pack(fill="x", pady=(0, 12))
        self.connection_text = ctk.CTkTextbox(metric, height=160, wrap="word",
            text_color=COLORS["text"], fg_color=COLORS["card"],
            font=ctk.CTkFont(family=FONT_MONO, size=12))
        self.connection_text.pack(fill="x", padx=12, pady=(0, 12))
        self.connection_text.configure(state="disabled")
        self.connection_text.bind('<Control-a>', self._select_connection_text)
        self.connection_text.bind('<Control-A>', self._select_connection_text)
        self._connection_menu = tk.Menu(self, tearoff=0)
        self._connection_menu.add_command(label="全选", command=self._select_connection_text)
        self._connection_menu.add_command(label="复制", command=lambda: self.connection_text._textbox.event_generate('<<Copy>>'))
        self.connection_text.bind('<Button-3>', lambda event: self._connection_menu.tk_popup(event.x_root, event.y_root))

        # 日志
        logcard = make_card(page, "运行日志")
        logcard.pack(fill="both", expand=True, pady=(0, 8))
        self.dash_log = LogPane(logcard, height=260)
        self.dash_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        return page

    def _fill_env_info(self):
        gpus = query_nvidia_smi()
        ram = total_ram_gb()
        lines = []
        if gpus:
            lines.append("显卡：" + "；".join(f"GPU {i} {n}（{v:.0f} GB）" for i, n, v in gpus))
        else:
            lines.append("显卡：未检测到 NVIDIA 显卡（nvidia-smi 无响应）")
        lines.append(f"内存：{ram:.0f} GB")
        ctx = [c for c in self.env.tables()[2]]
        lines.append(f"可用上下文档位：{', '.join(str(c) for c in ctx)}")
        lv = self.env.source_version()
        if lv:
            lines.append(f"Strata 版本：v{lv}")
        self.env_info.configure(text="\n".join(lines))
        dd = self.effective_data_dir()
        self.env_paths.configure(text=f"Strata 目录：{self.env.root}\n模型数据目录：{dd if dd else '（自动）'}")

    def effective_data_dir(self):
        """生效的模型数据目录：优先用户手动指定，其次 Strata 官方记录。

        误选到 models / packs / mtp 子目录时，自动回退到它们的父目录（数据根）。
        """
        manual = self.cfg.get("data_dir")
        if manual:
            p = Path(manual)
            if p.name.lower() in ("models", "packs", "mtp") and p.parent.is_dir():
                p = p.parent
            if p.is_dir():
                return str(p)
        dd = self.env.data_dir()
        return str(dd) if dd else ""

    def _detect_nearby_root(self):
        """在数据目录旁寻找已有的 Strata 安装目录（含 setup.py，优先带模型配置的）。

        例如数据目录 R:\\Strata-main\\Strata-data，其上级 R:\\Strata-main 里若有
        Strata-main\\setup.py 与 strata-*.json，即视为用户原有的完整安装。
        """
        dd = self.effective_data_dir()
        if not dd:
            return None
        base = Path(dd).parent
        cands = []
        try:
            if (base / "setup.py").is_file():
                cands.append(base)
            for c in sorted(base.iterdir(), key=lambda p: p.name.lower()):
                if not c.is_dir() or c.name.startswith("."):
                    continue
                try:
                    if (c / "setup.py").is_file():
                        cands.append(c)
                except OSError:
                    continue
        except OSError:
            return None
        for c in cands:                            # 优先含模型配置文件的目录
            try:
                if any(c.glob("strata-*.json")):
                    return c
            except OSError:
                continue
        return cands[0] if cands else None

    def _relocate_root_from_data(self):
        """未手动指定 Strata 目录时，自动改用「数据目录旁的已有安装」，让其中的模型配置生效。

        返回是否发生了切换（切换后调用方应重新扫描模型清单）。
        """
        if self.cfg.get("strata_dir"):
            return False
        hint = self._detect_nearby_root()
        if hint is None:
            return False
        try:
            if self.env.root is not None and Path(self.env.root).resolve() == hint.resolve():
                return False
        except OSError:
            pass
        self.env.hint_root = hint
        self.env.set_manual_root("")
        try:
            root = self.env.ensure_root(lambda s: self.dash_log.append(s))
            self.env._tables = None
            self.dash_log.append(f"已自动识别数据目录旁的 Strata 安装：{root}")
            return True
        except Exception as e:                     # noqa: BLE001
            self.dash_log.append(f"自动识别 Strata 安装失败：{e}")
            return False

    def _refresh_dashboard(self):
        names = []
        self._cfg_map = {}
        for ent in self.model_cache():
            label = ent.get("label") or ""
            if not label:
                continue
            if label in self._cfg_map:
                label = f"{label} — {Path(ent.get('path') or ent.get('dir') or '').name}"
            names.append(label)
            self._cfg_map[label] = ent
        if names:
            self.model_menu.configure(values=names)
            if self.model_var.get() not in names:
                self.model_var.set(names[0])
        else:
            self.model_menu.configure(values=["（暂无已安装模型）"])
            self.model_var.set("（暂无已安装模型）")
        mode = self.current_mode()
        self.mode_label.configure(text="模式：" + ("已有模型" if mode == "installed" else "全新安装"))
        self._fill_env_info()
        self._refresh_ctx_quick()
        self._load_model_optimization()
        self._refresh_cleanup_models()

    def _selected_cfg(self):
        label = self.model_var.get()
        return getattr(self, "_cfg_map", {}).get(label)

    def _selected_cfg_path(self):
        ent = self._selected_cfg()
        if not ent or not ent.get("path"):
            return None
        return Path(ent["path"])

    def _on_dashboard_model_change(self, _=None):
        self._refresh_ctx_quick()
        self._load_model_optimization()

    def _model_optimization(self, path=None):
        path = path or self._selected_cfg_path()
        saved = self.cfg.get('model_optimizations') or {}
        return dict(saved.get(str(path.resolve()), {})) if path else {}

    def _load_model_optimization(self):
        if hasattr(self, 'chat_template_var'):
            options = self._model_chat_template()
            self.chat_template_var.set(bool(options.get('enabled')))
            self.chat_template_path_var.set(options.get('path') or '')
            self._refresh_chat_template()
        if not hasattr(self, 'effort_var'):
            return
        options = self._model_optimization()
        self.effort_var.set(THINKING_LABELS.get(options.get('effort'), THINKING_LABELS['high']))
        self.optimization_mode_var.set(RUNTIME_MODE_LABELS.get(
            options.get('mode'), RUNTIME_MODE_LABELS['single']))
        budget = options.get('budget', 0)
        self.thinking_budget_var.set(THINKING_BUDGET_LABELS.get(budget, str(budget)))
        self.spec_var.set(ACCELERATION_LABELS.get(options.get('spec'), ACCELERATION_LABELS['original']))
        self.profile_var.set(bool(options.get('profile')))
        if hasattr(self, 'kv_precision_var'):
            self.kv_precision_var.set(RUNTIME_KV_LABELS.get(options.get('kv_precision'), RUNTIME_KV_LABELS['original']))
        if hasattr(self, 'kv_resident_var'):
            resident = options.get('kv_resident', 0)
            self.kv_resident_var.set({0: '全部驻留', 32768: '32K', 65536: '64K', 131072: '128K'}.get(resident, '全部驻留'))

    def _save_model_optimization(self, _=None):
        path = self._selected_cfg_path()
        if not path:
            return
        options = optimization_from_labels(
            self.effort_var.get(), self.optimization_mode_var.get(),
            self.thinking_budget_var.get(), self.spec_var.get(), self.profile_var.get())
        if hasattr(self, 'kv_resident_var'):
            options['kv_resident'] = {'全部驻留': 0, '32K': 32768, '64K': 65536, '128K': 131072}[self.kv_resident_var.get()]
        if hasattr(self, 'kv_precision_var'):
            options['kv_precision'] = {label: key for key, label in RUNTIME_KV_LABELS.items()}[self.kv_precision_var.get()]
        saved = dict(self.cfg.get('model_optimizations') or {})
        saved[str(path.resolve())] = options
        self.cfg.set('model_optimizations', saved)

    # ---------------- 启动 / 停止
    def _model_chat_template(self, path=None):
        path = path or self._selected_cfg_path()
        saved = self.cfg.get('model_chat_templates') or {}
        return dict(saved.get(str(path.resolve()), {})) if path else {}

    def _refresh_chat_template(self):
        enabled = bool(self.chat_template_var.get())
        self.chat_template_menu.configure(state='normal' if enabled else 'disabled')
        self.chat_template_choose.configure(state='normal' if enabled else 'disabled')
        path = self.chat_template_path_var.get()
        self.chat_template_menu.set('自选文件' if path else '内置 Froggeric v21.3')
        self.chat_template_status.configure(text=(
            (Path(path).name if path else '内置模板') + ' · 下次启动生效 · low/high 强度不适用'
            if enabled else '已关闭 · 使用模型原始模板'))

    def _save_chat_template(self, choice=None):
        if choice == '内置 Froggeric v21.3':
            self.chat_template_path_var.set('')
        if choice == '自选文件' and not self.chat_template_path_var.get():
            self._choose_chat_template()
            return
        path = self._selected_cfg_path()
        if path:
            saved = dict(self.cfg.get('model_chat_templates') or {})
            saved[str(path.resolve())] = {'enabled': bool(self.chat_template_var.get()),
                                        'path': self.chat_template_path_var.get()}
            self.cfg.set('model_chat_templates', saved)
        self._refresh_chat_template()

    def _choose_chat_template(self):
        path = filedialog.askopenfilename(title='选择聊天模板',
            filetypes=[('Jinja 聊天模板', '*.jinja *.jinja2'), ('所有文件', '*.*')])
        if path:
            try:
                validate_template(path)
            except Exception as error:
                messagebox.showerror('模板无法加载', str(error))
                self._refresh_chat_template()
                return
            self.chat_template_path_var.set(path)
            self._save_chat_template()
        else:
            self._refresh_chat_template()

    def start_model(self):
        if getattr(self, "_starting", False):
            self.toast("正在启动", "正在准备模型服务，请稍候。")
            return
        if self.server.running():
            self.toast("已在运行", "模型服务已经在运行中。")
            return
        ent = self._selected_cfg()
        if ent is None:
            self.toast("没有可启动的模型", "当前没有已安装的模型。请切换到「全新安装」模式安装模型，"
                                          "或在「网络与配置 → 目录设置」点「扫描硬盘」或「使用自动」重新扫描。")
            return
        if not ent.get("path"):
            self._configure_downloaded_model(ent)   # 已下载但还没有配置文件
            return
        cfg_path = Path(ent["path"])
        if self.cfg.get('managed_version_active') and self.env.root:
            managed_config = self.env.root / cfg_path.name
            if not managed_config.is_file():
                self.toast('版本配置不匹配', '所选环境没有此模型的配置，请在该环境重新导入模型。', 'warn')
                return
            cfg_path = managed_config
        if not self.cfg.get('strata_dir') and (cfg_path.parent / 'setup.py').is_file():
            selected_root = cfg_path.parent.resolve()
            if self.env.root != selected_root:
                self.env.root = selected_root
                self.env.hint_root = selected_root
                self.env._python = None
                self.env._tables = None
                self.env._mod = None
        try:
            cj = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            self.toast("配置读取失败", f"{cfg_path.name} 无法读取：{e}", "error")
            return
        launch_cfg = resolve_launch_config(cj, self.env.root or cfg_path.parent)
        exe = launch_cfg.get("exe")
        if not exe or not Path(str(exe)).is_file():
            if not messagebox.askyesno(
                    "配置路径已失效",
                    f"配置文件里记录的引擎路径不存在：\n{exe or '（未记录）'}\n\n"
                    "通常是因为 Strata 目录或模型数据目录被移动过（例如盘符从 G: 变成 R:）。\n\n"
                    "是否按当前目录重新生成配置？（已下载的模型文件会被复用，不会重新下载）"):
                return
            self._generate_config(ent, "重新生成配置",
                                  "当前配置：{spec}（家族 {fam}）\n数据目录：{dd}\n上下文：{ctx}\n\n"
                                  "将调用官方 setup.py 按当前目录重写该配置文件。")
            return
        port = int(cj.get("port") or 8080)
        if port_in_use(port):
            pid, pname = port_owner(port)
            desc = f"{pname or '未知进程'}（PID {pid}）" if pid else "未知进程"
            if self._is_our_server(pid):
                self.toast("端口被占用", f"端口 {port} 由启动器启动的模型服务占用，请先停止模型服务。", "warn")
                return
            if pid <= 4 or is_protected_process(pname):
                self.toast("端口被系统进程占用", f"端口 {port} 被系统进程 {desc} 占用，无法强制结束。", "error")
                return
            if not messagebox.askyesno("端口被占用",
                                       f"端口 {port} 被 {desc} 占用。\n\n"
                                       "是否强制结束该进程，并用网络配置中的端口继续启动？\n"
                                       "（网络配置的端口保持不变）"):
                self.dash_log.append(f"已取消启动：端口 {port} 被 {desc} 占用。"
                                     "可到「维护工具」页执行“修复端口占用”。")
                return
            ok, msg = kill_pid_force(pid)
            time.sleep(0.5)
            if not ok or port_in_use(port):
                self.dash_log.append(f"结束进程失败：{msg or '端口仍被占用'}"
                                     "（如权限不足，请以管理员身份运行启动器后重试）。")
                self.toast("启动已取消", f"未能释放端口 {port}，可到「维护工具」页重试。", "error")
                return
            self.dash_log.append(f"已强制结束 {desc}，端口 {port} 已释放，继续用原端口启动。")
        self.dash_log.clear()
        self.dash_log.append("正在启动模型服务…")
        self.dash_log.append("首次加载会把 34-55 GB 读入内存，期间电脑可能卡顿 1-3 分钟，属正常现象。")
        policy = {"context_compress_enabled": bool(self.compress_var.get()),
                  "fit_max_tokens": bool(self.fit_var.get())}
        browser = bool(self.cfg.get('open_browser'))
        vision_enabled = (bool(self.runtime_vision_var.get()) if hasattr(self, 'runtime_vision_var')
                          else bool(self.cfg.get('runtime_vision_enabled')))
        optimization = StrataLauncher._model_optimization(self, cfg_path)
        chat_template = StrataLauncher._model_chat_template(self, cfg_path)
        self._starting = True
        self._launch_cancel = threading.Event()
        cancel = self._launch_cancel
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.set_status('正在准备启动…', COLORS['warning'])
        def finish(error=None):
            self._starting = False
            if error:
                self.runtime['running'] = False
                self.start_btn.configure(state='normal')
                self.stop_btn.configure(state='disabled')
                self.set_status('已取消' if cancel.is_set() else '启动失败', COLORS['danger'])
                self.dash_log.append('启动失败：' + str(error))
                log_line('启动失败：' + str(error))
                if not cancel.is_set():
                    self.toast('启动失败', str(error), 'error')
                return
            if not self.server.running():
                self.start_btn.configure(state='normal')
                self.stop_btn.configure(state='disabled')
                self.runtime['running'] = False
                self.set_status('服务已退出，请查看日志', COLORS['danger'])
                return
            self.runtime['running'] = True
            self._context_sync_status = '独立上下文服务已自动接入，正在加载模型'
            self.set_status('启动中…', COLORS['warning'])
            self._refresh_compress_status()
            self._refresh_runtime_vision_status()
        def worker():
            try:
                if self.env.python() is None or not Path(str(self.env.python())).is_file():
                    self.env.ensure_venv(lambda line: self.ui(self.dash_log.append, line))
                if cancel.is_set() or getattr(self, '_closing', False):
                    raise RuntimeError('已取消启动')
                self._save_context_policy(cfg_path, policy)
                cmd = self.server.start(self.env, cfg_path, open_browser=browser,
                                        vision_enabled=vision_enabled,
                                        optimization=optimization,
                                        chat_template=chat_template,
                                        data_dir=(self.effective_data_dir() if hasattr(self, 'effective_data_dir') else None))
                if cancel.is_set() or getattr(self, '_closing', False):
                    self.server.stop()
                    raise RuntimeError('已取消启动')
                self.ui(self.dash_log.append, '$ ' + ' '.join(cmd))
                self.ui(finish)
                threading.Thread(target=self._drain_server_log, args=(self.server.log_queue,), daemon=True).start()
            except Exception as error:
                self.ui(finish, error)
        threading.Thread(target=worker, daemon=True, name='strata-launch').start()

    # ---------------- 配置生成（为已下载 / 路径失效的模型重写官方配置）
    def _generate_config(self, ent, title, intro, ctx=None):
        """调用官方 setup.py 生成或重写配置；已下载的模型文件会被复用。"""
        if self.env.root is None:
            self.toast("环境未就绪", "尚未定位到 Strata 源码目录，请先在「网络与配置 → 目录设置」"
                                     "用「扫描硬盘」或「浏览…」指定（该目录须含 setup.py）。", "warn")
            return
        if self.runner.running() or self.runtime.get("updating"):
            self.toast("有任务在运行", "请等待当前任务完成后再试。", "warn")
            return
        models, families, contexts = self.env.tables()
        fam, spec = ent.get("family"), ent.get("model")
        if not fam or not spec:
            self.toast("无法生成配置", "未能解析出该模型的家族/规格，请改用「安装 / 模型」页的向导。", "warn")
            return
        dd = self.effective_data_dir()
        if ctx is None:
            old = ent.get("context")
            last = self.cfg.get("last_context")
            if old in contexts:
                ctx = old
            elif last in contexts:
                ctx = last
            else:
                ctx = 32768 if 32768 in contexts else contexts[0]
        md = self.env.models_for(fam).get(spec, {})
        if md.get('contexts') and ctx not in md['contexts']:
            ctx = md['contexts'][0]
        text = intro.format(spec=spec, fam=fam, dd=dd or "（未指定）", ctx=ctx_label(ctx))
        if not messagebox.askyesno(title, text + "\n\n确定吗？"):
            return
        args = ["--yes", "--family", fam, "--model", spec, "--context", str(ctx), "--no-start"]
        args = adapt_install_args(args, md)
        if dd:
            args += ["--data-dir", dd]
        self.cfg.set("last_context", ctx)
        self.show_page("tools")

        def done(rc):
            if rc == 0:
                self.rescan_models()
                self.show_page("dashboard")
                self.toast("配置已生成", f"{spec} 的配置已生成，现在可以点击「启动模型」。")
            else:
                self.toast("生成失败", f"setup.py 退出码 {rc}，详见「维护工具」页输出。", "error")

        self._run_setup_tool(args, f"生成配置：{spec}", on_done=done)

    def _configure_downloaded_model(self, ent):
        """为「已下载但未配置」的模型生成配置。"""
        self._generate_config(ent, "生成模型配置",
                              "模型：{spec}（家族 {fam}）\n数据目录：{dd}\n上下文：{ctx}\n\n"
                              "该模型已下载，但还没有配置文件。将调用官方 setup.py 生成配置：\n"
                              "已下载的模型文件会被复用，不会重新下载（缺什么补什么）。")

    def _drain_server_log(self, logs=None):
        logs = logs if logs is not None else self.server.log_queue
        while True:
            line = logs.get()
            if line == "__EXIT__":
                self.ui(self._on_server_exit)
                return
            self.runtime["last_log"] = line
            self.ui(self.dash_log.append, line)

    def _on_server_exit(self):
        if getattr(self, "_starting", False) or self.server.running():
            return
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.runtime["running"] = False
        self._context_sync_status = '服务已停止；下次启动使用模型配置'
        self._refresh_ctx_quick()
        self._refresh_runtime_vision_status()
        self.set_status("已停止", COLORS["text_dim"])
        self.dash_log.append("— 服务已退出 —")

    def stop_model(self):
        if getattr(self, '_starting', False):
            self._launch_cancel.set()
            self.set_status('正在取消启动…', COLORS['warning'])
            return
        if not self.server.running():
            return
        if not messagebox.askyesno("停止模型", "确定要停止模型服务吗？"):
            return
        self.dash_log.append("正在停止模型服务…")
        self.server.stop()

    def restart_model(self):
        if self.server.running():
            self.dash_log.append("正在重启…")
            self.server.stop()
            self.after(1200, self.start_model)
        else:
            self.start_model()

    def _on_runtime_vision_toggle(self):
        self.cfg.set("runtime_vision_enabled", bool(self.runtime_vision_var.get()))
        self._refresh_runtime_vision_status(changed=True)

    def _refresh_runtime_vision_status(self, changed=False):
        if not hasattr(self, "runtime_vision_status"):
            return
        enabled = bool(self.runtime_vision_var.get())
        mode = "开启" if enabled else "关闭"
        if changed and self.server.running():
            suffix = "，重启模型后生效"
        elif self.server.running():
            suffix = "，当前运行已生效"
        else:
            suffix = "，下次启动时生效"
        self.runtime_vision_status.configure(text=mode + suffix,
                                             text_color=COLORS["warning"] if changed and self.server.running()
                                             else COLORS["text_dim"])

    def open_chat(self):
        url = self.server.local_url() + "/"
        try:
            os.startfile(url)                      # noqa: S606
        except Exception:                          # noqa: BLE001
            import webbrowser
            webbrowser.open(url)

    # ---------------- 上下文压缩开关
    def _on_compress_toggle(self):
        self._context_sync_status = "正在同步…"
        self._refresh_compress_status()
        policy = {"context_compress_enabled": bool(self.compress_var.get()),
                  "fit_max_tokens": bool(self.fit_var.get())}
        path = self.server.cfg_path if self.server.running() else self._selected_cfg_path()
        if path is None:
            self.cfg.update(**policy)
            self._context_sync_status = "已保存默认设置，选择模型后生效"
            self._refresh_compress_status()
            return
        generation = getattr(self, '_context_sync_generation', 0) + 1
        self._context_sync_generation = generation
        running = self.server.running()
        def worker():
            try:
                if running:
                    reply = self.server.set_context_policy(policy)
                    if reply.get('live') != policy:
                        raise RuntimeError('独立摘要服务未确认设置')
                else:
                    self._save_context_policy(path, policy)
                self.ui(finish, None)
            except Exception as e:
                self.ui(finish, str(e))
        def finish(error):
            if generation != self._context_sync_generation:
                return
            if error:
                self._context_sync_status = '未同步：' + error
                self.dash_log.append('上下文设置未同步：' + error)
            else:
                self.cfg.update(**policy)
                self._context_sync_status = '服务端已确认，下一次请求生效' if running else '已保存到模型，下次启动生效'
                self.dash_log.append(self._context_sync_status)
            self._refresh_compress_status()
        # Serialize rapid switch changes so the last choice wins on disk and server.
        if not hasattr(self, '_context_sync_lock'):
            self._context_sync_lock = threading.Lock()
        def ordered_worker():
            with self._context_sync_lock:
                if generation == self._context_sync_generation:
                    worker()
        threading.Thread(target=ordered_worker, daemon=True).start()

    def _save_context_policy(self, path, policy=None):
        if policy is None:
            policy = {"context_compress_enabled": bool(self.compress_var.get()),
                      "fit_max_tokens": bool(self.fit_var.get())}
        cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if all(cfg.get(k) == v for k, v in policy.items()):
            return
        shutil.copy2(path, str(path) + '.context.bak')
        cfg.update(policy)
        tmp = Path(str(path) + '.context.tmp')
        tmp.write_text(json.dumps(cfg, indent=1), encoding='utf-8')
        os.replace(tmp, path)

    def _refresh_compress_status(self):
        if not hasattr(self, "compress_status"):
            return
        on = bool(self.compress_var.get()) if hasattr(self, "compress_var") else bool(self.cfg.get("context_compress_enabled"))
        text = ('摘要压缩：开启' if on else '摘要压缩：关闭') + '；' + getattr(
            self, '_context_sync_status', '选择模型后显示生效状态')
        color = COLORS["warning"] if '未同步' in text or '正在' in text else COLORS["text_dim"]
        self.compress_status.configure(text=text, text_color=color)

    # ---------------- 定时状态轮询（探测放到后台线程，避免界面卡顿）
    def _tick(self):
        if self.server.running() and self.server.port:
            if not getattr(self, "_probe_busy", False):
                self._probe_busy = True
                threading.Thread(target=self._probe_worker, args=(self.server.port,), daemon=True).start()
        elif not self.server.running():
            self.set_status("未启动", COLORS["text_dim"])
        self._update_dash_conn()                   # 运行状态与连接信息合并刷新
        self.after(2500, self._tick)

    def _probe_worker(self, port):
        status, body = http_probe(self.server.local_url() + "/health")
        path = self.server.cfg_path
        if status == 200 and path:
            try:
                report = self.server.gateway.report()
                self.ui(self._apply_context_report, path, report)
            except Exception:
                pass
        self.ui(self._apply_probe, status, body, "")
        self._probe_busy = False

    def _apply_context_report(self, path, report):
        if not self.server.running() or path != self.server.cfg_path or not report.get('live'):
            return
        if '正在同步' in getattr(self, '_context_sync_status', '') or '未同步' in getattr(self, '_context_sync_status', ''):
            return
        live = report['live']
        self.compress_var.set(bool(live.get('context_compress_enabled')))
        self.fit_var.set(bool(live.get('fit_max_tokens')))
        stats = report.get('context_guard') or {}
        self._context_sync_status = '服务端已确认，下一次请求生效'
        if stats:
            self._context_sync_status += (f"；最近输入 {stats.get('prompt_before')} → {stats.get('prompt_after')} tokens，"
                                         f"摘要替换 {stats.get('summarized_messages', stats.get('removed_messages', 0))} 条，"
                                         f"{'复用缓存，' if stats.get('summary_cached') else ''}输出额度 {stats.get('effective_output')}")
        self._refresh_compress_status()

    def _apply_probe(self, status, body, info):
        if status == 200:
            self.set_status("运行中", COLORS["success"])
            self.runtime["health"] = body
        elif self.runtime["running"]:
            self.set_status("启动中…", COLORS["warning"])

    def _select_connection_text(self, event=None):
        self.connection_text.focus_set()
        self.connection_text._textbox.tag_add('sel', '1.0', 'end-1c')
        return 'break'

    def _render_connection_fields(self, fields):
        text = '\n'.join(f'{title}：{value or "未设置"}' for title, value in fields)
        if self.connection_text.get('1.0', 'end-1c') == text:
            return  # Preserve mouse selection during periodic status refreshes.
        self.connection_text.configure(state="normal")
        self.connection_text.delete('1.0', 'end')
        self.connection_text.insert('1.0', text)
        self.connection_text.configure(state="disabled")

    def _update_dash_conn(self):
        if not hasattr(self, "connection_text"):
            return
        running = bool(self.server.running()) and bool(self.server.port)
        cj = {}
        cfg_path = self.server.cfg_path if running else self._selected_cfg_path()
        if cfg_path:
            try:
                cj = json.loads(Path(cfg_path).read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                pass
        if running:
            active = getattr(self.server, 'active_config', {})
            # Keep the original display label, with the model ID actually served by the backend.
            if active.get('model_name'):
                cj = dict(cj)
                local = dict(cj.get('local_import') or {})
                if local:
                    local.setdefault('display_name', cj.get('model_name', ''))
                    cj['local_import'] = local
                cj['model_name'] = active['model_name']
            cj['host'] = self.server.host
        ips = local_ip_addresses() if cj.get('host') in ('0.0.0.0', '::') else []
        key = getattr(self.server, 'active_api_key', '') if running else (
            (cj.get('env') or {}).get('STRATA_API_KEY') or os.environ.get('STRATA_API_KEY') or cj.get('api_key') or '')
        fields = [('运行状态', self.status_chip.cget('text').removeprefix('● ').strip() if running else '未启动（以下为配置地址）')]
        fields += connection_fields(cj, actual_port=self.server.port if running else None, ips=ips, api_key=key)
        self._render_connection_fields(fields)

    def _is_our_server(self, pid):
        """判断占用端口的进程是否是本启动器启动的模型服务。"""
        proc = getattr(self.server, "proc", None)
        try:
            if pid and int(pid) == os.getpid() and getattr(self.server, 'gateway', None):
                return True
            return bool(pid) and proc is not None and proc.poll() is None and proc.pid == int(pid)
        except (OSError, ValueError):
            return False

    def open_quark_model_download(self):
        import webbrowser
        if not webbrowser.open(QUARK_MODEL_URL):
            self.toast("浏览器未打开", "请使用「复制链接」按钮，在浏览器中粘贴打开。", "warn")

    def copy_download_info(self, value, label):
        self.clipboard_clear()
        self.clipboard_append(value)
        self.toast("已复制" + label, "可粘贴到浏览器或网盘页面。", "success")

    # =================================================================== 安装页
    def _page_install(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.install_page = page

        download_card = make_card(page, "网盘模型下载")
        download_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(download_card, text="AI蛮子-Qwen3.8-Flash-Next-Strata启动器",
            text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
            anchor="w").pack(fill="x", padx=16, pady=(0, 5))
        ctk.CTkLabel(download_card, text=f"下载链接：{QUARK_MODEL_URL}\n提取码：{QUARK_MODEL_CODE}",
            text_color=COLORS["text"], font=ctk.CTkFont(family=FONT_MONO, size=12),
            justify="left", anchor="w").pack(fill="x", padx=16, pady=(0, 8))
        download_row = ctk.CTkFrame(download_card, fg_color="transparent")
        download_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(download_row, text="打开网盘下载", width=130,
            command=self.open_quark_model_download, fg_color=COLORS["accent"]).pack(side="left")
        ctk.CTkButton(download_row, text="复制链接", width=100, fg_color=COLORS["chip"],
            text_color=COLORS["text"],
            command=lambda: self.copy_download_info(QUARK_MODEL_URL, "链接")).pack(side="left", padx=8)
        ctk.CTkButton(download_row, text="复制提取码", width=110, fg_color=COLORS["chip"],
            text_color=COLORS["text"],
            command=lambda: self.copy_download_info(QUARK_MODEL_CODE, "提取码")).pack(side="left")

        local_card = make_card(page, "已有 GGUF 文件？从电脑导入模型")
        local_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(local_card, text=(
            "1. 点击下方按钮，选择放着同一个模型全部 .gguf 分片的文件夹。\n"
            "   例如 models\\IQ3_XXS 破限；不要选 models 总目录，也不用逐个选文件。\n"
            "2. 填写容易区分的名称，确认后等待准备完成；模型文件会直接使用，不再下载。\n"
            "3. 完成后到「概览」选择这个名称，点击「启动模型」。原版与破限版可分别使用。\n"
            "支持 Qwen3.8-Flash-Next GSQ-RCO 分片模型。首次准备可能联网补齐引擎或辅助资源。"),
            justify="left", wraplength=820, text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12)).pack(anchor="w", padx=16, pady=(0, 8))
        local_row = ctk.CTkFrame(local_card, fg_color="transparent")
        local_row.pack(fill="x", padx=16, pady=(0, 12))
        self.local_import_btn = ctk.CTkButton(local_row, text="导入本地 GGUF 模型…", width=190,
            command=self.import_local_model, fg_color=COLORS["accent"])
        self.local_import_btn.pack(side="left")
        ctk.CTkButton(local_row, text="查看新手说明", width=130, fg_color=COLORS["chip"],
            text_color=COLORS["text"], command=self.show_local_import_help).pack(side="left", padx=8)

        # --- 模式 A：已安装
        self.installed_card = make_card(page, "已安装的模型")
        self.installed_card.pack(fill="x", pady=(0, 12))
        self.installed_info = ctk.CTkLabel(self.installed_card, text="", text_color=COLORS["text"], justify="left",
                                           font=ctk.CTkFont(family=FONT, size=12), wraplength=820)
        self.installed_info.pack(anchor="w", padx=16, pady=(0, 8))
        arow = ctk.CTkFrame(self.installed_card, fg_color="transparent")
        arow.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(arow, text="在概览页启动", width=130, height=34, corner_radius=8,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=lambda: self.show_page("dashboard")).pack(side="left")
        ctk.CTkButton(arow, text="再装一个模型 / 改设置", width=190, height=34, corner_radius=8,
                      fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self._force_fresh_wizard).pack(side="left", padx=8)

        # --- 模式 B：安装向导
        self.wizard_card = make_card(page, "全新安装向导")
        self.wizard_card.pack(fill="x", pady=(0, 12))
        inner = ctk.CTkFrame(self.wizard_card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0, 12))

        # 1 硬件
        r, c = make_field(inner, "第 1 步 · 硬件体检", "安装前会调用 Strata 官方 setup.py --check 检查显卡、驱动、内存与磁盘。")
        r.pack(fill="x", pady=(0, 10))
        self.hw_label = ctk.CTkLabel(c, text="尚未检查", text_color=COLORS["text"], justify="left",
                                     font=ctk.CTkFont(family=FONT, size=12), wraplength=780)
        self.hw_label.pack(anchor="w")
        ctk.CTkButton(c, text="开始体检", width=110, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self.run_hardware_check).pack(anchor="w", pady=(6, 0))

        # 2 家族
        r, c = make_field(inner, "第 2 步 · 选择模型家族")
        r.pack(fill="x", pady=(0, 10))
        self.family_var = ctk.StringVar(value=self.cfg.get("last_family"))
        self.family_box = c
        self.family_radios = {}
        self._rebuild_family_radios()

        # 3 规格
        r, c = make_field(inner, "第 3 步 · 选择量化规格（体积 / 内存 / 质量）")
        r.pack(fill="x", pady=(0, 10))
        self.model_var2 = ctk.StringVar(value=self.cfg.get("last_model"))
        self.model_menu2 = ctk.CTkOptionMenu(c, variable=self.model_var2, values=[""], width=520, height=34,
                                             corner_radius=8, fg_color=COLORS["chip"], text_color=COLORS["text"],
                                             button_color="#dde5f7", button_hover_color="#cfdaf5",
                                             dropdown_text_color=COLORS["text"],
                                             font=ctk.CTkFont(family=FONT, size=12),
                                             command=lambda v: self._on_model_change())
        self.model_menu2.pack(anchor="w")
        self.model_detail = ctk.CTkLabel(c, text="", text_color=COLORS["text_dim"], justify="left",
                                         font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.model_detail.pack(anchor="w", pady=(4, 0))
        self._on_family_change()
        iq4 = ctk.CTkFrame(c, fg_color="transparent")
        iq4.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(iq4, text="选择 IQ4 模型（UD-IQ4_XS）", width=220,
                      command=self._select_iq4_install, fg_color=COLORS["accent"]).pack(side="left")
        ctk.CTkButton(iq4, text="查看官方模型文件", width=140,
                      fg_color=COLORS["chip"], text_color=COLORS["text"],
                      command=lambda: __import__('webbrowser').open(IQ4_MODEL_URL)).pack(side="left", padx=8)
        ctk.CTkLabel(c, text="IQ4 下载约 94 GB，需较大磁盘空间与内存；选择后按下方安装步骤下载。",
                     text_color=COLORS["text_dim"], font=ctk.CTkFont(family=FONT, size=11),
                     wraplength=780).pack(anchor="w", pady=(4, 0))

        # 4 上下文
        r, c = make_field(inner, "第 4 步 · 选择上下文长度",
                          "按模型提供上下文档位；DeepSeek 默认采用官方验证的 8K。")
        r.pack(fill="x", pady=(0, 10))
        self.ctx_var = ctk.StringVar(value=ctx_label(self.cfg.get("last_context")))
        self.ctx_menu = ctk.CTkOptionMenu(c, variable=self.ctx_var,
                                          values=[ctx_label(x) for x in self.env.tables()[2]],
                                          width=200, height=34, corner_radius=8, fg_color=COLORS["chip"],
                                          text_color=COLORS["text"], button_color="#dde5f7",
                                          button_hover_color="#cfdaf5", dropdown_text_color=COLORS["text"],
                                          font=ctk.CTkFont(family=FONT, size=12))
        self.ctx_menu.pack(anchor="w")
        self.ctx_note = ctk.CTkLabel(c, text="", text_color=COLORS["text_dim"], justify="left",
                                     font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.ctx_note.pack(anchor="w", pady=(4, 0))

        # 5 图片 / KV / 实验
        r, c = make_field(inner, "第 5 步 · 读图能力与 KV 缓存精度")
        r.pack(fill="x", pady=(0, 10))
        self.vision_var = ctk.StringVar(value=self.cfg.get("install_vision") or "gpu")
        self.kv_var = ctk.StringVar(value=self.cfg.get("install_kv") or "int8")
        self._option_row(c, "读图", self.vision_var, VISION_CHOICES)
        self._option_row(c, "KV 精度", self.kv_var, KV_CHOICES)
        self.esp_var = ctk.BooleanVar(value=bool(self.cfg.get("install_esp")))
        ctk.CTkCheckBox(c, text="启用拒答方向投影（实验功能）", variable=self.esp_var,
                        text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=12),
                        checkbox_width=20, checkbox_height=20).pack(anchor="w", pady=(6, 0))
        vector_row = ctk.CTkFrame(c, fg_color="transparent")
        vector_row.pack(fill="x", pady=(6, 0))
        self.esp_path_var = ctk.StringVar(value=self.cfg.get("install_esp_path") or "")
        ctk.CTkEntry(vector_row, textvariable=self.esp_path_var, width=430,
                     placeholder_text="留空使用 Strata 默认向量，或选择 Huihui 方向文件").pack(side="left", fill="x", expand=True)
        ctk.CTkButton(vector_row, text="选择方向文件…", width=125,
                      command=self.choose_projection_file).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(c, text="自定义文件需匹配 Qwen3.8-Flash-Next；改变拒答倾向，不降低内存需求。",
                     text_color=COLORS["text_dim"], wraplength=700).pack(anchor="w", pady=(4, 0))

        # 6 端口与局域网
        r, c = make_field(inner, "第 6 步 · 端口与网络访问")
        r.pack(fill="x", pady=(0, 10))
        netrow = ctk.CTkFrame(c, fg_color="transparent")
        netrow.pack(fill="x")
        ctk.CTkLabel(netrow, text="端口", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.port_entry = ctk.CTkEntry(netrow, width=100, height=32, corner_radius=8, text_color=COLORS["text"],
                                       font=ctk.CTkFont(family=FONT, size=12),
                                       placeholder_text="8080")
        self.port_entry.insert(0, str(self.cfg.get("last_port")))
        self.port_entry.pack(side="left", padx=(8, 18))
        self.lan_var = ctk.BooleanVar(value=bool(self.cfg.get("last_lan")))
        ctk.CTkCheckBox(netrow, text="允许局域网内其他设备访问（0.0.0.0）", variable=self.lan_var,
                        text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=12),
                        checkbox_width=20, checkbox_height=20,
                        command=self._on_lan_toggle).pack(side="left")
        keyrow = ctk.CTkFrame(c, fg_color="transparent")
        keyrow.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(keyrow, text="API 密钥", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.key_entry = ctk.CTkEntry(keyrow, width=330, height=32, corner_radius=8, text_color=COLORS["text"],
                                      font=ctk.CTkFont(family=FONT_MONO, size=11),
                                      placeholder_text="开启局域网时自动生成，可自行修改")
        if self.lan_var.get() and self.cfg.get("last_api_key"):
            self.key_entry.insert(0, self.cfg.get("last_api_key"))
        self.key_entry.pack(side="left", padx=(8, 8))
        self.key_gen_btn = ctk.CTkButton(keyrow, text="生成", width=64, height=32, corner_radius=8,
                                         fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                                         font=ctk.CTkFont(family=FONT, size=12), command=self._gen_key)
        self.key_gen_btn.pack(side="left")
        self.net_note = ctk.CTkLabel(c, text="", text_color=COLORS["text_dim"], justify="left",
                                     font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.net_note.pack(anchor="w", pady=(6, 0))
        self._on_lan_toggle()                      # 默认本地访问：禁用密钥输入

        # 7 数据目录
        r, c = make_field(inner, "第 7 步 · 模型数据目录",
                          "体积以所选规格为准。DeepSeek 约 239–508 GB；请手动选择空间足够的目录（必填）。")
        r.pack(fill="x", pady=(0, 10))
        drow = ctk.CTkFrame(c, fg_color="transparent")
        drow.pack(fill="x")
        self.data_entry = ctk.CTkEntry(drow, width=520, height=32, corner_radius=8, text_color=COLORS["text"],
                                       font=ctk.CTkFont(family=FONT, size=12),
                                       placeholder_text="必填：用「浏览…」或「扫描硬盘」选择目录")
        cur = self.cfg.get("data_dir") or (str(self.env.data_dir()) if self.env.data_dir() else "")
        if cur:
            self.data_entry.insert(0, cur)
        self.data_entry.pack(side="left")
        ctk.CTkButton(drow, text="浏览…", width=72, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._browse_data_dir).pack(side="left", padx=8)
        ctk.CTkButton(drow, text="扫描硬盘", width=84, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._scan_data_dir_install).pack(side="left")
        self.disk_note = ctk.CTkLabel(c, text="", text_color=COLORS["text_dim"], justify="left",
                                      font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.disk_note.pack(anchor="w", pady=(6, 0))
        self._update_disk_note()

        # 高级
        self.adv_open = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(inner, text="显示高级选项（低内存模式 / 多 GPU / 并行 / 显存预留 / 草稿词表）",
                        variable=self.adv_open, text_color=COLORS["text"],
                        font=ctk.CTkFont(family=FONT, size=12), checkbox_width=20, checkbox_height=20,
                        command=self._toggle_advanced).pack(anchor="w", pady=(4, 6))
        self.adv_frame = ctk.CTkFrame(inner, fg_color="#f8fafd", corner_radius=10, border_width=1,
                                      border_color=COLORS["border"])
        self._build_advanced(self.adv_frame)

        # 8 安装
        r, c = make_field(inner, "第 8 步 · 开始安装")
        r.pack(fill="x", pady=(4, 8))
        btnrow = ctk.CTkFrame(c, fg_color="transparent")
        btnrow.pack(fill="x")
        self.install_btn = ctk.CTkButton(btnrow, text="开始下载并安装", width=160, height=38, corner_radius=8,
                                         fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                         font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
                                         command=self.run_install)
        self.install_btn.pack(side="left")
        self.cancel_btn = ctk.CTkButton(btnrow, text="取消", width=88, height=38, corner_radius=8,
                                        fg_color=COLORS["danger"], hover_color="#c22f2f", state="disabled",
                                        font=ctk.CTkFont(family=FONT, size=13), command=self.cancel_install)
        self.cancel_btn.pack(side="left", padx=8)
        self.hf_var = ctk.BooleanVar(value=bool(self.cfg.get("install_hf_mirror", self.cfg.get("use_hf_mirror"))))
        ctk.CTkCheckBox(btnrow, text="使用 hf-mirror.com 镜像加速下载", variable=self.hf_var,
                        text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=12),
                        checkbox_width=20, checkbox_height=20,
                        command=lambda: self.cfg.set("use_hf_mirror", self.hf_var.get())).pack(side="left", padx=8)

        inst_log_card = make_card(page, "安装日志")
        inst_log_card.pack(fill="both", expand=True, pady=(0, 8))
        self.install_log = LogPane(inst_log_card, height=300)
        self.install_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        return page

    def _option_row(self, parent, label, var, choices):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label, width=70, text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        for text, val in choices:
            ctk.CTkRadioButton(row, text=text, variable=var, value=val, text_color=COLORS["text"],
                               font=ctk.CTkFont(family=FONT, size=12)).pack(side="left", padx=(0, 14))

    def _build_advanced(self, parent):
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=12)
        grid.grid_columnconfigure(1, weight=1)

        def row(r, text, widget_fn):
            ctk.CTkLabel(grid, text=text, text_color=COLORS["text"], anchor="w",
                         font=ctk.CTkFont(family=FONT, size=12)).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 10))
            w = widget_fn()
            w.grid(row=r, column=1, sticky="w", pady=4)
            return w

        self.gpu_entry = row(0, "指定 GPU 序号", lambda: ctk.CTkEntry(
            grid, width=180, height=30, corner_radius=8, text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12), placeholder_text="如 0 或 0,2（留空自动）"))
        if self.cfg.get("install_gpu"):
            self.gpu_entry.insert(0, self.cfg.get("install_gpu"))
        self.lowram_var = ctk.StringVar(value=self.cfg.get("install_lowram") or "auto")
        row(1, "低内存模式", lambda: ctk.CTkOptionMenu(
            grid, variable=self.lowram_var, values=[t for t, _ in LOWRAM_CHOICES], width=220, height=30,
            corner_radius=8, fg_color=COLORS["chip"], text_color=COLORS["text"], button_color="#dde5f7",
            button_hover_color="#cfdaf5", dropdown_text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12)))
        self.budget_entry = row(2, "RAM 预算（GiB）", lambda: ctk.CTkEntry(
            grid, width=180, height=30, corner_radius=8, text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12), placeholder_text="仅 UD-Q4_K_XL / UD-IQ4_XS 有效"))
        if self.cfg.get("install_ram_budget"):
            self.budget_entry.insert(0, self.cfg.get("install_ram_budget"))
        self.vram_entry = row(3, "显存预留（MiB）", lambda: ctk.CTkEntry(
            grid, width=180, height=30, corner_radius=8, text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12), placeholder_text="默认 500"))
        if self.cfg.get("install_vram_reserve"):
            self.vram_entry.insert(0, self.cfg.get("install_vram_reserve"))
        self.parallel_entry = row(4, "并行请求数", lambda: ctk.CTkEntry(
            grid, width=180, height=30, corner_radius=8, text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12), placeholder_text="默认 1（逐个处理）"))
        if self.cfg.get("install_parallel"):
            self.parallel_entry.insert(0, self.cfg.get("install_parallel"))
        self.draft_var = ctk.StringVar(value=self.cfg.get("install_draft_vocab") or "cjk")
        row(5, "草稿词表", lambda: ctk.CTkOptionMenu(
            grid, variable=self.draft_var, values=[t for t, _ in DRAFT_CHOICES], width=260, height=30,
            corner_radius=8, fg_color=COLORS["chip"], text_color=COLORS["text"], button_color="#dde5f7",
            button_hover_color="#cfdaf5", dropdown_text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12)))
        self.kvstream_var = ctk.StringVar(value=self.cfg.get("install_kv_streaming") or "auto")
        row(6, "KV 流式加载", lambda: ctk.CTkOptionMenu(
            grid, variable=self.kvstream_var, values=[t for t, _ in KVSTREAM_CHOICES], width=220, height=30,
            corner_radius=8, fg_color=COLORS["chip"], text_color=COLORS["text"], button_color="#dde5f7",
            button_hover_color="#cfdaf5", dropdown_text_color=COLORS["text"],
            font=ctk.CTkFont(family=FONT, size=12)))

    def _toggle_advanced(self):
        if self.adv_open.get():
            self.adv_frame.pack(fill="x", padx=0, pady=(0, 8))
        else:
            self.adv_frame.pack_forget()

    # ---- 向导交互
    def _force_fresh_wizard(self):
        self.cfg.set("mode", "fresh")
        self.apply_mode()
        self.show_page("install")

    def _rebuild_family_radios(self):
        """按当前加载到的真实家族表重建单选项（首次构造时用的是内置兜底表）。"""
        if not hasattr(self, "family_box"):
            return
        for rb in self.family_radios.values():
            rb.destroy()
        families = self.env.tables()[1]
        self.family_radios = {}
        for key, d in families.items():
            rb = ctk.CTkRadioButton(self.family_box, text=f"{d.get('title', key)} —— {d.get('about', '')}",
                                    variable=self.family_var, value=key, text_color=COLORS["text"],
                                    font=ctk.CTkFont(family=FONT, size=12),
                                    command=self._on_family_change)
            rb.pack(anchor="w", pady=2)
            self.family_radios[key] = rb
        if self.family_var.get() not in families:
            self.family_var.set(next(iter(families)))
        self._on_family_change()

    def _on_family_change(self):
        self.cfg.set("last_family", self.family_var.get())
        if not hasattr(self, "model_menu2"):       # 构造顺序：家族单选项先于规格下拉框建立
            return
        models = self.env.models_for(self.family_var.get())
        values = list(models)
        if not values:
            values = list(self.env.tables()[0])
        self.model_menu2.configure(values=values)
        if self.model_var2.get() not in values:
            self.model_var2.set(values[0])
        self._on_model_change()

    def _select_iq4_install(self):
        models, families, _ = self.env.tables()
        if 'unsloth' not in families or 'UD-IQ4_XS' not in models:
            self.toast('当前 Strata 版本不支持 IQ4', '请先在「维护工具」更新 Strata，再选择 IQ4 模型。', 'warn')
            return
        self.family_var.set('unsloth')
        self._on_family_change()
        self.model_var2.set('UD-IQ4_XS')
        self._on_model_change()

    def show_optimization_help(self):
        messagebox.showinfo('思考、运行与加速设置 · 新手说明',
            '深入思考、均衡思考和快速回答都会保留模型思考；客户端明确指定思考方式时以客户端为准。\n\n'
            '单任务优先适合个人使用。多任务同时运行允许两路同时生成，但更占显存，单个任务未必更快。\n\n'
            '长对话快速恢复需要客户端主动调用 /slots/0 的保存/恢复接口；不能与多任务同时运行一起使用。\n\n'
            '单次思考长度限制每次请求可用于思考的 token 数；应低于客户端的最大输出 token。\n\n'
            '增强加速和记住模型加速结果属于可选高级设置，建议使用实际任务比较速度。'
        )

    def _on_model_change(self):
        models = self.env.models_for(self.family_var.get())
        name = self.model_var2.get()
        d = models.get(name, {})
        if not d:
            return
        self.cfg.set("last_model", name)
        bits = []
        if d.get("download_gb"):
            bits.append(f"下载约 {d['download_gb']:.0f} GB")
        if d.get("ram_gb"):
            bits.append(f"建议内存 {d['ram_gb']} GB")
        if d.get("experimental"):
            bits.append("实验性")
        if d.get("budget"):
            bits.append("需要 RAM 预算（部分专家从 SSD 读取）")
        if d.get("vision"):
            bits.append("支持读图")
        if d.get('multi_file'):
            bits.append('独立引擎；Windows 需 NVIDIA RTX 30/40/50 系列；至少 12GB 显存；不支持读图')
        elif d.get('nvidia_only'):
            bits.append('仅支持 NVIDIA 显卡')
        elif d.get('own_path'):
            bits.append('支持 NVIDIA / AMD；当前不支持 Intel Arc')
        ctk_text = f"说明：{d.get('about', '')}\n" + ("　|　".join(bits) if bits else "")
        self.model_detail.configure(text=ctk_text)
        if hasattr(self, 'ctx_menu'):
            contexts = d.get('contexts') or self.env.tables()[2]
            self.ctx_menu.configure(values=[ctx_label(x) for x in contexts])
            if ctx_value(self.ctx_var.get()) not in contexts:
                self.ctx_var.set(ctx_label(contexts[0]))
            self.ctx_note.configure(text=('DeepSeek 使用独立引擎；当前采用官方已验证的 8K 上下文，纯文字模式。'
                if d.get('multi_file') else '安装选项会按所选模型适配；具体硬件支持由官方安装程序检查。'))
        self._update_disk_note()

    def _sync_key_field(self, entry, btn, enabled):
        """本地访问时禁用 API 密钥输入（不需要密钥）；开启局域网访问时启用。"""
        try:
            entry.configure(state="normal")
            if not enabled:
                entry.delete(0, "end")
            if btn is not None:
                btn.configure(state="normal" if enabled else "disabled")
            entry.configure(state="normal" if enabled else "disabled")
        except Exception:                          # noqa: BLE001
            pass

    def _on_lan_toggle(self, sync_other=True):
        lan = self.lan_var.get()
        self._sync_key_field(self.key_entry, self.key_gen_btn, lan)
        if lan:
            if not self.key_entry.get().strip():
                self.key_entry.insert(0, self.cfg.get("last_api_key") or generate_api_key())
                self.cfg.set("last_api_key", self.key_entry.get().strip())   # 自动生成的密钥持久化，重启不丢
            self.net_note.configure(text="已开启局域网访问：已自动生成 API 密钥（可点「生成」更换）。"
                                         "启动后可用「防火墙放行」按钮开放端口。")
        else:
            self.net_note.configure(text="仅本机可访问（127.0.0.1）：无需 API 密钥，最安全。")
        if sync_other and hasattr(self, "net_lan_var"):
            # 与「网络与配置」页的开关联动，避免两处保存同一设置时互相覆盖
            self.net_lan_var.set(lan)
            self._net_toggle(sync_other=False)

    def _gen_key(self):
        self.key_entry.delete(0, "end")
        self.key_entry.insert(0, generate_api_key())
        self.cfg.set("last_api_key", self.key_entry.get().strip())   # 记住新密钥，重启不丢

    def _fill_entry(self, entry, text):
        if entry is None:
            return
        entry.delete(0, "end")
        if text:
            entry.insert(0, str(text))

    def save_all_settings(self, source="手动保存", write_model=False):
        """收集当前界面上的启动器设置并写入本机配置文件。

        write_model=True（手动点「保存设置」）时，还会把网络设置一步到位写入选中模型的配置。
        """
        if hasattr(self, "family_var"):
            self.cfg.set("last_family", self.family_var.get())
        if hasattr(self, "model_var2"):
            self.cfg.set("last_model", self.model_var2.get())
        if hasattr(self, "ctx_var"):
            ctx = ctx_value(self.ctx_var.get())
            if ctx:
                self.cfg.set("last_context", ctx)
        if hasattr(self, "vision_var"):
            self.cfg.set("install_vision", self.vision_var.get())
        if hasattr(self, "kv_var"):
            self.cfg.set("install_kv", self.kv_var.get())
        if hasattr(self, "esp_var"):
            self.cfg.set("install_esp", bool(self.esp_var.get()))
        if hasattr(self, "esp_path_var"):
            self.cfg.set("install_esp_path", self.esp_path_var.get().strip())
        if hasattr(self, "hf_var"):
            self.cfg.set("install_hf_mirror", bool(self.hf_var.get()))
            self.cfg.set("use_hf_mirror", bool(self.hf_var.get()))
        if hasattr(self, "port_entry"):
            port = self.port_entry.get().strip()
            if port.isdigit():
                self.cfg.set("last_port", int(port))
        if hasattr(self, "lan_var"):
            self.cfg.set("last_lan", bool(self.lan_var.get()))
        if hasattr(self, "key_entry"):
            key = self.key_entry.get().strip()
            self.cfg.set("last_api_key", key if key else (self.cfg.get("last_api_key") or ""))
        if hasattr(self, "data_entry"):
            self.cfg.set("data_dir", self.data_entry.get().strip())
        if hasattr(self, "strata_dir_entry"):
            self.cfg.set("strata_dir", self.strata_dir_entry.get().strip())
        if hasattr(self, "data_dir_entry2"):
            self.cfg.set("data_dir", self.data_dir_entry2.get().strip() or self.cfg.get("data_dir"))
        if hasattr(self, "net_lan_var"):
            self.cfg.set("last_lan", bool(self.net_lan_var.get()))
        if hasattr(self, "net_port_entry"):
            port = self.net_port_entry.get().strip()
            if port.isdigit():
                self.cfg.set("last_port", int(port))
        if hasattr(self, "net_key_entry"):
            key = self.net_key_entry.get().strip()
            if key:
                self.cfg.set("last_api_key", key)
        if hasattr(self, "browser_var"):
            self.cfg.set("open_browser", bool(self.browser_var.get()))
        if hasattr(self, "compress_var"):
            self.cfg.set("context_compress_enabled", bool(self.compress_var.get()))
        if hasattr(self, "fit_var"):
            self.cfg.set("fit_max_tokens", bool(self.fit_var.get()))
        if hasattr(self, "runtime_vision_var"):
            self.cfg.set("runtime_vision_enabled", bool(self.runtime_vision_var.get()))
        if hasattr(self, "gpu_entry"):
            self.cfg.set("install_gpu", self.gpu_entry.get().strip())
        if hasattr(self, "lowram_var"):
            self.cfg.set("install_lowram", self.lowram_var.get())
        if hasattr(self, "budget_entry"):
            self.cfg.set("install_ram_budget", self.budget_entry.get().strip())
        if hasattr(self, "vram_entry"):
            self.cfg.set("install_vram_reserve", self.vram_entry.get().strip())
        if hasattr(self, "parallel_entry"):
            self.cfg.set("install_parallel", self.parallel_entry.get().strip())
        if hasattr(self, "draft_var"):
            self.cfg.set("install_draft_vocab", self.draft_var.get())
        if hasattr(self, "kvstream_var"):
            self.cfg.set("install_kv_streaming", self.kvstream_var.get())
        if hasattr(self, "adv_open"):
            self.cfg.set("install_advanced_open", bool(self.adv_open.get()))
        self.cfg.save()
        hint = ""
        if write_model and hasattr(self, "net_lan_var"):
            # 一步到位：把网络设置（开关 / 端口 / 密钥）写入选中模型的配置，重启模型后真正生效
            hint = "（网络设置已同步写入模型配置）" if self._net_save(silent=True) \
                else "（未选中模型，网络设置仅记入启动器偏好）"
        self.dash_log.append(f"启动器设置已保存（{source}）：{LAUNCHER_CONFIG}{hint}")
        return True

    def restore_all_settings(self):
        """把本机配置文件中记录的上一次启动器设置回填到当前界面。"""
        if hasattr(self, "family_var"):
            self.family_var.set(self.cfg.get("last_family") or "qwen")
        if hasattr(self, "model_var2"):
            self.model_var2.set(self.cfg.get("last_model") or "IQ2_XS")
        if hasattr(self, "ctx_var"):
            self.ctx_var.set(ctx_label(self.cfg.get("last_context")))
        if hasattr(self, "vision_var"):
            self.vision_var.set(self.cfg.get("install_vision") or "gpu")
        if hasattr(self, "kv_var"):
            self.kv_var.set(self.cfg.get("install_kv") or "int8")
        if hasattr(self, "esp_var"):
            self.esp_var.set(bool(self.cfg.get("install_esp")))
        if hasattr(self, "esp_path_var"):
            self.esp_path_var.set(self.cfg.get("install_esp_path") or "")
        if hasattr(self, "hf_var"):
            self.hf_var.set(bool(self.cfg.get("install_hf_mirror", self.cfg.get("use_hf_mirror"))))
        if hasattr(self, "port_entry"):
            self._fill_entry(self.port_entry, str(self.cfg.get("last_port") or "8080"))
        if hasattr(self, "lan_var"):
            self.lan_var.set(bool(self.cfg.get("last_lan")))
        if hasattr(self, "key_entry"):
            self._fill_entry(self.key_entry, self.cfg.get("last_api_key") or "")
        if hasattr(self, "data_entry"):
            self._fill_entry(self.data_entry, self.cfg.get("data_dir") or "")
        if hasattr(self, "strata_dir_entry"):
            self._fill_entry(self.strata_dir_entry, self.cfg.get("strata_dir") or "")
        if hasattr(self, "data_dir_entry2"):
            self._fill_entry(self.data_dir_entry2, self.cfg.get("data_dir") or "")
        if hasattr(self, "net_lan_var"):
            self.net_lan_var.set(bool(self.cfg.get("last_lan")))
        if hasattr(self, "net_port_entry"):
            self._fill_entry(self.net_port_entry, str(self.cfg.get("last_port") or "8080"))
        if hasattr(self, "net_key_entry"):
            self._fill_entry(self.net_key_entry, self.cfg.get("last_api_key") or "")
        if hasattr(self, "browser_var"):
            self.browser_var.set(bool(self.cfg.get("open_browser")))
        if hasattr(self, "compress_var"):
            self.compress_var.set(bool(self.cfg.get("context_compress_enabled")))
        if hasattr(self, "fit_var"):
            self.fit_var.set(bool(self.cfg.get("fit_max_tokens")))
        if hasattr(self, "runtime_vision_var"):
            self.runtime_vision_var.set(bool(self.cfg.get("runtime_vision_enabled")))
            self._refresh_runtime_vision_status()
            self._refresh_compress_status()
        if hasattr(self, "gpu_entry"):
            self._fill_entry(self.gpu_entry, self.cfg.get("install_gpu") or "")
        if hasattr(self, "lowram_var"):
            self.lowram_var.set(self.cfg.get("install_lowram") or "auto")
        if hasattr(self, "budget_entry"):
            self._fill_entry(self.budget_entry, self.cfg.get("install_ram_budget") or "")
        if hasattr(self, "vram_entry"):
            self._fill_entry(self.vram_entry, self.cfg.get("install_vram_reserve") or "")
        if hasattr(self, "parallel_entry"):
            self._fill_entry(self.parallel_entry, self.cfg.get("install_parallel") or "")
        if hasattr(self, "draft_var"):
            self.draft_var.set(self.cfg.get("install_draft_vocab") or "cjk")
        if hasattr(self, "kvstream_var"):
            self.kvstream_var.set(self.cfg.get("install_kv_streaming") or "auto")
        if hasattr(self, "adv_open"):
            self.adv_open.set(bool(self.cfg.get("install_advanced_open")))
        if hasattr(self, "lan_var"):
            self._on_lan_toggle()
        if hasattr(self, "net_lan_var"):
            self._net_toggle()
        if hasattr(self, "data_entry"):
            self._update_disk_note()
        self.dash_log.append("已恢复上次保存的启动器设置。")

    def _browse_data_dir(self):
        from tkinter import filedialog
        p = filedialog.askdirectory(title="选择模型数据目录（空间需求见所选规格）")
        if p:
            self.data_entry.delete(0, "end")
            self.data_entry.insert(0, p)
            self._update_disk_note()

    def _update_disk_note(self):
        if not hasattr(self, "disk_note"):
            return
        models = self.env.models_for(self.family_var.get())
        need = models.get(self.model_var2.get(), {}).get("download_gb") or 0
        need += models.get(self.model_var2.get(), {}).get('prepare_gb') or 0
        path = self.data_entry.get().strip() if hasattr(self, "data_entry") else ""
        if path:
            free = dir_free_gb(path)
            okk = "足够" if free >= need else "不足！"
            self.disk_note.configure(text=f"该目录剩余 {free:.0f} GB，本次需要约 {need:.0f} GB —— 空间{okk}",
                                     text_color=COLORS["text_dim"] if free >= need else COLORS["danger"])
        else:
            auto = pick_largest_free_drive()
            self.disk_note.configure(
                text=f"尚未选择目录。剩余空间最大的可用位置是 {auto}（约 {dir_free_gb(auto.anchor):.0f} GB）——"
                     "仅供参考，需你手动选定后才会使用。",
                text_color=COLORS["text_dim"])

    def run_hardware_check(self):
        if self.runner.running():
            return
        if self.env.root is None:
            self.toast("环境未就绪", "Strata 环境尚未准备完成，请稍候。", "warn")
            return
        self.install_log.append("=== 硬件体检（setup.py --check）===")
        self.hw_label.configure(text="检查中…")
        try:
            py = self.env.ensure_venv(lambda s: self.ui(self.install_log.append, s))
        except Exception as e:                     # noqa: BLE001
            self.hw_label.configure(text="检查失败：" + str(e))
            self.toast("运行环境创建失败", str(e), "error")
            return
        cmd = [py, str(self.env.root / "setup.py"), "--check", "--yes"]
        self.runner.run(cmd, self.env.root,
                        lambda line: self.ui(self.install_log.append, line),
                        lambda rc, err: self.ui(self._on_check_done, rc, err))

    def _on_check_done(self, rc, err):
        self.install_log.append(f"=== 体检结束（退出码 {rc}）===")
        gpus = query_nvidia_smi()
        ram = total_ram_gb()
        text = []
        if gpus:
            text.append("显卡：" + "；".join(f"GPU {i} {n}（{v:.0f} GB）" for i, n, v in gpus))
        else:
            text.append("显卡：未检测到可用 NVIDIA 显卡")
        text.append(f"内存：{ram:.0f} GB")
        text.append("（详细结论见左侧安装日志）")
        self.hw_label.configure(text="\n".join(text))

    # ---- 安装
    def _build_install_args(self):
        models = self.env.models_for(self.family_var.get())
        fam = self.family_var.get()
        model = self.model_var2.get()
        ctx = ctx_value(self.ctx_var.get())
        if ctx is None:
            raise ValueError("上下文格式无法识别。")
        vision = self.vision_var.get()
        kv = self.kv_var.get()
        args = ["--yes", "--family", fam, "--model", model, "--context", str(ctx),
                "--vision", vision, "--kv", kv, "--no-start"]

        # 网络
        port = self.port_entry.get().strip() or "8080"
        if not port.isdigit():
            raise ValueError("端口必须是数字。")
        args += ["--port", port]
        if self.lan_var.get():
            key = self.key_entry.get().strip()
            if not key:                            # 未填密钥时自动生成（官方要求局域网必须带密钥）
                key = generate_api_key()
                self._fill_entry(self.key_entry, key)
                self.cfg.set("last_api_key", key)
            args += ["--host", "0.0.0.0", "--api-key", key]
        else:
            args += ["--host", "127.0.0.1"]

        # 数据目录：必须由用户指定（不做自动挑盘，避免把未经确认的路径写进模型配置）
        dd = self.data_entry.get().strip()
        if not dd:
            raise ValueError("请先选择「模型数据目录」（用「浏览…」手动选择，或用「扫描硬盘」查找已有目录）。")
        args += ["--data-dir", dd]

        # 实验向量（默认关闭；仅在勾选时传递 on）
        args += projection_setup_args(self.esp_var.get(),
                                      self.esp_path_var.get() if hasattr(self, "esp_path_var") else '')

        # 高级
        if self.adv_open.get():
            gpu = self.gpu_entry.get().strip()
            if gpu:
                if "," in gpu:
                    args += ["--gpus", gpu]
                else:
                    args += ["--gpu", gpu]
            lowram = dict(LOWRAM_CHOICES).get(self.lowram_var.get(), self.lowram_var.get())
            if lowram and lowram != "auto":
                args += ["--low-ram", lowram]
            budget = self.budget_entry.get().strip()
            if budget:
                try:
                    if float(budget) <= 0:
                        raise ValueError
                    args += ["--resident-budget-gib", budget]
                except ValueError:
                    raise ValueError("RAM 预算必须是大于 0 的数字（GiB）。")
            vram = self.vram_entry.get().strip()
            if vram:
                if not vram.isdigit() or int(vram) < 0:
                    raise ValueError("显存预留必须是不小于 0 的整数（MiB）。")
                args += ["--vram-reserve-mib", vram]
            par = self.parallel_entry.get().strip()
            if par:
                if not par.isdigit() or int(par) < 1:
                    raise ValueError("并行请求数必须是 1 以上的整数。")
                args += ["--parallel", par]
            draft = dict(DRAFT_CHOICES).get(self.draft_var.get(), self.draft_var.get())
            if draft and draft != "cjk":
                args += ["--draft-vocab", draft]
            kvs = dict(KVSTREAM_CHOICES).get(self.kvstream_var.get(), self.kvstream_var.get())
            if kvs and kvs != "auto":
                args += ["--kv-streaming", kvs]
        if model not in models:
            raise ValueError('该规格不属于所选模型家族，请重新选择。')
        md = models[model]
        args = adapt_install_args(args, md)
        ctx = int(args[args.index('--context') + 1])
        vision = args[args.index('--vision') + 1]
        return args, md, ctx, vision, port

    def choose_projection_file(self):
        filename = filedialog.askopenfilename(title="选择拒答方向 GGUF 文件",
            filetypes=[("GGUF 方向文件", "*.gguf")])
        if filename:
            try:
                projection_setup_args(True, filename)
            except (ValueError, OSError) as error:
                self.toast("方向文件无效", str(error), "warn")
                return
            self.esp_path_var.set(filename)
            self.esp_var.set(True)

    def show_local_import_help(self):
        messagebox.showinfo("本地模型导入 · 新手说明",
            "GGUF 是模型文件，分片是一个大模型拆开的几份，必须全部放在同一个文件夹。\n\n"
            "支持 GSQ-RCO Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S、Coder IQ1_M，以及 UD-IQ4_XS / UD-Q4_K_XL。\n"
            "支持单文件或完整分片；普通 Q4_K_M 等量化尚不受 Strata 引擎支持。Q4 内存需求较高，32GB 内存不能保证运行。\n"
            "Coder 请选择含两个完整分片的 IQ1_M 文件夹；12GB 显存 + 32GB 内存建议先用 16K 上下文、关闭读图、低内存自动模式。\n\n"
            "你的两个模型需要分别导入：\n"
            "R:\\Strata-main\\Strata-data\\models\\IQ3_S 破限\n"
            "R:\\Strata-main\\Strata-data\\models\\IQ3_XXS 破限\n\n"
            "先在安装向导第 5 步选择读图模式：GPU 读图（推荐）、CPU 读图或关闭；再点击「导入本地 GGUF 模型」。\n"
            "初次不知道怎么选，可以保留默认设置。模型数据目录应选 Strata-data，不能选单个分片。\n\n"
            "导入会建立独立的准备文件和启动配置，不会搬走模型，也不会覆盖原版。"
            "首次准备可能下载引擎、依赖或辅助资源；请等待日志显示完成。\n\n"
            "完成后在概览页选模型名称并启动。以后直接启动即可。"
            "WebUI 可上传图片；视频会在浏览器抽取最多 8 张画面（不含声音），需要启用读图。\n\n"
            "要修改导入模型的设置，请重新导入同一个文件夹。移动模型后也需要重新导入。"
            "导入失败时先看日志，检查分片是否下载完整、格式是否受支持。")

    def import_local_model(self):
        if self.runner.running() or self.runtime.get("updating"):
            self.toast("有任务在运行", "请等待当前任务完成后再导入。", "warn")
            return
        if self.env.root is None:
            self.toast("环境未就绪", "请先在网络与配置页选择 Strata 源码目录。", "warn")
            return
        if self.server.running():
            self.toast("请先停止模型", "停止当前模型后再导入，避免准备文件正在被使用。", "warn")
            return
        folder = filedialog.askdirectory(title="选择一个模型全部 GGUF 分片所在的文件夹",
            initialdir=str(Path(self.effective_data_dir()) / 'models') if self.effective_data_dir() else None)
        if not folder:
            return
        try:
            info = inspect_model(folder)
            detected_vector = find_huihui_vector(folder) if info['family'] == 'coder' else None
            if detected_vector and not self.esp_path_var.get().strip():
                self.esp_path_var.set(str(detected_vector))
                self.esp_var.set(True)
            args, _, ctx, vision, port = self._build_install_args()
        except (ValueError, OSError) as error:
            self.toast("无法导入", str(error), "warn")
            return
        label = simpledialog.askstring("给模型起个名字", "这个名称会显示在概览页，例如 IQ3_XXS 破限：",
            initialvalue=Path(folder).name, parent=self)
        if label is None:
            return
        label = label.strip()
        if not label or len(label) > 80:
            self.toast("名称有误", "请输入 1 至 80 个字的模型名称。", "warn")
            return
        if not messagebox.askyesno("确认导入本地模型",
            f"名称：{label}\n规格：{info['model']}（{len(info['shards'])} 个分片）\n"
            f"文件夹：{folder}\n上下文：{ctx_label(ctx)}\n\n"
            f"拒答方向投影：{(self.esp_path_var.get() or 'Strata 默认向量') if self.esp_var.get() else '关闭'}\n\n"
            "直接使用已有文件，建立独立准备目录与启动配置。首次可能联网补齐引擎和辅助资源。\n"
            "请保持窗口开启，进度会显示在安装日志中。\n\n开始准备吗？"):
            return
        self._force_fresh_wizard()
        self.save_all_settings("本地导入前自动保存")
        self.install_log.clear()
        self.install_log.append("=== 正在准备本地模型（不重新下载模型分片）===")
        self.install_btn.configure(state="disabled")
        self.local_import_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        try:
            py = self.env.ensure_venv(lambda s: self.ui(self.install_log.append, s))
            source = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'strata_local_import.py'
            helper = self.env.root / '.launcher-local-import.py'
            helper.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
            cmd = [py, str(helper), str(self.env.root), folder,
                   self.data_entry.get().strip(), label, *args]
        except Exception as error:
            self._on_install_done(-1, str(error))
            return
        self.runtime['installing'] = True
        self._local_import_label = label
        self.runner.run(cmd, self.env.root,
            lambda line: self.ui(self.install_log.append, line),
            lambda rc, err: self.ui(self._on_install_done, rc, err),
            env={'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
                 **({'HF_ENDPOINT': HF_MIRROR} if self.hf_var.get() else {})})

    def run_install(self):
        if self.runner.running():
            self.toast("正在安装", "已有安装任务在运行。", "warn")
            return
        if self.env.root is None:
            self.toast("环境未就绪", "Strata 环境尚未准备完成，请稍候。", "warn")
            return
        try:
            args, md, ctx, vision, port = self._build_install_args()
        except ValueError as e:
            self.toast("参数有误", str(e), "warn")
            return

        need = (md.get("download_gb") or 0) + (md.get('prepare_gb') or 0)
        target = Path(self.data_entry.get().strip())
        free = dir_free_gb(target.anchor if not target.exists() else target)
        if free and free < need:
            if not messagebox.askyesno("磁盘空间可能不足",
                                       f"目标目录剩余 {free:.0f} GB，本次模型约需 {need:.0f} GB。\n\n仍要继续吗？"):
                return
        warn = (f"即将开始安装：\n\n"
                f"模型：{md.get('about', '')}\n"
                f"下载体积：约 {need:.0f} GB（可断点续传）\n"
                f"数据目录：{target}\n"
                f"上下文：{ctx}\n"
                f"读图：{vision}\n\n"
                f"下载与准备时间较长（视网速可能需要数小时），期间请保持本窗口开启。\n\n确定开始吗？")
        if not messagebox.askyesno("确认安装", warn):
            return

        # 记录配置：安装前自动保存当前所有启动器设置，保证下次可恢复。
        self.save_all_settings("安装前自动保存")

        self.install_log.clear()
        self.install_log.append("=== 开始安装（调用 Strata 官方 setup.py）===")
        self.install_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        try:
            py = self.env.ensure_venv(lambda s: self.ui(self.install_log.append, s))
        except Exception as e:                     # noqa: BLE001
            self.install_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.toast("运行环境创建失败", str(e), "error")
            return
        cmd = [py, str(self.env.root / "setup.py")] + args
        self.install_log.append("$ " + " ".join(cmd))
        env = {"HF_ENDPOINT": HF_MIRROR, "PYTHONUNBUFFERED": "1"} if self.hf_var.get() else {"PYTHONUNBUFFERED": "1"}
        if self.hf_var.get():
            self.install_log.append(f"已启用镜像：HF_ENDPOINT={HF_MIRROR}")
        self.runtime["installing"] = True
        self.runner.run(cmd, self.env.root,
                        lambda line: self.ui(self.install_log.append, line),
                        lambda rc, err: self.ui(self._on_install_done, rc, err), env=env)

    def cancel_install(self):
        if not self.runner.running():
            return
        if not messagebox.askyesno("取消安装",
                                   "已下载的部分会被保留（支持断点续传）。\n确定取消吗？"):
            return
        self.install_log.append("正在取消…")
        self.runner.cancel()

    def _on_install_done(self, rc, err):
        local_label = getattr(self, '_local_import_label', None)
        self._local_import_label = None
        self.local_import_btn.configure(state="normal")
        self.install_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.runtime["installing"] = False
        if err:
            self.install_log.append("启动安装进程失败：" + err)
            self.toast("安装失败", err, "error")
            return
        if rc != 0:
            self.install_log.append(f"=== 安装中断（退出码 {rc}）。已完成的步骤会被跳过，重新执行即可续传。===")
            return
        self.install_log.append("=== 安装完成 ===")
        self.cfg.set("mode", "installed")
        self.rescan_models()                       # 刚装完，显式重扫一次以刷新模型列表
        if local_label:
            self.model_var.set(local_label)
        if messagebox.askyesno("安装完成", "模型已安装完成。\n\n现在启动模型吗？\n"
                                           "（首次加载会占用大量内存，电脑可能卡顿 1-3 分钟）"):
            self.show_page("dashboard")
            self.after(300, self.start_model)

    # =================================================================== 网络与配置页
    def _page_network(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)

        # ---- 目录设置
        dir_card = make_card(page, "目录设置")
        dir_card.pack(fill="x", pady=(0, 12))
        dinner = ctk.CTkFrame(dir_card, fg_color="transparent")
        dinner.pack(fill="x", padx=16, pady=(0, 12))

        srow = ctk.CTkFrame(dinner, fg_color="transparent")
        srow.pack(fill="x")
        ctk.CTkLabel(srow, text="Strata 目录", width=92, anchor="w", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.strata_dir_entry = ctk.CTkEntry(srow, width=400, height=32, corner_radius=8,
                                             text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=11),
                                             placeholder_text="留空 = 自动定位（内置 / 已释放的源码）")
        self.strata_dir_entry.pack(side="left", padx=8)
        ctk.CTkButton(srow, text="浏览…", width=64, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._pick_strata_dir).pack(side="left")
        ctk.CTkButton(srow, text="扫描硬盘", width=84, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._scan_strata_dir).pack(side="left", padx=8)
        ctk.CTkButton(srow, text="使用自动", width=76, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._auto_strata_dir).pack(side="left", padx=8)
        ctk.CTkButton(srow, text="应用", width=64, height=32, corner_radius=8, fg_color=COLORS["accent"],
                      hover_color=COLORS["accent_hover"], font=ctk.CTkFont(family=FONT, size=12),
                      command=self._apply_strata_dir).pack(side="left")

        mrow = ctk.CTkFrame(dinner, fg_color="transparent")
        mrow.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(mrow, text="模型数据目录", width=92, anchor="w", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.data_dir_entry2 = ctk.CTkEntry(mrow, width=400, height=32, corner_radius=8,
                                            text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=11),
                                            placeholder_text="留空 = 不指定（由官方 setup.py 决定；建议手动选择）")
        self.data_dir_entry2.pack(side="left", padx=8)
        ctk.CTkButton(mrow, text="浏览…", width=64, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._pick_data_dir).pack(side="left")
        ctk.CTkButton(mrow, text="扫描硬盘", width=84, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._scan_data_dir).pack(side="left", padx=8)
        ctk.CTkButton(mrow, text="使用自动", width=76, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self._auto_data_dir).pack(side="left", padx=8)
        ctk.CTkButton(mrow, text="保存", width=64, height=32, corner_radius=8, fg_color=COLORS["accent"],
                      hover_color=COLORS["accent_hover"], font=ctk.CTkFont(family=FONT, size=12),
                      command=self._save_data_dir).pack(side="left")

        self.dir_status = ctk.CTkLabel(dinner, text="", text_color=COLORS["text_dim"], justify="left",
                                       font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.dir_status.pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(dinner, text="说明：手动指定 Strata 目录后，启动器会优先使用该目录中的源码；"
                                  "模型数据目录会作为安装模型时的下载位置，安装时必须指定。\n"
                                  "「扫描硬盘」会查找磁盘上含 setup.py 的源码目录与含 models/packs/mtp 的数据目录，由你挑选；"
                                  "切换模式不会扫描磁盘。",
                     text_color=COLORS["text_dim"], justify="left",
                     font=ctk.CTkFont(family=FONT, size=11), wraplength=780).pack(anchor="w", pady=(4, 0))

        card = make_card(page, "局域网访问（把服务开放给手机 / 其他电脑）")
        card.pack(fill="x", pady=(0, 12))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0, 12))

        self.net_lan_var = ctk.BooleanVar(value=bool(self.cfg.get("last_lan")))
        ctk.CTkCheckBox(inner, text="允许局域网内其他设备访问（写入模型配置 host=0.0.0.0）",
                        variable=self.net_lan_var, text_color=COLORS["text"],
                        font=ctk.CTkFont(family=FONT, size=12), checkbox_width=20, checkbox_height=20,
                        command=self._net_toggle).pack(anchor="w")

        krow = ctk.CTkFrame(inner, fg_color="transparent")
        krow.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(krow, text="API 密钥", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.net_key_entry = ctk.CTkEntry(krow, width=360, height=32, corner_radius=8, text_color=COLORS["text"],
                                          font=ctk.CTkFont(family=FONT_MONO, size=11),
                                          placeholder_text="开启局域网时自动生成，可自行修改")
        self.net_key_entry.pack(side="left", padx=8)
        self.net_key_gen_btn = ctk.CTkButton(krow, text="生成", width=64, height=32, corner_radius=8,
                                             fg_color=COLORS["chip"], text_color=COLORS["text"],
                                             hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                                             command=self._net_gen_key)
        self.net_key_gen_btn.pack(side="left")

        prow = ctk.CTkFrame(inner, fg_color="transparent")
        prow.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(prow, text="端口", text_color=COLORS["text"],
                     font=ctk.CTkFont(family=FONT, size=12)).pack(side="left")
        self.net_port_entry = ctk.CTkEntry(prow, width=100, height=32, corner_radius=8, text_color=COLORS["text"],
                                           font=ctk.CTkFont(family=FONT, size=12), placeholder_text="8080")
        self.net_port_entry.insert(0, str(self.cfg.get("last_port")))
        self.net_port_entry.pack(side="left", padx=8)
        ctk.CTkButton(prow, text="保存到模型配置", width=140, height=32, corner_radius=8,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self._net_save).pack(side="left", padx=8)
        ctk.CTkButton(prow, text="防火墙放行该端口", width=150, height=32, corner_radius=8,
                      fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self._net_firewall).pack(side="left")

        self.net_status = ctk.CTkLabel(inner, text="", text_color=COLORS["text_dim"], justify="left",
                                       font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.net_status.pack(anchor="w", pady=(10, 0))
        self._net_toggle()                         # 初始化：默认本地访问 → 禁用密钥输入

        conn_card = make_card(page, "连接信息")
        conn_card.pack(fill="x", pady=(0, 12))
        self.conn_label = ctk.CTkLabel(conn_card, text="", text_color=COLORS["text"], justify="left",
                                       font=ctk.CTkFont(family=FONT, size=12), wraplength=820)
        self.conn_label.pack(anchor="w", padx=16, pady=(0, 8))
        ctk.CTkButton(conn_card, text="刷新连接信息", width=130, height=32, corner_radius=8,
                      fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self._refresh_conn).pack(anchor="w", padx=16, pady=(0, 12))

        pref_card = make_card(page, "启动器偏好")
        pref_card.pack(fill="x", pady=(0, 12))
        inner2 = ctk.CTkFrame(pref_card, fg_color="transparent")
        inner2.pack(fill="x", padx=16, pady=(0, 12))
        self.browser_var = ctk.BooleanVar(value=bool(self.cfg.get("open_browser")))
        ctk.CTkCheckBox(inner2, text="启动模型后自动打开 Strata 聊天页面", variable=self.browser_var,
                        text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=12),
                        checkbox_width=20, checkbox_height=20,
                        command=lambda: self.cfg.set("open_browser", self.browser_var.get())).pack(anchor="w")
        ctk.CTkLabel(inner2, text=f"启动器配置：{LAUNCHER_CONFIG}", text_color=COLORS["text_dim"],
                     font=ctk.CTkFont(family=FONT_MONO, size=11)).pack(anchor="w", pady=(8, 0))
        btn_row = ctk.CTkFrame(inner2, fg_color="transparent")
        btn_row.pack(anchor="w", pady=(8, 0))
        ctk.CTkButton(btn_row, text="保存设置", width=92, height=30, corner_radius=8,
                      fg_color=COLORS["accent"], text_color="white", hover_color="#3f465c",
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=lambda: self.save_all_settings("手动保存", write_model=True)).pack(side="left")
        ctk.CTkButton(btn_row, text="恢复上次设置", width=108, height=30, corner_radius=8,
                      fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self.restore_all_settings).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(inner2, text="保存设置会把当前页面与安装向导里的配置写入上方文件；恢复上次设置会回填这些内容。",
                     text_color=COLORS["text_dim"], font=ctk.CTkFont(family=FONT, size=11),
                     wraplength=760, justify="left").pack(anchor="w", pady=(8, 0))
        return page

    # ---------------- 目录设置
    def _refresh_dir_fields(self):
        if not hasattr(self, "strata_dir_entry"):
            return
        sd = self.cfg.get("strata_dir") or ""
        self.strata_dir_entry.delete(0, "end")
        if sd:
            self.strata_dir_entry.insert(0, sd)
        dd = self.cfg.get("data_dir") or ""
        self.data_dir_entry2.delete(0, "end")
        if dd:
            self.data_dir_entry2.insert(0, dd)
        lv = self.env.source_version()
        bits = [f"当前生效的 Strata 目录：{self.env.root or '（未就绪）'}"]
        if not sd:
            bits.append("定位方式：自动（优先使用数据目录旁已有的安装）")
        if lv:
            bits.append(f"版本 v{lv}")
        bits.append(f"模型数据目录：{self.effective_data_dir() or '（未指定）'}")
        self.dir_status.configure(text="　｜　".join(bits))

    def _pick_strata_dir(self):
        init = self.strata_dir_entry.get().strip() or str(self.env.root or APP_DIR)
        d = filedialog.askdirectory(title="选择 Strata 源码目录（该目录下应包含 setup.py）", initialdir=init)
        if not d:
            return
        self.strata_dir_entry.delete(0, "end")
        self.strata_dir_entry.insert(0, d)
        self._apply_strata_dir()

    def _auto_strata_dir(self):
        self.strata_dir_entry.delete(0, "end")
        self._apply_strata_dir()

    def _apply_strata_dir(self):
        path = self.strata_dir_entry.get().strip()
        if path and not (Path(path) / "setup.py").is_file():
            sub = self.env.descend_root(path)
            if sub is not None:
                if not messagebox.askyesno(
                        "发现嵌套的源码目录",
                        f"{path}\n\n该目录下没有 setup.py，但在子目录中发现了：\n{sub}\n\n是否改用该子目录作为 Strata 目录？"):
                    self._refresh_dir_fields()
                    return
                path = str(sub)
                self.strata_dir_entry.delete(0, "end")
                self.strata_dir_entry.insert(0, path)
            elif not messagebox.askyesno(
                    "目录可能无效",
                    f"{path}\n\n该目录下未找到 setup.py。\n仍要使用该目录吗？"):
                self._refresh_dir_fields()
                return
        self.cfg.set("strata_dir", path)
        self.env.set_manual_root(path)
        self.dash_log.append(f"已切换 Strata 目录：{path or '（自动定位）'}，正在重新定位环境…")
        self.show_page("dashboard")
        self.bootstrap()
        self.rescan_models(note=f"Strata 目录：{path or '自动定位'}")

    def _pick_data_dir(self):
        init = self.data_dir_entry2.get().strip() or self.effective_data_dir() or str(APP_DIR)
        d = filedialog.askdirectory(title="选择模型数据目录（models / packs / mtp 所在目录）", initialdir=init)
        if not d:
            return
        self.data_dir_entry2.delete(0, "end")
        self.data_dir_entry2.insert(0, d)
        self._save_data_dir()

    def _auto_data_dir(self):
        self.data_dir_entry2.delete(0, "end")
        self._save_data_dir()

    def _save_data_dir(self):
        path = self.data_dir_entry2.get().strip()
        if path:
            p = Path(path)
            if p.name.lower() in ("models", "packs", "mtp") and p.parent.is_dir():
                fixed = str(p.parent)
                if messagebox.askyesno(
                        "路径像是子目录",
                        f"你选择的是：\n{path}\n\n这是数据目录下的 {p.name} 子目录。\n"
                        f"模型数据目录应指向包含 models / packs / mtp 的根目录：\n{fixed}\n\n"
                        "是否改用该根目录？"):
                    path = fixed
                    self.data_dir_entry2.delete(0, "end")
                    self.data_dir_entry2.insert(0, path)
        self.cfg.set("data_dir", path)
        if hasattr(self, "data_entry"):
            self.data_entry.delete(0, "end")
            if path:
                self.data_entry.insert(0, path)
            self._update_disk_note()
        self._relocate_root_from_data()            # 数据目录旁的已有安装：自动采用，让其中配置生效
        self._refresh_dir_fields()
        self._fill_env_info()
        self.rescan_models()
        self.dir_status.configure(
            text=f"已保存模型数据目录：{path or '（未指定，安装时必填）'}。安装模型时将把模型下载到该目录。",
            text_color=COLORS["success"])

    # ---------------- 扫描硬盘（后台线程，结果由用户挑选）
    def _start_disk_scan(self, kind, on_found, note_label=None):
        if getattr(self, "_scanning", False):
            return
        self._scanning = True
        if note_label is not None:
            note_label.configure(text="正在扫描磁盘，请稍候…", text_color=COLORS["text_dim"])

        def work():
            try:
                sources, datas = scan_disk_roots()
                self.ui(self._finish_disk_scan, on_found, sources if kind == "source" else datas,
                        None, note_label)
            except Exception as e:                 # noqa: BLE001
                self.ui(self._finish_disk_scan, on_found, None, str(e), note_label)

        threading.Thread(target=work, daemon=True).start()

    def _finish_disk_scan(self, on_found, items, err, note_label):
        self._scanning = False
        if err:
            if note_label is not None:
                note_label.configure(text=f"扫描失败：{err}", text_color=COLORS["danger"])
            return
        if not items:
            if note_label is not None:
                note_label.configure(text="未在磁盘上找到候选目录，请用「浏览…」手动选择。",
                                     text_color=COLORS["warning"])
            return
        if note_label is not None:
            note_label.configure(text=f"扫描到 {len(items)} 个候选目录。", text_color=COLORS["text_dim"])
        on_found(items)

    def _scan_strata_dir(self):
        self._start_disk_scan("source", self._scanned_strata, self.dir_status)

    def _scanned_strata(self, items):
        chosen = items[0] if len(items) == 1 else choose_from_list(
            self, "选择 Strata 源码目录",
            "扫描到以下含 setup.py 的 Strata 源码目录，请选择：", items)
        if not chosen:
            return
        self.strata_dir_entry.delete(0, "end")
        self.strata_dir_entry.insert(0, chosen)
        self._apply_strata_dir()

    def _scan_data_dir(self):
        self._start_disk_scan("data", self._scanned_data_dir, self.dir_status)

    def _scanned_data_dir(self, items):
        chosen = items[0] if len(items) == 1 else choose_from_list(
            self, "选择模型数据目录",
            "扫描到以下模型数据目录（含 models/packs/mtp），请选择：", items)
        if not chosen:
            return
        self.data_dir_entry2.delete(0, "end")
        self.data_dir_entry2.insert(0, chosen)
        self._save_data_dir()

    def _scan_data_dir_install(self):
        self._start_disk_scan("data", self._scanned_data_dir_install, getattr(self, "disk_note", None))

    def _scanned_data_dir_install(self, items):
        chosen = items[0] if len(items) == 1 else choose_from_list(
            self, "选择模型数据目录",
            "扫描到以下模型数据目录（含 models/packs/mtp），请选择：", items)
        if not chosen:
            return
        self.data_entry.delete(0, "end")
        self.data_entry.insert(0, chosen)
        self._update_disk_note()

    # ---------------- 上下文快捷设置（概览页，启动前随时可调）
    def _refresh_ctx_quick(self, _=None):
        if hasattr(self, 'fit_var') and not self.server.running():
            path = self._selected_cfg_path()
            if path:
                try:
                    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
                    self.compress_var.set(bool(cfg.get('context_compress_enabled', self.cfg.get('context_compress_enabled'))))
                    self.fit_var.set(bool(cfg.get('fit_max_tokens', self.cfg.get('fit_max_tokens'))))
                    self._context_sync_status = '当前模型设置；启动后在服务端生效'
                    self._refresh_compress_status()
                except (OSError, ValueError):
                    pass
        if not hasattr(self, "ctx_quick_menu"):
            return
        _models, _families, contexts = self.env.tables()
        vals = [ctx_label(c) for c in contexts]
        self.ctx_quick_menu.configure(values=vals)
        ent = self._selected_cfg() if hasattr(self, "_cfg_map") else None
        if ent is None:
            ents = self.model_cache()
            ent = ents[0] if ents else None
        if ent:
            md = self.env.models_for(ent.get('family')).get(ent.get('model'), {})
            if md.get('contexts'):
                contexts = md['contexts']
                vals = [ctx_label(c) for c in contexts]
                self.ctx_quick_menu.configure(values=vals)
        self._quick_ent = ent
        if ent is None:
            # 全新模式：尚未安装模型，这里选的是安装时要用的值
            self.ctx_quick_menu.configure(state="normal")
            last = self.cfg.get("last_context")
            if last is not None and ctx_label(last) in vals:
                self.ctx_quick_menu.set(ctx_label(last))
            elif vals:
                self.ctx_quick_menu.set(vals[0])
            self.ctx_quick_info.configure(text="尚未安装模型：此处选择的是安装时要使用的上下文。",
                                          text_color=COLORS["text_dim"])
            self.ctx_quick_status.configure(text="")
            return
        cur = ent.get("context")
        unconfigured = not ent.get("path")
        if unconfigured:
            txt = f"当前模型：{ent.get('label', '—')}　｜　上下文：尚未生成配置"
        else:
            txt = f"当前模型：{ent.get('label', '—')}　｜　上下文：{ctx_label(cur) if cur else '未知'}"
            if ent.get("kv"):
                txt += f"　｜　KV 精度：{ent['kv']}"
        self.ctx_quick_info.configure(text=txt, text_color=COLORS["text"])
        if cur and ctx_label(cur) in vals:
            self.ctx_quick_menu.set(ctx_label(cur))
        elif vals:
            last = self.cfg.get("last_context")
            if unconfigured and last is not None and ctx_label(last) in vals:
                self.ctx_quick_menu.set(ctx_label(last))
            else:
                self.ctx_quick_menu.set(vals[0])
        ready = bool(ent.get("family") and ent.get("model"))
        self.ctx_quick_menu.configure(state="normal" if ready else "disabled")
        if not ready:
            status, color = "无法识别该配置，请用安装向导调整。", COLORS["warning"]
        elif unconfigured:
            status, color = ("该模型已下载但尚未生成配置：选好上下文后点「应用上下文」即可生成配置"
                             "（不会重新下载模型）。"), COLORS["success"]
        else:
            status, color = "", COLORS["text_dim"]
        self.ctx_quick_status.configure(text=status, text_color=color)

    def _open_config_file(self):
        ent = getattr(self, "_quick_ent", None)
        if ent is None or not ent.get("path"):
            self.tool_open_config()
            return
        try:
            subprocess.Popen(["notepad", str(ent.get("path", ""))])
        except Exception as e:                     # noqa: BLE001
            self.toast("打开失败", str(e), "error")

    def _apply_local_quick_context(self, ent, new_ctx):
        path = Path(ent['path'])
        restart = self.server.running() and Path(self.server.cfg_path).resolve() == path.resolve()
        note = "当前模型将自动重启并重新加载，期间服务暂时不可用。" if restart else "下次启动该模型时生效。"
        if not messagebox.askyesno("修改本地模型上下文",
                f"新上下文：{ctx_label(new_ctx)}\n{note}\n\n无需重新导入；较大的上下文会占用更多内存。确定应用吗？"):
            return
        try:
            saved = save_local_context(path, new_ctx)
        except (OSError, ValueError, IndexError) as error:
            self.toast("修改上下文失败", str(error), "error")
            return
        ent.update(local_context_choices(saved))
        self.cfg.set("last_context", new_ctx)
        self.cfg.set("cached_models", self.model_cache())
        self._refresh_ctx_quick()
        self.ctx_quick_status.configure(text=f"已应用上下文 {ctx_label(new_ctx)}；" +
            ("正在重启模型服务。" if restart else "下次启动生效。"), text_color=COLORS["success"])
        if restart:
            self.restart_model()

    def _apply_quick_context(self):
        new_ctx = ctx_value(self.ctx_quick_menu.get())
        if new_ctx is None:
            return
        ent = getattr(self, "_quick_ent", None)
        if ent is None:                            # 全新模式：只记录，安装时使用
            self.cfg.set("last_context", new_ctx)
            if hasattr(self, "ctx_var"):
                self.ctx_var.set(ctx_label(new_ctx))
            self.ctx_quick_status.configure(text=f"已记录：安装模型时将使用上下文 {ctx_label(new_ctx)}。",
                                            text_color=COLORS["success"])
            return
        if not (ent.get("family") and ent.get("model")):
            self.ctx_quick_status.configure(text="无法从该配置解析出家族/规格，请改用「安装 / 模型」页的向导调整。",
                                            text_color=COLORS["danger"])
            return
        if not ent.get("path"):                    # 已下载但未配置：生成配置，并使用所选上下文
            self._generate_config(ent, "生成模型配置",
                                  "模型：{spec}（家族 {fam}）\n数据目录：{dd}\n上下文：{ctx}\n\n"
                                  "该模型已下载但还没有配置文件，将调用官方 setup.py 生成：\n"
                                  "已下载的模型文件会被复用，不会重新下载。", ctx=new_ctx)
            return
        if self.runner.running() or self.runtime.get("updating"):
            self.ctx_quick_status.configure(text="有任务正在运行，请稍后再试。", text_color=COLORS["warning"])
            return
        old = ent.get("context")
        if old == new_ctx:
            self.ctx_quick_status.configure(text=f"当前上下文已经是 {ctx_label(new_ctx)}，无需修改。",
                                            text_color=COLORS["text_dim"])
            return
        if getattr(self, '_starting', False):
            self.ctx_quick_status.configure(text="模型正在启动，请加载完成后再调整。", text_color=COLORS["warning"])
            return
        try:
            local = json.loads(Path(ent['path']).read_text(encoding='utf-8-sig')).get('local_import')
        except (OSError, ValueError) as error:
            self.toast("读取配置失败", str(error), "error")
            return
        if local:
            self._apply_local_quick_context(ent, new_ctx)
            return
        dd = self.effective_data_dir()
        note = ""
        if self.server.running():
            note = "\n注意：模型服务正在运行，修改后需要重启服务才会生效。\n"
        if not messagebox.askyesno("修改上下文",
                                   f"配置：{Path(ent.get('path', '')).name}\n"
                                   f"当前上下文：{ctx_label(old) if old else '未知'}\n"
                                   f"新上下文：{ctx_label(new_ctx)}\n"
                                   f"模型数据目录：{dd or '（由官方自动判断）'}\n{note}\n"
                                   "将调用官方 setup.py 重写该配置：已下载的模型文件会被复用，不会重新下载。\n"
                                   "上下文超过 256K 时官方会自动启用 RoPE 扩展。\n\n确定应用吗？"):
            return
        args = ["--yes", "--family", ent["family"], "--model", ent["model"],
                "--context", str(new_ctx), "--no-start"]
        if dd:
            args += ["--data-dir", dd]
        self.cfg.set("last_context", new_ctx)
        if hasattr(self, "ctx_var"):
            self.ctx_var.set(ctx_label(new_ctx))

        def done(rc):
            if rc == 0:
                ent["context"] = new_ctx          # 直接更新缓存，避免再扫一遍磁盘
                self.cfg.set("cached_models", self.model_cache())
            self._refresh_ctx_quick()
            if rc == 0:
                self.ctx_quick_status.configure(text=f"已应用：上下文 {ctx_label(new_ctx)}。重启模型服务后生效。",
                                                text_color=COLORS["success"])
                self.toast("上下文已修改", f"已改为 {ctx_label(new_ctx)}，重启模型服务后生效。")
            else:
                self.ctx_quick_status.configure(text=f"修改失败（退出码 {rc}），详见「维护工具」页输出。",
                                                text_color=COLORS["danger"])

        self.show_page("tools")
        self._run_setup_tool(args, f"修改上下文为 {ctx_label(new_ctx)}", on_done=done)

    def _net_toggle(self, sync_other=True):
        lan = self.net_lan_var.get()
        self._sync_key_field(self.net_key_entry, self.net_key_gen_btn, lan)
        if lan:
            if not self.net_key_entry.get().strip():
                self.net_key_entry.insert(0, self.cfg.get("last_api_key") or generate_api_key())
                self.cfg.set("last_api_key", self.net_key_entry.get().strip())   # 自动生成的密钥持久化，重启不丢
            self.net_status.configure(text="已开启局域网访问：已自动生成 API 密钥（可点「生成」更换），"
                                           "点「保存到模型配置」写入。",
                                      text_color=COLORS["text_dim"])
        else:
            self.net_status.configure(text="本地访问（127.0.0.1）：不需要 API 密钥。",
                                      text_color=COLORS["text_dim"])
        if sync_other and hasattr(self, "lan_var"):
            # 与安装向导里的同名开关保持联动，避免两处保存同一设置时互相覆盖
            self.lan_var.set(lan)
            self._on_lan_toggle(sync_other=False)

    def _net_gen_key(self):
        self.net_key_entry.delete(0, "end")
        self.net_key_entry.insert(0, generate_api_key())
        self.cfg.set("last_api_key", self.net_key_entry.get().strip())   # 记住新密钥，重启不丢

    def _net_save(self, silent=False):
        """把网络设置写入选中模型的配置；silent=True 时不弹窗、不刷状态（供「保存设置」一键同步用）。"""
        cfg_path = self._selected_cfg_path()
        if cfg_path is None:
            if not silent:
                self.toast("没有选中模型", "请先在概览页选择一个已安装的模型。", "warn")
            return False
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            port = self.net_port_entry.get().strip() or "8080"
            if not port.isdigit():
                raise ValueError("端口必须是数字。")
            if self.net_lan_var.get():
                key = self.net_key_entry.get().strip()
                if not key:                        # 未填密钥时自动生成（官方要求局域网必须带密钥）
                    key = generate_api_key()
                    self._fill_entry(self.net_key_entry, key)
                cfg["host"] = "0.0.0.0"
                cfg["api_key"] = key
            else:
                cfg["host"] = "127.0.0.1"
                cfg.pop("api_key", None)           # 本地访问无需密钥，清除历史残留
            cfg["port"] = int(port)
            tmp = cfg_path.with_name(cfg_path.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            os.replace(tmp, cfg_path)
        except (OSError, ValueError) as e:
            if silent:
                self.dash_log.append(f"网络设置写入模型配置失败：{e}")
            else:
                self.toast("保存失败", str(e), "error")
            return False
        # 同步记录启动器偏好（含局域网开关状态），保证重启后界面能恢复
        self.cfg.update(last_port=int(port),
                        last_lan=bool(self.net_lan_var.get()),
                        last_api_key=(self.net_key_entry.get().strip()
                                      or self.cfg.get("last_api_key") or ""))
        self._refresh_conn()
        if not silent:
            self.net_status.configure(text=f"已保存到 {cfg_path.name}（host={cfg['host']}, port={cfg['port']}）。"
                                           "重启模型后生效。", text_color=COLORS["success"])
        return True

    def _net_firewall(self):
        port = self.net_port_entry.get().strip() or "8080"
        if not port.isdigit():
            self.toast("端口无效", "端口必须是数字。", "warn")
            return
        script = (f'netsh advfirewall firewall add rule name="Strata Server {port}" dir=in action=allow '
                  f'protocol=TCP localport={port}')
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"Start-Process -Verb RunAs -Wait -WindowStyle Hidden cmd -ArgumentList '/c {script}'"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                self.net_status.configure(text=f"已在防火墙放行 TCP {port}（入站）。", text_color=COLORS["success"])
            else:
                self.net_status.configure(text="放行未完成（可能拒绝了管理员授权）。", text_color=COLORS["warning"])
        except Exception as e:                     # noqa: BLE001
            self.net_status.configure(text="放行失败：" + str(e), text_color=COLORS["danger"])

    def _refresh_conn(self):
        cfg_path = self._selected_cfg_path()
        port = self.cfg.get("last_port") or 8080
        host = "127.0.0.1"
        key = ""
        if cfg_path is not None:
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
                port = cfg.get("port") or port
                host = cfg.get("host") or host
                key = cfg.get("api_key") or ""
            except (OSError, ValueError):
                pass
            self.net_port_entry.delete(0, "end")
            self.net_port_entry.insert(0, str(port))
            try:
                self.net_key_entry.configure(state="normal")
            except Exception:                      # noqa: BLE001
                pass
            self.net_key_entry.delete(0, "end")
            if key:
                self.net_key_entry.insert(0, key)
            self.net_lan_var.set(host == "0.0.0.0")
        self._net_toggle()                         # 同步密钥输入启用/禁用状态
        lines = [f"本机地址：http://127.0.0.1:{port}", f"接口：/v1/chat/completions （OpenAI 兼容）"]
        if host == "0.0.0.0":
            ips = local_ip_addresses()
            lines += [f"局域网地址：http://{ip}:{port}" for ip in ips] or ["局域网地址：未获取到本机 IP"]
            lines.append("API 密钥：" + (key or "（未设置，请至少设置一个）"))
        else:
            lines.append("当前仅本机可访问。")
        self.conn_label.configure(text="\n".join(lines))

    # =================================================================== 维护工具页
    def _page_tools(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.tools_log = None

        # ---- Strata 版本与更新
        up_card = make_card(page, "Strata 版本与更新")
        up_card.pack(fill="x", pady=(0, 12))
        uinner = ctk.CTkFrame(up_card, fg_color="transparent")
        uinner.pack(fill="x", padx=16, pady=(0, 12))

        self.ver_label = ctk.CTkLabel(uinner, text="正在读取版本…", text_color=COLORS["text"], justify="left",
                                      font=ctk.CTkFont(family=FONT, size=12), wraplength=780)
        self.ver_label.pack(anchor="w")

        urow = ctk.CTkFrame(uinner, fg_color="transparent")
        urow.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(urow, text="检查最新版本", width=120, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self.check_update).pack(side="left")
        ctk.CTkButton(urow, text="安装所选版本", width=140, height=32, corner_radius=8,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      font=ctk.CTkFont(family=FONT, size=12),
                      command=self.do_update).pack(side="left", padx=8)
        ctk.CTkButton(urow, text="打开发布页", width=100, height=32, corner_radius=8, fg_color=COLORS["chip"],
                      text_color=COLORS["text"], hover_color="#dde5f7", font=ctk.CTkFont(family=FONT, size=12),
                      command=self.open_release_page).pack(side="left")

        self.auto_update_var = ctk.BooleanVar(value=bool(self.cfg.get("auto_update")))
        ctk.CTkCheckBox(uinner, text="启动时检查更新（固定版本时仅提示；未固定时独立安装新版）",
                        variable=self.auto_update_var, text_color=COLORS["text"],
                        font=ctk.CTkFont(family=FONT, size=12), checkbox_width=20, checkbox_height=20,
                        command=lambda: self.cfg.set("auto_update", self.auto_update_var.get())).pack(anchor="w",
                                                                                                    pady=(10, 0))
        self.up_status = ctk.CTkLabel(uinner, text="", text_color=COLORS["text_dim"], justify="left",
                                      font=ctk.CTkFont(family=FONT, size=11), wraplength=780)
        self.up_status.pack(anchor="w", pady=(6, 0))

        versions = ctk.CTkFrame(uinner, fg_color="transparent")
        versions.pack(fill="x", pady=(12, 0))
        ctk.CTkLabel(versions, text="选择 Strata 发布版本（配套引擎自动匹配）").pack(anchor="w")
        self.version_choice = ctk.StringVar(value=self.cfg.get('preferred_release') or '')
        self.version_menu = ctk.CTkComboBox(versions, variable=self.version_choice,
                                          values=[], width=360,
                                          command=lambda value: self.cfg.set('preferred_release', self._chosen_release()))
        self.version_menu.pack(anchor="w", pady=6)
        self.version_list_status = ctk.CTkLabel(versions, text='正在动态获取官方版本…',
                                               text_color=COLORS['text_dim'], wraplength=650, justify='left')
        self.version_list_status.pack(anchor='w')
        row = ctk.CTkFrame(versions, fg_color="transparent")
        row.pack(fill="x")
        for index, (title, action) in enumerate([('刷新版本', self.refresh_versions), ('检查兼容性', self.check_version),
                              ('安装所选版本', self.install_version), ('切换已安装版本', self.switch_version),
                              ('回退', self.rollback_version), ('清理所选版本', self.clean_version)]):
            ctk.CTkButton(row, text=title, width=140, command=action).grid(row=index // 3, column=index % 3,
                                                                      padx=(0, 6), pady=4, sticky='w')
        self.installed_version_choice = ctk.StringVar(value='请选择已安装环境')
        self.installed_version_menu = ctk.CTkComboBox(versions, variable=self.installed_version_choice,
                                                     values=['请选择已安装环境'], width=440)
        self.installed_version_menu.pack(anchor='w', pady=6)
        ctk.CTkButton(versions, text='使用列表中的环境', command=self.use_installed_version).pack(anchor='w')
        self.version_pin = ctk.BooleanVar(value=bool(self.cfg.get('version_pinned', True)))
        ctk.CTkCheckBox(versions, text="固定所选版本（自动更新仅提示）", variable=self.version_pin,
                        command=lambda: self.cfg.set('version_pinned', self.version_pin.get())).pack(anchor="w", pady=8)
        ctk.CTkLabel(versions, text="安装使用独立目录；切换后下次启动生效。历史版缺少签名源时会停止安装。",
                    text_color=COLORS['text_dim']).pack(anchor="w")

        card = make_card(page, "维护工具")
        card.pack(fill="x", pady=(0, 12))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0, 12))

        tools = [
            ("更新引擎与依赖", "独立安装上方所选发布版本的配套引擎与依赖，完成后手动切换。", self.tool_update),
            ("校准性能", "为本机调优引擎参数（约 5-10 分钟），完成后按默认启动。", self.tool_calibrate),
            ("重新体检", "再次运行 setup.py --check 输出体检结论。", self.tool_check),
            ("打开 Strata 目录", "打开引擎源码目录（配置、日志所在处）。", lambda: self._open_path(self.env.root)),
            ("打开模型数据目录", "打开模型文件（models / packs / mtp）所在目录。", self._open_data_dir),
            ("打开配置文件", "用记事本打开当前模型的 strata-*.json 运行配置。", self.tool_open_config),
            ("修复端口占用", "检测并强制结束占用端口的进程，保持端口设置不变。", self.tool_fix_port),
            ("上下文压缩自检", "验证摘要替换、缓存与历史保护规则，使用模拟摘要，不加载模型。",
             self.tool_selftest),
        ]
        grid = ctk.CTkFrame(inner, fg_color="transparent")
        grid.pack(fill="x")
        for i, (title, desc, cmd) in enumerate(tools):
            b = ctk.CTkButton(grid, text=title, width=160, height=34, corner_radius=8, anchor="w",
                              fg_color=COLORS["chip"], text_color=COLORS["text"], hover_color="#dde5f7",
                              font=ctk.CTkFont(family=FONT, size=12), command=cmd)
            b.grid(row=i, column=0, sticky="w", pady=3)
            ctk.CTkLabel(grid, text=desc, text_color=COLORS["text_dim"], font=ctk.CTkFont(family=FONT, size=11),
                         wraplength=560, justify="left").grid(row=i, column=1, sticky="w", padx=12, pady=3)

        cleanup = make_card(page, "模型数据清理")
        cleanup.pack(fill="x", pady=(0, 12))
        clean_inner = ctk.CTkFrame(cleanup, fg_color="transparent")
        clean_inner.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(clean_inner, text=(
            "选择模型后先扫描预览。可仅删除运行配置，也可彻底删除该模型独占的 GGUF、packs 和启动器缓存。\n"
            "共享 MTP、视觉文件、引擎文件以及其他模型仍在使用的数据会自动保护。"),
            text_color=COLORS["text_dim"], justify="left", wraplength=800,
            font=ctk.CTkFont(family=FONT, size=11)).pack(anchor="w", pady=(0, 8))
        clean_row = ctk.CTkFrame(clean_inner, fg_color="transparent")
        clean_row.pack(fill="x")
        self.cleanup_model_var = ctk.StringVar(value="（暂无可清理模型）")
        self.cleanup_model_menu = ctk.CTkOptionMenu(
            clean_row, variable=self.cleanup_model_var, values=["（暂无可清理模型）"],
            width=420, height=34, corner_radius=8, fg_color=COLORS["chip"],
            text_color=COLORS["text"], button_color="#dde5f7", button_hover_color="#cfdaf5",
            dropdown_text_color=COLORS["text"], font=ctk.CTkFont(family=FONT, size=12))
        self.cleanup_model_menu.pack(side="left")
        self.cleanup_scan_btn = ctk.CTkButton(
            clean_row, text="扫描并预览", width=120, height=34, corner_radius=8,
            fg_color=COLORS["danger"], hover_color="#c22f2f",
            command=self.preview_model_cleanup, font=ctk.CTkFont(family=FONT, size=12))
        self.cleanup_scan_btn.pack(side="left", padx=8)
        self._refresh_cleanup_models()

        log_card = make_card(page, "工具输出")
        log_card.pack(fill="both", expand=True, pady=(0, 8))
        self.tools_log = LogPane(log_card, height=280)
        self.tools_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.after(1000, self._refresh_version_environments)
        self.after(1100, self.refresh_versions)
        return page

    def _refresh_version_environments(self):
        try:
            store = self._version_store()
            state = store.state()
            mapping = {}
            for path, entry in state['environments'].items():
                status = '当前' if path == state.get('current') else '可回退' if path == state.get('previous') else '已安装'
                if not entry.get('verified') and not entry.get('original'):
                    status = '安装未完成'
                label = f"{entry['release']} / 引擎 {entry['engine']} / {status} / {Path(path).name}"
                mapping[label] = path
            self._installed_version_map = mapping
            self.installed_version_menu.configure(values=list(mapping) or ['暂无已安装环境'])
            selected = next((label for label, path in mapping.items() if path == state.get('current')), None)
            if selected:
                self.installed_version_choice.set(selected)
        except Exception as exc:
            self._version_log(f'读取已安装环境失败：{exc}')

    def use_installed_version(self):
        try:
            path = self._installed_version_map.get(self.installed_version_choice.get())
            if not path:
                raise RuntimeError('请从已安装列表选择环境。')
            self._activate_version(path)
        except Exception as exc:
            self.toast('切换失败', str(exc), 'warn')

    def _version_store(self):
        store = VersionStore(APP_DIR)
        current = self.env.root or self.env.find_root()
        if current and str(Path(current).resolve()) not in store.state()['environments']:
            store.register(current, installed_release(current) or '原有环境', original=True)
            if not store.state().get('current'):
                store.activate(current)
        return store

    def _version_log(self, text):
        if self.tools_log is not None:
            self.ui(self.tools_log.append, text)

    def refresh_versions(self):
        if getattr(self, '_version_refreshing', False):
            return
        self._version_refreshing = True
        self.version_list_status.configure(text='正在获取官方版本及历史分页…')
        def worker():
            try:
                store = VersionStore(APP_DIR)
                cached = store.catalog()
                if cached:
                    self.ui(self._apply_release_list, cached, f'已加载缓存 {len(cached)} 个版本，正在联网刷新…')
                try:
                    releases = store.catalog(refresh=True)
                    if not releases:
                        raise RuntimeError('官方列表为空')
                    source = releases[0].get('source', '官方 API')
                    self.ui(self._apply_release_list, releases, f'来源：{source}；已获取 {len(releases)} 个发布版本，由你选择安装。')
                    self._version_log(f'动态版本列表已刷新：{len(releases)} 个版本（{source}）。')
                except Exception as exc:
                    message = f'联网获取失败；保留缓存 {len(cached)} 个版本。' if cached else '联网获取失败，暂无版本列表；可重试或手动输入发布标签。'
                    self.ui(self.version_list_status.configure, text=message)
                    self._version_log(f'{message}\n{exc}')
            except Exception as exc:
                self._version_log(f'版本列表不可用，可手动输入正式发布标签：{exc}')
                self.ui(self.version_list_status.configure, text=f'版本列表读取失败：{exc}')
            finally:
                self._version_refreshing = False
        threading.Thread(target=worker, daemon=True).start()

    def _apply_release_list(self, releases, status):
        current = self.version_choice.get()
        current = getattr(self, '_release_labels', {}).get(current, current)
        labels = {f"{r['version']} · 引擎 {r.get('engine') or '安装时核验'}": r['version'] for r in releases}
        self._release_labels = labels
        self.version_menu.configure(values=list(labels))
        selected = next((label for label, version in labels.items() if version == current), None)
        if selected:
            self.version_choice.set(selected)
        elif not current and labels:
            self.version_choice.set(next(iter(labels)))
        self.version_list_status.configure(text=status)

    def _chosen_release(self):
        value = self.version_choice.get()
        return normalize_release(getattr(self, '_release_labels', {}).get(value, value))

    def check_version(self):
        try:
            version = self._chosen_release()
        except ValueError as exc:
            self.toast('版本无效', str(exc), 'warn')
            return
        def worker():
            try:
                with urllib.request.urlopen(pinned_feed(version), timeout=20) as response:
                    data = response.read()
                if not isinstance(json.loads(data), dict):
                    raise RuntimeError('官方组件源格式无效')
                self._version_log(f'v{version}：官方组件源可访问，继续执行官方签名与硬件预检。')
                root = self.env.root or self.env.find_root()
                if not root or not (Path(root) / 'strata.py').is_file():
                    self._version_log('当前环境没有官方组件检查接口；安装所选源码后继续预检。')
                    return
                python = self.env.python()
                if not python:
                    raise RuntimeError('缺少 Python 运行环境')
                with tempfile.TemporaryDirectory(prefix='strata-version-check-') as work:
                    result = subprocess.run([python, str(Path(root) / 'strata.py'), 'update', '--check',
                                             '--no-control', '--feed', pinned_feed(version),
                                             '--components-dir', work, '--output', 'json'],
                                            cwd=root, capture_output=True, text=True, encoding='utf-8',
                                            errors='replace', timeout=120,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                self._version_log(result.stdout or result.stderr)
                if result.returncode:
                    raise RuntimeError(f'官方兼容性检查未通过（{result.returncode}）')
                self._version_log('官方组件预检通过；具体模型加载与真实推理仍需启动后验证。')
            except Exception as exc:
                self._version_log(f'v{version} 暂不可安装：未取得对应官方组件源：{exc}')
        threading.Thread(target=worker, daemon=True).start()

    def install_version(self):
        if self.runtime.get('updating') or self.server.running() or getattr(self, '_starting', False):
            self.toast('暂不能安装', '请等待维护任务结束并停止模型服务。', 'warn')
            return
        try:
            version = self._chosen_release()
        except ValueError as exc:
            self.toast('版本无效', str(exc), 'warn')
            return
        self.cfg.set('preferred_release', version)
        self.runtime['updating'] = True
        threading.Thread(target=self._install_version_worker, args=(version,), daemon=True).start()

    def _install_version_worker(self, version):
        try:
            store = self._version_store()
            source = self.env.ensure_root(self._version_log)
            # Resolve the release-specific feed before downloading or preparing an environment.
            with urllib.request.urlopen(pinned_feed(version), timeout=20) as response:
                json.load(response)
            target = store.root / f'release-{version}-{time.time_ns()}'
            with tempfile.TemporaryDirectory(prefix='strata-source-') as work:
                archive = Path(work) / 'source.zip'
                download_source(f'https://github.com/{GITHUB_REPO}/archive/refs/tags/v{version}.zip',
                                archive, self._version_log)
                extracted = extract_source(archive, Path(work) / 'source')
                shutil.copytree(extracted, target, dirs_exist_ok=True)
            if not (target / 'strata_cli/update_cmd.py').is_file():
                raise RuntimeError('此历史版本缺少签名组件更新接口，未切换环境。')
            copy_configs(source, target)
            store.register(target, version)
            env = StrataEnv()
            env.root = target
            py = env.ensure_venv(self._version_log)
            helper = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'strata_update.py'
            process_env = dict(os.environ, STRATA_LAUNCHER_RELEASE=version,
                               PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
            self._version_log(f'安装 v{version}，配套引擎 {source_engine_version(target)}：{target}')
            with subprocess.Popen([py, '-u', str(helper), str(target)], cwd=target, env=process_env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding='utf-8', errors='replace',
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as proc:
                for line in proc.stdout:
                    self._version_log(line.rstrip())
                if proc.wait():
                    raise RuntimeError('目标环境安装失败，当前环境保持可用。详见工具输出。')
            check_exact_engines(target)
            validate_config_flags(target)
            service_check = subprocess.run([py, str(target / 'serve/server.py'), '--help'], cwd=target,
                                           env=process_env, capture_output=True, timeout=60,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if service_check.returncode:
                raise RuntimeError('此版本服务接口或依赖不完整，未开放切换：' + service_check.stderr.decode('utf-8', 'replace')[-1200:])
            record_release(target, version)
            store.register(target, version, verified=True)
            self._version_log('安装与文件校验完成；可点击“切换已安装版本”。真实模型推理尚未验证。')
        except Exception as exc:
            self._version_log(f'[安装失败] {exc}')
        finally:
            self.runtime['updating'] = False
            self.ui(self._refresh_version_environments)

    def _activate_version(self, path):
        if self.runtime.get('updating') or self.server.running() or getattr(self, '_starting', False):
            raise RuntimeError('请先停止服务并等待维护任务完成。')
        store = self._version_store()
        entry = store.state()['environments'].get(str(Path(path).resolve()), {})
        if entry.get('verified'):
            check_exact_engines(path)
            validate_config_flags(path)
        elif not entry.get('original'):
            raise RuntimeError('此环境尚未完成安装校验。')
        current = self.env.root or self.env.find_root()
        if current and Path(current).resolve() != Path(path).resolve():
            differences = []
            from strata_update import model_configs
            for old_path, old in model_configs(current).items():
                target_config = Path(path) / old_path.name
                if not target_config.exists():
                    differences.append(f'{old_path.name}：此环境没有配置，需要重新导入')
                    continue
                new = json.loads(target_config.read_text(encoding='utf-8-sig'))
                for key in ('arch', 'parallel', 'drafting'):
                    if old.get(key) != new.get(key):
                        differences.append(f'{old_path.name}：{key} {old.get(key)} → {new.get(key)}')
                def flag_value(config, flag):
                    args = config.get('args', [])
                    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None
                for flag in ('--max-context', '--kv', '--kv-resident', '--draft', '--mtp'):
                    before, after = flag_value(old, flag), flag_value(new, flag)
                    if before != after:
                        differences.append(f'{old_path.name}：{flag} {before} → {after}')
            if differences:
                self._version_log('切换配置差异：\n' + '\n'.join(differences))
                if not messagebox.askyesno('确认版本配置差异', '\n'.join(differences[:16]) + '\n\n使用目标环境保存的独立配置？'):
                    return
        path = store.activate(path)
        self.cfg.set('strata_dir', path)
        self.cfg.set('managed_version_active', not entry.get('original', False))
        self.cfg.set('cached_scan_version', 0)
        self.env.set_manual_root(path)
        self.env.root = Path(path)
        self._after_source_updated()
        self._refresh_version_environments()
        self._version_log(f'已选择环境：{path}；下次启动使用此环境。')

    def switch_version(self):
        try:
            version = self._chosen_release()
            candidates = [(path, entry) for path, entry in self._version_store().state()['environments'].items()
                          if entry.get('release') == version and entry.get('verified')]
            if not candidates:
                raise RuntimeError('所选版本尚未完成安装，请先安装。')
            path, _ = max(candidates, key=lambda item: item[1]['installed_at'])
            self._activate_version(path)
        except Exception as exc:
            self.toast('切换失败', str(exc), 'warn')

    def rollback_version(self):
        try:
            path = self._version_store().state().get('previous')
            if not path:
                raise RuntimeError('还没有可回退环境。')
            self._activate_version(path)
        except Exception as exc:
            self.toast('回退失败', str(exc), 'warn')

    def clean_version(self):
        try:
            if self.runtime.get('updating') or self.server.running():
                raise RuntimeError('请先停止服务并等待维护任务结束。')
            version = self._chosen_release()
            store = self._version_store()
            paths = [path for path, entry in store.state()['environments'].items() if entry.get('release') == version]
            if not paths:
                raise RuntimeError('没有找到此版本的已登记环境。')
            if messagebox.askyesno('清理版本', '删除以下独立环境（当前和回退环境受保护）：\n' + '\n'.join(paths)):
                for path in paths:
                    store.remove(path)
                    self._version_log(f'已清理环境：{path}')
                self._refresh_version_environments()
        except Exception as exc:
            self.toast('清理未完成', str(exc), 'warn')

    def _refresh_cleanup_models(self):
        if not hasattr(self, 'cleanup_model_menu'):
            return
        entries = [ent for ent in self.model_cache() if ent.get('path')]
        labels, mapping = [], {}
        for ent in entries:
            label = ent.get('label') or Path(ent['path']).stem
            shown = label
            if shown in mapping:
                shown = f"{label} — {Path(ent['path']).name}"
            labels.append(shown)
            mapping[shown] = ent
        self._cleanup_cfg_map = mapping
        values = labels or ['（暂无可清理模型）']
        self.cleanup_model_menu.configure(values=values,
                                          state='normal' if labels else 'disabled')
        if self.cleanup_model_var.get() not in values:
            self.cleanup_model_var.set(values[0])

    def _cleanup_is_busy(self):
        return (self.server.running() or getattr(self, '_starting', False) or
                self.runner.running() or self.runtime.get('updating'))

    def preview_model_cleanup(self):
        if self._cleanup_is_busy():
            self.toast('暂时不能清理', '模型正在运行、启动、安装或更新，请停止并等待当前任务结束。', 'warn')
            return
        ent = getattr(self, '_cleanup_cfg_map', {}).get(self.cleanup_model_var.get())
        if not ent or not ent.get('path'):
            self.toast('没有可清理模型', '请先刷新已安装模型列表。', 'warn')
            return
        data_dir = self.effective_data_dir()
        if not data_dir or not Path(data_dir).is_dir():
            self.toast('模型数据目录无效', '请先在「网络与配置」中选择正确的模型数据目录。', 'error')
            return
        self.cleanup_scan_btn.configure(state='disabled', text='正在扫描…')
        if self.tools_log:
            self.tools_log.append('正在扫描模型专属数据：' + str(ent.get('label') or ent['path']))

        def worker():
            try:
                path = Path(ent['path']).resolve()
                raw = json.loads(path.read_text(encoding='utf-8-sig'))
                selected = resolve_launch_config(raw, self.env.root or path.parent)
                configs = []
                for other in self.env.installed_configs():
                    try:
                        candidate = json.loads(Path(other).read_text(encoding='utf-8-sig'))
                        configs.append((Path(other), resolve_launch_config(
                            candidate, self.env.root or Path(other).parent)))
                    except (OSError, ValueError):
                        continue
                plan = build_cleanup_plan(path, selected, configs, Path(data_dir))
                self.ui(self._show_cleanup_preview, ent, plan)
            except Exception as error:              # noqa: BLE001
                self.ui(self.toast, '扫描失败', str(error), 'error')
            finally:
                self.ui(lambda: self.cleanup_scan_btn.configure(state='normal', text='扫描并预览'))
        threading.Thread(target=worker, daemon=True, name='strata-cleanup-scan').start()

    def _show_cleanup_preview(self, ent, plan):
        if not plan.items:
            self.toast('没有找到文件', '该模型没有可清理的配置或专属数据。', 'warn')
            return
        dialog = ctk.CTkToplevel(self)
        dialog.title('模型数据清理预览')
        dialog.geometry(restore_geometry('860x560+99999+99999', monitor_workareas(self)))
        dialog.transient(self)
        dialog.grab_set()
        label = ent.get('label') or Path(ent['path']).stem
        ctk.CTkLabel(dialog, text=f'模型：{label}\n预计可释放：{format_file_size(plan.reclaim_bytes)}',
                     justify='left', anchor='w', text_color=COLORS['text'],
                     font=ctk.CTkFont(family=FONT, size=14, weight='bold')).pack(
                         fill='x', padx=18, pady=(16, 8))
        lines = []
        for item in plan.items:
            mark = '可删除' if item.deletable else '已保护'
            kind = '配置' if item.scope == 'config' else '模型数据'
            suffix = f'；{item.reason}' if item.reason else ''
            lines.append(f'[{mark}] [{kind}] {format_file_size(item.size)}\n{item.path}{suffix}\n')
        box = ctk.CTkTextbox(dialog, wrap='word', font=ctk.CTkFont(family=FONT_MONO, size=11))
        box.pack(fill='both', expand=True, padx=18, pady=8)
        box.insert('1.0', '\n'.join(lines))
        box.configure(state='disabled')
        ctk.CTkLabel(dialog, text='彻底删除会永久移除模型文件，网络盘通常无法进入回收站。',
                     text_color=COLORS['danger'], font=ctk.CTkFont(family=FONT, size=11)).pack(
                         anchor='w', padx=18, pady=(2, 8))
        buttons = ctk.CTkFrame(dialog, fg_color='transparent')
        buttons.pack(fill='x', padx=18, pady=(0, 16))
        ctk.CTkButton(buttons, text='仅删除配置（保留模型数据）', width=210,
                      fg_color=COLORS['chip'], text_color=COLORS['text'],
                      command=lambda: self._confirm_model_cleanup(ent, plan, False, dialog)).pack(side='left')
        ctk.CTkButton(buttons, text='彻底删除模型及专属数据', width=210,
                      fg_color=COLORS['danger'], hover_color='#c22f2f',
                      command=lambda: self._confirm_model_cleanup(ent, plan, True, dialog)).pack(
                          side='left', padx=8)
        ctk.CTkButton(buttons, text='取消', width=80, fg_color=COLORS['chip'],
                      text_color=COLORS['text'], command=dialog.destroy).pack(side='right')

    def _confirm_model_cleanup(self, ent, plan, delete_data, dialog):
        label = ent.get('label') or Path(ent['path']).stem
        if delete_data:
            answer = simpledialog.askstring(
                '确认永久删除',
                f'将永久删除“{label}”的专属模型数据，预计释放 {format_file_size(plan.reclaim_bytes)}。\n\n'
                f'此操作无法恢复。请输入完整模型名称确认：\n{label}', parent=dialog)
            if answer != label:
                if answer is not None:
                    self.toast('名称不一致', '未执行删除。请输入预览窗口显示的完整模型名称。', 'warn')
                return
        elif not messagebox.askyesno(
                '仅删除模型配置',
                f'将删除“{label}”的运行配置与启动器设置。\n模型文件会保留，以后可以重新生成配置。\n\n确定继续吗？',
                parent=dialog):
            return
        dialog.destroy()
        self._execute_model_cleanup(ent, plan, delete_data)

    def _execute_model_cleanup(self, ent, plan, delete_data):
        if self._cleanup_is_busy():
            self.toast('清理已取消', '检测到模型或维护任务已经开始运行。', 'warn')
            return
        self.cleanup_scan_btn.configure(state='disabled', text='正在删除…')
        label = ent.get('label') or Path(ent['path']).stem

        def worker():
            result = execute_cleanup_plan(plan, delete_data)
            self.ui(self._finish_model_cleanup, ent, label, plan, delete_data, result)
        threading.Thread(target=worker, daemon=True, name='strata-model-cleanup').start()

    def _finish_model_cleanup(self, ent, label, plan, delete_data, result):
        self.cleanup_scan_btn.configure(state='normal', text='扫描并预览')
        saved = dict(self.cfg.get('model_optimizations') or {})
        saved.pop(str(Path(ent['path']).resolve()), None)
        self.cfg.set('model_optimizations', saved)
        if delete_data:
            self.rescan_models()
        else:
            remaining = [item for item in self.model_cache()
                         if str(Path(item.get('path') or '').resolve()) != str(Path(ent['path']).resolve())]
            self.cfg.set('cached_models', remaining)
            self.apply_mode()
        self._refresh_cleanup_models()
        for path in result.deleted:
            if self.tools_log:
                self.tools_log.append('已删除：' + str(path))
        if result.failed:
            details = '\n'.join(f'{path}：{error}' for path, error in result.failed)
            if self.tools_log:
                self.tools_log.append('部分文件删除失败：\n' + details)
            self.toast('部分清理失败', f'已删除 {len(result.deleted)} 项，失败 {len(result.failed)} 项。详见工具输出。', 'warn')
        else:
            released = format_file_size(plan.reclaim_bytes) if delete_data else '模型数据已保留'
            self.toast('清理完成', f'{label}\n已删除 {len(result.deleted)} 项；{released}。')

    def _open_path(self, p):
        if not p:
            return
        try:
            os.startfile(str(p))                   # noqa: S606
        except Exception as e:                     # noqa: BLE001
            self.toast("打开失败", str(e), "error")

    def _open_data_dir(self):
        dd = self.effective_data_dir()
        if dd and Path(dd).is_dir():
            self._open_path(dd)
        else:
            self.toast("模型数据目录不存在", "尚未确定模型数据目录，可在「网络与配置 → 目录设置」中指定。", "warn")
            self._open_path(self.env.root)

    def tool_open_config(self):
        cfg_path = self._selected_cfg_path()
        if cfg_path is None:
            self.toast("没有选中模型", "请先在概览页选择一个已安装的模型。", "warn")
            return
        try:
            subprocess.Popen(["notepad", str(cfg_path)])
        except Exception as e:                     # noqa: BLE001
            self.toast("打开失败", str(e), "error")

    def _run_setup_tool(self, args, title, on_done=None):
        if self.runner.running():
            self.toast("有任务在运行", "请等待当前任务结束。", "warn")
            return
        if self.env.root is None:
            self.toast("环境未就绪", "Strata 环境尚未准备完成。", "warn")
            return
        try:
            py = self.env.ensure_venv(lambda s: self.ui(self.tools_log.append, s))
        except Exception as e:                     # noqa: BLE001
            self.toast("运行环境创建失败", str(e), "error")
            return
        self.tools_log.append(f"=== {title} ===")
        cmd = [py, str(self.env.root / "setup.py")] + args
        self.tools_log.append("$ " + " ".join(cmd))

        def _finish(rc, err):
            self.ui(self.tools_log.append, f"=== 结束（退出码 {rc}）===")
            if on_done is not None:
                self.ui(on_done, rc)

        self.runner.run(cmd, self.env.root,
                        lambda line: self.ui(self.tools_log.append, line), _finish)

    def tool_update(self):
        self.install_version()

    def tool_calibrate(self):
        if not messagebox.askyesno("校准性能", "校准约需 5-10 分钟，期间会占用显卡。\n确定开始吗？"):
            return
        self._run_setup_tool(["--calibrate", "--no-start", "--yes"], "校准性能")

    def tool_check(self):
        self._run_setup_tool(["--check", "--yes"], "硬件体检")

    def tool_fix_port(self):
        """检测占用端口的进程，确认后强制结束，保持网络配置中的端口不变。"""
        port = int(self.cfg.get("last_port") or 8080)
        cfg_path = self._selected_cfg_path()
        if cfg_path is not None:
            try:
                cj = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
                port = int(cj.get("port") or port)
            except (OSError, ValueError):
                pass
        if not port_in_use(port):
            self.tools_log.append(f"端口 {port} 当前可用，无需修复。")
            return
        pid, pname = port_owner(port)
        desc = f"{pname or '未知进程'}（PID {pid}）" if pid else "未知进程"
        if pid == os.getpid():
            self.tools_log.append(f"端口 {port} 由本启动器自身占用，请关闭启动器后重试。")
            return
        if self._is_our_server(pid):
            self.tools_log.append(f"端口 {port} 由正在运行的模型服务占用，请先在概览页停止模型服务。")
            self.toast("端口被模型服务占用", "请先在概览页停止模型服务，再修复端口。", "warn")
            return
        if pid <= 4 or is_protected_process(pname):
            self.tools_log.append(f"端口 {port} 被系统进程 {desc} 占用，为安全起见不结束该进程。")
            self.toast("无法结束系统进程", f"端口被系统进程 {desc} 占用。", "warn")
            return
        if not messagebox.askyesno("修复端口占用",
                                   f"端口 {port} 被 {desc} 占用。\n\n"
                                   "将强制结束该进程，并保持网络配置中的端口不变。\n确定继续吗？"):
            self.tools_log.append(f"已取消：端口 {port} 仍被 {desc} 占用。")
            return
        ok, msg = kill_pid_force(pid)
        time.sleep(0.5)
        if ok and not port_in_use(port):
            self.tools_log.append(f"已强制结束 {desc}，端口 {port} 已释放；网络配置端口保持不变。")
            self.toast("修复完成", f"端口 {port} 已释放，可直接启动模型。")
        else:
            self.tools_log.append(f"结束进程失败：{msg or '端口仍被占用'}"
                                  "（如权限不足，请以管理员身份运行启动器后重试）")
            self.toast("修复失败", "未能释放端口，可能需要管理员权限。", "error")

    def tool_selftest(self):
        """后台运行上下文压缩自检，结果输出到「工具输出」面板。"""
        if self.runtime.get("selftesting"):
            self.toast("自检进行中", "上一次自检尚未结束，请稍候。", "warn")
            return
        self.runtime["selftesting"] = True
        self.tools_log.append("=== 上下文压缩自检开始 ===")

        def _worker():
            def sink(line):
                self.ui(self.tools_log.append, line)
            try:
                from strata_selftest import run_context_selftest
                res = run_context_selftest(sink)
                if res["failed"] == 0:
                    self.ui(self.toast, "自检通过",
                            "上下文压缩功能自检全部通过（%d 项）。" % res["passed"], "info")
                else:
                    self.ui(self.toast, "自检未通过",
                            "失败 %d 项：\n%s" % (res["failed"], "\n".join(res["failures"])), "error")
            except Exception as e:                     # noqa: BLE001
                self.ui(self.tools_log.append, "自检异常：" + str(e))
                self.ui(self.toast, "自检异常", str(e), "error")
            finally:
                self.runtime["selftesting"] = False
                self.ui(self.tools_log.append, "=== 上下文压缩自检结束 ===")

        threading.Thread(target=_worker, daemon=True).start()

    # ---------------- 版本检查与自动更新
    def check_update(self, auto=False):
        if self.runtime.get("updating"):
            return
        self._remote_version = self._remote_version or KNOWN_LATEST
        if not auto:
            self.up_status.configure(text="正在联网检查最新版本…", text_color=COLORS["text_dim"])
        threading.Thread(target=self._check_update_worker, args=(bool(auto),), daemon=True).start()

    def _check_update_worker(self, auto):
        root = self.env.root or self.env.find_root()
        if root:
            self.env.root = root
        local = installed_release(root) if root else ""
        if local:
            try:
                verify_engines(root, full=False)
            except Exception:
                local = ""
        local = local or read_local_version(root)
        remote, url, published, err = "", GITHUB_RELEASES_PAGE, "", ""
        try:
            remote, url, published = remote_latest_release()
        except Exception as e:                     # noqa: BLE001
            err = str(e)
            remote = self.cfg.get("last_remote_version") or KNOWN_LATEST
        self.ui(self._apply_update_info, local, remote, url, published, err, bool(auto))

    def _apply_update_info(self, local, remote, url, published, err, auto):
        if remote:
            self._remote_version = remote
        if url:
            self._remote_url = url
        newer = bool(local and remote and version_tuple(remote) > version_tuple(local))
        if local and remote:
            self.cfg.update(last_remote_version=remote, last_update_check=time.strftime("%Y-%m-%d %H:%M"))
        bits = [f"本地 Strata 版本（已校验发布 / 源码内部版本）：{'v' + local if local else '（未识别，源码布局可能不同）'}",
                f"最新版本：{'v' + remote if remote else '未知'}"]
        if published:
            bits.append(f"发布时间：{published[:10]}")
        bits.append("发现新版本，建议更新。" if newer else ("已是最新版本。" if (local and remote) else ""))
        self.ver_label.configure(text="\n".join(b for b in bits if b))
        if err:
            self.up_status.configure(text=f"联网检查失败：{err}（“最新版本”为已知版本，可点「打开发布页」确认）",
                                     text_color=COLORS["warning"])
        elif not local:
            self.up_status.configure(text="无法读取本地版本号，可直接点「立即更新 Strata」重下官方源码。",
                                     text_color=COLORS["warning"])
        else:
            self.up_status.configure(text="", text_color=COLORS["text_dim"])
        if auto and self.cfg.get("auto_update") and not self.cfg.get('version_pinned', True):
            # 仅以安装目录中经过校验的发布记录判断，忽略旧版误写的成功缓存。
            if err:
                self.up_status.configure(text=f"自动更新已跳过：联网检查失败（{err}）。",
                                         text_color=COLORS["warning"])
                self.tools_log.append("[更新] 自动更新跳过：联网检查失败。")
            elif not local:
                self.tools_log.append("[更新] 自动更新跳过：无法读取本地版本号，可手动点击「立即更新」。")
            elif remote and version_tuple(remote) > version_tuple(local):
                self.up_status.configure(text="自动更新已开启，检测到新版本，正在更新…",
                                         text_color=COLORS["warning"])
                self.tools_log.append(f"[更新] 检测到新版本 v{remote}（当前 v{local}），开始自动更新。")
                self._start_update(auto=True)
            else:
                self.up_status.configure(text="已是最新版本，自动更新已跳过。",
                                         text_color=COLORS["text_dim"])
                self.tools_log.append(f"[更新] 已是最新版本（v{local or remote}），自动更新跳过。")

    def open_release_page(self):
        url = self._remote_url or GITHUB_RELEASES_PAGE
        try:
            os.startfile(url)                      # noqa: S606
        except Exception:                          # noqa: BLE001
            import webbrowser
            webbrowser.open(url)

    def do_update(self):
        self.install_version()

    def _start_update(self, auto=False):
        if auto:
            if any(entry.get('release') == self._remote_version and entry.get('verified')
                   for entry in self._version_store().state()['environments'].values()):
                self._version_log('最新发布环境已安装，可从已安装列表切换。')
                return
            self.version_choice.set(self._remote_version)
        self.install_version()

    def _update_worker(self, auto):
        def log(s):
            if self.tools_log is not None:
                self.ui(self.tools_log.append, s)
        try:
            self.env.ensure_root(log)
            target, _url, _published = remote_latest_release()
            self._update_target = target
            log(f"目标发布版本：v{target}（独立于 CMake 内部版本）")
            self.env.update_from_zip(log, version=target)
            self.env._tables = None
            nv = self.env.source_version()
            log(f"源码版本：v{nv}" if nv else "源码已更新（未能识别版本号）。")
        except Exception as e:                     # noqa: BLE001
            log("更新源码失败：" + str(e))
            self.ui(self._update_done, -1)
            return
        self.ui(self._after_source_updated)
        try:
            py = self.env.ensure_venv(log)
        except Exception as e:                     # noqa: BLE001
            log("准备运行环境失败：" + str(e))
            self.ui(self._update_done, -2)
            return
        helper = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "strata_update.py"
        cmd = [py, "-u", str(helper), str(self.env.root)]
        log("$ " + " ".join(cmd))
        self.runner.run(cmd, self.env.root,
                        lambda line: self.ui(self.tools_log.append, line),
                        lambda rc, err: self.ui(self._update_done, rc if err is None else -3),
                        env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})

    def _after_source_updated(self):
        try:
            _models, _families, contexts = self.env.tables()
            self.ctx_menu.configure(values=[ctx_label(x) for x in contexts])
            self._rebuild_family_radios()
        except Exception as e:                     # noqa: BLE001
            log_line(f"[update] {e}")
        self._fill_env_info()
        self._refresh_dir_fields()
        self.rescan_models()

    def _update_done(self, rc):
        self.runtime["updating"] = False
        if self.tools_log is not None:
            self.tools_log.append(f"=== 更新结束（退出码 {rc}）===")
        if rc == 0:
            target = getattr(self, "_update_target", "")
            try:
                if not target:
                    raise RuntimeError("本次更新缺少明确的发布标签，无法确认成功。")
                report = verify_engines(self.env.root, full=False)
                record_release(self.env.root, target)
            except Exception as exc:
                self.cfg.set("last_auto_updated_version", "")
                self.up_status.configure(text=f"更新未完成：{exc}", text_color=COLORS["warning"])
                if self.tools_log is not None:
                    self.tools_log.append(f"[更新校验失败] {exc}；未记录成功，下次可以重试。")
                self.toast("更新未完成", str(exc), "warn")
                return
            self._remote_version = target
            self.cfg.set("last_auto_updated_version", target)
            self.up_status.configure(text="发布源码与引擎校验通过。", text_color=COLORS["success"])
            self.ver_label.configure(text=f"已安装发布 v{target}；源码内部版本 v{self.env.source_version()}。")
            if self.tools_log is not None:
                for line in report:
                    self.tools_log.append("[更新校验] " + line)
            self.toast("更新完成", f"Strata 发布 v{target} 已安装并通过校验。")
        elif rc == -1:
            self.up_status.configure(text="源码更新失败，详见工具输出。", text_color=COLORS["danger"])
        else:
            self.up_status.configure(text=f"更新未完成（退出码 {rc}），详见工具输出。", text_color=COLORS["warning"])

    # =================================================================== 帮助页
    def _page_help(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        card = make_card(page, "使用帮助")
        card.pack(fill="x", pady=(0, 12))
        text = (
            "一、两种模式\n"
            "  · 已有模型：启动器检测到已安装的模型配置时自动进入。此模式只做启动、停止、监控、改配置与维护。\n"
            "  · 全新安装：没有检测到模型时进入。按向导依次选择家族、规格、上下文、读图与网络，然后开始下载。\n"
            "  顶部「切换模式」按钮可随时手动切换。\n\n"
            "二、关于目录（全部由你决定，启动器不写死路径）\n"
            "  · Strata 目录：留空时自动定位（内置或已释放的源码）；也可用「浏览…」手动选择，\n"
            "    或用「扫描硬盘」自动查找含 setup.py 的源码目录。\n"
            "  · 模型数据目录：用「浏览…」手动选择，或用「扫描硬盘」查找已有的 models/packs/mtp 目录；\n"
            "    安装模型时必须选定，启动器不会替你指定路径（模型约 59-112 GB）。\n\n"
            "三、版本更新\n"
            "  在「维护工具」刷新正式版本列表，或输入发布标签（例如 0.2.10），检查后安装。\n"
            "  每个发布使用独立的源码、依赖和配套引擎；完成后从已安装列表切换，下次启动生效。\n"
            "  默认固定版本；可离线切换、回退。当前和回退环境受到清理保护，模型文件继续共享。\n"
            "  历史版本没有官方签名组件源、硬件不兼容或参数不支持时，会提示原因。\n\n"
            "四、首次启动为什么很慢\n"
            "  首次加载会把 34-55 GB 的模型读入内存，电脑可能卡顿 1-3 分钟，这是正常现象，请不要关闭窗口。\n\n"
            "五、局域网访问（本地启动无需 API 密钥）\n"
            "  · 本地访问（127.0.0.1，默认）：不需要 API 密钥，最安全。\n"
            "  · 局域网访问：Strata 官方要求监听地址不是 127.0.0.1 时必须带 API 密钥；\n"
            "    本启动器会自动生成密钥（无需手动输入），并提供「防火墙放行该端口」按钮。\n"
            "  修改后需要重启模型服务才会生效。\n\n"
            "六、下载中断\n"
            "  Hugging Face 下载支持断点续传。中断后重新点击「开始下载并安装」即可继续，已完成步骤会被跳过。\n"
            "  网络较慢时可勾选「使用 hf-mirror.com 镜像加速下载」。\n\n"
            "七、接口\n"
            "  服务启动后提供 OpenAI 兼容接口：http://127.0.0.1:端口/v1/chat/completions\n"
            "  健康检查：/health　状态：/status　指标：/metrics　卸载模型：POST /unload\n"
        )
        ctk.CTkLabel(card, text=text, text_color=COLORS["text"], justify="left",
                     font=ctk.CTkFont(family=FONT, size=12)).pack(anchor="w", padx=16, pady=(0, 14))
        return page

    def _refresh_install_page(self):
        mode = self.current_mode()
        if mode == "installed":
            self.wizard_card.pack_forget()
            self.installed_card.pack(fill="x", pady=(0, 12))
            ents = self.model_cache()
            if ents:
                items = []
                for e in ents[:8]:
                    if not e.get("path"):
                        items.append(f"· {e.get('label', '—')}（未配置，可在「概览」页点启动或应用上下文生成配置）")
                        continue
                    name = Path(e.get("path", "")).name or "—"
                    ctx = e.get("context")
                    ctx_txt = f"，上下文 {ctx_label(ctx)}" if ctx else ""
                    items.append(f"· {e.get('label', '—')}（配置 {name}{ctx_txt}）")
                self.installed_info.configure(text="检测到以下已安装模型：\n" + "\n".join(items) +
                                                   "\n\n直接在「概览」页选择并启动即可。")
            else:
                self.installed_info.configure(text="未检测到模型配置文件。若模型已下载，请先在「网络与配置 → 目录设置」"
                                                   "确认「模型数据目录」指向 Strata-data 根目录（如 R:\\Strata-main\\Strata-data），"
                                                   "再点「使用自动」重新扫描；也可点击下方按钮安装新模型。")
        else:
            self.installed_card.pack_forget()
            self.wizard_card.pack(fill="x", pady=(0, 12))
            self._on_family_change()
            self._refresh_conn()

    def on_close(self):
        if self.server.running():
            if not messagebox.askyesno("退出", "模型服务正在运行，退出会停止它。\n确定退出吗？"):
                return
            self.server.stop()
        self._closing = True
        if getattr(self, '_launch_cancel', None):
            self._launch_cancel.set()
        self.save_all_settings("退出自动保存")
        self.cfg.set("window_maximized", self.state() == "zoomed")
        self.cfg.set("window_geometry", self._normal_geometry if self.state() == "zoomed" else self.geometry())
        self.destroy()


def main():
    app = StrataLauncher()
    app.mainloop()
    return 0


if __name__ == "__main__":
    if any(a in ("--self-test", "--selftest", "-t") for a in sys.argv[1:]):
        # 无界面自检：验证上下文压缩功能，返回码 0 表示全部通过。
        from strata_selftest import run_context_selftest
        _res = run_context_selftest(print)
        sys.exit(0 if _res["failed"] == 0 else 1)
    try:
        sys.exit(main())
    except Exception as exc:                       # noqa: BLE001
        log_line("启动器异常：" + str(exc))
        try:
            messagebox.showerror("启动器异常", str(exc))
        except Exception:                          # noqa: BLE001
            pass
        sys.exit(1)
