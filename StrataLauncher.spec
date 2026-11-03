# -*- mode: python ; coding: utf-8 -*-
"""Strata 启动器打包配置（PyInstaller，单文件模式）。

运行依赖、源码包与新手说明封装进 EXE，运行时由 PyInstaller 解压到临时目录。
模型与 Strata 安装数据继续保存在外部持久目录。
"""
import os
from PyInstaller.utils.hooks import collect_all

datas = []
datas.append(('bundled_chat_template.jinja', '.'))
datas.append(('strata_build_info.json', '.'))
datas.append(('strata_versions.py', '.'))
datas.append(('多版本引擎管理-V3.5.md', '.'))
datas.append(('strata_launch_support.py', '.'))
datas.append(('strata_catalog.py', '.'))
datas.append(('strata_model_catalog.json', '.'))
datas.append(('全模型支持说明-V3.3.md', '.'))
binaries = []
hiddenimports = []

_tmp = collect_all("customtkinter")
datas += _tmp[0]
binaries += _tmp[1]
hiddenimports += _tmp[2]

# 内置 Strata 源码包（若存在则一并打包，离线也能安装）
for _z in ("Strata-main.zip", "strata_legacy_restore.json", "strata_local_import.py", "strata_update.py", "strata_download.py", "strata_web_extension.js", "strata_monitor_extension.js", "strata_code_preview.js", "本地模型导入说明.md", "启动器优化与IQ4模型说明.md", "聊天模板与Q4导入说明.md", "V2.4完整版使用与验证说明.md"):
    if os.path.exists(_z):
        datas.append((_z, "."))

a = Analysis(
    ["strata_launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="AI蛮子-Strata全模型启动器V3.5.1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
