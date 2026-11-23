# Strata 启动器 V3.7.17

独立 Windows 中文启动器，非 Strata 官方发行版。

下载：https://github.com/xumanzi/strata-launcher/releases/tag/v3.7.17

## 源码构建

本源码包包含当前启动器源码、测试、Strata-baseline-0.2.20.zip 和本版配套原生 Inkling 运行组件；不包含模型、用户配置、令牌或已安装的用户环境。

在解压后的目录使用 Windows x64 Python：

```powershell
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m unittest discover -p "test_strata_*.py"
python -m PyInstaller --noconfirm StrataLauncher.spec
```

直接运行：`python strata_launcher.py`。原生推理还需要兼容的硬件、驱动和 Strata 组件。

运行方式可选择“速度优先（已调优）”；思考方式可单独选择“关闭思考（最快）”。现有上下文不会因预设自动缩短。资源不足时以引擎提示为准。

许可与第三方归属见 LICENSE-Strata.txt、THIRD_PARTY_NOTICES.md 及组件内的许可文件。

## 仓库与完整构建包

仓库不提交大体积二进制或内置上游 ZIP。完整构建请下载本版 Release 中的 `strata-launcher-v3.7.17-source.zip`，其中提供本版配套组件和上游源码 ZIP。
