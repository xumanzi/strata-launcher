# AI 蛮子 Strata 启动器

Windows 中文图形启动器，用于管理 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的安装、模型、服务及多个引擎版本。本项目是独立启动器，非 Strata 官方发行版。

## 下载与使用

前往 [Releases](https://github.com/xumanzi/strata-launcher/releases) 下载 `AI-Manzi-Strata-Launcher-v3.5.1-windows-x64.exe`，放在有写入权限的目录运行。模型及运行环境需要另外准备；启动器内含的 Strata 源码不等于已安装推理引擎。

在“维护工具”中，启动器自动动态获取官方发布版本和历史分页。选择版本后先检查兼容性，再安装并切换。安装使用独立源码、依赖和配套引擎，支持固定版本、离线切换和回退。

GitHub API 限流时会读取官方发布页；联网失败时保留缓存并显示原因。刷新版本列表不会改动用户已经选择的版本。

## 主要功能

- 模型安装、本地 GGUF 导入和服务启停。
- 多版本引擎安装、版本校验、切换、回退与清理保护。
- OpenAI 兼容接口、上下文管理、运行监控及聊天页面扩展。
- 版本环境共享模型文件，保留各自运行配置。

## 从源码运行

需要 Windows、64 位 Python 3.10+（建议 3.12 或 3.13）及适合所选引擎的硬件和驱动。

```powershell
python -m pip install -r requirements.txt
python strata_launcher.py
```

构建可执行文件：

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm StrataLauncher.spec
```

如需将官方源码内置到构建中，可另外下载所需 Strata 发布标签的源码 ZIP，命名为 `Strata-main.zip` 放在源码根目录。此仓库不提交模型、已安装引擎、虚拟环境或大体积上游源码包。

## 验证与限制

V3.5.1 在开发工作区通过 176 项 Python 回归测试、打包自检和界面自动加载检查。官方 API 与发布页备用通道均实际读取到 63 个正式版本（发布准备时的结果，后续数量会变化）。

可在源码目录运行独立版本管理测试：

```powershell
python -m unittest test_strata_versions -v
```

部分历史服务集成测试需要额外的上游源码测试夹具，不包含在此仓库中。模拟接口测试不代表已完成所有真实模型、GPU 和驱动组合的推理验证。历史版本缺少签名组件源或接口时会拒绝安装；未来版本可能需要补充适配。

## 上游与第三方

Strata 的版权及 MIT 许可保留于 `LICENSE-Strata.txt`。相关依赖和上游资源遵循各自许可；参见 `THIRD_PARTY_NOTICES.md`。本仓库未另行指定启动器自有代码的开源许可。
