# 上游与第三方说明

本启动器集成并调用 Niko1221/Strata 的安装和服务接口：

- 项目：https://github.com/Niko1221/Strata
- 版权：Copyright (c) 2026 Niko1221 and the Strata contributors
- 许可：MIT，完整文本见 `LICENSE-Strata.txt`。

发行版中的内置 Strata 源码包保留其许可文件。模型和推理引擎由相应上游提供，遵循各自的使用条件。

源码运行主要依赖 CustomTkinter 和 Jinja2；打包使用 PyInstaller，并携带 Python 运行时及依赖组件。各组件版权与许可归原作者所有。本说明不重新授权第三方组件。
