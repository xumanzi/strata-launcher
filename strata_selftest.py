def run_context_selftest(log=None):
    """Exercise bundled memory compression with a simulated summary, without loading a model."""
    from strata_memory import ContextCompressor
    compressor = ContextCompressor()
    count = lambda ms: sum(len(m.get('content', '')) for m in ms)
    old = [{'role': 'system', 'content': 's' * 50},
           {'role': 'user', 'content': 'a' * 1200},
           {'role': 'assistant', 'content': 'b' * 1500},
           {'role': 'user', 'content': 'c' * 200}]
    def summarize(text, cap):
        return 'goal=repair startup; completed=check engine; pending=benchmark'
    def compact(messages, fn=summarize):
        return compressor.compact(messages, count, 4096, 1800, fn, lambda text: len(text) + 200)
    kept, stats = compact(old)
    checks = [('旧历史被摘要替换', stats['summarized_messages'] == 2 and 'pending=benchmark' in kept[1]['content']),
              ('保留系统指令与最近问题', kept[0] == old[0] and kept[-1] == old[-1]),
              ('不修改客户端原始历史', len(old) == 4)]
    tools = [old[0], {'role': 'user', 'content': 'u' * 1500},
             {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1'}]},
             {'role': 'tool', 'tool_call_id': 'call1', 'content': 't' * 1500}, old[-1]]
    result, report = compact(tools)
    checks.append(('完整工具轮次进入摘要', result[-1] == old[-1] and report['summarized_messages'] == 3))
    single = [old[0], {'role': 'user', 'content': 'x' * 5000}]
    result, report = compact(single)
    checks.append(('不截断单条大输入', result == single and report['removed_messages'] == 0))
    _, report = compact(old, lambda text, cap: (_ for _ in ()).throw(RuntimeError('unexpected re-generation')))
    checks.append(('相同历史复用摘要缓存', report['summary_cached'] and report['summary_calls'] == 0))
    import sys
    from strata_launch_support import http_probe, port_in_use, port_owner, kill_pid_force, is_protected_process
    names = ('http_probe', 'port_in_use', 'port_owner', 'kill_pid_force', 'is_protected_process')
    entry = sys.modules.get('__main__')
    if not hasattr(entry, 'StrataLauncher'):
        entry = sys.modules.get('strata_launcher')
    if entry is not None and hasattr(entry, 'StrataLauncher'):
        namespace = entry.StrataLauncher.start_model.__globals__
        checks.append(('启动按钮所需公共函数完整', all(callable(namespace.get(name)) for name in names)))
        checks.append(('界面异常可见', 'report_callback_exception' in vars(entry.StrataLauncher)))
    else:
        checks.append(('启动辅助模块完整', all(callable(fn) for fn in (http_probe, port_in_use, port_owner, kill_pid_force, is_protected_process))))
    failures = [name for name, ok in checks if not ok]
    from strata_catalog import bundled_tables, family_models
    try:
        models, families, contexts = bundled_tables()
        checks.append(('内置完整官方模型目录', 'deepseek41' in families and 'qwen36' in families
                       and sum(len(family_models(models, f)) for f in families) >= 25))
        checks.append(('DeepSeek 独立规格与上下文', all(row.get('multi_file') and row.get('contexts') == [8192]
                       for row in family_models(models, 'deepseek41').values())
                       and len(family_models(models, 'deepseek41')) >= 4))
    except (OSError, ValueError, KeyError):
        checks.append(('内置完整官方模型目录', False))
    failures = [name for name, ok in checks if not ok]
    if log:
        for name, ok in checks:
            log(('通过：' if ok else '失败：') + name)
        log('此自检使用模拟摘要；真实模型的摘要内容和质量需在实际模型上确认。')
    return {'passed': len(checks) - len(failures), 'failed': len(failures), 'failures': failures}
