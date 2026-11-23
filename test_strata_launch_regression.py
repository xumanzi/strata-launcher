import builtins,json,queue,socket,symtable,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import strata_launcher as launcher

class LaunchRegression(unittest.TestCase):
 def test_launcher_has_no_undefined_global_calls(self):
  source=Path('strata_launcher.py').read_text(encoding='utf-8-sig')
  table=symtable.symtable(source,'launcher','exec')
  known={x.get_name() for x in table.get_symbols() if x.is_assigned() or x.is_imported()}|set(dir(builtins))|{'__file__'}
  bad=set();pending=[table]
  while pending:
   table=pending.pop();pending.extend(table.get_children())
   bad.update(x.get_name() for x in table.get_symbols() if x.is_global() and x.is_referenced() and x.get_name() not in known)
  self.assertEqual(bad,set())
 def test_start_button_reaches_server_start(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.json';path.write_text(json.dumps({'exe':launcher.sys.executable,'port':54321}))
   launched=threading.Event();errors=[]
   server=SimpleNamespace(log_queue=queue.Queue(),running=lambda:False,start=lambda *a,**kw:launched.set() or ['native','server'])
   widget=SimpleNamespace(configure=lambda **kw:None)
   app=SimpleNamespace(server=server,_selected_cfg=lambda:{'path':str(path)},env=SimpleNamespace(root=Path(d),python=lambda:launcher.sys.executable),
    cfg=SimpleNamespace(get=lambda k:False),dash_log=SimpleNamespace(clear=lambda:None,append=lambda s:None),toast=lambda *a:errors.append(a),
    start_btn=widget,stop_btn=widget,compress_var=SimpleNamespace(get=lambda:True),fit_var=SimpleNamespace(get=lambda:True),
    _save_context_policy=lambda *a:None,set_status=lambda *a:None,runtime={},_refresh_compress_status=lambda:None,
    _drain_server_log=lambda *a:None,ui=lambda fn,*a:fn(*a))
   launcher.StrataLauncher.start_model(app)
   self.assertTrue(launched.wait(3),errors)
 def test_health_probe_exists_and_avoids_environment_proxy(self):
  from test_strata_gateway import Backend
  backend=Backend()
  try:
   with patch.dict('os.environ',{'HTTP_PROXY':'http://127.0.0.1:1','NO_PROXY':''}):
    self.assertEqual(launcher.http_probe('http://127.0.0.1:%s/health'%backend.server.server_port)[0],200)
  finally:backend.stop()
 def test_port_detection(self):
  with socket.socket() as sock:
   sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
   self.assertTrue(launcher.port_in_use(port))
  self.assertFalse(launcher.port_in_use(port))
 def test_callback_error_visible_and_logged(self):
  errors=[];app=SimpleNamespace(toast=lambda *a:errors.append(a))
  with patch.object(launcher,'log_line') as log:
   try:raise RuntimeError('test launch error')
   except RuntimeError as e:launcher.StrataLauncher.report_callback_exception(app,type(e),e,e.__traceback__)
   self.assertTrue(log.called);self.assertTrue(errors)
 def test_dependency_missing_detection(self):
  with patch('importlib.util.find_spec',return_value=None),patch('subprocess.run') as run:
   run.return_value=SimpleNamespace(returncode=0)
   launcher._auto_install_dependencies()
   self.assertTrue(run.called)

 def test_relative_engine_resolved_against_config_cwd(self):
  from strata_launch_support import resolve_launch_config
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();(root/'engine').mkdir();(root/'engine/strata.exe').touch()
   cfg={'exe':'engine/strata.exe','cwd':str(root)}
   adapted=resolve_launch_config(cfg,root)
   self.assertEqual(Path(adapted['exe']),root/'engine/strata.exe')
   self.assertEqual(cfg['exe'],'engine/strata.exe')
 def test_native_flash_next_recovers_missing_expert_profile(self):
  from strata_launch_support import resolve_launch_config
  with tempfile.TemporaryDirectory() as d:
   root = Path(d).resolve()
   shipped = root / 'data' / 'expert-profile.bin'
   shipped.parent.mkdir()
   shipped.touch()
   missing = str(root / 'retired-release.bin')
   cfg = {'backend': 'cuda', 'model_name': 'Qwen3.8-Flash-Next-IQ3_S',
          'local_import': {'family': 'qwen'},
          'args': ['--native', 'model.gguf', '--expert-cache', 'auto',
                   '--spec', '4', '--expert-profile', missing]}
   resolved = resolve_launch_config(cfg, root)
   self.assertEqual(resolved['args'], cfg['args'][:-1] + [str(shipped)])
   self.assertEqual(cfg['args'][-1], missing)
   custom = root / 'custom.bin'
   custom.touch()
   cfg['args'][-1] = str(custom)
   self.assertEqual(resolve_launch_config(cfg, root)['args'], cfg['args'])

 def test_moved_installation_paths_resolved_without_mutation(self):
  from strata_launch_support import resolve_launch_config
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve()/'Strata';root.mkdir();(root/'engine').mkdir();(root/'engine/strata.exe').touch()
   data=root.parent/'data';data.mkdir();(data/'model.bin').touch()
   cfg={'cwd':r'G:\old\Strata','exe':r'G:\old\Strata\engine\strata.exe','args':[r'G:\old\data\model.bin'],'api_key':'preserve'}
   adapted=resolve_launch_config(cfg,root)
   self.assertEqual(Path(adapted['exe']),root/'engine/strata.exe')
   self.assertEqual(Path(adapted['cwd']),root)
   self.assertEqual(Path(adapted['args'][0]),data/'model.bin')
   self.assertEqual(adapted['api_key'],'preserve')
 def test_path_adaptation_does_not_persist_runtime_paths(self):
  from strata_gateway import ContextGateway
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);original=root/'original.json';runtime=root/'runtime.json'
   source={'exe':'G:/old/strata.exe','cwd':'G:/old','tokenizer':'G:/tokenizer','args':['--model','G:/model.bin','--vram-reserve-mib','700'],'port':8080}
   original.write_text(json.dumps(source))
   clone={**source,'exe':'Z:/new/strata.exe','cwd':'Z:/new','tokenizer':'Z:/tokenizer','args':['--model','Z:/model.bin','--vram-reserve-mib','800'],'port':9999,'fit_max_tokens':False,'sampling':{'temperature':.3}}
   runtime.write_text(json.dumps(clone))
   gateway=ContextGateway('127.0.0.1',0,1,{'context_compress_enabled':True,'fit_max_tokens':True},cfg_path=original,runtime_config=runtime)
   try:
    gateway.mirror_config({'set':{'sampling.temperature':.3,'vram_reserve_mib':800}})
    saved=json.loads(original.read_text())
    for key in ('exe','cwd','tokenizer'):self.assertEqual(saved[key],source[key])
    self.assertEqual(saved['args'],['--model','G:/model.bin','--vram-reserve-mib','800'])
    self.assertEqual(saved['port'],8080)
   finally:gateway.stop()

 def test_slow_launch_does_not_block_button_and_shows_progress(self):
  import time
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.json';path.write_text(json.dumps({'exe':launcher.sys.executable,'port':54321}))
   launched=threading.Event();states=[]
   def slow(*a,**kw):time.sleep(.4);launched.set();return ['native','server']
   widget=SimpleNamespace(configure=lambda **kw:states.append(kw))
   app=SimpleNamespace(server=SimpleNamespace(log_queue=queue.Queue(),running=lambda:False,start=slow),_selected_cfg=lambda:{'path':str(path)},env=SimpleNamespace(root=Path(d),python=lambda:launcher.sys.executable),
    cfg=SimpleNamespace(get=lambda k:False),dash_log=SimpleNamespace(clear=lambda:None,append=lambda s:None),toast=lambda *a:None,
    start_btn=widget,stop_btn=widget,compress_var=SimpleNamespace(get=lambda:True),fit_var=SimpleNamespace(get=lambda:True),
    _save_context_policy=lambda *a:None,set_status=lambda *a:None,runtime={},_refresh_compress_status=lambda:None,
    _drain_server_log=lambda *a:None,ui=lambda fn,*a:fn(*a))
   before=time.monotonic();launcher.StrataLauncher.start_model(app)
   self.assertLess(time.monotonic()-before,.25)
   self.assertIn({'state':'disabled'},states)
   self.assertTrue(launched.wait(3))
 def test_new_installation_defaults_to_summary_protection(self):
  self.assertTrue(launcher.AppConfig.DEFAULTS['context_compress_enabled'])

if __name__=='__main__':unittest.main()
