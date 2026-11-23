import unittest, tempfile, json
from pathlib import Path
from unittest.mock import Mock
import strata_update as u

class UpdateTests(unittest.TestCase):
 def tree(self, d):
  r=Path(d); (r/'setup.py').write_text('MIN_ENGINE = (0, 1, 46)\n'); (r/'CMakeLists.txt').write_text('project(strata VERSION 0.1.48.1)')
  e=r/'engine'; e.mkdir(); (e/'strata.exe').write_bytes(b'engine=0.1.44.1\n')
  (r/'strata-test.json').write_text(json.dumps({'exe':str(e/'strata.exe'),'args':[]}))
  return r,e
 def test_release_identity_independent_from_cmake(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d); u.record_release(r,'0.2.3.1'); self.assertEqual(u.installed_release(r),'0.2.3.1')
   (r/'setup.py').write_text('changed'); self.assertEqual(u.installed_release(r),'')
 def test_missing_build_cannot_pass_verification(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d)
   with self.assertRaisesRegex(RuntimeError,'BUILD.json'): u.verify_engines(r)
 def test_old_engine_cannot_pass_even_with_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d); (e/'BUILD.json').write_text('{"version":"0.1.44.1"}')
   with self.assertRaisesRegex(RuntimeError,'0.1.46'): u.verify_engines(r)
 def test_compatible_engine_passes_despite_release_number(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d); (e/'BUILD.json').write_text('{"version":"0.1.46"}'); self.assertTrue(u.verify_engines(r))
 def test_missing_build_uses_official_download_not_fabricated_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d); download=Mock(return_value=e/'strata.exe')
   mod={'PREBUILT_URL':'official','get_prebuilt':download,'update_engine_gpu':lambda toolkit:{'arch':86},'pip_cuda_libs':Mock()}
   u.repair_missing_metadata(r,mod)
   download.assert_called_once_with('official',{'arch':86},'gpu',updating=True,toolkit=13)
   self.assertFalse((e/'BUILD.json').exists())
 def test_moved_installation_checks_resolved_engine(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d)
   (r/'strata-test.json').write_text(json.dumps({'cwd':r'Q:\OldStrata','exe':r'Q:\OldStrata\engine\strata.exe','args':[]}))
   self.assertEqual(list(u.engine_targets(r)),[(e/'strata.exe').resolve()])
 def test_failed_download_is_failure(self):
  with tempfile.TemporaryDirectory() as d:
   r,e=self.tree(d); mod={'PREBUILT_URL':'official','get_prebuilt':Mock(return_value=None),'update_engine_gpu':lambda toolkit:{'arch':86}}
   with self.assertRaises(RuntimeError): u.repair_missing_metadata(r,mod)

class UpdateProcessTests(unittest.TestCase):
 def test_helper_repairs_missing_metadata_and_checks_subprocess(self):
  import subprocess,sys
  with tempfile.TemporaryDirectory() as d:
   r,e=UpdateTests().tree(d)
   setup = """from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
MIN_ENGINE = (0, 1, 46)
PREBUILT_URL = 'fixture'
def update_engine_gpu(toolkit): return {'arch': 86}
def pip_cuda_libs(toolkit): pass
def get_prebuilt(base, gpu, vision, updating=False, toolkit=13):
    (ROOT/'engine/BUILD.json').write_text(json.dumps({'version':'0.1.46'}))
    (ROOT/'engine/strata.exe').write_bytes(b'engine=0.1.46\\n')
    return ROOT/'engine/strata.exe'
if __name__ == '__main__': print('setup --update completed')
"""
   (r/'setup.py').write_text(setup)
   result=subprocess.run([sys.executable,str(Path(u.__file__).resolve()),str(r)],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   self.assertTrue(u.verify_engines(r))

class LauncherUpdateTests(unittest.TestCase):
 def app(self, root):
  from types import SimpleNamespace
  events={}
  cfg=SimpleNamespace(set=lambda k,v:events.__setitem__(k,v))
  label=SimpleNamespace(configure=Mock())
  app=SimpleNamespace(runtime={'updating':True},tools_log=SimpleNamespace(append=Mock()),
   env=SimpleNamespace(root=root,source_version=lambda:'0.1.48.1'),cfg=cfg,
   up_status=label,ver_label=SimpleNamespace(configure=Mock()),toast=Mock(),_update_target='0.2.3.1')
  return app,events
 def test_zero_exit_with_old_engine_does_not_mark_success(self):
  from strata_launcher import StrataLauncher
  with tempfile.TemporaryDirectory() as d:
   r,e=UpdateTests().tree(d); app,events=self.app(r)
   StrataLauncher._update_done(app,0)
   self.assertEqual(events['last_auto_updated_version'],'')
   self.assertEqual(u.installed_release(r),'')
   self.assertEqual(app.toast.call_args.args[0],'更新未完成')
 def test_success_records_release_not_cmake(self):
  from strata_launcher import StrataLauncher
  with tempfile.TemporaryDirectory() as d:
   r,e=UpdateTests().tree(d); (e/'BUILD.json').write_text('{"version":"0.1.46"}')
   app,events=self.app(r); StrataLauncher._update_done(app,0)
   self.assertEqual(events['last_auto_updated_version'],'0.2.3.1')
   self.assertEqual(u.installed_release(r),'0.2.3.1')
 def test_tag_archive_preserves_model_and_venv(self):
  from strata_launcher import StrataEnv
  from unittest.mock import patch
  import zipfile
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)/'install'; r.mkdir(); (r/'setup.py').write_text('old')
   (r/'.venv').mkdir(); (r/'.venv'/'keep').write_text('venv')
   (r/'strata-model.json').write_text('config'); (r/'model.gguf').write_text('model')
   env=StrataEnv(); env.root=r; urls=[]
   def download(url,dest,log):
    urls.append(url)
    with zipfile.ZipFile(dest,'w') as z:
     z.writestr('Strata-0.2.3.1/setup.py','MIN_ENGINE = (0, 1, 46)')
     z.writestr('Strata-0.2.3.1/CMakeLists.txt','project(strata VERSION 0.1.48.1)')
   env.download=download
   with patch('strata_launcher.APP_DIR',Path(d)):
    env.update_from_zip(version='0.2.3.1')
   self.assertTrue(urls[0].endswith('/refs/tags/v0.2.3.1.zip'))
   for file,value in [('.venv/keep','venv'),('strata-model.json','config'),('model.gguf','model')]:
    self.assertEqual((r/file).read_text(),value)
 def test_retry_uses_verified_source_without_redownloading(self):
  from strata_launcher import StrataEnv
  from unittest.mock import Mock
  with tempfile.TemporaryDirectory() as d:
   r,e=UpdateTests().tree(d); env=StrataEnv(); env.root=r; env.download=Mock(side_effect=AssertionError('must not redownload'))
   u.record_release(r,'0.2.3.1',u.SOURCE_MARKER)
   env.update_from_zip(version='0.2.3.1')
   env.download.assert_not_called()
 def test_startup_check_only_reports_new_release(self):
  from strata_launcher import StrataLauncher
  from types import SimpleNamespace
  cfg={'auto_update':True,'version_pinned':False,'last_auto_updated_version':'0.2.3.1'}
  app=SimpleNamespace(_remote_version='',cfg=SimpleNamespace(get=cfg.get,update=lambda **kw:None),
   ver_label=SimpleNamespace(configure=Mock()),up_status=SimpleNamespace(configure=Mock()),
   tools_log=SimpleNamespace(append=Mock()),_start_update=Mock())
  StrataLauncher._apply_update_info(app,'0.1.48.1','0.2.3.1','', '', '',True)
  app._start_update.assert_not_called()

class LatestSetupTests(unittest.TestCase):
 def test_latest_setup_imports_sibling_control_and_refreshes_its_modules(self):
  import sys
  from strata_launcher import StrataEnv
  saved={k:v for k,v in sys.modules.items() if k=='strata_setup' or k.startswith(('strata_control','strata_cli'))}
  try:
   with tempfile.TemporaryDirectory() as d:
    for name in ('first','second'):
     r=Path(d)/name; (r/'strata_control').mkdir(parents=True)
     (r/'strata_control/__init__.py').write_text('TOKEN = '+repr(name))
     (r/'setup.py').write_text("from strata_control import TOKEN\nMODELS={'fixture': {'id':TOKEN}}\nFAMILIES={}\nCONTEXTS=[32768]\n".replace('\\n','\n'))
     env=StrataEnv(); env.root=r
     self.assertEqual(env.tables()[0].get('fixture',{}).get('id'),name)
     self.assertIsNotNone(env._mod)
  finally:
   for k in list(sys.modules):
    if k=='strata_setup' or k.startswith(('strata_control','strata_cli')): sys.modules.pop(k,None)
   sys.modules.update(saved)

class ProductionSourceTests(unittest.TestCase):
 def test_update_excludes_harness_development_e2e_scripts(self):
  from strata_launcher import StrataEnv
  from unittest.mock import patch
  import zipfile
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)/'install'; r.mkdir(); (r/'setup.py').write_text('old')
   env=StrataEnv(); env.root=r
   def download(url,dest,log):
    with zipfile.ZipFile(dest,'w') as z:
     z.writestr('Strata-0.2.3.1/setup.py','MIN_ENGINE = (0, 1, 46)')
     z.writestr('Strata-0.2.3.1/CMakeLists.txt','project(strata VERSION 0.1.48.1)')
     z.writestr('Strata-0.2.3.1/integrations/harness/tool_flow_e2e.py','developer test')
     z.writestr('Strata-0.2.3.1/integrations/harness/_bridge.py','runtime bridge')
   env.download=download
   with patch('strata_launcher.APP_DIR',Path(d)):
    env.update_from_zip(version='0.2.3.1')
   self.assertFalse((r/'integrations/harness/tool_flow_e2e.py').exists())
   self.assertEqual((r/'integrations/harness/_bridge.py').read_text(),'runtime bridge')
   self.assertEqual(env.source_version(),'0.1.48.1')

if __name__=='__main__': unittest.main()
