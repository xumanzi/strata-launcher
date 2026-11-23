import json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import strata_update as u

class NativeUpdateTests(unittest.TestCase):
 def fixture(self,d):
  r=Path(d).resolve(); (r/'setup.py').write_text('MIN_ENGINE = (0, 1, 46)'); (r/'CMakeLists.txt').write_text('project(strata VERSION 0.1.48.1)')
  old=r/'engine'; old.mkdir(); (old/'strata.exe').write_bytes(b'old')
  cfg=r/'strata-model.json'; data={'exe':str(old/'strata.exe'),'args':['--model','keep.gguf'],'api_key':'keep','vision':{'exe':str(old/'strata-vision.exe')},'lib_dirs':[str(old)]}; cfg.write_text(json.dumps(data))
  pkg=r/'.strata-launcher-components'/'auto'/'engine'/'0.1.48-cuda'; pkg.mkdir(parents=True)
  (pkg/'strata.exe').write_bytes(b'new'); (pkg/'strata-vision.exe').write_bytes(b'vision')
  files=[{'path':name,'bytes':(pkg/name).stat().st_size,'sha256':hashlib.sha256((pkg/name).read_bytes()).hexdigest()} for name in ['strata.exe','strata-vision.exe']]
  (pkg/'engine-component.json').write_text(json.dumps({'version':'0.1.48','entrypoint':'strata.exe','backend':'cuda','files':files,'signature':{'key_id':'fixture'}}))
  return r,cfg,pkg,data
 def test_manifest_without_build_is_valid_and_unused_old_engine_is_ignored(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=self.fixture(d); data['exe']=str(p/'strata.exe'); c.write_text(json.dumps(data))
   self.assertIn('0.1.48',u.verify_engines(r)[0])
 def test_manifest_tampering_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=self.fixture(d); data['exe']=str(p/'strata.exe'); c.write_text(json.dumps(data)); (p/'strata.exe').write_bytes(b'bad')
   with self.assertRaises(RuntimeError): u.verify_engines(r)
 def test_component_activation_preserves_model_settings_and_old_binary(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=self.fixture(d); u.activate_components(r,{c:p})
   saved=json.loads(c.read_text()); self.assertEqual(saved['exe'],str(p/'strata.exe')); self.assertEqual(saved['api_key'],'keep'); self.assertEqual(saved['args'],data['args'])
   self.assertEqual(saved['vision']['exe'],str(p/'strata-vision.exe')); self.assertEqual((r/'engine/strata.exe').read_bytes(),b'old')
   self.assertEqual(json.loads(Path(str(c)+'.update.bak').read_text()),data)
 def test_failed_activation_rolls_back_all_configs(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=self.fixture(d); c2=r/'strata-second.json'; c2.write_text(json.dumps(data)); original=u.atomic_json
   def fail(path,value):
    if path==c2: raise OSError('fixture locked config')
    original(path,value)
   with patch.object(u,'atomic_json',side_effect=fail):
    with self.assertRaises(OSError): u.activate_components(r,{c:p,c2:p})
   self.assertEqual(json.loads(c.read_text()),data)
 def test_external_backend_is_excluded_from_native_engine_repair_and_verification(self):
  with tempfile.TemporaryDirectory() as d:
   root,path,pkg,config=self.fixture(d)
   config.update(engine='llama.cpp',exe=str(root/'llama-server.exe'),args=['--model','mini.gguf','--ctx-size','4096'])
   path.write_text(json.dumps(config))
   with patch.object(u,'load_component_api') as loader:
    u.repair_active_configs(root,path)
    loader.assert_not_called()
   self.assertEqual(u.engine_targets(root),{})
   from strata_versions import validate_config_flags
   parser=root/'src/program/generate.cpp'; parser.parent.mkdir(parents=True)
   parser.write_text('const char * arg = "--native";')
   validate_config_flags(root)
   self.assertEqual(json.loads(path.read_text()),config)
 def test_repair_prefers_recorded_auto_store_even_if_cuda_backend_is_known(self):
  with tempfile.TemporaryDirectory() as d:
   root,path,pkg,config=self.fixture(d)
   config.update(backend='cuda',exe=str(pkg/'strata.exe'))
   path.write_text(json.dumps(config))
   ec=SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(root=pkg)),verify_signature=Mock())
   with patch.object(u,'load_component_api',return_value=ec):
    u.repair_active_configs(root,path)
   ec.resolve.assert_called_once_with(root/'.strata-launcher-components'/'auto',backend='cuda',full=True)
 def test_unique_cuda_store_is_used_for_legacy_config_without_backend(self):
  with tempfile.TemporaryDirectory() as d:
   root,path,pkg,config=self.fixture(d)
   auto=root/'.strata-launcher-components'/'auto'
   auto.replace(auto.with_name('cuda'))
   self.assertEqual(u.component_slot(root,config),'cuda')
 def test_cli_result_must_explicitly_confirm_engine(self):
  with self.assertRaises(RuntimeError): u.require_engine_result({'ok':True,'results':[{'component':'engine','status':'none-for-this-pc'}]})
  u.require_engine_result({'ok':True,'results':[{'component':'engine','status':'current'}]})
 def test_cross_process_lock_rejects_second_updater(self):
  import subprocess,sys
  with tempfile.TemporaryDirectory() as d:
   with u.launcher_update_lock(Path(d)):
    code='import strata_update as u; from pathlib import Path;\nwith u.launcher_update_lock(Path('+repr(d)+')): pass'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    self.assertNotEqual(r.returncode,0)
 def test_native_flow_does_not_use_legacy_zip_or_locked_feed(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=self.fixture(d)
   ec=SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(root=p,exe=p/'strata.exe')),verify_signature=Mock())
   with patch.object(u,'run_signed_cli',return_value={'ok':True,'results':[{'component':'engine','status':'updated'}]}) as cli,patch.object(u,'load_component_api',return_value=ec):
    u.update_signed_components(r)
   args=cli.call_args.args
   self.assertIn('.strata-launcher-components',str(args[1])); self.assertNotIn('engine-feed',str(args[1]))
   self.assertEqual(json.loads(c.read_text())['exe'],str(p/'strata.exe'))

class NativeCliTests(unittest.TestCase):
 def test_real_child_process_uses_jsonl_and_explicit_store(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve(); script=r/'strata.py'
   script.write_text("""import json,sys,os
from pathlib import Path
Path(__file__).with_name('call.json').write_text(json.dumps({'argv':sys.argv,'signed':os.environ.get('STRATA_REQUIRE_SIGNED_COMPONENTS')}))
print(json.dumps({'event':'phase','message':'fixture checking signature'}),flush=True)
print(json.dumps({'ok':True,'results':[{'component':'engine','status':'current'}]}),flush=True)
""")
   store=r/'.strata-launcher-components/auto'
   result=u.run_signed_cli(r,store,log=lambda s:None)
   call=json.loads((r/'call.json').read_text()); self.assertEqual(call['signed'],'1')
   self.assertIn(str(store),call['argv']); self.assertIn('--no-control',call['argv']); self.assertIn('jsonl',call['argv'])
   self.assertTrue(result['ok'])
 def test_busy_child_retries_without_removing_its_lock(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve(); store=r/'.strata-launcher-components/auto'; store.mkdir(parents=True)
   lock=store/'update.lock'; lock.write_bytes(b'keep')
   (r/'strata.py').write_text("""import json,sys
from pathlib import Path
p=Path(__file__).with_name('attempt.txt'); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n))
if n<3:
 print(json.dumps({'ok':False,'error':{'code':7,'message':'another update is already running'}})); sys.exit(7)
print(json.dumps({'ok':True,'results':[{'component':'engine','status':'current'}]}))
""")
   with patch.object(u.time,'sleep') as sleep:
    u.run_signed_cli(r,store,log=lambda s:None)
   self.assertEqual(sleep.call_count,2); self.assertEqual(lock.read_bytes(),b'keep')
 def test_next_update_reuses_auto_store_after_backend_is_known(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=NativeUpdateTests().fixture(d); u.activate_components(r,{c:p})
   ec=SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(root=p,exe=p/'strata.exe')),verify_signature=Mock())
   with patch.object(u,'run_signed_cli',return_value={'ok':True,'results':[{'component':'engine','status':'current','installed_before':{'version':'0.1.48'}}]}) as cli,patch.object(u,'load_component_api',return_value=ec):
    u.update_signed_components(r,log=lambda s:None)
   self.assertEqual(cli.call_args.args[1],r/'.strata-launcher-components/auto')
 def test_resolved_version_must_match_official_result(self):
  with tempfile.TemporaryDirectory() as d:
   r,c,p,data=NativeUpdateTests().fixture(d)
   ec=SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(root=p,exe=p/'strata.exe')),verify_signature=Mock())
   with patch.object(u,'run_signed_cli',return_value={'ok':True,'results':[{'component':'engine','status':'updated','installed_after':{'version':'0.1.49'}}]}),patch.object(u,'load_component_api',return_value=ec):
    with self.assertRaises(RuntimeError): u.update_signed_components(r,log=lambda s:None)
   self.assertEqual(json.loads(c.read_text()),data)

if __name__=='__main__': unittest.main()
