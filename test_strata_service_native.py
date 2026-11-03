import unittest, tempfile, shutil, subprocess, socket, json, time, http.client, io, tarfile
from pathlib import Path
from strata_gateway import ContextGateway
from strata_migration import restore_legacy_patch
class NativeIntegration(unittest.TestCase):
 def test_browser_readiness_requires_strata_identity(self):
  from strata_launch_support import strata_health_ready
  self.assertFalse(strata_health_ready(200, '{"status":"ok"}'))
  self.assertFalse(strata_health_ready(200, '<html>llama.cpp</html>'))
  self.assertFalse(strata_health_ready(503, '{"service":"strata"}'))
  self.assertTrue(strata_health_ready(200, '{"status":"ok","service":"strata"}'))
 def test_unpatched_versions(self):
  for version, source in [('0.1.39',Path('.strata-src')),('0.1.40',Path('.strata-compat-src'))]:
   with self.subTest(version=version), tempfile.TemporaryDirectory() as d:
    root=Path(d)
    if version=='0.1.39':
     # Keep the entire historical runtime consistent after the working checkout is upgraded.
     archive=subprocess.check_output(['git','archive','HEAD'],cwd=source)
     with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
      members=[m for m in tf.getmembers() if m.name.startswith(('serve/','tools/','strata_control/')) or ('/' not in m.name and m.name.endswith('.py'))]
      tf.extractall(root,members=members,filter='data')
    else:
     for folder in ('serve','tools'):
      shutil.copytree(source/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__','*.bak'))
     for p in source.glob('*.py'): shutil.copy2(p,root/p.name)
    for name in ('server.py','runconfig.py'):
     data=(subprocess.check_output(['git','show','HEAD:serve/'+name],cwd=source) if version=='0.1.39' else (Path('.strata-compat-original')/name).read_bytes())
     (root/'serve'/name).write_bytes(data)
    before=(root/'serve/server.py').read_bytes()
    with socket.socket() as s: s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    logfile=open(root/'server.log','w')
    proc=subprocess.Popen(['python',str(root/'serve/server.py'),'--engine','mock','--host','127.0.0.1','--port',str(port),'--script','goal=repair startup; pending=benchmark'],stdout=logfile,stderr=subprocess.STDOUT)
    gateway=ContextGateway('127.0.0.1',0,port,{'context_compress_enabled':True,'fit_max_tokens':True});gateway.start()
    try:
     for i in range(80):
      try:
       c=http.client.HTTPConnection('127.0.0.1',port,timeout=1);c.request('GET','/health');r=c.getresponse();r.read();c.close()
       if r.status==200:break
      except OSError:pass
      time.sleep(.1)
     else:self.fail((root/'server.log').read_text())
     history=[{'role':'user','content':'old fact '+ 'x'*22000},{'role':'assistant','content':'y'*10000},{'role':'user','content':'latest question'}]
     for path,body in [('/v1/chat/completions',{'messages':history,'max_tokens':100}),('/v1/messages',{'messages':history,'max_tokens':100}),('/v1/responses',{'input':history,'max_output_tokens':100})]:
      body.update(model='qwen3.8-flash-next',stream=False,chat_template_kwargs={'enable_thinking':False})
      c=http.client.HTTPConnection('127.0.0.1',gateway.port,timeout=40);c.request('POST',path,json.dumps(body),{'Content-Type':'application/json'});r=c.getresponse();data=r.read();c.close();self.assertEqual(r.status,200,data.decode()[:2000]);self.assertGreater(gateway.report()['context_guard']['summarized_messages'],0)
     self.assertEqual((root/'serve/server.py').read_bytes(),before)
    finally:
     gateway.stop();proc.terminate();proc.wait(timeout=10);logfile.close()
 def test_restore_only_known_patch(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'serve').mkdir()
   for name in ('server.py','runconfig.py'):
    shutil.copy2(Path('.strata-compat-src/serve')/name,root/'serve'/name)
    shutil.copy2(Path('.strata-compat-original')/name,root/'serve'/(name+'.context-original.bak'))
   self.assertTrue(restore_legacy_patch(root))
   self.assertFalse(restore_legacy_patch(root))
   for name in ('server.py','runconfig.py'):self.assertEqual((root/'serve'/name).read_bytes(),(Path('.strata-compat-original')/name).read_bytes())
 def test_launcher_owns_public_service_and_native_child(self):
  from strata_launcher import ServerProcess, StrataLauncher
  from unittest.mock import patch
  from types import SimpleNamespace
  import sys
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for folder in ('serve','tools'):
    shutil.copytree(Path('.strata-compat-src')/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__','*.bak'))
   for p in Path('.strata-compat-src').glob('*.py'):shutil.copy2(p,root/p.name)
   for name in ('server.py','runconfig.py'):shutil.copy2(Path('.strata-compat-original')/name,root/'serve'/name)
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));public=sock.getsockname()[1]
   model_folder=root/'IQ3_XXS 破限';model_folder.mkdir()
   first=model_folder/'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf';first.write_bytes(b'GGUF')
   second=model_folder/first.name.replace('00001-of','00002-of');second.write_bytes(b'GGUF')
   full_name='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS'
   config=root/'model.json';config.write_text(json.dumps({'exe':sys.executable,'port':public,'host':'127.0.0.1','api_key':'test-key','context_compress_enabled':True,'fit_max_tokens':True,
    'model_name':'IQ3_XXS 破限','local_import':{'folder':str(model_folder)},
    'vision':{'exe':'strata-vision.exe','mmproj':'mmproj.gguf','model':str(first),'gpu':True},
    'args':['--native',str(first),'--ple-gguf',str(second),'--vision']}))
   saved=config.read_bytes();server=ServerProcess();popen=subprocess.Popen;captured={};browser_urls=[]
   def mock_engine(cmd,**kwargs):
    runtime=Path(cmd[cmd.index('--config')+1]);captured['runtime']=runtime
    native=json.loads(runtime.read_text());self.assertFalse(native['context_compress_enabled']);self.assertFalse(native['fit_max_tokens']);self.assertEqual(native['host'],'127.0.0.1');self.assertNotEqual(native['port'],public)
    self.assertNotIn('--vision',native['args']);self.assertNotIn('vision',native)
    self.assertEqual(native['model_name'],full_name)
    if __import__('os').name=='nt':self.assertTrue(native['args'][3].isascii())
    self.assertTrue(Path(native['args'][3]).samefile(second))
    self.assertEqual(Path(native['args'][1]).name,first.name)
    self.assertEqual(Path(native['args'][3]).name,second.name)
    cmd=list(cmd);cmd[cmd.index('--engine')+1]='mock'
    return popen(cmd,**kwargs)
   try:
    errors=[]
    widget=SimpleNamespace(configure=lambda **kw:None)
    app=SimpleNamespace(server=server,_selected_cfg=lambda:{'path':str(config)},env=SimpleNamespace(root=root,python=lambda:sys.executable),
     cfg={'open_browser':True},dash_log=SimpleNamespace(clear=lambda:None,append=lambda line:None),toast=lambda *args:errors.append(args),
     start_btn=widget,stop_btn=widget,compress_var=SimpleNamespace(get=lambda:True),fit_var=SimpleNamespace(get=lambda:True),
     runtime_vision_var=SimpleNamespace(get=lambda:False),
     _refresh_runtime_vision_status=lambda *args:None,
     _save_context_policy=lambda path,policy:StrataLauncher._save_context_policy(None,path,policy),
     set_status=lambda *args:None,runtime={},_refresh_compress_status=lambda:None,_drain_server_log=lambda *args:None,ui=lambda fn,*args:fn(*args))
    with patch('subprocess.Popen',side_effect=mock_engine), patch('webbrowser.open', side_effect=browser_urls.append):
     StrataLauncher.start_model(app)
     for _ in range(50):
      if server.running() and not getattr(app,'_starting',True):break
      if errors:self.fail(str(errors))
      time.sleep(.1)
     else:self.fail('button did not start native backend')
     self.assertEqual(server.port,public)
     for _ in range(80):
      c=http.client.HTTPConnection('127.0.0.1',public,timeout=2);c.request('GET','/health');r=c.getresponse();r.read();c.close()
      if r.status==200:break
      time.sleep(.1)
     else:self.fail('native mock did not become healthy')
     for _ in range(30):
      if browser_urls:break
      time.sleep(.1)
     self.assertEqual(len(browser_urls), 1)
     self.assertTrue(browser_urls[0].startswith(server.local_url() + '/?strata-ui='))
    c=http.client.HTTPConnection('127.0.0.1',public,timeout=2);c.request('GET','/v1/models',headers={'Authorization':'Bearer test-key'});r=c.getresponse();models=json.loads(r.read());c.close()
    self.assertEqual(r.status,200)
    self.assertEqual(models['data'][0]['id'],full_name)
    self.assertEqual(config.read_bytes(),saved)
    server.set_context_policy({'context_compress_enabled':False,'fit_max_tokens':False})
    self.assertEqual(json.loads(config.read_text())['port'],public)
    self.assertEqual(server.gateway.report()['live'],{'context_compress_enabled':False,'fit_max_tokens':False})
    for route, payload in [('/config', {'set': {'sampling.temperature': 0.6, 'fit_max_tokens': True, 'open_browser': True}}), ('/settings', {'defaults': {'temperature': 0.4}})]:
     c=http.client.HTTPConnection('127.0.0.1', public, timeout=5)
     c.request('POST', route, json.dumps(payload), {'Content-Type':'application/json','Authorization':'Bearer test-key','Origin':server.local_url()})
     r=c.getresponse();data=r.read();c.close();self.assertEqual(r.status,200,data.decode())
    original=json.loads(config.read_text());native=json.loads(captured['runtime'].read_text())
    self.assertEqual(original['port'],public)
    self.assertTrue(original['fit_max_tokens'])
    self.assertFalse(native['fit_max_tokens'])
    self.assertEqual(original['sampling']['temperature'],0.6)
    shared=json.loads(config.with_suffix('.shared-settings.json').read_text())
    self.assertEqual(shared['temperature'],0.4)
   finally:
    server.stop()
    for _ in range(50):
     if not captured['runtime'].exists():break
     time.sleep(.1)
    self.assertFalse(captured['runtime'].exists())
    with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',public),timeout=1)
if __name__=='__main__':unittest.main()
