import unittest,tempfile,threading,time,re
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import Mock
import strata_download as d

DATA=bytes(range(256))*1024
class DownloadTests(unittest.TestCase):
 def server(self,mode='range'):
  counts={'active':0,'peak':0,'ranges':[]}; lock=threading.Lock()
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args): pass
   def do_GET(self):
    span=self.headers.get('Range'); match=re.fullmatch(r'bytes=(\d+)-(\d+)',span or '')
    if not match or mode=='ignore':
     self.send_response(200); self.send_header('Content-Length',str(len(DATA))); self.end_headers(); self.wfile.write(DATA); return
    start,end=map(int,match.groups()); end=min(end,len(DATA)-1)
    with lock: counts['ranges'].append((start,end)); counts['active']+=1; counts['peak']=max(counts['peak'],counts['active'])
    try:
     self.send_response(206); self.send_header('Content-Range',f'bytes {start+1 if mode=="bad" and end>start else start}-{end}/{len(DATA)}'); self.send_header('ETag','"test-v1"'); self.send_header('Content-Length',str(end-start+1)); self.end_headers()
     for i in range(start,end+1,4096): self.wfile.write(DATA[i:min(i+4096,end+1)]); time.sleep(.05 if mode=='slow' else .001)
    except ConnectionError: pass
    finally:
     with lock: counts['active']-=1
  httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler); t=threading.Thread(target=httpd.serve_forever,daemon=True); t.start(); self.addCleanup(httpd.server_close); self.addCleanup(httpd.shutdown)
  return f'http://127.0.0.1:{httpd.server_port}/archive',counts
 def test_parallel_ranges_preserve_existing_prefix_and_correct_content(self):
  url,c=self.server()
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'engine.part'; p.write_bytes(DATA[:12345]); progress=[]
   d.parallel_transfer(url,p,12345,len(DATA),lambda done,total:progress.append(done),None,5,workers=4)
   self.assertEqual(p.read_bytes(),DATA); self.assertGreater(c['peak'],1); self.assertEqual(progress[-1],len(DATA))
 def test_cancelled_segments_resume_on_next_attempt(self):
  url,c=self.server('slow'); cancel=threading.Event()
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'engine.part'; p.write_bytes(DATA[:123])
   def progress(done,total):
    if done>32000: cancel.set()
   with self.assertRaises(d.DownloadCancelled): d.parallel_transfer(url,p,123,len(DATA),progress,cancel,5,workers=4)
   self.assertEqual(p.read_bytes(),DATA[:123]); folder=p.with_name(p.name+'.ranges')
   self.assertTrue(any(x.stat().st_size>0 for x in folder.glob('segment-*.part')))
   d.parallel_transfer(url,p,123,len(DATA),None,None,5,workers=4)
   self.assertEqual(p.read_bytes(),DATA)
 def test_ignored_range_leaves_original_partial_untouched(self):
  url,c=self.server('ignore')
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'engine.part'; p.write_bytes(DATA[:11])
   with self.assertRaises(d.RangeUnavailable): d.parallel_transfer(url,p,11,len(DATA),None,None,5)
   self.assertEqual(p.read_bytes(),DATA[:11])
 def test_cancelled_download_does_not_mark_preallocated_file_complete(self):
  url,c=self.server(); cancel=threading.Event(); cancel.set()
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'engine.part'; p.write_bytes(DATA[:7])
   with self.assertRaises(d.DownloadCancelled): d.parallel_transfer(url,p,7,len(DATA),None,cancel,5)
   self.assertEqual(p.stat().st_size,7)
 def test_wrong_content_range_never_corrupts_official_partial(self):
  url,c=self.server('bad')
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'engine.part'; p.write_bytes(DATA[:17])
   with self.assertRaises(d.RangeUnavailable): d.parallel_transfer(url,p,17,len(DATA),None,None,5)
   self.assertEqual(p.read_bytes(),DATA[:17])
 def test_official_transport_falls_back_without_replacing_checksum_validation(self):
  transfer=type('Transfer',(),{})(); original=Mock(); transfer._transfer=lambda location,partial,have,size,progress,cancel,timeout: original(location,partial,have,size,progress,cancel,timeout); transfer._opener=None; transfer.check_url_policy=Mock(); transfer.kind_of=lambda url:'https'
  d.install_transport(transfer,threshold=1)
  from unittest.mock import patch
  with patch.object(d,'parallel_transfer',side_effect=d.RangeUnavailable('no ranges')):
   transfer._transfer('https://example.test/a',Path('fixture'),0,100,None,None,5)
  original.assert_called_once()

class SourceDownloadTests(unittest.TestCase):
 def test_source_download_writes_complete_response(self):
  case=DownloadTests(); url,_=case.server('ignore'); self.addCleanup(case.doCleanups)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'source.zip'
   self.assertEqual(d.download_source(url,path,log=lambda value:None),path)
   self.assertEqual(path.read_bytes(),DATA)
 def test_source_uses_only_official_codeload_then_original(self):
  url='https://github.com/Niko1221/Strata/archive/refs/tags/v0.2.3.1.zip'
  self.assertEqual(d.source_urls(url),['https://codeload.github.com/Niko1221/Strata/zip/refs/tags/v0.2.3.1',url])
  self.assertEqual(d.source_urls('https://example.test/file.zip'),['https://example.test/file.zip'])

if __name__=='__main__': unittest.main()

def test_official_transfer_with_new_options_is_not_replaced():
    from types import SimpleNamespace

    def official(location, partial, have, size, progress, cancel, timeout, *,
                 first=True, what='the download', say=None):
        return first, what, say

    transfer = SimpleNamespace(_transfer=official, _opener=None,
                               kind_of=lambda url: 'https', check_url_policy=Mock())
    d.install_transport(transfer, threshold=1)
    assert transfer._transfer is official
    assert transfer._transfer('https://example.test/a', Path('fixture'), 0, 100,
                              None, None, 5, first=False, what='engine') == (False, 'engine', None)
