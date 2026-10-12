import sys,unittest
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mms_translation as m
from starlette.applications import Starlette
from starlette.testclient import TestClient

class Tokenizer:
    def encode(self,text,out_type): return text.split()
    def decode(self,tokens): return ' '.join(tokens)
class Model:
    def translate_batch(self,batch,**kw):
        assert batch[0][0]=='<2en>' and batch[0][-1]=='</s>'
        assert kw['beam_size']==1 and kw['max_decoding_length']==128
        return [SimpleNamespace(hypotheses=[['Attempted','English']])]
class Tests(unittest.TestCase):
    def setUp(self):
        self.engine=m.EnglishTranslator(); self.engine.tokenizer=Tokenizer();self.engine.model=Model()
        self.client=TestClient(Starlette(routes=m.translation_routes(self.engine)))
    def test_success(self):
        r=self.client.post('/mms/translate',json={'text':'Ka lawm tuk.'})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['text'],'Attempted English');self.assertTrue(r.json()['experimental'])
    def test_invalid(self):
        for payload in ({},{'text':None},{'text':5},{'text':''},{'text':'x'*1201},[],False,{'text':'a '*260}):
            self.assertEqual(self.client.post('/mms/translate',json=payload).status_code,400,payload)
    def test_bad_json(self):
        for body in (b'bad',b'\xff'):
            self.assertEqual(self.client.post('/mms/translate',content=body).status_code,400)
    def test_oversize(self):
        self.assertEqual(self.client.post('/mms/translate',content=b'x'*8193).status_code,413)
    def test_busy_and_recovery(self):
        self.engine.lock.acquire()
        self.assertEqual(self.client.post('/mms/translate',json={'text':'hi'}).status_code,429)
        self.engine.lock.release()
        self.assertEqual(self.client.post('/mms/translate',json={'text':'hi'}).status_code,200)
    def test_failure_isolated_and_no_text_leak(self):
        def fail(): raise RuntimeError('private text')
        self.engine.model=None;self.engine._load=fail
        r=self.client.post('/mms/translate',json={'text':'private source'})
        self.assertEqual(r.status_code,503);self.assertNotIn('private',r.text);self.assertFalse(self.engine.lock.locked())
    def test_truncation_flag(self):
        self.engine.model.translate_batch=lambda *a,**kw:[SimpleNamespace(hypotheses=[['x']*128])]
        self.assertTrue(self.client.post('/mms/translate',json={'text':'hi'}).json()['truncated'])
if __name__=='__main__':unittest.main()
