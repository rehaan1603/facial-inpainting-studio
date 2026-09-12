"""Exercise HTTP boundary behavior without loading or mocking model predictions."""
import base64,io,json,sys,threading,unittest
from pathlib import Path
from http.client import HTTPConnection
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'webapp'))
import server as studio

def png(size=(32,32),value=255):
    stream=io.BytesIO();Image.new('RGB',size,(value,)*3).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()

class WebBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=studio.ThreadingHTTPServer(('127.0.0.1',0),studio.Handler)
        cls.port=cls.server.server_port
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):studio.ACTIVE=False
    def request(self,path='/api/inpaint',payload=None,headers=None,method='POST'):
        h={'Content-Type':'application/json','X-Local-Token':studio.TOKEN}
        h.update(headers or {});con=HTTPConnection('127.0.0.1',self.port,timeout=5)
        con.request(method,path,json.dumps(payload) if payload is not None else None,h)
        response=con.getresponse();data=response.read();status=response.status;con.close()
        return status,data
    def valid(self):return {'image':png(),'mask':png(),'backbone':'lama','mode':'painted'}
    def test_cross_origin_and_host_rejected(self):
        self.assertEqual(self.request(payload=self.valid(),headers={'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.request(headers={'Host':'evil.example'})[0],403)
    def test_token_required(self):self.assertEqual(self.request(headers={'X-Local-Token':'wrong'})[0],403)
    def test_reference_count_and_mask_mode(self):
        data=self.valid();data['backbone']='reference';data['references']=[png(),png()]
        self.assertEqual(self.request(payload=data)[0],400)
        data['references']=[png()]*5;self.assertEqual(self.request(payload=data)[0],400)
        data['references']=[png()]*3;data['mode']='learned';self.assertEqual(self.request(payload=data)[0],400)
        data['mode']='painted'
        with patch.object(studio,'run_job'):
            self.assertEqual(self.request(payload=data)[0],202)
        studio.ACTIVE=False
    def test_non_object_json_is_client_error(self):self.assertEqual(self.request(payload=[])[0],400)
    def test_invalid_blending_rejected(self):
        data=self.valid();data['blend']='unknown';self.assertEqual(self.request(payload=data)[0],400)
    def test_reference_detail_validation_and_dispatch(self):
        data=self.valid();data.update(backbone='reference',references=[png()]*3,detail='unsupported')
        self.assertEqual(self.request(payload=data)[0],400)
        for detail in ['standard','detailed']:
            data['detail']=detail;studio.ACTIVE=False
            called=threading.Event()
            with patch.object(studio,'run_job',side_effect=lambda *args:called.set()) as worker:
                self.assertEqual(self.request(payload=data)[0],202)
                self.assertTrue(called.wait(5),'The admitted background job was not dispatched')
                self.assertEqual(worker.call_args.args[-1],detail)
    def test_mismatched_and_empty_masks_rejected(self):
        data=self.valid();data['mask']=png((16,16));self.assertEqual(self.request(payload=data)[0],400)
        data['mask']=png(value=0);self.assertEqual(self.request(payload=data)[0],400)
    def test_invalid_image_and_model_rejected(self):
        data=self.valid();data['image']='data:image/png;base64,@@';self.assertEqual(self.request(payload=data)[0],400)
        data=self.valid();data['backbone']='untrusted';self.assertEqual(self.request(payload=data)[0],400)
    def test_oversized_body_rejected_before_read(self):self.assertEqual(self.request(headers={'Content-Length':'16000001'})[0],413)
    def test_path_cannot_read_project_files(self):
        self.assertEqual(self.request(path='/../configs/local.json',method='GET')[0],404)
        self.assertEqual(self.request(path='/runs/'+'a'*32+'/../../configs/local.json',method='GET')[0],404)
    def test_job_admission_and_busy_response(self):
        with patch.object(studio,'run_job'):
            status,body=self.request(payload=self.valid());self.assertEqual(status,202)
            job_id=json.loads(body)['id'];self.assertEqual(studio.JOBS[job_id]['status'],'queued')
            self.assertEqual(self.request(payload=self.valid())[0],409)
        studio.ACTIVE=False

if __name__=='__main__':unittest.main()
