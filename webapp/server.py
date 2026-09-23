"""Loopback-only web interface connected to the project's real GPU inpainting engine."""
import argparse,base64,binascii,hashlib,io,json,mimetypes,os,re,secrets,subprocess,sys,threading,time,traceback,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from PIL import Image,ImageOps,UnidentifiedImageError
ROOT=Path(__file__).resolve().parents[1];STATIC=Path(__file__).parent/'dist';RUNS=ROOT/'outputs/webapp_runs'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
Image.MAX_IMAGE_PIXELS=16_000_000
JOBS={};LOCK=threading.Lock();TOKEN=os.environ.get('INPAINTING_LOCAL_TOKEN') or secrets.token_urlsafe(32);ACTIVE=False

def decode_image(value,mask=False):
    if not isinstance(value,str) or not value.startswith('data:image/png;base64,'):raise ValueError('Images must be PNG data URLs.')
    blob=base64.b64decode(value.split(',',1)[1],validate=True)
    with Image.open(io.BytesIO(blob)) as im:
        if im.width*im.height>16_000_000 or min(im.size)<16:raise ValueError('Use an image between 16 pixels and 16 megapixels.')
        im.load();im=ImageOps.exif_transpose(im)
        return im.convert('L' if mask else 'RGB').copy()

def run_reference_job(job_id,source,mask,mode,references,blend,detail='standard',selection=False,confidence=None,strength=.99,restoration=False):
    job=JOBS[job_id];folder=RUNS/job_id;folder.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from pilot import morph
    geometry=None
    if confidence is not None:
        from webapp.geometry import fit_evidence
        source,mask,confidence,geometry=fit_evidence(source,mask,confidence)
    source=source.resize((512,512),Image.Resampling.LANCZOS);supplied=np.asarray(mask.resize((512,512),Image.Resampling.NEAREST))>=128
    if mode=='learned':raise ValueError('Use the painted or expanded mask with reference photos.')
    effective=morph(supplied,8) if mode=='expand' else supplied
    source.save(folder/'input.png');Image.fromarray(supplied.astype('uint8')*255).save(folder/'mask.png');Image.fromarray(effective.astype('uint8')*255).save(folder/'effective_mask.png')
    paths=[]
    for index,reference in enumerate(references):
        path=folder/f'reference_{index+1}.png';reference.save(path);paths.append(path)
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);python=cache/'reference_env_v2/Scripts/python.exe';progress=folder/'progress.json'
    if not python.is_file():raise ValueError('The reference environment is not installed. Run scripts/prepare_reference_runtime_v2.py with the project Python, then restart the studio.')
    command=[str(python),str(ROOT/'scripts/reference_inpaint.py'),'--image',str(folder/'input.png'),'--mask',str(folder/'effective_mask.png'),'--references',*[str(p) for p in paths],'--output',str(folder/'result.png'),'--progress',str(progress),'--blend',blend,'--strength',str(strength),'--model-resolution','1024' if detail=='detailed' else '512']
    if selection:command[1]=str(ROOT/'scripts/mask_aware_inpaint.py')
    if restoration:
        python=cache/'refldm_env_v1/Scripts/python.exe'
        if not python.is_file():raise ValueError('The experimental restoration environment is not installed.')
        command=[str(python),str(ROOT/'scripts/refldm_restore.py'),'--image',str(folder/'input.png'),'--mask',str(folder/'effective_mask.png'),'--references',*[str(p) for p in paths],'--output',str(folder/'result.png')]
    job.update(status='running',message='Preparing the reference photos…');start=time.monotonic()
    with (folder/'inference.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            if progress.exists():
                try:job['message']=json.loads(progress.read_text()).get('message','Reconstructing…')
                except (ValueError,OSError):pass
            if time.monotonic()-start>1800:process.terminate();process.wait();raise ValueError('Reference reconstruction timed out. See the saved inference log.')
            time.sleep(.5)
    if process.returncode:
        log_text=(folder/'inference.log').read_text(encoding='utf-8',errors='replace')
        if 'Application Control policy has blocked this file' in log_text:
            raise ValueError('Windows blocked a required reference-model component. The reference environment needs repair. LaMa remains available as a separate single-image option.')
        detail=json.loads(progress.read_text()).get('error') if progress.exists() else None
        raise ValueError(detail or 'Reference reconstruction failed. The saved inference log has details.')
    metadata=json.loads((folder/'result.json').read_text());metadata['mask_mode']=mode
    if geometry is not None:metadata['upload_geometry']=geometry
    result_name='result.png'
    if confidence is not None:
        from src.preservation.confidence import preserve_observation
        confidence=confidence.resize((512,512),Image.Resampling.NEAREST)
        confidence.save(folder/'confidence.png')
        observed=np.asarray(source);generated=np.asarray(Image.open(folder/'result.png').convert('RGB'))
        preserved=preserve_observation(observed,generated,effective,np.asarray(confidence,dtype=float)/255)
        result_name='preserved.png';Image.fromarray(preserved).save(folder/result_name)
        metadata.update(experimental_mode='user_confidence_preservation',parent_output_sha256=metadata['result_sha256'],confidence_sha256=hashlib.sha256((folder/'confidence.png').read_bytes()).hexdigest(),result_sha256=hashlib.sha256((folder/result_name).read_bytes()).hexdigest(),scope='Experimental user-supplied evidence blending; not verified recovery of the true face.')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
    if selection:
        diagnostics=metadata.get('reference_selection',{})
        job.update(warnings=diagnostics.get('warnings',[]),selected_reference_count=len(diagnostics.get('selected_paths',[])))
    job.update(status='complete',message='Evidence-weighted reconstruction ready' if confidence is not None else 'Reference-guided reconstruction ready',result=f'/runs/{job_id}/{result_name}',input=f'/runs/{job_id}/input.png',mask=f'/runs/{job_id}/effective_mask.png',metadata=f'/runs/{job_id}/metadata.json',seconds=round(metadata['inference_seconds_including_offload'],2),resolution=512)

def run_job(job_id,source,mask,backbone,mode,references=None,blend='poisson',detail='standard',confidence=None,strength=.99):
    global ACTIVE
    job=JOBS[job_id];folder=RUNS/job_id
    try:
        if backbone in ['reference','reference_select','refldm']:return run_reference_job(job_id,source,mask,mode,references,blend,detail,selection=backbone=='reference_select',confidence=confidence,strength=strength,restoration=backbone=='refldm')
        import numpy as np
        import torch
        from inpaint import Inpainter,RefinerPredictor
        from pilot import morph
        torch.set_num_threads(4);job.update(status='running',message=f'Loading {"LaMa" if backbone=="lama" else "ResShift"}…')
        resolution=512 if backbone=='lama' else 256
        folder.mkdir(parents=True,exist_ok=False);source=source.resize((resolution,resolution),Image.Resampling.LANCZOS);mask=mask.resize((resolution,resolution),Image.Resampling.NEAREST)
        observed=np.asarray(source).astype('float32')/255;supplied=np.asarray(mask)>=128
        if not supplied.any():raise ValueError('Paint or upload a mask before running inpainting.')
        effective=morph(supplied,8) if mode=='expand' else supplied
        refiner_path=None
        if mode=='learned':
            refiner_path=ROOT/'outputs/learned_refiner/generic_17/best.pt';effective=RefinerPredictor(refiner_path).probability(observed,supplied)>=.5
        if not effective.any():raise ValueError('The learned refiner selected no pixels. Try Use mask exactly or Expand mask by 8 px.')
        source.save(folder/'input.png');Image.fromarray(supplied.astype('uint8')*255).save(folder/'mask.png');Image.fromarray(effective.astype('uint8')*255).save(folder/'effective_mask.png')
        model=Inpainter(backbone);job['message']='Reconstructing the masked area…';start=time.perf_counter();pred=model(observed,effective,17);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if not np.isfinite(pred).all():raise RuntimeError('The model returned invalid pixels. Please try another image or model.')
        Image.fromarray((pred.clip(0,1)*255).round().astype('uint8')).save(folder/'result.png')
        metadata={'backbone':backbone,'mask_mode':mode,'resolution':[resolution,resolution],'seed':17,'inference_seconds':elapsed,'input_sha256':hashlib.sha256((folder/'input.png').read_bytes()).hexdigest(),'refiner_sha256':hashlib.sha256(refiner_path.read_bytes()).hexdigest() if refiner_path else None,'scope':'Local research inference. Generated hidden facial content is a prediction, not verified recovery.'}
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2));del model;torch.cuda.empty_cache()
        job.update(status='complete',message='Restoration ready',result=f'/runs/{job_id}/result.png',input=f'/runs/{job_id}/input.png',mask=f'/runs/{job_id}/effective_mask.png',metadata=f'/runs/{job_id}/metadata.json',seconds=round(elapsed,2),resolution=resolution)
    except Exception as exc:
        traceback.print_exc();message=str(exc) if isinstance(exc,ValueError) else 'Inpainting could not finish. The local server log has details; try again with LaMa.'
        job.update(status='error',message=message)
    finally:
        with LOCK:ACTIVE=False

class Handler(BaseHTTPRequestHandler):
    server_version='InpaintingStudio/1.0'
    def send(self,status,body,content_type='application/json'):
        if not isinstance(body,bytes):body=json.dumps(body).encode()
        self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.send_header('Cross-Origin-Resource-Policy','same-origin');self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'");self.end_headers();self.wfile.write(body)
    def trusted_host(self):return self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']
    def do_GET(self):
        if not self.trusted_host():return self.send(403,{'error':'Local access only.'})
        path=urlparse(self.path).path
        if path=='/api/session':return self.send(200,{'token':TOKEN,'busy':ACTIVE})
        if path=='/api/demo':
            if parse_qs(urlparse(self.path).query).get('reference')==['1']:
                folder=ROOT/'outputs/reference_examples/identity_916'
                files=[folder/'observed.png',folder/'mask.png',*[folder/f'reference_{i}.png' for i in range(1,5)]]
                if not all(p.is_file() for p in files):return self.send(404,{'error':'Reference sample unavailable. Upload a face and matching reference photos.'})
                encode=lambda p:'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
                return self.send(200,{'image':encode(files[0]),'mask':encode(files[1]),'references':[encode(p) for p in files[2:]]})
            folder=ROOT/'outputs/benchmark_v2/cases/10371_brush'
            if not (folder/'observed.png').exists():return self.send(404,{'error':'Sample unavailable. Upload a face to start.'})
            encode=lambda p:'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
            return self.send(200,{'image':encode(folder/'observed.png'),'mask':encode(folder/'true_mask.png')})
        if path.startswith('/api/jobs/'):
            job=JOBS.get(path.rsplit('/',1)[-1]);return self.send(200,job) if job else self.send(404,{'error':'Run not found. Start a new run.'})
        match=re.fullmatch(r'/runs/([a-f0-9]{32})/(input\.png|result\.png|preserved\.png|mask\.png|effective_mask\.png|metadata\.json)',path)
        if match:
            job_id,name=match.groups();file=RUNS/job_id/name
            if not file.is_file():return self.send(404,{'error':'Result not available.'})
        else:
            name='index.html' if path=='/' else path.removeprefix('/')
            if name not in ['index.html','app.js','framing.js','style.css','favicon.svg','confidence.html','confidence.js','confidence.css']:return self.send(404,{'error':'Not found.'})
            file=STATIC/name
        if not file.is_file():return self.send(404,{'error':'Not found.'})
        return self.send(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
    def do_POST(self):
        global ACTIVE
        if not self.trusted_host():return self.send(403,{'error':'Local access only.'})
        origin=self.headers.get('Origin');allowed=[f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']
        if origin and origin not in allowed:return self.send(403,{'error':'Cross-origin requests are disabled.'})
        if self.headers.get('X-Local-Token')!=TOKEN:return self.send(403,{'error':'Refresh the page and try again.'})
        if urlparse(self.path).path!='/api/inpaint':return self.send(404,{'error':'Not found.'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send(415,{'error':'Expected JSON.'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=16_000_000:return self.send(413,{'error':'Image is too large. Use a smaller image.'})
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
            backbone=data.get('backbone');mode=data.get('mode')
            if backbone not in ['lama','resshift','reference','reference_select','refldm'] or mode not in ['painted','expand','learned']:raise ValueError('Choose a supported model and mask mode.')
            source=decode_image(data.get('image'));mask=decode_image(data.get('mask'),True)
            if source.size!=mask.size:raise ValueError('Image and mask dimensions must match.')
            if mask.getextrema()[1]<128:raise ValueError('Paint or upload a mask first.')
            references=[];blend=data.get('blend','poisson');detail=data.get('detail','standard')
            if blend not in ['poisson','hard']:raise ValueError('Choose supported edge blending.')
            if detail not in ['standard','detailed']:raise ValueError('Choose standard or detailed processing.')
            if backbone in ['reference','reference_select','refldm']:
                values=data.get('references')
                minimum=1 if backbone=='reference_select' else 3
                if not isinstance(values,list) or not minimum<=len(values)<=4:raise ValueError(f'Upload {minimum} to 4 reference photos of the same person.')
                if mode=='learned':raise ValueError('Use the painted or expanded mask with reference photos.')
                references=[decode_image(value) for value in values]
            confidence=None;strength=.99
            if 'confidence' in data:
                if backbone not in ['reference','refldm'] or mode!='painted' or detail!='standard':raise ValueError('Evidence maps require standard reference reconstruction with the exact painted mask.')
                confidence=decode_image(data['confidence'],True)
                if confidence.size!=source.size:raise ValueError('Evidence map dimensions must match the image.')
                import numpy as np
                supplied=np.asarray(mask)>=128;weights=np.asarray(confidence)
                if not np.all(weights[~supplied]==255):raise ValueError('Evidence outside the damage mask must be white (keep).')
                if backbone=='refldm' and np.any(weights[supplied]==0):raise ValueError('Blur/noise restoration cannot fill missing areas. Choose missing-area reconstruction instead.')
                strength=data.get('strength',.99)
                if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not .5<=strength<=1:raise ValueError('Strength must be between 0.5 and 1.')
            elif 'strength' in data:raise ValueError('Custom strength requires an evidence map.')
            if backbone=='refldm' and confidence is None:raise ValueError('Blur/noise restoration requires a partial-damage evidence map.')
        except (ValueError,TypeError,UnidentifiedImageError,Image.DecompressionBombError,binascii.Error,OSError) as exc:return self.send(400,{'error':str(exc)})
        with LOCK:
            if ACTIVE:return self.send(409,{'error':'Another inpainting run is using the GPU. Try again when it finishes.'})
            ACTIVE=True
        job_id=uuid.uuid4().hex;JOBS[job_id]={'id':job_id,'status':'queued','message':'Preparing your image…'}
        threading.Thread(target=run_job,args=(job_id,source,mask,backbone,mode,references,blend,detail),kwargs={'confidence':confidence,'strength':strength} if confidence is not None else {},daemon=True).start();self.send(202,{'id':job_id})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args();RUNS.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);print(f'Inpainting Studio: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
