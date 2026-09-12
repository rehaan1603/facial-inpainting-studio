"""Resume the predeclared 1,728-row OSOR development comparison, then report it."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch,lpips
from osor_adapter import OSORLocal,CACHE
from area_matched_v3 import corrupt
from pilot import metrics
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.hub.set_dir(str(CACHE/'torch'))
    out=ROOT/'outputs/osor_evaluation_v1';out.mkdir(exist_ok=True)
    case_path=ROOT/'outputs/area_matched_v3_margin12/cases.json';cases=[c for c in json.loads(case_path.read_text()) if c['partition']=='assessment']
    assert len(cases)==432 and len({c['identity'] for c in cases})==48
    paths=['scripts/evaluate_osor.py','scripts/osor_adapter.py','scripts/area_matched_v3.py','scripts/pilot.py','research/OSOR_COMPARISON_PROTOCOL.md','research/osor_downloads.json','research/osor_runtime.json','research/osor-environment-lock.txt','data/manifests/celebahq_reviewed_v2.csv','outputs/area_matched_v3_margin12/cases.json']
    signature={p:digest(ROOT/p) for p in paths};guard=out/'signature.json'
    if guard.exists():assert json.loads(guard.read_text())==signature,'Protocol changed; create a versioned run.'
    else:guard.write_text(json.dumps(signature,indent=2))
    sources={int(r['hq_id']):r for r in csv.DictReader((ROOT/'data/manifests/celebahq_reviewed_v2.csv').open())}
    model=OSORLocal();(out/'model_loading.json').write_text(json.dumps(model.load_audit,indent=2));perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[]
    tensor=lambda a:torch.from_numpy(a.transpose(2,0,1).copy()).cuda()[None]*2-1
    for index,c in enumerate(cases):
        chunk=out/f"{c['hq_id']}_{c['location']}_{c['missing_pixels']}.json"
        if chunk.exists():rows.extend(json.loads(chunk.read_text()));continue
        source=sources[int(c['hq_id'])];assert source['split']=='val' and digest(source['image_path'])==c['source_sha256']
        target=np.array(Image.open(source['image_path']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
        folder=ROOT/c['mask_directory']
        def read_mask(name):
            path=folder/f'{name}.png';assert digest(path)==c['mask_sha256'][name]
            return np.array(Image.open(path))>0
        true=read_mask('true');observed=corrupt(target,true,c['seed'],c['center_xy']);current=[]
        for condition in c['conditions']:
            supplied=read_mask(condition);start=time.perf_counter();pred,alpha=model(observed,supplied,c['seed']);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            with torch.inference_mode():value=float(perceptual(tensor(pred),tensor(target)).item())
            assert np.isfinite(value)
            current.append({'method':'osor_direct','hq_id':c['hq_id'],'identity':c['identity'],'location':c['location'],'missing_pixels':c['missing_pixels'],'condition':condition,'seed':c['seed'],'full_face_lpips':value,'seconds_with_offload':elapsed,**metrics(pred,target,observed,true,supplied)})
        temp=chunk.with_suffix('.tmp');temp.write_text(json.dumps(current));temp.replace(chunk);rows.extend(current)
        (out/'progress.json').write_text(json.dumps({'completed_cases':index+1,'total_cases':len(cases),'completed_rows':len(rows),'status':'running'}))
        if (index+1)%12==0:print(f'OSOR {index+1}/{len(cases)} cases ({len(rows)} rows)',flush=True)
    assert len(rows)==1728
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (out/'progress.json').write_text(json.dumps({'completed_cases':432,'total_cases':432,'completed_rows':1728,'status':'complete'}))
    from report_osor import main as report
    report()
if __name__=='__main__':main()
