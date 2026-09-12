"""Frozen, exploratory test: 64 HQ identity labels and 64 LaPa images, real-object composites."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import shift
import torch,lpips
from inpaint import Inpainter,RefinerPredictor
from pilot import morph,metrics
from control_sweep import feather

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/object_test'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'scripts/inpaint.py',ROOT/'scripts/control_sweep.py',ROOT/'scripts/pilot.py',ROOT/'research/object_assets.json']
    paths += [ROOT/f'data/manifests/{d}_reviewed_v2.csv' for d in ['celebahq','lapa']]
    for variant in ['generic','preservation_weighted']:
        paths += [ROOT/f'outputs/learned_refiner/{variant}_17/best.pt',ROOT/f'outputs/refiner_evaluation/{variant}_17_selection.json']
    signature={str(p.relative_to(ROOT)):digest(p) for p in paths}
    if (OUT/'protocol.json').exists():
        saved=json.loads((OUT/'protocol.json').read_text());assert saved['signature']==signature,'Frozen inputs changed';return saved
    assets=json.loads((ROOT/'research/object_assets.json').read_text())['assets'];cases=[]
    for dataset in ['celebahq','lapa']:
        rows=list(csv.DictReader((ROOT/f'data/manifests/{dataset}_reviewed_v2.csv').open()))
        rows=[r for r in rows if r['split']=='test'];rows.sort(key=lambda r:hashlib.sha256(('object-test-v1:'+r['source_sha256']).encode()).hexdigest())
        selected=[];seen=set()
        for row in rows:
            unit=row.get('identity',row['source_sha256'])
            if unit in seen:continue
            seen.add(unit);selected.append(row)
            if len(selected)==64:break
        assert len(selected)==64
        for i,row in enumerate(selected):
            crop=None
            if dataset=='lapa':
                xy=np.loadtxt(row['landmark_path'],skiprows=1);assert xy.shape==(106,2) and np.isfinite(xy).all()
                center=(xy.min(0)+xy.max(0))/2;side=1.5*max(xy.max(0)-xy.min(0));assert side>0
                crop=[int(round(center[0]-side/2)),int(round(center[1]-side/2)),int(round(center[0]+side/2)),int(round(center[1]+side/2))]
            cases.append({'dataset':dataset,'unit':row.get('identity',row['source_sha256']),'image_path':row['image_path'],'source_sha256':row['source_sha256'],'crop':crop,'asset':assets[i%len(assets)],'seed':710000+i,'case_id':f'{dataset}_{i:03d}'})
    report={'signature':signature,'cases':cases,'protocol':'64 hash-selected test identity labels from HQ and 64 hash-selected LaPa test images. One object per image, 12 fixed assets reused; 4 error conditions. HQ resized; LaPa landmark bounding square padded by factor 1.5 then resized to 256. No target parsing or landmark input to inference.','methods':['unchanged','supplied','dilate8','tuned (LaMa 8/0, ResShift 12/4)','generic seed17','preservation_weighted seed17'],'thresholds':{v:json.loads((ROOT/f'outputs/refiner_evaluation/{v}_17_selection.json').read_text())['threshold'] for v in ['generic','preservation_weighted']},'interpretation':'Exploratory descriptive test, no preset practical-effect margin or confirmatory hypothesis. No changes after reading results. Single training seed here; three seeds on development. Image/identity bootstrap conditional on 12 fixed assets, not uncertainty over unseen objects. No real-capture or identity-fidelity claim.'}
    (OUT/'protocol.json').write_text(json.dumps(report,indent=2));return report

def make_case(c):
    assert digest(c['image_path'])==c['source_sha256'];assert digest(c['asset']['path'])==c['asset']['asset_sha256']
    with Image.open(c['image_path']) as im:
        im=im.convert('RGB')
        if c['crop'] is not None:im=im.crop(tuple(c['crop']))
        target=np.array(im.resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
    rng=np.random.default_rng(c['seed'])
    with Image.open(c['asset']['path']) as im:
        im=im.convert('RGBA');scale=float(rng.integers(100,145))/max(im.size);im=im.resize(tuple(max(1,round(n*scale)) for n in im.size),Image.Resampling.LANCZOS)
        cx,cy=map(int,rng.integers(105,151,2));canvas=Image.new('RGBA',(256,256));canvas.paste(im,(cx-im.width//2,cy-im.height//2));rgba=np.array(canvas)
    true=rgba[...,3]>=128;assert true.any() and (~true).any()
    observed=np.where(true[...,None],rgba[...,:3].astype('float32')/255,target)
    masks={'accurate':true,'under':morph(true,-4),'over':morph(true,4),'shift':shift(true.astype('float32'),(6,-6),order=0,mode='constant')>.5}
    return target,observed,true,masks

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');a=p.parse_args();protocol=freeze()
    if a.freeze_only:print('Frozen',len(protocol['cases']),'test images before inference');return
    torch.set_num_threads(4);local=json.loads((ROOT/'configs/local.json').read_text());torch.hub.set_dir(str(Path(local['cache'])/'torch'));perceptual=lpips.LPIPS(net='alex').cuda().eval()
    refiners={v:RefinerPredictor(ROOT/f'outputs/learned_refiner/{v}_17/best.pt') for v in protocol['thresholds']}
    @torch.inference_mode()
    def score(pred,target):
        tensor=lambda x:torch.from_numpy(x.transpose(2,0,1).copy()).cuda()[None]*2-1
        return float(perceptual(tensor(pred),tensor(target)).item())
    records=[]
    for backbone in ['lama','resshift']:
        inpainter=Inpainter(backbone)
        for i,c in enumerate(protocol['cases']):
            chunk=OUT/f"{backbone}_{c['case_id']}.json"
            if chunk.exists():records.extend(json.loads(chunk.read_text()));continue
            target,observed,true,masks=make_case(c);current=[];pictures=[target,observed]
            for condition,supplied in masks.items():
                methods={'unchanged':(np.zeros_like(true),0),'supplied':(supplied,0),'dilate8':(morph(supplied,8),0)}
                if backbone=='resshift':methods['tuned']=(morph(supplied,12),4)
                for v,refiner in refiners.items():methods[v]=(refiner.probability(observed,supplied)>=protocol['thresholds'][v],0)
                for method,(effective,width) in methods.items():
                    pred=inpainter(observed,effective,c['seed']);alpha=feather(effective,width)[...,None];pred=alpha*pred+(1-alpha)*observed
                    current.append({'dataset':c['dataset'],'unit':c['unit'],'case_id':c['case_id'],'asset_id':c['asset']['asset_id'],'category':c['asset']['category'],'condition':condition,'backbone':backbone,'method':method,'missing_fraction':float(true.mean()),'full_face_lpips':score(pred,target),'effective_fp_fraction':float((effective&~true).mean()),'effective_fn_fraction':float((~effective&true).mean()),**metrics(pred,target,observed,true,supplied)})
                    if condition=='under':pictures.append(pred)
            tmp=chunk.with_suffix('.tmp');tmp.write_text(json.dumps(current));tmp.replace(chunk);records.extend(current)
            if c['case_id'] in ['celebahq_000','lapa_000']:
                labels=['target','observed','unchanged','supplied','dilate8']+(['tuned'] if backbone=='resshift' else [])+list(refiners)
                sheet=Image.new('RGB',(256*len(pictures),280),'white');draw=ImageDraw.Draw(sheet)
                for j,(im,label) in enumerate(zip(pictures,labels)):
                    sheet.paste(Image.fromarray((im.clip(0,1)*255).round().astype('uint8')),(j*256,24));draw.text((j*256+5,5),label,fill='black')
                sheet.save(OUT/f"preview_{backbone}_{c['dataset']}.png")
            if (i+1)%16==0:print(backbone,i+1,len(protocol['cases']),flush=True)
        del inpainter;torch.cuda.empty_cache()
    assert len(records)==128*4*11
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    print('Finished exploratory object test',len(records),'rows')
if __name__=='__main__':main()
