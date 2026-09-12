import csv
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from resshift_adapter import ResShiftFace
from pilot import make_case,metrics

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/local.json').read_text())
rows={r['hq_id']:r for r in csv.DictReader((ROOT/'data/manifests/celebahq_clean.csv').open())}
cases=json.loads((ROOT/'outputs/pilot_clean_v1/cases.json').read_text())[:4]
model=ResShiftFace();out=ROOT/'outputs/resshift_smoke';out.mkdir(parents=True,exist_ok=True)
records=[];preview=[]
torch.cuda.reset_peak_memory_stats()
for c in cases:
    r=rows[c['hq_id']]
    with Image.open(r['image_path']) as im: target=np.array(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255
    obs,true,masks=make_case(target,c['seed'])
    torch.cuda.synchronize(); start=time.perf_counter()
    pred=model(obs,true,seed=c['seed'])
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    records.append({'hq_id':c['hq_id'],'seconds':seconds,**metrics(pred,target,obs,true,true)})
    Image.fromarray((pred*255).round().astype('uint8')).save(out/f"{c['hq_id']}.png")
    preview.append([target,obs,pred])
    print('ResShift smoke:',c['hq_id'],seconds,flush=True)
canvas=Image.new('RGB',(768,280*len(preview)),'white');draw=ImageDraw.Draw(canvas)
for row,ims in enumerate(preview):
    for col,im in enumerate(ims):
        canvas.paste(Image.fromarray((im*255).round().astype('uint8')),(col*256,row*280+24))
        draw.text((col*256+4,row*280+3),['Target','Observed','ResShift exact mask'][col],fill='black')
canvas.save(out/'preview.png')
report={'records':records,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'scope':'Four-image compatibility smoke test on previously used validation cases; not a model ranking or official reproduction.'}
(ROOT/'research/resshift_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
