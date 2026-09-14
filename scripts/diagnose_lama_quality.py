"""Inspect local LaMa runs and compare 256/512 inference on a fixed development case."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from inpaint import Inpainter

ROOT=Path(__file__).resolve().parents[1]
def main():
    recent=[]
    for path in (ROOT/'outputs/webapp_runs').glob('*/metadata.json'):
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('backbone')=='lama':
            recent.append((path.stat().st_mtime,path.parent,data))
    for _,folder,meta in sorted(recent,reverse=True)[:5]:
        print('Recent LaMa:',folder,'mode:',meta.get('mask_mode'),flush=True)
    if '--list-only' in sys.argv:
        return
    torch.set_num_threads(4)
    model=Inpainter('lama')
    print('Model forward:',model.model.code[:2500],flush=True)
    case=ROOT/'outputs/reference_diagnostics_v1/identity_620'
    out=ROOT/'outputs/lama_resolution_diagnostic_v1'
    out.mkdir(exist_ok=True)
    observed=Image.open(case/'observed.png').convert('RGB')
    mask=Image.open(case/'mask.png').convert('L')
    target=Image.open(case/'target.png').convert('RGB')
    rows=[]
    for resolution in [256,512]:
        source=np.asarray(observed.resize((resolution,resolution),Image.Resampling.LANCZOS)).astype('float32')/255
        binary=np.asarray(mask.resize((resolution,resolution),Image.Resampling.NEAREST))>=128
        start=time.perf_counter()
        pred=model(source,binary)
        torch.cuda.synchronize()
        path=out/f'result_{resolution}.png'
        Image.fromarray((pred*255).round().astype('uint8')).save(path)
        rows.append({'resolution':resolution,'seconds':time.perf_counter()-start,
                     'visible_exact':bool(np.array_equal(pred[~binary],source[~binary]))})
        print(rows[-1],flush=True)
    observed.save(out/'observed.png');target.save(out/'target.png');mask.save(out/'mask.png')
    (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
