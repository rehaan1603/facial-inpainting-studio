"""Warm local inference timing including tensor transfer, excluding disk IO and model loading."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from inpaint import Inpainter,RefinerPredictor
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(4);folder=ROOT/'outputs/benchmark_v2/cases/10371_brush'
    with Image.open(folder/'observed.png') as im:observed=np.asarray(im).astype('float32')/255
    with Image.open(folder/'under.png') as im:mask=np.asarray(im)>0
    refiner=RefinerPredictor(ROOT/'outputs/learned_refiner/preservation_weighted_17/best.pt');results=[]
    for name in ['lama','resshift']:
        model=Inpainter(name)
        for learned in [False,True]:
            def run():
                effective=refiner.probability(observed,mask)>=.35 if learned else mask
                return model(observed,effective,17)
            for _ in range(3):run()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(20):
                start=time.perf_counter();pred=run();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            assert np.isfinite(pred).all()
            results.append({'backbone':name,'refinement':learned,'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        del model;torch.cuda.empty_cache()
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'resolution':256,'batch_size':1,'warmups':3,'repeats':20,'scope':'Warm single-image inference including preprocessing tensors and CPU/GPU transfers, excluding image disk IO and model loading. One fixed development example, no background evaluation running. Refiner remains resident for all measurements. Not total project compute.','results':results}
    (ROOT/'research/inference_efficiency.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
