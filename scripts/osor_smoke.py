"""Functional OSOR check on an already-observed development sample; no quality claim."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from osor_adapter import OSORLocal
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(4);out=ROOT/'outputs/osor_smoke';out.mkdir(exist_ok=True)
    folder=ROOT/'outputs/benchmark_v2/cases/10371_brush'
    x=np.array(Image.open(folder/'observed.png').convert('RGB')).astype('float32')/255
    m=np.array(Image.open(folder/'under.png').convert('L'))>=128
    model=OSORLocal();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();pred,alpha=model(x,m,17);torch.cuda.synchronize();elapsed=time.perf_counter()-start
    Image.fromarray((pred.clip(0,1)*255).round().astype('uint8')).save(out/'result.png');Image.fromarray(alpha).save(out/'alpha.png')
    report={**model.load_audit,'seed':17,'resolution':[256,256],'seconds_including_cpu_gpu_transfers':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'output_sha256':hashlib.sha256((out/'result.png').read_bytes()).hexdigest(),'scope':'Author inference with memory offload; default direct decoded output without RGB paste-back. Single functional check, not a comparative benchmark or claimed paper reproduction.'}
    (ROOT/'research/osor_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
