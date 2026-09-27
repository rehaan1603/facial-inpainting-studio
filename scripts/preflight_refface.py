"""Verify strict author loading and synthetic-image inference equivalence."""
import ast
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.refface_baseline import RefFaceBaseline


def main():
    cache = Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    source, parser = cache/'refface_source_v1', cache/'refface_parser_source_v1'
    weights = cache/'refface_weights_v1'
    source_commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    parser_commit = subprocess.check_output(['git','-C',str(parser),'rev-parse','HEAD'],text=True).strip()
    assert source_commit == '0f1ad75677cc8fae4ae14d878e4c6cfce9365f28'
    assert parser_commit == 'd2e684cf1588b46145635e8fe7bcc29544e5537e'
    for checkout in [source, parser]:
        assert not subprocess.check_output(['git','-C',str(checkout),'diff'],text=True)
    records = json.loads((ROOT/'research/refface_downloads_v1.json').read_text())['files']
    records += [json.loads((ROOT/'research/refface_parser_download_v1.json').read_text())]
    for r in records: assert sha(weights/r['name']) == r['sha256']
    torch.set_num_threads(4); torch.manual_seed(17)
    start = time.monotonic()
    model = RefFaceBaseline(source, parser, weights)
    # Extract exactly the author inference method, avoiding unrelated Trainer imports.
    tree = ast.parse((source/'trainer.py').read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'test')
    module = ast.Module(body=[method], type_ignores=[]); ast.fix_missing_locations(module)
    namespace = {'torch':torch, 'F':F}
    exec(compile(module, str(source/'trainer.py'), 'exec'), namespace)
    obs = torch.rand(1,3,256,256,device='cuda')*2-1
    ref = torch.rand_like(obs)*2-1
    mask = torch.zeros(1,1,256,256,device='cuda'); mask[:,:,64:192,64:192] = 1
    labels = torch.randint(0,19,mask.shape,device='cuda')
    dummy = SimpleNamespace(netG=model.generator, arcface=model.arcface,
        gen_segmap=lambda lab: torch.zeros(1,19,256,256,device='cuda').scatter_(1,lab.long(),1.))
    with torch.no_grad():
        author = namespace['test'](dummy, obs, obs*(1-mask), ref, mask, labels)
        adapted = model.native(obs,mask,ref,labels)*mask+obs*(1-mask)
    exact = torch.equal(author, adapted)
    assert exact, 'Adapter changes author inference'
    # Exercise reference parsing on synthetic RGB only, never dataset targets.
    import numpy as np
    from PIL import Image
    fake = Image.fromarray(np.full((512,512,3),128,dtype=np.uint8))
    parsed = model.reference_labels(fake)
    assert parsed.shape == (1,1,256,256) and parsed.min()>=0 and parsed.max()<=18
    receipt = {'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':source_commit,'parser_commit':parser_commit,'source_checkouts_unmodified':True,
        'all_three_models_strict_loaded':True,'native_author_bit_exact':exact,
        'native_max_abs_difference':float((author-adapted).abs().max()),
        'reference_parser_exercised':True,'dataset_pixels_read':False,
        'weight_sha256':{r['name']:r['sha256'] for r in records},
        'wrapper_sha256':sha(ROOT/'src/preservation/refface_baseline.py'),
        'preflight_sha256':sha(__file__),'torch':torch.__version__,
        'seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Synthetic preflight only. No reconstruction accuracy or novelty result.'}
    write_new(ROOT/'research/refface_preflight_v1.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
