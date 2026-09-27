"""Frozen author reference-guided inpainting screen; no clean target access."""
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.refface_baseline import RefFaceBaseline

BASE = ROOT/'outputs/refface_baseline_v1'
PROTOCOL = ROOT/'research/protocols/refface_baseline_v1.json'


def main():
    cfg = json.loads(PROTOCOL.read_text())
    manifest = ROOT/'outputs/distortion_aware_v1/inference_manifest.json'
    cases = {c['case_id']:c for c in json.loads(manifest.read_text())['cases']}
    previous = ROOT/'outputs/reference_context_v2/comparison.json'
    controls = {(r['case_id'],r['seed'],r['mode']):r for r in json.loads(previous.read_text())['rows']}
    preflight = json.loads((ROOT/'research/refface_preflight_v1.json').read_text())
    assert preflight['native_author_bit_exact'] and preflight['all_three_models_strict_loaded']
    assert preflight['wrapper_sha256'] == sha(ROOT/'src/preservation/refface_baseline.py')
    cache = Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    for folder, key in [('refface_source_v1','source_commit'),('refface_parser_source_v1','parser_commit')]:
        assert subprocess.check_output(['git','-C',str(cache/folder),'rev-parse','HEAD'],text=True).strip()==cfg[key]
        assert not subprocess.check_output(['git','-C',str(cache/folder),'diff'],text=True)
    for name,digest in preflight['weight_sha256'].items():assert sha(cache/'refface_weights_v1'/name)==digest
    for cid in cfg['cases']:
        c=cases[cid]; assert c['identity'] not in cfg['reserved_identities'] and len(c['references'])==4
        for path,digest in [(c['observed'],c['observed_sha256']),(c['mask'],c['mask_sha256'])]+[(r['path'],r['sha256']) for r in c['references']]:assert sha(path)==digest
    BASE.mkdir(exist_ok=False)
    write_new(BASE/'signature.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(PROTOCOL),'runner_sha256':sha(__file__),
        'wrapper_sha256':sha(ROOT/'src/preservation/refface_baseline.py'),
        'preflight_sha256':sha(ROOT/'research/refface_preflight_v1.json'),
        'inference_manifest_sha256':sha(manifest),'prior_comparison_sha256':sha(previous),
        'source_commit':cfg['source_commit'],'parser_commit':cfg['parser_commit'],
        'weight_sha256':preflight['weight_sha256'],'inference_scope':cfg['inference_scope']})
    torch.set_num_threads(4);torch.manual_seed(17);torch.set_grad_enabled(False)
    model=RefFaceBaseline(cache/'refface_source_v1',cache/'refface_parser_source_v1',cache/'refface_weights_v1')
    rows=[]
    for cid in cfg['cases']:
        c=cases[cid]
        with Image.open(c['observed']) as image:observed=image.convert('RGB')
        with Image.open(c['mask']) as image:missing=np.asarray(image.convert('L'))>=128
        for mode in cfg['arms']:
            key=f'{cid}_s17_{mode}'
            row={'key':key,'case_id':cid,'identity':c['identity'],'seed':17,'mode':mode,'status':'failed'}
            if mode in ['scaffold','reference_context']:
                prior=controls[(cid,17,mode)];assert prior['status']=='complete' and sha(prior['output'])==prior['output_sha256']
                row.update(status='complete',output=prior['output'],output_sha256=prior['output_sha256'],reused_control=True)
            elif mode=='observed':
                row.update(status='complete',output=c['observed'],output_sha256=c['observed_sha256'],reused_control=True)
            else:
                start=time.monotonic()
                try:
                    ref_index=int(mode.split('_')[1]);ref=c['references'][ref_index]
                    with Image.open(ref['path']) as image:reference=image.convert('RGB')
                    torch.cuda.reset_peak_memory_stats()
                    result,native,meta=model.reconstruct(observed,missing,reference)
                    folder=BASE/'generations';folder.mkdir(exist_ok=True)
                    output=folder/(key+'.png');raw=folder/(key+'_native.png')
                    result.save(output);native.save(raw)
                    row.update(status='complete',output=str(output),output_sha256=sha(output),native_sha256=sha(raw),
                        reference_index=ref_index,reference_input_sha256=ref['sha256'],
                        generation_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),**meta)
                except Exception as error:
                    row.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
                row['attempt_seconds']=time.monotonic()-start
            rows.append(row);write_new(BASE/'receipts'/(key+'.json'),row)
            print('REFFACE',len(rows),'/28',key,row['status'],flush=True)
    write_new(BASE/'comparison.json',{'rows':rows,'signature_sha256':sha(BASE/'signature.json'),'final_test_used':False})


if __name__=='__main__':main()
