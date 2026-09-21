"""Real GPU zero-gain control; new development-only outputs, no historical overwrite."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from src.local_correspondence.latent_fusion import LocalLatentPipeline

def main():
    from reference_inpaint import reconstruct
    out=ROOT/'outputs/local_zero_gain_v1'
    if out.exists():raise ValueError('Preserve previous check; use a separately versioned script for another run')
    out.mkdir()
    manifest=ROOT/'outputs/distortion_aware_v1/inference_manifest.json'
    c=next(c for c in json.loads(manifest.read_text())['cases'] if c['case_id']=='1306_mixed')
    assert c['identity']=='1306' and c['split']=='development_screen'
    runtime={};refs=[r['path'] for r in c['references']]
    settings=dict(seed=17,steps=30,scale=.8,strength=.99,blend='poisson')
    reconstruct(c['observed'],c['mask'],refs,out/'baseline.png',runtime=runtime,**settings)
    runtime['pipeline']=LocalLatentPipeline(runtime['pipeline'],np.ones((4,64,64),np.float32),np.zeros((64,64),np.float32))
    reconstruct(c['observed'],c['mask'],refs,out/'zero_gain.png',runtime=runtime,**settings)
    checks={suffix:sha(out/('baseline'+suffix))==sha(out/('zero_gain'+suffix)) for suffix in ['.png','_raw.png','_hard.png','_input.png','_effective_mask.png']}
    result={'date':'2026-09-21','case_id':c['case_id'],'settings':settings,'pixel_file_equivalence':checks,'passed':all(checks.values()),'baseline_sha256':sha(out/'baseline.png'),'zero_gain_sha256':sha(out/'zero_gain.png'),'source_sha256':sha(__file__),'wrapper_sha256':sha(ROOT/'src/local_correspondence/latent_fusion.py'),'generator_sha256':sha(ROOT/'scripts/reference_inpaint.py'),'final_test_used':False,'scope':'One matched full-GPU equivalence pair; no quality superiority claim'}
    write_new(out/'verification.json',result);write_new(ROOT/'research/local_zero_gain_verification_v1.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise AssertionError('Zero-gain outputs differ')
if __name__=='__main__':main()
