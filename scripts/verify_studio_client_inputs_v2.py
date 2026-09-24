"""Post-fix client input checks, separate from the frozen unfamiliar audit."""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from unfamiliar_studio_audit_v2 import cases
from verify_studio_fixes_v2 import png, request


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recover-restoration',action='store_true');args=parser.parse_args()
    suffix='_recovery' if args.recover_restoration else ''
    out = ROOT/('outputs/studio_client_inputs_v2'+suffix); out.mkdir(exist_ok=False)
    lookup = {c['case_id']: c for c in cases()}; rows=[]
    plan = [('3182_removal','resshift','square'),('1037_removal','resshift','square'),
            ('3182_mixed','refldm','square'),('3182_mixed','refldm','rectangular_references')]
    if args.recover_restoration:plan=[p for p in plan if p[1]=='refldm']
    for case, model, framing in plan:
        c=lookup[case]; assert sha(c['observed'])==c['observed_sha256'] and sha(c['mask'])==c['mask_sha256']
        source=Image.open(c['observed']).convert('RGB');mask=Image.open(c['mask']).convert('L')
        payload=dict(image=png(source),mask=png(mask),backbone=model,mode='painted',detail='standard',blend='poisson')
        if model=='refldm':
            refs=[]
            for index,r in enumerate(c['references']):
                assert sha(r['path'])==r['sha256']; image=Image.open(r['path']).convert('RGB')
                if framing=='rectangular_references': image=image.crop((56,0,456,512))
                image.save(out/(framing+f'_reference_{index+1}.png'));refs.append(png(image))
            confidence=Image.fromarray(np.where(np.asarray(mask)>=128,128,255).astype('uint8'))
            payload.update(references=refs,confidence=png(confidence))
        session=request('/api/session');assert not session['busy']
        row=dict(case_id=case,backbone=model,framing=framing,status='failed')
        try:
            job=request('/api/inpaint',payload,session['token']);row['job_id']=job['id'];start=time.monotonic()
            while True:
                result=request('/api/jobs/'+job['id'])
                if result['status'] in ['complete','error']:break
                if time.monotonic()-start>1900:raise TimeoutError(job['id'])
                time.sleep(1)
            if result['status']!='complete':raise RuntimeError(result['message'])
            folder=ROOT/'outputs/webapp_runs'/job['id'];file=folder/Path(result['result']).name
            a=np.asarray(Image.open(file).convert('RGB'));b=np.asarray(Image.open(folder/'input.png').convert('RGB'))
            m=np.asarray(Image.open(folder/'effective_mask.png'))>=128;metadata=json.loads((folder/'metadata.json').read_text())
            row.update(status='complete',output_sha256=sha(file),seconds=result['seconds'],known_pixels_unchanged=bool(np.array_equal(a[~m],b[~m])))
            assert row['known_pixels_unchanged']
            if model=='refldm':
                prep=metadata['studio_reference_preparation'];row['reference_preparation']=prep
                if framing=='square':
                    assert all(r['original_sha256']==r['prepared_sha256'] for r in prep)
                    old=json.loads((ROOT/'outputs/unfamiliar_studio_audit_v2/rows/3182_mixed__refldm_partial.json').read_text())
                    row['byte_matches_pre_fix_default']=row['output_sha256']==old['output_sha256']
                else:assert all(r['geometry']['operation']=='face_centered_square_crop' for r in prep)
        except Exception as error:row.update(status='failed',error=str(error))
        rows.append(row);write_new(out/(case+'_'+model+'_'+framing+'.json'),row);print(case,model,framing,row['status'],flush=True)
    write_new(ROOT/('research/studio_client_inputs_verification_v2'+suffix+'.json'),dict(scope='Post-fix geometry and restoration validation. No generalization or accuracy claim. Initial failures remain in the original receipt.',server_sha256=sha(ROOT/'webapp/server.py'),worker_sha256=sha(ROOT/'scripts/studio_inference_worker.py'),rows=rows,final_test_used=False))


if __name__=='__main__':main()
