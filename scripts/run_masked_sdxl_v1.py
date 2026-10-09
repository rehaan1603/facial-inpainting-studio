"""Existing SDXL/FaceID baseline on frozen masked-task cases; no training."""
import argparse,json,time
from pathlib import Path
from reference_inpaint import reconstruct,sha,ROOT,CACHE

BASE=ROOT/'outputs/masked_task_v1'
OUT=ROOT/'outputs/masked_sdxl_v1'


def main(limit,resolution=512):
    global OUT
    if resolution==1024:OUT=ROOT/'outputs/masked_sdxl_resolution_v1'
    cases=json.loads((BASE/'cases.json').read_text());runtime={}
    refs={}
    for c in cases:
        for condition in ['clean','wrong_patch']:
            paths=[ROOT/'outputs/target_compatibility_v1/images'/c['identity']/name
                   for name in [condition+'.png','reference_1.png','reference_2.png','reference_3.png']]
            refs[c['identity']+'/'+condition]=[dict(path=str(p),sha256=sha(p)) for p in paths]
    config=dict(cases_sha256=sha(BASE/'cases.json'),source_sha256=sha(ROOT/'scripts/reference_inpaint.py'),
                adapter_sha256=sha(CACHE/'reference_models/ip-adapter-faceid-portrait_sdxl.bin'),
                seeds=[17,29,43] if resolution==512 else [17],conditions=['clean','wrong_patch'] if resolution==512 else ['clean'],steps=30,scale=.8,strength=1.,guidance=5.,
                resolution=resolution,blend='hard',memory='model',expected=48 if resolution==512 else 8,references=refs,
                scope='Existing mask-conditioned SDXL FaceID baseline, different architecture/settings from ReF-LDM; no superiority claim')
    if OUT.exists():assert json.loads((OUT/'config.json').read_text())==config
    else:
        OUT.mkdir();(OUT/'config.json').write_text(json.dumps(config,indent=2))
    record=OUT/'rows.json';rows=json.loads(record.read_text()) if record.exists() else []
    for r in rows:
        if r['status']=='complete':assert sha(OUT/r['file'])==r['sha256']
    done={(r['identity'],r['location'],r['condition'],r['seed']) for r in rows};count=0
    for seed in config['seeds']:
      for condition in config['conditions']:
       for c in cases:
        key=(c['identity'],c['location'],condition,seed)
        if key in done:continue
        if limit and count>=limit:return
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        rr=config['references'][c['identity']+'/'+condition]
        for ref in rr:assert sha(ref['path'])==ref['sha256']
        name='_'.join(map(str,key));path=OUT/(name+'.png');assert not path.exists() and not path.with_suffix('.json').exists()
        row=dict(identity=c['identity'],location=c['location'],condition=condition,seed=seed,file=path.name)
        start=time.monotonic()
        try:
            meta=reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in rr],path,
                             seed=seed,steps=30,scale=.8,memory='model',blend='hard',strength=1.,runtime=runtime,model_resolution=resolution)
            row.update(status='complete',sha256=sha(path),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,
                       peak_allocated_bytes=meta['peak_allocated_bytes'])
        except Exception as exc:
            row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
            if not isinstance(exc,ValueError) or not str(exc).startswith('Reference '):
                rows.append(row);record.write_text(json.dumps(rows,indent=2));raise
        rows.append(row);record.write_text(json.dumps(rows,indent=2));count+=1
        print('OUTPUT',len(rows),name,row['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--resolution',type=int,choices=[512,1024],default=512);a=p.parse_args();main(a.limit,a.resolution)
