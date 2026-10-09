"""Export existing metric/hash evidence; no photographs, features or inference."""
import ast, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/releases/2026-10-10'
EXPERIMENTS=[
    'masked_sdxl_v1','masked_sdxl_resolution_v1','focused_reference_finish_v1',
    'focused_eye_guidance_v1','focused_eye_scheduler_v1',
    'focused_eye_guidance_replication_v2','focused_eye_appearance_v1',
    'focused_eye_factorial_v1','focused_eye_texture_v1']
ROW_KEYS=['identity','location','condition','reference_condition','seed','guidance',
          'appearance_scale','scheduler','status','sha256','raw_sha256','seconds',
          'peak_allocated_bytes','reused','mechanism_status','metrics',
          'target_detections','output_detections']
TEST_LOGS={
    'webapp':'github_release_web_tests_20261010.log',
    'finishing':'github_release_finishing_tests_20261010.log',
    'spatial_mechanism':'github_release_spatial_tests_20261010.log',
    'spatial_statistics':'github_release_spatial_statistics_tests_20261010.log'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    experiments={};verified_images=0
    for name in EXPERIMENTS:
        folder=ROOT/'outputs'/name;p=folder/'summary.json';source=read(p)
        rows=[]
        for row in source['rows']:
            if row['status']=='complete':
                assert sha(folder/row['file'])==row['sha256'],(name,row['file'])
                verified_images+=1
            public={k:row[k] for k in ROW_KEYS if k in row}
            # Failure text remains in local source; preserve failure status without local paths.
            if row['status']!='complete':public['failure_record']='Retained in local source summary'
            rows.append(public)
        experiments[name]=dict(source_summary_sha256=sha(p),
            source_summary=f'outputs/{name}/summary.json',
            attempted=source['attempted'],expected=source['expected'],
            all_attempted=source['all_attempted'],all_completed=source['all_completed'],
            config_sha256=source.get('config_sha256'),scorer_sha256=source.get('scorer_sha256'),
            summary=source['summary'],rows=rows)
    write('metrics.json',dict(date='2026-10-10',
        scope='Development masked SDXL comparisons, with all attempts and null metrics retained; protocols differ by experiment.',
        condition_note='Guidance replication control/candidate labels are CFG arms; reference_condition is clean for both.',
        photographs_included=False,experiments=experiments))
    website=read(ROOT/'outputs/gentler_reference_website_v1/receipt.json')
    assert len(website)==4
    public=[]
    for row in website:
        assert row['status']=='complete' and row['raw_exact'] and row['finished_exact'] and row['visible_exact']
        assert sha(row['output'])==row['sha256']
        public.append({k:row[k] for k in ['identity','gentler_details','job_id','status','seconds','sha256',
                                         'visible_exact','finished_exact','raw_exact','metadata_sha256']})
    write('website_replays.json',dict(date='2026-10-10',rows=public,photographs_included=False))
    tests={}
    for name,log in TEST_LOGS.items():
        p=ROOT/'outputs'/log;text=p.read_text(encoding='utf-8')
        count=re.search(r'Ran (\d+) tests?',text)
        assert count and re.search(r'^OK(?:\s|$)',text,re.M),name
        assert 'FAILED (' not in text and 'Traceback (most recent call last)' not in text,name
        tests[name]=dict(tests=int(count.group(1)),passed=True,log_sha256=sha(p))
    # Parsing verifies syntax without importing models, writing bytecode or training.
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    python=[ROOT/p for p in paths if p.endswith('.py') and (ROOT/p).is_file()]
    for p in python:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p.relative_to(ROOT)))
    guidance=read(ROOT/'research/focused_eye_guidance_replication_v2_verification.json');assert guidance['passed']
    appearance=read(ROOT/'research/focused_eye_appearance_verification_v1.json');assert all(r['passed'] for r in appearance['experiments'].values())
    texture=read(ROOT/'research/focused_eye_texture_verification_v1.json')
    assert not texture['selected'] and all(r['zero_exact'] and r['regeneration_exact'] and r['visible_exact'] for r in texture['checks'])
    frozen=sha(ROOT/'scripts/reference_inpaint.py')
    assert frozen=='6fa4fe9c7742c8a1ec69d1586cc98a07573b1b85e5bebf0ad4c899b75a782daf'
    receipt=dict(date='2026-10-10',tests=tests,total_targeted_tests=sum(r['tests'] for r in tests.values()),
        python_files_parsed=len(python),completed_image_hashes_verified=verified_images,
        frozen_reference_generator_sha256=frozen,website_replays=4,
        guidance_verification_sha256=sha(ROOT/'research/focused_eye_guidance_replication_v2_verification.json'),
        appearance_verification_sha256=sha(ROOT/'research/focused_eye_appearance_verification_v1.json'),
        texture_verification_sha256=sha(ROOT/'research/focused_eye_texture_verification_v1.json'),
        publication_scope='Code, documentation and metric/hash metadata; restricted photos, personal uploads, arrays, environments and downloaded weights remain local.',
        limits='Targeted release verification, not a fresh inference/evaluation of every historical model or independent proof of facial reconstruction correctness.')
    write('verification.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
