"""Verify the completed local delivery without repeating model evaluation."""
import csv,hashlib,json,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks={'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    for name,count in [('refiner_evaluation',13824),('object_test',5632),('area_matched_v3_margin12_evaluation',5184)]:
        p=ROOT/f'outputs/{name}/metrics.csv';rows=list(csv.DictReader(p.open()));assert len(rows)==count
        checks[name]={'rows':count,'sha256':digest(p)}
    protocol=json.loads((ROOT/'outputs/object_test/protocol.json').read_text())
    qa=json.loads((ROOT/'research/object_protocol_verification.json').read_text())
    assert qa['cases']==128 and qa['protocol_sha256']==digest(ROOT/'outputs/object_test/protocol.json')
    for name,sha in protocol['signature'].items():assert digest(ROOT/name)==sha,name
    for name,sha in json.loads((ROOT/'outputs/refiner_evaluation/signature.json').read_text()).items():assert digest(ROOT/name)==sha,name
    review=json.loads((ROOT/'research/near_duplicate_review_v2.json').read_text());assert review['pairs_reviewed']==496 and review['pending_review']==0
    local=json.loads((ROOT/'configs/local.json').read_text());cache=Path(local['cache'])
    expected={'big-lama.pt':'344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9','resshift_inpainting_face_s4.pth':'bf3a1691d4ad46958f7abe847f9afce632400112d9e9cd8fb5253c33da4cec02','celeba256_vq_f4_dim3_face.pth':'770a2d8888ad51d84ceeda5f9b64ee19b2688bc1e88d1139e3a1eea0a3b9501a'}
    for name,sha in expected.items():assert digest(cache/'models'/name)==sha,name
    checks['backbone_weight_hashes']=expected
    for p in (ROOT/'outputs/learned_refiner').glob('*/training.json'):
        data=json.loads(p.read_text());assert data['steps']==1500 and p.with_name('best.pt').exists()
        for name,sha in data['signature'].items():assert digest(ROOT/name)==sha,name
    assert len(list((ROOT/'outputs/learned_refiner').glob('*/training.json')))==6
    for name in ['FINAL_RESULTS.md','MANUSCRIPT.md','final_results.json','inference_efficiency.json','DATA_AND_LICENSES.md','PROJECT_STATUS.md']:
        p=ROOT/'research'/name;assert p.is_file() and p.stat().st_size>100
    result=json.loads((ROOT/'research/final_results.json').read_text())
    for name,sha in result['raw_sha256'].items():assert digest(ROOT/name)==sha
    completed=subprocess.run([sys.executable,'-m','unittest','discover','-s','scripts','-p','test_*.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'research/test_results.txt').write_text(completed.stdout+completed.stderr);assert completed.returncode==0,completed.stderr
    count=re.search(r'Ran (\d+) tests?',completed.stderr)
    assert count,completed.stderr
    skipped=re.search(r'OK \(skipped=(\d+)\)',completed.stderr)
    skipped_count=int(skipped.group(1)) if skipped else 0
    checks['tests']=f'{int(count.group(1))-skipped_count} tests passed; {skipped_count} skipped; full output in research/test_results.txt'
    checks['scope']='Scoped local software and experiments verified. Scientific novelty, comprehensive comparisons, and publication readiness remain unresolved in PROJECT_STATUS.md.'
    (ROOT/'research/completion_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
