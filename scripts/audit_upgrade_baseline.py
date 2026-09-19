"""Record the existing architecture/artifact boundary without changing old work."""
import hashlib,json,subprocess
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    counts={}
    for name in ['scripts','configs','data','outputs','research','tests','webapp']:
        paths=[p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        counts[name]={'files':len(paths),'bytes':sum(p.stat().st_size for p in paths),
                      'extensions':dict(Counter(p.suffix for p in paths))}
    protected={}
    for folder in ['scripts','configs','research']:
        for p in (ROOT/folder).glob('*'):
            if p.is_file() and p.name not in ['local.json','GENERALIZATION_AND_RESTORATION_UPGRADE_PLAN.md','audit_upgrade_baseline.py']:
                protected[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    checkpoints=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (ROOT/'outputs').rglob('*.pt')]
    result={'date':'2026-09-19','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'inventory':counts,'existing_source_hashes':protected,'project_checkpoints':checkpoints,
            'scope':'Pre-upgrade snapshot. Does not contain photos or secrets. New work must not replace signed outputs.'}
    destination=ROOT/'research/upgrade_baseline_audit_v1.json'
    if destination.exists():
        raise FileExistsError('Audit already frozen')
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'inventory':counts,'checkpoints':len(checkpoints)},indent=2))
if __name__=='__main__':main()
