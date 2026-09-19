"""Verify pre-upgrade source/checkpoint hashes, allowing only declared live changes."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new

def main():
    audit=json.loads((ROOT/'research/upgrade_baseline_audit_v1.json').read_text())
    allowed={'scripts/test_webapp.py'}
    changed=[];missing=[];verified=0
    for relative,digest in audit['existing_source_hashes'].items():
        path=ROOT/relative
        if not path.exists():missing.append(relative)
        elif sha(path)!=digest:changed.append(relative)
        else:verified+=1
    checkpoint_results=[]
    for record in audit['project_checkpoints']:
        path=ROOT/record['path'];checkpoint_results.append({'path':record['path'],'unchanged':sha(path)==record['sha256']})
    result={'verified_existing_sources':verified,'changed_sources':changed,'missing_sources':missing,
            'undeclared_changes':sorted(set(changed)-allowed),'checkpoints':checkpoint_results}
    print(json.dumps(result,indent=2))
    if missing or result['undeclared_changes'] or not all(c['unchanged'] for c in checkpoint_results):raise RuntimeError('Preservation failed')
    write_new(ROOT/'outputs/generalization_v1/preservation_verification.json',result)

if __name__=='__main__':main()
