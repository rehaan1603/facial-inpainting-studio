"""Stage the detailed research record, excluding photographs and downloaded models."""
import hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.json','.jsonl','.csv','.md','.svg','.txt'}

def main():
    entries=[];excluded={};total=0
    candidates=list((ROOT/'outputs').rglob('*'))+list((ROOT/'data/manifests').glob('*'))
    secret=re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for path in sorted(candidates):
        if not path.is_file():continue
        relative=path.relative_to(ROOT).as_posix()
        trained=relative.startswith(('outputs/learned_refiner/','outputs/refiner_extension_v1/')) and path.name in ['best.pt','latest.pt']
        if path.suffix not in ALLOWED and not trained:
            excluded[path.suffix]=excluded.get(path.suffix,0)+1;continue
        data=path.read_bytes()
        if path.suffix in ALLOWED:
            if b'data:image/' in data or b'<image ' in data:raise RuntimeError(f'Embedded image needs separate review: {relative}')
            if secret.search(data):raise RuntimeError(f'Credential pattern needs review: {relative}')
        if len(data)>=90_000_000:raise RuntimeError(f'File is too large for this source checkpoint: {relative}')
        entries.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'kind':'locally_trained_checkpoint' if trained else 'research_record'})
        total+=len(data)
    inventory={'scope':'Detailed private research checkpoint. No source, uploaded or generated face photographs; no downloaded generative or reference-model weights; no environment or local path configuration. Metadata includes source paths and supplied dataset identity labels. Prepared and partial runs remain labelled as such.','files':len(entries),'bytes':total,'excluded_file_counts_by_extension':excluded,'inventory':entries}
    (ROOT/'research/REPOSITORY_ARTIFACT_INVENTORY.json').write_text(json.dumps(inventory,indent=2))
    spec=ROOT/'.git/artifact-paths'
    spec.write_bytes(b'\0'.join(e['path'].encode() for e in entries)+b'\0')
    subprocess.run(['git','add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul'],cwd=ROOT,check=True)
    print(json.dumps({'staged_artifacts':len(entries),'bytes':total,'excluded':excluded}))

if __name__=='__main__':main()
