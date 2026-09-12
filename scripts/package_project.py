"""Create a local code/results/checkpoint archive; exclude source face images and backbone weights."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    assert (ROOT/'research/completion_checks.json').exists(),'Verify delivery first'
    files=set()
    for folder in ['scripts','research','configs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.ps1','.md','.json','.txt'] and p.name!='local.json':files.add(p)
    for name in ['README.md','requirements-lock.txt','.gitignore','Start Studio.cmd']:files.add(ROOT/name)
    for p in (ROOT/'webapp').rglob('*'):
        if p.is_file() and p.suffix in ['.py','.ps1','.md','.json','.html','.css','.js','.svg']:files.add(p)
    for p in (ROOT/'data/manifests').glob('*_reviewed_v2.csv'):files.add(p)
    for folder in ['refiner_evaluation','object_test','area_matched_v3_margin12_evaluation','extension_evaluation_v1']:
        for name in ['metrics.csv','strata.csv','protocol.json','signature.json','tradeoff.svg']:
            p=ROOT/'outputs'/folder/name
            if p.exists():files.add(p)
    files.update((ROOT/'outputs/refiner_evaluation').glob('*_selection.json'))
    files.update((ROOT/'outputs/extension_evaluation_v1').glob('*_selection.json'))
    for name in ['best.pt','training.json']:files.update((ROOT/'outputs/learned_refiner').glob('*/'+name))
    for name in ['best.pt','training.json']:files.update((ROOT/'outputs/refiner_extension_v1').glob('*/'+name))
    files={p for p in files if not p.name.startswith('archive_receipt')}
    inventory={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    out=ROOT/'facial_inpainting_local_delivery_v2.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in inventory:z.write(ROOT/name,name)
        z.writestr('ARCHIVE_INVENTORY.json',json.dumps(inventory,indent=2))
        z.writestr('ARCHIVE_SCOPE.txt','Local private delivery v2. Includes the browser app, project code, reports, completed experiment metrics, and initial/extended small trained refiner checkpoints. Excludes source faces, uploaded faces and references, object photographs, generative and reference-model weights, cache, virtual environments and local path config. These remain on the original laptop. The reference-photo feature uses existing pretrained methods; no novelty or publication claim. This is not a self-contained or cleared public release. See README, webapp/README and research/PROJECT_STATUS.md.\n')
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    (ROOT/'research/archive_receipt_v2.json').write_text(json.dumps({'path':out.name,'files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scope':'Local code, results and small refiner-checkpoint backup; no public redistribution performed.'},indent=2));print(out)
if __name__=='__main__':main()
