"""Collect all thirteen prepared local identity sets without publishing photos."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=Path.home()/'Downloads/Celeb TEST data'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cases=json.loads((ROOT/'outputs/reference_diagnostics_v1/manifest.json').read_text(encoding='utf-8'))['cases']
    engineering_folder=ROOT/'outputs/reference_examples/identity_916'
    engineering=json.loads((engineering_folder/'manifest.json').read_text(encoding='utf-8'))
    jobs=[]
    inventory=[]
    for case in [*cases,engineering]:
        identity=case['identity_label']
        folder=DEST/f'Person {identity}'
        for photo in case['images']:
            name=('Comparison_original' if photo['role']=='target' else photo['role'])+'_'+photo['hq_id']+'.jpg'
            jobs.append((Path(photo['source_path']),folder/name,photo['source_sha256']))
        for kind in ['observed','mask']:
            if identity=='916':
                source=engineering_folder/case[kind]['file']
                expected=case[kind]['sha256']
            else:
                source=ROOT/case[kind]
                expected=case[kind+'_sha256']
            jobs.append((source,folder/'Ready to test - input and mask'/f'{kind}.png',expected))
        inventory.append({'identity_label':identity,'folder':folder.name,'original_photos':len(case['images']),
                          'scope':'engineering example' if identity=='916' else 'existing development example'})
    # Validate the complete plan before copying; never replace differing user files.
    for source,destination,expected in jobs:
        if sha(source)!=expected:
            raise ValueError(f'Source hash mismatch: {source}')
        if destination.exists() and sha(destination)!=expected:
            raise ValueError(f'Existing destination differs: {destination}')
    for source,destination,expected in jobs:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            shutil.copy2(source,destination)
        if sha(destination)!=expected:
            raise ValueError(f'Copy verification failed: {destination}')
    instructions='''CELEB TEST DATA - LOCAL PRACTICE SETS

Each Person folder contains five original dataset photographs sharing a CelebA identity annotation.
For a ready-made test:
1. Open http://127.0.0.1:8765/ and choose Use 3-4 reference photos.
2. Upload observed.png and mask.png from that person's Ready to test subfolder.
3. Add that same person's four reference_ JPG images.
4. Choose Use mask exactly and reconstruct.
5. Compare with Comparison_original. Do NOT add that original as a reference.

These are all 13 currently prepared sets (12 development identities and 1 engineering example),
not every identity in the full dataset. They are for practice, not a fresh scientific test split.
The generator was not locally trained on these sets. Keep dataset photographs local.
'''
    readme=DEST/'START HERE.txt'
    if not readme.exists():
        readme.write_text(instructions,encoding='utf-8')
    index=DEST/'SET INDEX.json'
    content=json.dumps({'sets':inventory,'verified_images':len(jobs),'files':[{'file':p.relative_to(DEST).as_posix(),'sha256':h} for _,p,h in jobs]},indent=2)+'\n'
    if index.exists() and json.loads(index.read_text(encoding='utf-8'))!=json.loads(content):
        raise ValueError('Existing index differs; photo copies are verified but index was not replaced')
    index.write_text(content,encoding='utf-8')
    print(f'Verified {len(inventory)} sets and {len(jobs)} images in {DEST}')

if __name__=='__main__':
    main()
