"""Copy three existing dataset test packs to Downloads; preserve source files."""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
IDS=['263','6681','10149']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cases={(c['identity'],c['location']):c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text())}
    refs=json.loads((ROOT/'outputs/masked_sdxl_v1/config.json').read_text())['references']
    DEST.mkdir(exist_ok=False);records=[]
    for i,ident in enumerate(IDS,1):
        folder=DEST/f'Set {i:02d} - Person {ident}';folder.mkdir()
        reference_folder=folder/'Reference photos';reference_folder.mkdir()
        original=folder/'Original - comparison only.png';source=Path(cases[ident,'eyes']['target'])
        assert sha(source)==cases[ident,'eyes']['target_sha256'];shutil.copy2(source,original)
        files=[dict(file=original.name,source=str(source),sha256=sha(original),role='evaluation_only')]
        for n,ref in enumerate(refs[ident+'/clean'],1):
            assert sha(ref['path'])==ref['sha256'];p=reference_folder/f'Reference {n:02d}.png';shutil.copy2(ref['path'],p)
            assert sha(p)!=sha(original);files.append(dict(file=str(p.relative_to(folder)),source=ref['path'],sha256=sha(p),role='reference'))
        for location in ['eyes','mouth']:
            test=folder/(location.capitalize()+' test');test.mkdir()
            c=cases[ident,location]
            for key,name in [('input','Damaged input.png'),('mask','Mask - white means reconstruct.png')]:
                assert sha(c[key])==c[key+'_sha256'];p=test/name;shutil.copy2(c[key],p)
                files.append(dict(file=str(p.relative_to(folder)),source=c[key],sha256=sha(p),role=key))
        (folder/'START HERE.txt').write_text(
            f'Set {i:02d}, dataset person label {ident}\n\n'
            'Open http://127.0.0.1:8765/ and choose reference-guided inpainting.\n'
            'Start with Eyes test/Damaged input.png and its matching white-hole mask.\n'
            'Add all FOUR photos from Reference photos. Use the supplied/painted mask without expansion.\n'
            'The mouth test is an additional test using the same four references.\n'
            'Original - comparison only.png is the evaluation target. Do not upload it as an additional reference.\n'
            'Use the matching mask from the same test folder; a white pixel marks the missing region.\n'
            'These are real dataset references, not generated angle variants. Outputs can differ in gaze/expression.\n'
            'Keep this dataset-photo pack local; repository restoration links remain separate.\n',encoding='utf-8')
        records.append(dict(set=i,identity=ident,files=files))
    (DEST/'manifest.json').write_text(json.dumps(dict(source_dataset='Existing CelebAMask-HQ development cases',sets=records),indent=2))
    (DEST/'START HERE.txt').write_text('Three reference-reconstruction test sets\n\nEach person has four reference photos, an erased-eye test, an erased-mouth test and an original for comparison only.\nStart with the Eyes test in each set. All source files were copied; originals are unchanged.\nResults from the focused improvement experiment will be added in separate labelled folders.\n',encoding='utf-8')
    print(DEST)


if __name__=='__main__':main()
