"""Copy verified comparisons into fresh local folders under the three test sets."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/masked_sdxl_decoded_continuation_eval_v1'
DEST=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
IDS=['263','6681','10149']
NAMES={'control3':'Current CFG3','local':'Neutral local appearance','prior':'Prior 64-update scorer',
       'epsilon':'Epsilon continuation','rgb':'Decoded RGB continuation'}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    summary=json.loads((SOURCE/'summary.json').read_text());paired=json.loads((SOURCE/'paired_results.json').read_text())
    assert paired['summary_sha256']==sha(SOURCE/'summary.json')
    for position,ident in enumerate(IDS,1):
        person=DEST/f'Set {position:02d} - Person {ident}'
        assert person.is_dir()
        target=person/'Experimental - Decoded RGB 16 updates'
        if target.exists():raise ValueError('Refusing overwrite of local comparison set')
        target.mkdir(exist_ok=False);files=[]
        def copy(src,name,expected=None):
            if expected:assert sha(src)==expected
            dest=target/name;shutil.copy2(src,dest);assert sha(src)==sha(dest)
            files.append(dict(file=name,sha256=sha(dest)))
        for row in summary['rows']:
            if row['identity']!=ident or row['status']!='complete':continue
            arm=row['condition'];path=SOURCE/row['file']
            copy(path,NAMES[arm]+' - Finished.png',row['sha256'])
            copy(path.with_name(path.stem+'_raw.png'),NAMES[arm]+' - Raw.png',row['raw_sha256'])
        copy(SOURCE/'sheets'/f'{ident}_finished_comparison.png','Finished comparison.png')
        copy(SOURCE/'sheets'/f'{ident}_raw_comparison.png','Raw comparison.png')
        copy(SOURCE/'report.md','All three people - Whole and missing-region values.md')
        (target/'READ FIRST.txt').write_text(
            'Experimental offline comparisons for the existing three-person test set.\n'
            'Use the existing input, supplied mask and real reference folders when testing the studio.\n'
            'These candidate weights are not deployed in the website. Existing methods remain available.\n'
            'Both new arms continued the same prior checkpoint for16 updates on eight OTHER training identities.\n'
            'Decoded RGB is compared with equally trained epsilon, prior checkpoint, neutral local and current CFG3.\n'
            'Read the report for all successes, regressions and visual decisions. A whole-image SSIM is not missing-eye accuracy.\n'
            'Original portraits are evaluation-only, never use them as reconstruction references.\n'
            f'Numerical gate: {paired["numerical_gate"]}. Visual review: {paired["visual_review_passed"]}.\n',encoding='utf8')
        (target/'copy_receipt.json').write_text(json.dumps(dict(identity=ident,files=files,
            source_summary_sha256=sha(SOURCE/'summary.json')),indent=2)+'\n',encoding='utf8')
        print(str(target))


if __name__=='__main__':main()
