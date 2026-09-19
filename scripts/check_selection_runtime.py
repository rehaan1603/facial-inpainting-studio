"""Real CPU detector checks on runtime paths, including malformed reference sets."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.reference_selection.reference_analyzer import ReferenceAnalyzer
from src.reference_selection.selection_baselines import select
from src.research_integrity import ROOT,write_new

def main():
    out=ROOT/'outputs/generalization_v1/runtime_checks';out.mkdir(exist_ok=False)
    case=json.loads((ROOT/'outputs/generalization_v1/cases/manifest.json').read_text())['cases'][0]
    photo=out/'arbitrary_user_photo.png'
    with Image.open(case['references'][0]['path']) as im:im.save(photo)
    Image.new('RGB',(512,512),(120,120,120)).save(out/'no_face.png')
    (out/'broken.png').write_text('not an image',encoding='utf-8')
    analyzer=ReferenceAnalyzer()
    mixed=analyzer.analyze(case['observed'],case['mask'],[photo,out/'no_face.png',out/'broken.png',photo])
    assert [r['valid'] for r in mixed['references']]==[True,False,False,False]
    assert select(mixed)==[str(photo.resolve())]
    missing=analyzer.analyze(out/'no_face.png',case['mask'],[photo])
    assert missing['input_face_count']==0 and len(select(missing))==1
    invalid=analyzer.analyze(case['observed'],case['mask'],[out/'no_face.png'])
    assert select(invalid)==[]
    for r in mixed['references']:
        if r['valid']:assert np.isfinite(r['mask_aware_score'])
    write_new(out/'checks.json',{'status':'passed','mixed_validity':mixed,'no_input_face':missing,'no_valid_reference':invalid,
                                'scope':'Actual CPU detection/recognition; no mocked predictions; no GPU generation in this check'})
    print('Real runtime checks passed: arbitrary filename, one valid reference, blank/corrupt/duplicate rejection, no-input-face fallback, no-valid-reference result.')

if __name__=='__main__':main()
