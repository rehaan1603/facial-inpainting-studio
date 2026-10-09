"""One-factor Poisson comparison on the three fixed reference test identities."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reference_blending import harmonize_reference
from extended_evaluation_metrics import ROOT,sha


def main():
    base=ROOT/'outputs/masked_task_v1';source=ROOT/'outputs/masked_sdxl_resolution_v1';out=ROOT/'outputs/focused_reference_finish_v1'
    cases={(c['identity'],c['location']):c for c in json.loads((base/'cases.json').read_text())}
    rows=json.loads((source/'rows.json').read_text());out.mkdir(exist_ok=False)
    config=json.loads((source/'config.json').read_text());config.update(expected=6,
        source_config_sha256=sha(source/'config.json'),source_rows_sha256=sha(source/'rows.json'),blend='poisson',
        selected_identities=['263','6681','10149'],change='Only postprocessing: existing Poisson boundary harmonization applied to unchanged1024 raw generations',
        hypothesis='Boundary colour consistency improves without changing generation or references',
        selection_scope='User-requested three usable-reference development cases; not generalization evaluation',
        blending_source_sha256=sha(ROOT/'scripts/reference_blending.py'))
    (out/'config.json').write_text(json.dumps(config,indent=2));result=[]
    for r in rows:
        if r['identity'] not in config['selected_identities']:continue
        assert r['status']=='complete';c=cases[r['identity'],r['location']]
        raw=source/r['file'].replace('.png','_raw.png');assert sha(raw)==r['raw_sha256']
        observed=np.asarray(Image.open(c['input']).convert('RGB'));mask=np.asarray(Image.open(c['mask']))>127
        generated=np.asarray(Image.open(raw).convert('RGB'));rgb,meta=harmonize_reference(generated,observed,mask,'poisson')
        assert not meta['fallback'] and np.array_equal(rgb[~mask],observed[~mask])
        p=out/r['file'];Image.fromarray(rgb).save(p)
        Image.fromarray(generated).save(out/raw.name)
        (out/p.with_suffix('.json').name).write_text(json.dumps(dict(blending=meta,raw_source=str(raw),raw_sha256=sha(raw)),indent=2))
        result.append(dict(r,sha256=sha(p),raw_sha256=sha(out/raw.name),postprocess='poisson'))
    assert len(result)==6;(out/'rows.json').write_text(json.dumps(result,indent=2))
    print('Six unchanged raw outputs recomposited; visible pixels exact; no training')


if __name__=='__main__':main()
