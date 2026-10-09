"""Verify saved provenance and assemble local scientific comparison sheets."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from reference_intervention_core import ROOT,sha

OUT=ROOT/'outputs/spatial_reference_rejection_v1'


def main():
    result=json.loads((OUT/'scores.json').read_text())
    complete=json.loads((OUT/'completion.json').read_text())
    config=json.loads((OUT/'config.json').read_text())
    replay=json.loads((OUT/'replays.json').read_text())
    rows=result['rows']
    assert len(rows)==72 and len(replay)==8 and all(r['replay_identical'] for r in replay)
    assert result['counts']['scored_rows']==72
    assert sha(OUT/'evaluation/rows.json')==complete['rows_sha256']
    assert sha(ROOT/'scripts/spatial_reference_rejection_v1.py')==config['mechanism_sha256']
    assert sha(ROOT/'scripts/evaluate_spatial_reference_rejection_v1.py')==config['script_sha256']
    prior=json.loads((ROOT/'outputs/identity_timestep_v5/timestep_scaled/scores.json').read_text())['rows']
    for r in rows+replay:
        assert r['status']=='complete' and sha(OUT/'evaluation'/r['file'])==r['sha256']
        if r['arm']=='baseline':
            old=next(p for p in prior if all(p[k]==r[k] for k in ['identity','arm','seed','condition']))
            assert old['sha256']==r['sha256'] and old['metrics']==r['metrics']
        elif r['arm']!='neutral':
            assert r['attention_calls']==800
            assert r['token_counts'] and all(total==count*4 for total,count in r['token_counts'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    images=[]
    for identity in config['identities']:
        for condition in config['conditions']:
            sheet=Image.new('RGB',(1000,716),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,5),f'Identity {identity} | {condition} | 17, 29, 43 inference seeds',font=font,fill='black')
            for index,label in enumerate(['Ground truth','Degraded input','Baseline','Oracle region','Random corner']):draw.text((index*200+5,32),label,font=font,fill='black')
            base=ROOT/'outputs/reference_intervention_pilot_v1/images'/identity
            for j,seed in enumerate(config['seeds']):
                files=[base/'target.png',base/'input.png']+[OUT/'evaluation'/next(r['file'] for r in rows if r['identity']==identity and r['condition']==condition and r['seed']==seed and r['arm']==a) for a in ['baseline','oracle_region','random_corner']]
                y=62+j*217
                for i,file in enumerate(files):
                    with Image.open(file) as image:sheet.paste(image.convert('RGB').resize((192,192),Image.Resampling.LANCZOS),(i*200+4,y))
                draw.text((8,y+194),f'Seed {seed}',font=font,fill='black')
            path=OUT/f'comparison_{identity}_{condition}.png';sheet.save(path)
            images.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    receipt=dict(status='numeric_and_provenance_checks_pass',rows=72,new_outputs=48,baseline_controls_verified=24,
        exact_neutral_replays=8,mechanism_tests=4,paired_statistics_tests=2,active_attention_calls_per_output=800,
        scope='No visual judgment implied by this receipt. Review the eight sheets separately.',
        scores_sha256=sha(OUT/'scores.json'),config_sha256=sha(OUT/'config.json'),sheets=images)
    (ROOT/'research/spatial_reference_rejection_v1_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
