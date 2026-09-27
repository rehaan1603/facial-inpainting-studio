"""Complete reporting for the prespecified scaffold-conditioning intervention."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.context_statistics import identity_means, contrast, holm
from report_distortion_aware import METRICS

BASE = ROOT / 'outputs/scaffold_conditioning_v1'


def main():
    cfg = json.loads((ROOT / 'research/protocols/scaffold_conditioning_v1.json').read_text())
    data = json.loads((BASE / 'evaluation.json').read_text()); rows = data['rows']
    assert len(rows)==40 and len({r['key'] for r in rows})==40
    arms = {a: [r for r in rows if r['mode']==a] for a in cfg['arms']}
    summaries = {}
    for arm, group in arms.items():
        entry = {'rows': len(group), 'scored': sum(r['evaluation']['status']=='complete' for r in group)}
        for metric in METRICS:
            values, excluded = identity_means(group, metric, cfg['seeds'])
            entry[metric] = {'mean': float(np.mean(list(values.values()))) if values else None,
                             'identity_count': len(values), 'excluded': excluded}
        summaries[arm] = entry
    candidate = cfg['primary_candidate']
    primary = {f'{candidate}_minus_{a}/{m}': contrast(arms[candidate], arms[a], m, cfg['seeds'])
               for a in cfg['primary_controls'] for m in cfg['primary_metrics']}
    holm(primary)
    def mean(arm, seed, metric):
        values = [r['evaluation'].get('metrics', {}).get(metric) for r in arms[arm] if r['seed']==seed]
        return float(np.mean(values)) if len(values)==4 and all(v is not None for v in values) else None
    checks = []
    for seed in cfg['seeds']:
        for control in cfg['primary_controls']:
            pairs = {m: [mean(a, seed, m) for a in [candidate, control]] for m in ['facenet_cosine','hole_mae','facenet_gallery_cosine','lpips']}
            if any(v is None for p in pairs.values() for v in p): passed = {'coverage': False}
            elif control == 'masked_lq_no_reference_context':
                passed = {'identity_reference_benefit': pairs['facenet_cosine'][0]>pairs['facenet_cosine'][1],
                          'hole_reference_noninferiority': pairs['hole_mae'][0]<=pairs['hole_mae'][1]}
            else:
                passed = {'identity_gain_0.005': pairs['facenet_cosine'][0]>=pairs['facenet_cosine'][1]+.005,
                          'hole_error_reduction_1pct': pairs['hole_mae'][0]<=pairs['hole_mae'][1]*.99,
                          'gallery_noninferiority_0.002': pairs['facenet_gallery_cosine'][0]>=pairs['facenet_gallery_cosine'][1]-.002,
                          'lpips_noninferiority_1pct': pairs['lpips'][0]<=pairs['lpips'][1]*1.01}
            checks.append({'seed': seed, 'control': control, 'metrics_candidate_control': pairs, 'checks': passed, 'pass': all(passed.values())})
    coverage = all(r['evaluation']['status']=='complete' and all(r['evaluation']['metrics']['detections'][n]['status']=='ok'
        and r['evaluation']['metrics'][n+'_gallery_valid_count']==3 for n in ['facenet','arcface_conditioning']) for r in rows)
    preservation = all(r['evaluation'].get('metrics',{}).get('known_pixels_unchanged') is True for r in rows)
    pass_gate = coverage and preservation and all(g['pass'] for g in checks)
    new = [r for r in rows if r['mode'] not in ['scaffold','reference_context']]
    summary = {'date': '2026-09-25', 'identities': 4, 'previously_observed': True, 'seeds': cfg['seeds'],
               'new_generation_count': len(new), 'new_generation_complete': sum(r['status']=='complete' for r in new),
               'reused_controls': 16, 'equivalence_calls': 2, 'scored': sum(r['evaluation']['status']=='complete' for r in rows),
               'summaries': summaries, 'primary_contrasts': primary,
               'gate': {'passed': bool(pass_gate), 'coverage': coverage, 'preservation': preservation, 'checks': checks},
               'equivalence': json.loads((BASE / 'gain_zero_equivalence.json').read_text()),
               'protocol_sha256': sha(ROOT / 'research/protocols/scaffold_conditioning_v1.json'),
               'evaluation_sha256': sha(BASE / 'evaluation.json'), 'reporter_sha256': sha(__file__),
               'statistics_sha256': sha(ROOT / 'src/preservation/context_statistics.py'), 'final_test_used': False}
    write_new(ROOT / 'research/scaffold_conditioning_results_v1.json', summary)
    safe = []
    for r in rows:
        if r['status']=='complete': assert sha(r['output'])==r['output_sha256']
        out = {k:v for k,v in r.items() if k not in ['output','evaluation','traceback']}
        score = r['evaluation']; out['evaluation'] = {k:v for k,v in score.items() if k not in ['metrics','target_detections','gallery_detections']}
        if 'metrics' in score:
            out['evaluation']['metrics'] = {k:v for k,v in score['metrics'].items() if k!='detections'}
            out['evaluation']['detections'] = {n:{k:v for k,v in d.items() if k in ['face_count','status']} for n,d in score['metrics']['detections'].items()}
        safe.append(out)
    write_new(ROOT / 'research/scaffold_conditioning_evidence_v1.json', {'signature': json.loads((BASE / 'signature.json').read_text()),
        'evaluation_signature': json.loads((BASE / 'evaluation_signature.json').read_text()), 'rows': safe})
    fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
    lines = ['# Masked scaffold conditioning — 25 September 2026', '',
             '**Progression gate: '+('passed numerically; independent confirmation and visual review required' if pass_gate else 'failed')+'.**', '',
             'The preceding cascade improved identity but increased pixel error. This separately frozen follow-up suppresses low-quality scaffold latent cells overlapping missing pixels. Real references and the late original-context correction remain fixed. The global control matches the mean conditioning multiplier, not exact latent norm. A reference-disabled arm tests reference benefit. No clean truth enters inference.', '',
             f"{summary['new_generation_complete']}/24 new generations, two equivalence calls, 16 reused controls and {summary['scored']}/40 scored rows. Complete detector/gallery coverage: {coverage}. Exact outside-mask preservation: {preservation}.", '',
             '| Arm | FaceNet ↑ | Gallery ↑ | Hole MAE ↓ | LPIPS ↓ |', '|---|---:|---:|---:|---:|']
    for a,s in summaries.items(): lines.append('| '+a+' | '+' | '.join(fmt(s[m]['mean']) for m in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips'])+' |')
    lines += ['', '| Primary contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |', '|---|---:|---:|---|---:|']
    for k,c in primary.items(): lines.append(f"| {k} | {c['identities']} | {fmt(c['mean_delta'])} | {c['ci95']} | {fmt(c['p_holm'])} |")
    lines += ['', 'Eight tests specified before generation. Seeds are averaged within identity. Four identities cannot support a two-sided exact p below 0.125; the intervals are unstable. The same already observed cases informed the follow-up hypothesis, so this is development exploration, not independent confirmation. All scores, coverage and per-seed gates remain in JSON.', '',
              'Latent masking does not guarantee semantic locality because the encoder has a broad receptive field. Missing input-latent values were set to zero without training the model on spatially dropped conditioning; out-of-distribution behavior is a plausible failure mode. No radius/strength search or clean-target tuning was performed in this protocol.', '',
              'Mask-aware conditioning, personal reference restoration and reliability-aware guidance have prior art. No first-in-literature, publication-ready superiority or arbitrary-client accuracy claim follows. Final identities are untouched. See `NOVELTY_ESTABLISHMENT_20260925.md` for visual findings, related-work boundaries and the resulting decision.']
    (ROOT / 'research/SCAFFOLD_CONDITIONING_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    evaluation = {c['case_id']:c for c in json.loads((ROOT/'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    for cid in cfg['cases']:
        for seed in cfg['seeds']:
            paths = [('clean scoring target',evaluation[cid]['target']['path'])]+[(a,next(r.get('output') for r in arms[a] if r['case_id']==cid and r['seed']==seed)) for a in cfg['arms']]
            sheet = Image.new('RGB',(768,560),'white'); draw=ImageDraw.Draw(sheet)
            for i,(label,path) in enumerate(paths):
                x,y=i%3*256,i//3*280;draw.text((x+4,y+4),label,fill='black')
                if path:
                    with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((256,256)),(x,y+24))
                else:draw.text((x+4,y+70),'FAILED',fill='red')
            dest=BASE/'review'/f'{cid}_s{seed}.jpg';dest.parent.mkdir(exist_ok=True);sheet.save(dest)
    print(json.dumps({'gate_passed':bool(pass_gate),'scored':summary['scored'],'means':{a:{m:s[m]['mean'] for m in ['facenet_cosine','hole_mae','lpips']} for a,s in summaries.items()}},indent=2))


if __name__=='__main__':main()
