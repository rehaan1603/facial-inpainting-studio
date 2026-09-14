"""Paired identity inference with explicit coverage and familywise correction."""
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEASURES = ['facenet_cosine','arcface_conditioning_cosine','niqe','brisque','ssim_rgb',
            'psnr_rgb','hole_psnr','lpips','hole_mae','visible_mae']

def paired(values, seed=20260914):
    delta = np.asarray(values,dtype=float)
    if not len(delta):
        return {'n':0,'mean':None,'ci95':None,'p_exact':None}
    if not np.isfinite(delta).all() or len(delta)>12:
        raise ValueError('Expected at most twelve finite identity differences')
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    signs = np.asarray(list(itertools.product([-1,1],repeat=len(delta))))
    null = np.abs((signs*delta).mean(1))
    p = float(np.mean(null>=abs(delta.mean())-1e-12))
    return {'n':len(delta),'mean':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'p_exact':p}

def holm(pvalues):
    order = sorted(range(len(pvalues)),key=lambda i:pvalues[i])
    adjusted = [None]*len(pvalues)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running,min(1.,(len(order)-rank)*pvalues[index]))
        adjusted[index] = running
    return adjusted

def key(count,scale,strength,compositor):
    return f'refs{count}_scale{float(scale)}_strength{float(strength)}_{compositor}'

def main():
    files = [ROOT/f'research/extended_evaluation_{suite}_v1.json' for suite in ['original','count']]
    datasets = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    rows = [row for dataset in datasets for row in dataset['rows']]
    assert len(rows)==144
    groups = {}
    for row in rows:
        group = key(row['reference_count'],row['scale'],row['strength'],row['compositor'])
        identity = row['identity']
        if identity in groups.setdefault(group,{}):
            raise ValueError('Duplicate identity within configuration')
        groups[group][identity] = row
    summaries = {}
    for name, group in groups.items():
        assert len(group)==12
        summaries[name] = {}
        for metric in MEASURES:
            values = [r[metric] for r in group.values() if r[metric] is not None]
            summaries[name][metric] = {'valid':len(values),'total':len(group),
                                       'mean':float(np.mean(values)) if values else None}
    specifications = []
    for strength in [1.,.99]:
        for compositor in ['hard','poisson']:
            specifications.append(('reference on - off',key(4,.8,strength,compositor),key(4,0,strength,compositor)))
    for scale in [0.,.8]:
        for compositor in ['hard','poisson']:
            specifications.append(('strength .99 - 1.0',key(4,scale,.99,compositor),key(4,scale,1.,compositor)))
        for strength in [1.,.99]:
            specifications.append(('Poisson - hard',key(4,scale,strength,'poisson'),key(4,scale,strength,'hard')))
    for a,b in [(2,1),(4,1),(4,2)]:
        for compositor in ['hard','poisson']:
            specifications.append((f'{a} references - {b}',key(a,.8,.99,compositor),key(b,.8,.99,compositor)))
    contrasts, tests = [], []
    for label,a,b in specifications:
        assert set(groups[a])==set(groups[b])
        record = {'label':label,'a':a,'b':b,'metrics':{}}
        for metric in MEASURES:
            ids = sorted(i for i in groups[a] if groups[a][i][metric] is not None and groups[b][i][metric] is not None)
            result = paired([groups[a][i][metric]-groups[b][i][metric] for i in ids])
            result['identities'] = ids
            record['metrics'][metric] = result
            if result['p_exact'] is not None:
                tests.append(result)
        contrasts.append(record)
    for result, corrected in zip(tests,holm([t['p_exact'] for t in tests])):
        result['p_holm_all_tests'] = corrected
    import hashlib
    report = {'scope':'exploratory_development_only','scored_rows':144,'identities':12,
              'generated_candidates':72,'holm_family_tests':len(tests),
              'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'means':summaries,'paired_contrasts':contrasts}
    (ROOT/'research/extended_ablation_analysis_v1.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = ['# Extended evaluation and ablations — development results','',
             'All 144 outputs (72 generated candidates) are scored across twelve identity labels. Rows and compositors are correlated, not independent subjects. ArcFace uses the conditioning checkpoint; FaceNet is the separate evaluator.','',
             '| Configuration | FaceNet | ArcFace* | NIQE ↓ | BRISQUE ↓ | SSIM ↑ | Hole PSNR ↑ | LPIPS ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    display = ['facenet_cosine','arcface_conditioning_cosine','niqe','brisque','ssim_rgb','hole_psnr','lpips']
    for name, measures in summaries.items():
        cells = []
        for m in display:
            v = measures[m]
            cells.append('unscorable' if v['mean'] is None else f"{v['mean']:.4f} ({v['valid']}/12)")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '*ArcFace is a conditioning-encoder diagnostic, not independent identity verification. Cosine similarities are not recognition percentages. NIQE and BRISQUE are full-image statistical quality proxies; neither establishes facial correctness.','',
              '## Paired FaceNet ablations','',
              '| A minus B | Paired n | Difference | Descriptive 95% interval | Exact p | Holm p |',
              '|---|---:|---:|---|---:|---:|']
    for c in contrasts:
        v=c['metrics']['facenet_cosine']
        if v['n']:
            lines.append(f"| {c['a']} minus {c['b']} | {v['n']} | {v['mean']:+.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['p_exact']:.4f} | {v['p_holm_all_tests']:.4f} |")
    lines += ['',f"Holm correction covers all {len(tests)} metric/contrast tests; bootstrap intervals are descriptive and unadjusted. Full per-identity data, coverage, quality-control scores and every metric contrast are retained in JSON. Detection failures are not imputed.", '',
              '## Limits','',
              'Fixed nested reference subsets confound photo count with the evidence in those photos; this is not optimal selection or learned fusion. One seed, small eye masks, twelve development identities, and unknown pretraining overlap remain limitations. The strength comparison also changes effective denoising steps. No FID, low-FAR recognition, final-test or novelty claim is supported.','',
              '## Reproduction','',
              'Run prepare_extended_evaluation.py with the project Python; run run_reference_count_ablation.py with reference_env_v2 Python; run evaluate_extended_diagnostics.py --suite original and --suite count with evaluation_env_v1 Python, then analyze_extended_ablations.py. Original local images/checkpoints and frozen manifests are required. Public hosting and app defaults are unchanged.']
    (ROOT/'research/EXTENDED_ABLATION_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Analyzed {len(rows)} rows and {len(tests)} paired metric tests',flush=True)

if __name__=='__main__':
    main()
