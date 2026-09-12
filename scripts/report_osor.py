"""Report the entire OSOR comparison; never promote partial chunks to final results."""
import csv,hashlib,json
from pathlib import Path
from report_final import means,paired,table
ROOT=Path(__file__).resolve().parents[1]
def main():
    path=ROOT/'outputs/osor_evaluation_v1/metrics.csv';rows=list(csv.DictReader(path.open()));assert len(rows)==1728
    fixed_path=ROOT/'outputs/area_matched_v3_margin12_evaluation/metrics.csv';fixed=list(csv.DictReader(fixed_path.open()))
    summaries=[{'method':'osor_direct',**means(rows)}];contrasts={}
    key=lambda r:(r['identity'],r['location'],r['missing_pixels'],r['condition'])
    for backbone,radius,feather in [('lama','8','0'),('resshift','8','0'),('resshift','12','4')]:
        selected=[r for r in fixed if (r['backbone'],r['radius'],r['feather'])==(backbone,radius,feather)]
        name=f'{backbone}_dilate{radius}_feather{feather}';summaries.append({'method':name,**means(selected)});contrasts[f'osor-minus-{name}']=paired(rows,selected,key,lambda r:r['identity'])
    strata=[]
    for field in ['condition','location','missing_pixels']:
        for value in sorted({r[field] for r in rows}):
            subset=[r for r in rows if r[field]==value];strata.append({'factor':field,'value':value,'rows':len(subset),**means(subset)})
    report={'summary':summaries,'contrasts':contrasts,'strata':strata,'raw_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,fixed_path]},'scope':'256-pixel validation comparison, author OSOR infer() with sequential offload and fp16 base variants cast to bf16. Not an original-paper benchmark reproduction; see OSOR_COMPARISON_PROTOCOL.md.'}
    (ROOT/'research/osor_results.json').write_text(json.dumps(report,indent=2))
    lines=['# OSOR development comparison','',report['scope'],'',table(summaries,['method']),'','## Paired contrasts','','First minus second; lower is better. Paired identity-label bootstrap, 2,000 resamples, conditional on fixed checkpoints and sampling seeds.','']
    for name,metrics in contrasts.items():
        lines.append(f'- {name}: '+ '; '.join(f"{metric} {d['mean']:+.6f} [{d['bootstrap_95_interval'][0]:+.6f}, {d['bootstrap_95_interval'][1]:+.6f}]" for metric,d in metrics.items()))
    lines+=['','## All OSOR strata','',table(strata,['factor','value','rows']),'','No target RGB or true mask enters OSOR inference. Direct VAE decoding can change visible pixels. Method-specific pretraining, domain and precision differences limit causal interpretation. No fresh-test or state-of-the-art claim follows from this validation comparison.']
    (ROOT/'research/OSOR_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('Complete OSOR comparison and report written',flush=True)
if __name__=='__main__':main()
