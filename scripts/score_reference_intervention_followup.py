"""Score the loss-scale follow-up and degraded-input controls offline."""
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from extended_evaluation_metrics import ROOT, Metrics, read_rgb, sha


def main():
    base = ROOT/'outputs/reference_intervention_pilot_v1'
    out = base/'scaled_followup'
    rows = json.loads((out/'rows.json').read_text())
    assert len(rows)==8
    if (out/'scores.json').exists():
        raise ValueError('Scores already exist')
    metrics = Metrics()
    records = []
    ids = sorted({r['identity'] for r in rows})
    for identity in ids:
        target = read_rgb(base/'images'/identity/'target.png')
        target_features, target_detections = metrics.features(target)
        inputs = [(r,out/r['file']) for r in rows if r['identity']==identity]
        inputs += [(dict(identity=identity,arm='damaged_input',condition='clean'),base/'images'/identity/'input.png')]
        for row,path in inputs:
            rgb = read_rgb(path,row.get('sha256'))
            features,detections = metrics.features(rgb)
            values = metrics.quality(rgb)
            for name,vector in features.items():
                values[name+'_cosine'] = float(np.dot(vector,target_features[name])) if vector is not None and target_features[name] is not None else None
            a,b = rgb.astype(float)/255,target.astype(float)/255
            values['mae'] = float(np.abs(a-b).mean())
            values['psnr'] = float(-10*np.log10(np.square(a-b).mean()))
            values['ssim'] = float(metrics.ssim(a,b,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11))
            with torch.no_grad():
                values['lpips'] = metrics.lpips(metrics.tensor(rgb)*2-1,metrics.tensor(target)*2-1).item()
            records.append(dict(row,metrics=values,target_detections=target_detections,output_detections=detections))
            print('Follow-up score',len(records),'/12',flush=True)
        panels = [('Target',base/'images'/identity/'target.png'),('Input',base/'images'/identity/'input.png'),
            ('Baseline corrupt',base/'evaluation'/(identity+'_baseline_corrupt.png')),
            ('Reconstruction corrupt',base/'evaluation'/(identity+'_reconstruction_corrupt.png')),
            ('Scaled clean',out/(identity+'_clean.png')),('Scaled corrupt',out/(identity+'_corrupt.png'))]
        sheet = Image.new('RGB',(768,568),'white')
        draw = ImageDraw.Draw(sheet)
        for i,(label,path) in enumerate(panels):
            x,y = i%3*256,i//3*284
            with Image.open(path) as image:
                sheet.paste(image.resize((256,256)),(x,y+28))
            draw.text((x+4,y+5),label,fill='black')
        sheet.save(out/(identity+'_sheet.jpg'))
    summary = {}
    for group in [('scaled_intervention','clean'),('scaled_intervention','corrupt'),('damaged_input','clean')]:
        entries = [r for r in records if (r['arm'],r['condition'])==group]
        summary['/'.join(group)] = {}
        for name in ['facenet_cosine','arcface_conditioning_cosine','lpips','mae','ssim','psnr']:
            values = [r['metrics'][name] for r in entries if r['metrics'][name] is not None]
            summary['/'.join(group)][name] = dict(mean=float(np.mean(values)) if values else None,n=len(values))
    result = dict(rows=records,summary=summary,script_sha256=sha(__file__),protocol_sha256=sha(out/'protocol.json'),
        status='Repeated development cohort, not independent validation', final_test_used=False)
    (out/'scores.json').write_text(json.dumps(result,indent=2))
    (ROOT/'research/reference_intervention_scaled_scores_v1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
