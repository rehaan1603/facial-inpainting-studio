"""Training-only MTCNN boxes; target pixels never become inference inputs."""
import json
import numpy as np
import torch
from PIL import Image
from extended_evaluation_metrics import ROOT,Metrics,sha
from identity_crop_v3 import crop_tensor,standardize

def main():
    out=ROOT/'outputs/identity_crop_v3';out.mkdir(exist_ok=False)
    base=ROOT/'outputs/reference_intervention_pilot_v1'
    m=Metrics();rows=[]
    for r in json.loads((base/'manifest.json').read_text())['cases']:
        if r['role']!='train':continue
        p=base/'images'/r['identity']/'target.png';im=Image.open(p).convert('RGB')
        boxes,probs=m.mtcnn.detect(im);n=0 if boxes is None else len(boxes)
        row=dict(identity=r['identity'],source_sha256=sha(p),face_count=n,status='ok' if n==1 else 'identity_loss_unavailable')
        if n==1:
            b=boxes[0];box=[int(max(b[0],0)),int(max(b[1],0)),int(min(b[2],im.width)),int(min(b[3],im.height))]
            raw=m.tensor(np.asarray(im).copy());crop=standardize(crop_tensor(raw,box));author=m.mtcnn.extract(im,boxes,save_path=None)
            with torch.no_grad():cos=float((m.facenet(crop)*m.facenet(author)).sum())
            row.update(box=box,probability=float(probs[0]),pixel_standardization_mae=float((crop-author).abs().mean()),embedding_cosine_vs_evaluator_crop=cos)
        rows.append(row);print(row['identity'],row['status'],flush=True)
    (out/'crops.json').write_text(json.dumps(dict(records=rows,script_sha256=sha(__file__),
        definition='Ground-truth training face box, zero margin; same bounds,160size,standardization; differentiable antialiased bilinear resize avoids PIL uint8 rounding; evaluation independently detects each output.',
        manifest_sha256=sha(base/'manifest.json')),indent=2))

if __name__=='__main__':main()
