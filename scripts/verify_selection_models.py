"""Recheck locally cached generator weights against the existing download ledger."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,write_new

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    rows=[]
    for r in json.loads((ROOT/'research/osor_downloads.json').read_text()):
        if r['repository']=='diffusers/stable-diffusion-xl-1.0-inpainting-0.1':
            actual=digest(r['path']);rows.append({'component':'sdxl','file':r['filename'],'revision':r['revision'],'sha256':actual,'matches_ledger':actual==r['sha256']})
    adapter=json.loads((ROOT/'research/reference_adapter_provenance.json').read_text());actual=digest(adapter['local_path'])
    rows.append({'component':'faceid_adapter','file':adapter['file'],'revision':adapter['revision'],'sha256':actual,'matches_ledger':actual==adapter['sha256']})
    faces=json.loads((ROOT/'research/reference_faces_provenance.json').read_text())
    for name in ['det_10g.onnx','w600k_r50.onnx']:
        actual=digest(cache/'reference_models/insightface/models/buffalo_l'/name)
        rows.append({'component':'face_analysis','file':name,'sha256':actual,'matches_ledger':actual==faces['files'][name]})
    write_new(ROOT/'outputs/generalization_v1/model_verification.json',{'files':rows,'scope':'Read-only verification during the experiment against the previously frozen download records'})
    if not rows or not all(r['matches_ledger'] for r in rows):raise RuntimeError('Model provenance mismatch')
    print('Verified',len(rows),'generator/detector/recognizer files against the prior ledger')

if __name__=='__main__':main()
