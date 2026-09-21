"""Fixed JPEG stress diagnostic of a previously failed development reference."""
import io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new

def main():
    from src.reference_selection.reference_analyzer import ReferenceAnalyzer
    out=ROOT/'outputs/reference_detection_diagnostic_v1';out.mkdir(exist_ok=False)
    source=ROOT/'outputs/distortion_extension_v1/roles/386/role_1.png'
    previous=ROOT/'outputs/distortion_extension_v1/roles/386/reference_1_degraded.png'
    protocol={'identity':'386','role':'reference_1','qualities':[95,75,50,35,20],'detector_size':[640,640],'threshold':'unchanged default','scope':'Observed development failure diagnosis; no replacement of historical failed rows; no clean target or gallery used','source_sha256':sha(source),'prior_degraded_sha256':sha(previous),'script_sha256':sha(__file__),'final_test_used':False}
    write_new(out/'protocol.json',protocol)
    detector=ReferenceAnalyzer().detector
    with Image.open(source) as im:original=im.convert('RGB').copy()
    variants=[('original',np.asarray(original)),('historical_jpeg20',np.asarray(Image.open(previous).convert('RGB')))]
    for q in protocol['qualities']:
        stream=io.BytesIO();original.save(stream,format='JPEG',quality=q);stream.seek(0)
        with Image.open(stream) as im:variants.append((f'jpeg{q}',np.asarray(im.convert('RGB')).copy()))
    rows=[]
    for label,rgb in variants:
        faces=detector.get(rgb[:,:,::-1].copy())
        rows.append({'variant':label,'face_count':len(faces),'scores':[float(f.det_score) for f in faces],'rgb_sha256':__import__('hashlib').sha256(rgb.tobytes()).hexdigest()})
    result={'protocol':protocol,'rows':rows,'final_test_used':False}
    write_new(out/'results.json',result);write_new(ROOT/'research/reference_detection_diagnostic_v1.json',result)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
