"""Explicit, frozen metric definitions shared by development evaluation suites."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))['cache'])
os.environ['TORCH_HOME'] = str(CACHE/'evaluation_models_v1')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rgb(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError(f'Image hash mismatch: {path}')
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()

class Metrics:
    def __init__(self):
        import pyiqa
        import lpips
        from insightface.app import FaceAnalysis
        from skimage.metrics import structural_similarity
        self.ssim = structural_similarity
        torch.set_num_threads(4)
        torch.manual_seed(20260914)
        for name in ['extended_evaluation_provenance_v1.json','identity_evaluator_provenance_v1.json']:
            record = json.loads((ROOT/'research'/name).read_text(encoding='utf-8'))
            base = CACHE/'identity_eval_v1' if name.startswith('identity') else CACHE
            for relative, expected in record['files'].items():
                if sha(base/relative) != expected:
                    raise ValueError(f'Model/source hash mismatch: {relative}')
        self.niqe = pyiqa.create_metric('niqe', device='cpu')
        self.brisque = pyiqa.create_metric('brisque', device='cpu')
        torch.hub.set_dir(str(CACHE/'torch'))
        self.lpips = lpips.LPIPS(net='alex').cpu().eval()
        self.arc = FaceAnalysis(name=str(CACHE/'reference_models/insightface/models/buffalo_l'),
                                allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])
        self.arc.prepare(ctx_id=-1, det_size=(640,640), det_thresh=.5)
        sys.path.insert(0,str(CACHE/'identity_eval_v1'))
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.mtcnn = MTCNN(image_size=160,margin=0,min_face_size=20,thresholds=[.6,.7,.7],
                           factor=.709,post_process=True,keep_all=True,device='cpu')
        self.facenet = InceptionResnetV1(classify=True,num_classes=8631)
        self.facenet.load_state_dict(torch.load(CACHE/'identity_eval_v1/vggface2.pt',map_location='cpu',weights_only=True))
        self.facenet.classify = False
        self.facenet.eval()

    @staticmethod
    def tensor(rgb):
        return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255

    def features(self, rgb):
        image = Image.fromarray(rgb)
        boxes, probabilities = self.mtcnn.detect(image)
        count = 0 if boxes is None else len(boxes)
        def status(n):
            return {'face_count':n,'status':'ok' if n==1 else 'no_face' if n==0 else 'multiple_faces'}
        mt = status(count)
        vec = None
        if count == 1:
            with torch.inference_mode():
                vec = self.facenet(self.mtcnn.extract(image,boxes,save_path=None))[0].numpy()
            mt.update(box=boxes[0].tolist(),probability=float(probabilities[0]))
        faces = self.arc.get(rgb[:,:,::-1].copy())
        arc = status(len(faces))
        av = None
        if len(faces)==1:
            av = faces[0].normed_embedding
            arc.update(box=faces[0].bbox.tolist(),probability=float(faces[0].det_score))
        return {'facenet':vec,'arcface_conditioning':av}, {'facenet':mt,'arcface_conditioning':arc}

    def quality(self, rgb):
        result = {}
        for name, model in [('niqe',self.niqe),('brisque',self.brisque)]:
            try:
                with torch.inference_mode():
                    value = float(model(self.tensor(rgb)).item())
                if not np.isfinite(value):
                    raise ValueError('Nonfinite metric')
                result[name] = value
            except Exception as error:
                result[name] = None
                result[name+'_failure'] = str(error)
        return result

    def score(self, rgb, target, observed, mask, target_vectors):
        if rgb.shape != target.shape or mask.shape != rgb.shape[:2] or not mask.any() or mask.all():
            raise ValueError('Invalid image/mask geometry')
        if not np.array_equal(rgb[~mask],observed[~mask]):
            raise ValueError('Known pixels changed')
        result = self.quality(rgb)
        vectors, detections = self.features(rgb)
        result['detections'] = detections
        for key, vector in vectors.items():
            target_vector = target_vectors[key]
            value = float(np.dot(vector,target_vector)) if vector is not None and target_vector is not None else None
            if value is not None and not np.isfinite(value):
                raise ValueError('Nonfinite identity similarity')
            result[key+'_cosine'] = value
        a, b = rgb.astype(np.float64)/255, target.astype(np.float64)/255
        error = a-b
        result.update(hole_mae=float(np.abs(error[mask]).mean()),visible_mae=float(np.abs(error[~mask]).mean()),
                      ssim_rgb=float(self.ssim(a,b,data_range=1,channel_axis=2,gaussian_weights=True,
                                              sigma=1.5,use_sample_covariance=False,win_size=11)))
        for name, mse in [('psnr_rgb',float((error**2).mean())),('hole_psnr',float((error[mask]**2).mean()))]:
            result[name] = float(-10*np.log10(mse)) if mse>0 else None
            if mse==0:
                result[name+'_status'] = 'perfect_infinite'
        with torch.inference_mode():
            result['lpips'] = float(self.lpips(self.tensor(rgb)*2-1,self.tensor(target)*2-1).item())
        return result
