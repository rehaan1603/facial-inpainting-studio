"""Runtime analysis: accepts only observed pixels, mask, and uploaded references."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from src.research_integrity import ROOT,sha
from src.degradation.face_region_masks import region_masks
from .region_mapper import mask_demand
from .quality_estimator import quality
from .pose_estimator import pose_proxy,similarity
from .visibility_estimator import visibility_proxy
from .mask_aware_scorer import score_reference

def load_rgb(path):
    with Image.open(path) as image:return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()

class ReferenceAnalyzer:
    def __init__(self, detector=None):
        self.config=json.loads((ROOT/'configs/mask_aware_selection_v1.json').read_text())
        if detector is None:
            from insightface.app import FaceAnalysis
            import cv2
            cv2.setNumThreads(2)
            cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
            detector=FaceAnalysis(name=str(cache/'reference_models/insightface/models/buffalo_l'),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'])
            detector.prepare(ctx_id=-1,det_size=(640,640))
        self.detector=detector

    def analyze(self, observed, mask, references):
        if not 1<=len(references)<=4:raise ValueError('Supply one to four reference photos')
        rgb=load_rgb(observed)
        with Image.open(mask) as image:binary=np.asarray(image.convert('L'))>=128
        if binary.shape!=rgb.shape[:2] or not binary.any():raise ValueError('needs_mask: paint a nonempty mask matching the input')
        faces=self.detector.get(rgb[:,:,::-1].copy());target=faces[0] if len(faces)==1 else None
        bbox=target.bbox if target is not None else None;kps=target.kps if target is not None else None
        demand,mapping=mask_demand(binary,bbox,kps);target_pose=pose_proxy(kps) if kps is not None else None
        face_union=np.logical_or.reduce(list(region_masks(binary.shape,bbox,kps).values()))
        visible=float((face_union & ~binary).sum()/max(face_union.sum(),1))
        use_target=target is not None and visible>=self.config['target_identity_min_visible_fraction']
        warnings=[]
        if target is None:warnings.append('Damaged input has no unique detectable face; using coarse frame regions and reference consensus.')
        if not use_target:warnings.append('Damaged target embedding disabled; identity-only ranking uses reference consensus.')
        records=[];embeddings={};seen=set()
        for index,path in enumerate(references):
            record={'index':index,'path':str(Path(path).resolve()),'valid':False}
            try:
                digest=sha(path);record['sha256']=digest
                if digest in seen:raise ValueError('duplicate_reference_file')
                ref=load_rgb(path)
                import hashlib
                decoded=hashlib.sha256(str(ref.shape).encode()+ref.tobytes()).hexdigest()
                if decoded in seen:raise ValueError('duplicate_decoded_pixels')
                seen.update([digest,decoded]);detected=self.detector.get(ref[:,:,::-1].copy())
                record['face_count']=len(detected)
                if len(detected)!=1:raise ValueError('reference_must_have_exactly_one_detectable_face')
                face=detected[0];embedding=face.normed_embedding
                if embedding is None or not np.isfinite(embedding).all():raise ValueError('invalid_embedding')
                pose=pose_proxy(face.kps);regions=region_masks(ref.shape[:2],face.bbox,face.kps)
                regional={name:dict(quality(ref,area),visibility_proxy=visibility_proxy(area,pose,name)) for name,area in regions.items()}
                record.update(valid=True,detection_score=float(face.det_score),bbox=face.bbox.tolist(),landmarks=face.kps.tolist(),
                              pose_proxy=pose,regions=regional,global_quality=quality(ref,np.logical_or.reduce(list(regions.values())))['quality'],pose_similarity=similarity(pose,target_pose))
                embeddings[index]=embedding
            except (OSError,ValueError) as error:record['rejection_reason']=str(error)
            records.append(record)
        pairs=[]
        for r in records:
            if not r['valid']:continue
            other=[float(np.dot(embeddings[r['index']],v)) for i,v in embeddings.items() if i!=r['index']]
            consensus=float(np.median(other)) if other else 0.
            r['reference_consensus_cosine']=consensus
            r['identity_compatibility']=float(np.dot(embeddings[r['index']],target.normed_embedding)) if use_target else consensus
            r['identity_source']='damaged_input' if use_target else 'reference_consensus'
            r['mask_aware_score'],r['components']=score_reference(r,demand,self.config['weights'])
            for i,v in embeddings.items():
                if i>r['index']:pairs.append({'a':r['index'],'b':i,'cosine':float(np.dot(embeddings[r['index']],v))})
        if any(p['cosine']<self.config['disagreement_cosine_threshold'] for p in pairs):warnings.append('Reference embeddings disagree. Check that all uploads show the same person; this is not identity verification.')
        if not embeddings:warnings.append('No usable references; reference generation cannot run.')
        return {'version':1,'config_sha256':sha(ROOT/'configs/mask_aware_selection_v1.json'),'observed_sha256':sha(observed),'mask_sha256':sha(mask),
                'input_face_count':len(faces),'visible_coarse_face_fraction':visible,'region_demand':demand,'mapping':mapping,
                'references':records,'pairwise_cosines':pairs,'warnings':warnings,'score_type':'uncalibrated heuristic','true_occlusion_estimated':False}
