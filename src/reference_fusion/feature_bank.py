import numpy as np
from src.reference_selection.reference_analyzer import ReferenceAnalyzer,load_rgb

def analyze_runtime(image,mask,references,analyzer=None):
    analyzer=analyzer or ReferenceAnalyzer()
    analysis=analyzer.analyze(image,mask,references)
    faces=analyzer.detector.get(load_rgb(image)[:,:,::-1].copy())
    face=faces[0] if len(faces)==1 else None
    geometry={'bbox':face.bbox.tolist() if face is not None else None,
              'landmarks':face.kps.tolist() if face is not None else None,
              'source':'damaged_input_detection' if face is not None else 'coarse_frame_fallback'}
    return analysis,geometry,analyzer
