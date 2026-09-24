"""Run an allowlisted studio model entry point with bounded OpenCV threading.

The research generator sources and model settings remain unchanged. Runtime
stability mitigation for intermittent native colour-conversion exceptions.
"""
import runpy
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    allowed={root/name for name in ['reference_inpaint.py','mask_aware_inpaint.py','studio_reference_inpaint.py','refldm_restore.py','studio_refldm_restore.py']}
    if len(sys.argv)<2:raise ValueError('A studio model entry point is required.')
    target=Path(sys.argv[1]).resolve()
    if target not in allowed:raise ValueError('Unsupported studio model entry point.')
    if target.name != 'refldm_restore.py':
        import cv2
        set_threads = cv2.setNumThreads
        # ReferenceAnalyzer requests two threads after this entry point. Keep the
        # mitigation effective for nested imports without editing research code.
        def bounded_threads(_requested):
            set_threads(1)
        cv2.setNumThreads = bounded_threads
        cv2.setNumThreads(1)
        print(f'Studio OpenCV {cv2.__version__}: {cv2.getNumThreads()} worker thread',flush=True)
    else:
        # The frozen ReFLDM environment does not use or install OpenCV. Reference
        # checks already ran in its parent, in the dedicated reference runtime.
        print('Studio ReFLDM worker: CPU reference checks completed in parent',flush=True)
    sys.argv=[str(target),*sys.argv[2:]]
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
