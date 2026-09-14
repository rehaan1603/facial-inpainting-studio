"""Install PyIQA separately and record its runtime and model provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
def main():
    cache = Path(json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))['cache'])
    env = cache/'evaluation_env_v1'
    python = env/'Scripts/python.exe'
    if not python.exists():
        subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
    (env/'Lib/site-packages/shared_runtime.pth').write_text(
        (cache/'reference_env_v2/Lib/site-packages').as_posix()+'\n'+
        (ROOT/'.venv/Lib/site-packages').as_posix()+'\n',encoding='utf-8')
    subprocess.run([str(python),'-m','pip','install','pyiqa==0.1.16','--no-deps'],check=True)
    # TORCH_HOME must precede the import: PyIQA captures its cache path on import.
    code = "import os; os.environ['TORCH_HOME']="+repr(str(cache/'evaluation_models_v1'))+"; import pyiqa; pyiqa.create_metric('niqe',device='cpu'); pyiqa.create_metric('brisque',device='cpu')"
    subprocess.run([str(python),'-X','utf8','-c',code],check=True)
    lock = subprocess.check_output([str(python),'-m','pip','freeze'],text=True,encoding='utf-8')
    lock_path=ROOT/'research/extended-evaluation-environment-v1-lock.txt'
    if lock_path.exists() and lock_path.read_text(encoding='utf-8')!=lock:
        raise ValueError('Evaluation runtime differs from the recorded environment; use a new version')
    lock_path.write_text(lock,encoding='utf-8')
    files = list((env/'Lib/site-packages/pyiqa').rglob('*.py'))
    files += list((cache/'evaluation_models_v1').rglob('*.mat')) + list((cache/'evaluation_models_v1').rglob('*.pth'))
    files += [cache/'reference_models/insightface/models/buffalo_l'/n for n in ['det_10g.onnx','w600k_r50.onnx']]
    record = {'pyiqa_version':'0.1.16','source':'https://github.com/chaofengc/IQA-PyTorch',
              'weight_source':'https://huggingface.co/chaofengc/IQA-PyTorch-Weights',
              'arcface_role':'Same checkpoint as conditioning, not independent verification',
              'files':{p.relative_to(cache).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    provenance_path=ROOT/'research/extended_evaluation_provenance_v1.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text(encoding='utf-8'))!=record:
        raise ValueError('Evaluation source/model hashes changed; refusing to replace frozen provenance')
    provenance_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
