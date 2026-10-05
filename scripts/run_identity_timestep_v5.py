import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/identity_timestep_v5'
GPU='C:/Users/rehaa/.cache/facial-inpainting/refldm_env_v1/Scripts/python.exe'
CPU='C:/Users/rehaa/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe'
OUT.mkdir(exist_ok=True)
status={'status':'waiting_for_training','completed':[]}
def save():(OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2))
save()
try:
 deadline=time.monotonic()+1800
 while not (OUT/'timestep_scaled/training.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Training receipt missing after 30 minutes')
  time.sleep(5)
 for env,stage in [(GPU,'evaluate'),(CPU,'score')]:
  status['status']=stage;save()
  with (OUT/(stage+'.log')).open('w') as log:
   subprocess.run([env,str(ROOT/'scripts'/f'{stage}_identity_timestep_v5.py'),'--arm','timestep_scaled'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  status['completed'].append(stage);save()
 status['status']='complete';save()
except Exception as e:
 status.update(status='failed',error=repr(e));save();raise
