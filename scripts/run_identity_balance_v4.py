"""Finish the declared v4 screen sequentially on GPU; preserve logs and failures."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/identity_balance_v4'
GPU=Path('C:/Users/rehaa/.cache/facial-inpainting/refldm_env_v1/Scripts/python.exe')
CPU=Path('C:/Users/rehaa/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe')
def run(env,script,arm):
 log=OUT/f'{script}_{arm}.log'
 with log.open('w') as handle:
  subprocess.run([str(env),str(ROOT/'scripts'/f'{script}.py'),'--arm',arm],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
def main():
 status={'status':'waiting_for_fixed_training','completed':[]}
 def save(): (OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2))
 save()
 try:
  deadline=time.monotonic()+1800
  while not (OUT/'balanced_fixed/training.json').exists():
   if time.monotonic()>deadline: raise TimeoutError('Fixed training did not finish within 30 minutes; inspect its log.')
   time.sleep(5)
  tasks=[(GPU,'train_identity_balance_v4','balanced_capped')]
  for arm in ['balanced_fixed','balanced_capped']:
   tasks.extend([(GPU,'evaluate_identity_balance_v4',arm),(CPU,'score_identity_balance_v4',arm)])
  for env,script,arm in tasks:
   status['status']=f'{script}/{arm}';save();run(env,script,arm)
   status['completed'].append(status['status']);save()
  status['status']='complete';save()
 except Exception as error:
  status.update(status='failed',error=repr(error));save();raise
if __name__=='__main__':main()
