"""Supervise only the fixed new training worker, with a 30-minute wall limit."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/masked_sdxl_compatibility_pilot_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    output=OUT/'training_supervisor.json'
    if output.exists() or (OUT/'training').exists(): raise ValueError('Existing training attempt; refusing overwrite/resume')
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    worker=ROOT/'scripts/train_masked_sdxl_compatibility_pilot_v1.py'
    protocol=ROOT/'research/MASKED_SDXL_COMPATIBILITY_PILOT_PREFLIGHT_V1.md'
    command=[str(cache/'reference_env_v2/Scripts/python.exe'),str(worker),'--train']
    record=dict(status='running',timeout_seconds=1800,command=command,
        worker_sha256=sha(worker),supervisor_sha256=sha(__file__),protocol_sha256=sha(protocol))
    OUT.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f: json.dump(record,f,indent=2)
    start=time.monotonic()
    with (OUT/'training.log').open('x',encoding='utf8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        record['worker_pid']=process.pid;save(output,record)
        try:
            record.update(returncode=process.wait(timeout=1800),timeout=False,status='finished')
        except subprocess.TimeoutExpired:
            result=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            record.update(status='failed',timeout=True,termination_returncode=result.returncode)
            save(output,record)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                record['termination_incomplete']=True
            receipt=OUT/'training/receipt.json'
            child=json.loads(receipt.read_text()) if receipt.exists() else {}
            child.update(status='failed',timed_out=True,error='Fixed 1800-second wall-clock budget exceeded; no complete candidate')
            receipt.parent.mkdir(parents=True,exist_ok=True);save(receipt,child)
    record['wall_seconds']=time.monotonic()-start;save(output,record)
    print(json.dumps(record,indent=2),flush=True)
    receipt=OUT/'training/receipt.json'
    child=json.loads(receipt.read_text()) if receipt.exists() else {}
    if record.get('timeout') or record.get('returncode')!=0 or child.get('status')!='complete': raise SystemExit(1)


if __name__=='__main__': main()
