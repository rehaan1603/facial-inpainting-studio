"""Bound the new decoded-loss gradient probe to 600 wall seconds, owned worker only."""
import json
import subprocess
import time
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/masked_sdxl_ssim_loss_probe_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    if OUT.exists():raise ValueError('Refusing overwrite/resume of previous probe')
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    worker=ROOT/'scripts/probe_masked_sdxl_ssim_loss_v1.py'
    protocol=ROOT/'research/MASKED_SSIM_CONTINUATION_PREFLIGHT_V1.md'
    record=dict(status='running',timeout_seconds=600,worker_sha256=sha(worker),protocol_sha256=sha(protocol),
        supervisor_sha256=sha(__file__))
    # Worker reserves its folder after immutable-input checks; supervisor uses a
    # separate fresh control folder so it cannot preempt the worker's no-overwrite rule.
    supervisor=ROOT/'outputs/masked_sdxl_ssim_loss_supervisor_v1'
    if supervisor.exists():raise ValueError('Previous supervisor attempt exists')
    supervisor.mkdir(parents=True,exist_ok=False);out=supervisor/'receipt.json';save(out,record)
    command=[str(cache/'reference_env_v2/Scripts/python.exe'),str(worker),'--probe']
    start=time.monotonic()
    with (supervisor/'worker.log').open('x',encoding='utf8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        record['worker_pid']=process.pid;save(out,record)
        try:
            record.update(status='finished',returncode=process.wait(timeout=600),timeout=False)
        except subprocess.TimeoutExpired:
            result=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],text=True,capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW)
            record.update(status='failed',timeout=True,termination_returncode=result.returncode);save(out,record)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:record['termination_incomplete']=True
            receipt=OUT/'receipt.json';child=json.loads(receipt.read_text()) if receipt.exists() else {}
            child.update(status='failed',timed_out=True,error='Fixed 600-second wall budget exceeded; no complete probe')
            receipt.parent.mkdir(parents=True,exist_ok=True);save(receipt,child)
    record['wall_seconds']=time.monotonic()-start;save(out,record)
    child=json.loads((OUT/'receipt.json').read_text()) if (OUT/'receipt.json').is_file() else {}
    print(json.dumps(record,indent=2),flush=True)
    if record.get('timeout') or record.get('returncode')!=0 or child.get('status')!='passed':raise SystemExit(1)


if __name__=='__main__':main()
