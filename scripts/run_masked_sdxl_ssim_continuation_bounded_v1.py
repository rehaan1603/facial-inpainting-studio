"""Run the two fixed continuation arms sequentially, each with 1800 wall seconds."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/masked_sdxl_ssim_continuation_v1'
LIMIT=1800
ARMS=['epsilon','ssim']


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,value):
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)


def read_child(path):
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}


def main():
    control=BASE/'supervision'
    if control.exists() or any((BASE/arm/'training').exists() for arm in ARMS):
        raise ValueError('Existing attempt; refusing overwrite/resume of either continuation')
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    worker=ROOT/'scripts/train_masked_sdxl_ssim_continuation_v1.py'
    protocol=ROOT/'research/MASKED_SSIM_CONTINUATION_PREFLIGHT_V1.md'
    record=dict(status='running',scope='Sequential matched epsilon/SSIM continuations; no quality-based arm selection',
        arms_order=ARMS,per_arm_timeout_seconds=LIMIT,worker_sha256=sha(worker),
        protocol_sha256=sha(protocol),supervisor_sha256=sha(__file__),arms=[],unrun_arms=ARMS.copy())
    control.mkdir(parents=True,exist_ok=False);output=control/'receipt.json';save(output,record)
    start=time.monotonic()
    for arm in ARMS:
        item=dict(arm=arm,status='starting',timeout_seconds=LIMIT)
        record['arms'].append(item);save(output,record)
        command=[str(cache/'reference_env_v2/Scripts/python.exe'),str(worker),'--arm',arm,'--train']
        arm_start=time.monotonic();child_path=BASE/arm/'training/receipt.json'
        try:
            with (control/(arm+'.log')).open('x',encoding='utf8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                item.update(status='running',worker_pid=process.pid,command=command)
                record['unrun_arms'].remove(arm);save(output,record)
                try:
                    item.update(returncode=process.wait(timeout=LIMIT),timeout=False)
                except subprocess.TimeoutExpired:
                    item.update(status='failed',timeout=True,error=f'Fixed {LIMIT}-second wall budget exceeded')
                    save(output,record)
                    if process.poll() is None:
                        termination=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                            text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                        item['termination_returncode']=termination.returncode
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:item['termination_incomplete']=True
                    # Preserve the last stage and completed-update count. Never proceed
                    # to another arm or trust a partial checkpoint after this timeout.
                    child=read_child(child_path)
                    child.update(status='failed',timed_out=True,error=item['error'])
                    child_path.parent.mkdir(parents=True,exist_ok=True);save(child_path,child)
                child=read_child(child_path)
                success=not item.get('timeout') and item.get('returncode')==0 and child.get('status')=='complete'
                item.update(status='complete' if success else 'failed',child_status=child.get('status'),
                    child_completed_updates=child.get('completed_updates'),wall_seconds=time.monotonic()-arm_start)
                if child_path.is_file():item['child_receipt_sha256']=sha(child_path)
                save(output,record)
                if not success:
                    record.update(status='failed',stop_reason=f'{arm} did not complete; subsequent arm remains unrun')
                    break
        except BaseException as exc:
            item.update(status='failed',error=f'{type(exc).__name__}: {exc}',wall_seconds=time.monotonic()-arm_start)
            record.update(status='failed',stop_reason=f'{arm} launch/supervision failed; no later arm attempted')
            save(output,record);break
    if all(item['status']=='complete' for item in record['arms']) and len(record['arms'])==2:
        record['status']='complete'
    record['wall_seconds']=time.monotonic()-start;save(output,record)
    print(json.dumps(record,indent=2),flush=True)
    if record['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
