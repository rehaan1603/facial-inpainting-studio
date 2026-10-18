"""Export photo-free prerequisite receipts, verifying immutable source pins."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    names=['spatial_qv_matched_path_probe_v3','spatial_qv_optimizer_control_probe_v4',
           'spatial_qv_repeat_isolation_probe_v5','spatial_qv_loss_order_probe_v6',
           'spatial_qv_loss_order_replay_v6']
    rows=[]
    for name in names:
        p=ROOT/'outputs'/name/'receipt.json'
        r=json.loads(p.read_text())
        if r['status'] not in ('passed','failed'): raise ValueError('Unfinished receipt: '+name)
        for source in r.get('source_pins',[]):
            if sha(source['path'])!=source['sha256']: raise ValueError('Changed source: '+source['path'])
        # Omit long gradient vectors, preserving comparison statistics and source hashes.
        compact={k:v for k,v in r.items() if not (isinstance(v,list) and len(v)>100)}
        rows.append(dict(name=name,receipt_sha256=sha(p),receipt=compact))
    out=ROOT/'research/spatial_qv_loss_order_evidence_v6.json'
    if out.exists(): raise ValueError('Refuse evidence overwrite')
    out.write_text(json.dumps(dict(scope='Prerequisite engineering checks only; no new quality scores or trained checkpoint',records=rows),indent=2)+'\n',encoding='utf8')
    print(out)
if __name__=='__main__': main()
