"""Validate the official archive and extract mapping files only; no test-image decoding."""
import hashlib,json,time,zipfile
from pathlib import Path
BASE=Path('C:/Users/rehaa/.cache/facial-inpainting/refldm_official_benchmark_v1')
asset=json.loads((BASE/'release_asset.json').read_text(encoding='utf-8-sig'))
p=BASE/'CelebA-Test-Ref.zip.partial';final=BASE/'CelebA-Test-Ref.zip'
status=BASE/'preparation_status.json'
def save(value):status.write_text(json.dumps(value,indent=2))
try:
 save({'status':'waiting_for_download','expected_bytes':asset['size']})
 deadline=time.monotonic()+7200
 while not final.exists() and (not p.exists() or p.stat().st_size!=asset['size']):
  if time.monotonic()>deadline:raise TimeoutError('Archive download incomplete after two hours')
  time.sleep(10)
 if final.exists():p=final
 save({'status':'validating_archive'})
 digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
 expected=asset.get('digest')
 if expected:assert expected=='sha256:'+digest,'Release digest mismatch'
 records=[];metadata=[]
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None,'Archive CRC failure'
  root=(BASE/'metadata').resolve()
  for name in z.namelist():
   records.append(name)
   if name.lower().endswith(('.csv','.txt','.json')):
    target=(root/name).resolve()
    if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name));metadata.append(name)
 if p!=final:p.rename(final)
 (BASE/'archive_members.json').write_text(json.dumps(records,indent=2))
 save({'status':'verified_metadata_ready','url':asset['browser_download_url'],'bytes':final.stat().st_size,'sha256':digest,'release_digest_verified':bool(expected),'crc_verified':True,'metadata_files':metadata,'archive_members':len(records),'test_images_decoded':0,'remaining':'Audit identity overlap and author preprocessing before official benchmark inference.'})
except Exception as e:save({'status':'failed','error':repr(e)});raise
