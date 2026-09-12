"""Prepare verified fp16 aliases and fixed prompt with the pinned author's encoder code."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);base=cache/'osor_models/sdxl'
    receipt=json.loads((ROOT/'research/osor_downloads.json').read_text())
    for row in receipt:
        p=Path(row['path']);assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    aliases=[]
    for p in base.rglob('*.fp16.safetensors'):
        alias=p.with_name(p.name.replace('.fp16',''))
        if not alias.exists():os.link(p,alias)
        assert os.path.samefile(p,alias);aliases.append(str(alias))
    repo=cache/'OSOR/osor-sdxlinpainting';prompt=cache/'osor_models/fixed_prompt.pt';config=cache/'osor_models/prompt_config.json'
    config.write_text(json.dumps({'model':{'sdxl_base':str(base),'prompt_embeds_path':str(prompt),'fixed_prompt':'Remove the instance of object'}}))
    subprocess.run([sys.executable,str(repo/'scripts/cache_prompts.py'),'--config',str(config)],check=True,cwd=repo)
    import torch
    state=torch.load(prompt,weights_only=True);assert state['prompt_embeds'].shape==(1,77,2048) and state['pooled_prompt_embeds'].shape==(1,1280)
    (ROOT/'research/osor_runtime.json').write_text(json.dumps({'source_commit':'d5b68e822996d61aae6ba612e5e9b5a14f576879','fp16_aliases':aliases,'prompt':state['prompt'],'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),'prompt_code_sha256':hashlib.sha256((repo/'scripts/cache_prompts.py').read_bytes()).hexdigest(),'note':'Official prompt encoder code; verified fp16 base weight variants exposed under standard filenames via hard links. Generator inference uses bfloat16, matching author inference dtype, after fp16 base loading.'},indent=2))
if __name__=='__main__':main()
