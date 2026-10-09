"""Bounded local ResShift LoRA pilot. Never overwrites author weights or prior studies."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/human_resshift_lora_v2'
SOURCE = Path.home()/'Downloads/Human dataset/Human Faces Dataset'
PACK = Path.home()/'Downloads/Human dataset/Ready to test - Human Faces v1'
SEED = 20261002


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare():
    from audit_near_duplicates import fingerprint, candidates
    OUT.mkdir(parents=True, exist_ok=False)
    rows = json.loads((ROOT/'outputs/human_faces_intake_20261001/manifest.json').read_text())
    rows = [r for r in rows if r['group'] == 'Real Images']
    original = {r['file']: r for r in rows}
    records = [fingerprint(dict(file=r['file'], image_path=str(SOURCE/r['file']),
                               source_sha256=r['sha256'])) for r in rows]
    prior = [json.loads(line) for line in (ROOT/'outputs/near_duplicate_audit/fingerprints.jsonl').read_text().splitlines()]
    used = {c['source_file'] for c in json.loads((PACK/'evaluation_manifest.json').read_text())['cases']}
    blocked = {i for i, r in enumerate(records) if r['file'] in used}
    # Saved fingerprints only: never decode prior reserved/evaluation images.
    hashes = [int(r['phash'], 16) for r in records + prior]
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    cross = []
    for a, b, distance in candidates(hashes):
        if a >= len(records):
            continue
        if b >= len(records):
            blocked.add(a)
            cross.append(dict(file=records[a]['file'], prior_index=b-len(records), distance=distance))
        else:
            parent[find(b)] = find(a)
    blocked_groups = {find(i) for i in blocked}
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(find(i), []).append(row)
    eligible = [g for k, g in groups.items() if k not in blocked_groups]
    eligible.sort(key=lambda group: hashlib.sha256(('human_lora_v1:' + min(r['source_sha256'] for r in group)).encode()).hexdigest())
    # Take one representative per near-duplicate component, then disjoint components per split.
    pool = [sorted(g, key=lambda r: r['source_sha256'])[0] for g in eligible]
    assert len(pool) >= 576, f'Insufficient non-overlapping photos: {len(pool)}'
    split = {'holdout': pool[560:576], 'validation': pool[16:48], 'train': pool[48:560]}
    assert not used.intersection(r['file'] for subset in split.values() for r in subset)
    for subset in split.values():
        for r in subset:
            r['intake_rgb_sha256'] = original[r['file']]['rgb_sha256']
    protocol = dict(seed=SEED, scope='Local development pilot; photograph-separated, identity separation/provenance unverified',
        source=str(SOURCE), source_resolution=[178,218], canvas=[256,256],
        interpolation='Aspect-preserving bicubic fit, gray padding; not HQ ground truth',
        excluded_generated=4630, excluded_prior_pack=list(sorted(used)),
        prior_fingerprint_count=len(prior), cross_near_duplicate_count=len(cross),
        excluded_component_count=len(blocked_groups), eligible_components=len(eligible),
        duplicate_rule='64-bit pHash Hamming <=6 connected components; excludes any component touching previous data or four prior test photographs. Not an identity test.',
        split=split, steps=256, accumulation=2, learning_rate=1e-4, rank=4,
        objective='Latent x-start MSE + 10 * LPIPS-Alex against decoded encoded target. Author loss structure with Alex instead of VGG; small LoRA pilot, not full author training.',
        previous_holdout_used_for_design='v1 16 holdout photos are now development. v2 uses next 16 untouched components, same 512 train and 32 validation photos.',
        masks='50% eye boxes with jitter, 25% random boxes, 25% free-form strokes',
        selection='Lowest fixed validation MSE+10*LPIPS objective among steps 0,64,128,192,256; holdout only after selection',
        promotion='Do not promote automatically. Compare fixed seed, raw outputs on 16 held-out photos and the 4 existing diagnostic photos.',
        script_sha256=sha(__file__))
    write(OUT/'protocol.json', protocol)
    write(OUT/'overlap_screen.json', dict(cross_matches=cross))
    print('PREPARED', {k: len(v) for k,v in split.items()}, 'excluded matches', len(cross), flush=True)


class LowRankLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        return y + self.up(self.down(x)) if self.enabled else y


def attach(model, rank):
    modules = []
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.endswith(('.attn.qkv', '.attn.proj')):
            parent_name, child = name.rsplit('.', 1)
            adapter = LowRankLinear(layer, rank)
            setattr(model.get_submodule(parent_name), child, adapter)
            modules.append((name, adapter))
    assert modules, 'No attention projections found'
    return modules


def state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_state(model, weights):
    params = dict(model.named_parameters())
    assert set(weights) == {k for k,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for key, value in weights.items():
            params[key].copy_(value.to(params[key]))


def image(row):
    path = SOURCE/row['file']
    assert sha(path) == row['source_sha256'], 'Source changed'
    with Image.open(path) as im:
        return np.asarray(ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'), (256,256),
                          method=Image.Resampling.BICUBIC, color=(128,128,128))).copy()


def mask(rng, kind=None):
    kind = rng.choice(['eyes','box','strokes'], p=[.5,.25,.25]) if kind is None else kind
    m = Image.new('L', (256,256)); draw = ImageDraw.Draw(m)
    if kind == 'eyes':
        x, y = 82+int(rng.integers(-10,11)), 102+int(rng.integers(-10,11))
        draw.rectangle((x,y,x+int(rng.integers(72,111)),y+int(rng.integers(26,57))),fill=255)
    elif kind == 'box':
        x,y = int(rng.integers(48,150)),int(rng.integers(60,170))
        draw.rectangle((x,y,min(215,x+int(rng.integers(30,85))),min(230,y+int(rng.integers(30,85)))),fill=255)
    else:
        points = [tuple(map(int, rng.integers([55,55],[205,225]))) for _ in range(4)]
        draw.line(points,fill=255,width=int(rng.integers(5,15)),joint='curve')
    return np.asarray(m).copy()/255


def wait_for_studio():
    from urllib.request import urlopen
    from urllib.error import URLError
    while True:
        try:
            with urlopen('http://127.0.0.1:8765/api/session', timeout=3) as response:
                busy = json.load(response)['busy']
        except URLError:
            return
        if not busy:
            return
        print('WAITING: studio inference active', flush=True)
        time.sleep(5)


def train():
    from resshift_adapter import ResShiftFace
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert not (OUT/'training.json').exists(), 'Use a new version rather than overwrite completed training'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    wait_for_studio()
    runtime = ResShiftFace()
    provenance = json.loads((ROOT/'research/resshift_provenance.json').read_text())
    train_images = [image(r) for r in protocol['split']['train']]
    val_images = [image(r) for r in protocol['split']['validation']]
    neutral_input = val_images[0].astype(np.float32)/255
    neutral_mask = mask(np.random.default_rng(SEED))
    baseline = runtime(neutral_input, neutral_mask, SEED)
    modules = attach(runtime.model, protocol['rank'])
    neutral = runtime(neutral_input, neutral_mask, SEED)
    assert np.array_equal(baseline, neutral), 'Zero adapter changed baseline'
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=protocol['learning_rate'], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    import lpips
    cache = Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    torch.hub.set_dir(str(cache/'torch'))
    with torch.random.fork_rng(devices=[0]):
        perceptual = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_images))
    def loss(rgb, missing, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/255
        m = torch.from_numpy(missing.astype(np.float32))[None,None].cuda()
        y = x*(1-m)*2-1
        t = torch.randint(0,4,(1,),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            losses, _, prediction = runtime.diffusion.training_losses(runtime.model,x*2-1,y,t,
                first_stage_model=runtime.vae, model_kwargs={'lq':y,'mask':m*2-1})
            decoded = runtime.diffusion.decode_first_stage(prediction, runtime.vae)
            with torch.no_grad():
                target_latent = runtime.diffusion.encode_first_stage(x*2-1,runtime.vae)
                target_decoded = runtime.diffusion.decode_first_stage(target_latent,runtime.vae)
            perception = perceptual(decoded.float(),target_decoded.float()).mean()
        return losses['mse'].float().mean() + 10*perception.float()
    def validate():
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            values = [float(loss(rgb, mask(np.random.default_rng(SEED+5000+i)),SEED+6000+i))
                      for i,rgb in enumerate(val_images)]
        assert np.isfinite(values).all()
        return float(np.mean(values))
    start = time.time()
    initial = validate()
    best, best_step = initial, 0
    torch.save(state(runtime.model),OUT/'adapter_step_0000.pt')
    history = [dict(step=0,validation_objective=initial)]
    print('START', dict(device=torch.cuda.get_device_name(),trainable=sum(p.numel() for p in parameters),
                        adapters=len(modules),validation_objective=initial),flush=True)
    # eval mode keeps author dropout/state frozen; gradient computation remains enabled.
    with (OUT/'training_steps.jsonl').open('a',encoding='utf-8') as log:
        for step in range(1,protocol['steps']+1):
            wait_for_studio()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(protocol['accumulation']):
                draw = (step-1)*protocol['accumulation']+micro
                rgb = train_images[int(order[draw % len(order)])].copy()
                if rng.random()<.5:
                    rgb = rgb[:,::-1].copy()
                value = loss(rgb,mask(rng))
                assert torch.isfinite(value), 'Nonfinite training loss'
                losses.append(float(value.detach()))
                scaler.scale(value/protocol['accumulation']).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            assert all(p.grad is None for p in runtime.vae.parameters())
            assert all(p.grad is None for p in runtime.model.parameters() if not p.requires_grad)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            assert scaler.get_scale() >= previous_scale, 'Skipped nonfinite update'
            row = dict(step=step,train_objective=float(np.mean(losses)),gradient_norm=float(norm))
            log.write(json.dumps(row)+'\n');log.flush()
            if step%16==0:
                print('STEP',step,'loss',round(row['train_objective'],6),'seconds',round(time.time()-start),flush=True)
            if step%64==0:
                score = validate()
                history.append(dict(step=step,validation_objective=score))
                torch.save(state(runtime.model), OUT/f'adapter_step_{step:04d}.pt')
                if score<best:
                    best,best_step=score,step
                write(OUT/'validation.json',history)
                print('VALIDATION',step,score,'selected',best_step,flush=True)
    weights = torch.load(OUT/f'adapter_step_{best_step:04d}.pt',map_location='cpu',weights_only=True)
    torch.save(weights,OUT/'adapter_selected.pt')
    write(OUT/'training.json',dict(status='complete',device=torch.cuda.get_device_name(),
        steps=protocol['steps'],training_examples_drawn=protocol['steps']*protocol['accumulation'],
        unique_training_photographs=min(len(train_images),protocol['steps']*protocol['accumulation']),
        trainable_parameters=sum(p.numel() for p in parameters),adapter_modules=[name for name,_ in modules],
        zero_adapter_exact=True,frozen_backbone_no_gradients=True,selected_step=best_step,
        validation=history,elapsed_seconds=time.time()-start,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        adapter_sha256=sha(OUT/'adapter_selected.pt'),base_provenance=provenance,
        protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),
        torch_version=str(torch.__version__),promotion='pending held-out comparison; not enabled in studio'))
    print('TRAINING COMPLETE selected',best_step,flush=True)


def generate():
    from resshift_adapter import ResShiftFace
    protocol=json.loads((OUT/'protocol.json').read_text())
    training=json.loads((OUT/'training.json').read_text())
    assert not (OUT/'generation.json').exists()
    assert sha(OUT/'adapter_selected.pt')==training['adapter_sha256']
    torch.set_num_threads(4)
    runtime=ResShiftFace()
    modules=attach(runtime.model,protocol['rank'])
    load_state(runtime.model,torch.load(OUT/'adapter_selected.pt',map_location='cpu',weights_only=True))
    cases=[]
    for i,row in enumerate(protocol['split']['holdout']):
        target=image(row)
        for kind in ['eyes','strokes']:
            m=mask(np.random.default_rng(SEED+10000+i),kind)>0
            observed=target.copy();observed[m]=0
            cases.append((f'holdout_{i+1:02d}_{kind}',observed,m,target,'new_holdout'))
    prior={c['case_id']:c for c in json.loads((PACK/'inference_manifest.json').read_text())['cases']}
    for c in json.loads((PACK/'evaluation_manifest.json').read_text())['cases']:
        if c['case_id'].endswith('central_missing'):
            p=prior[c['case_id']]
            for item in [p['input'],p['mask'],dict(path=c['target'],sha256=c['target_sha256'])]:
                assert sha(item['path'])==item['sha256']
            # Match model-native 256 input; do not use target for inference.
            obs=np.asarray(Image.open(p['input']['path']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).copy()
            m=np.asarray(Image.open(p['mask']['path']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
            obs[m]=0
            target=np.asarray(Image.open(c['target']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).copy()
            cases.append((c['case_id'],obs,m,target,'previous_diagnostic'))
    rows=[]
    for case_id,observed,m,target,group in cases:
        folder=OUT/'evaluation'/case_id;folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(target).save(folder/'target.png')
        Image.fromarray(observed).save(folder/'input.png')
        Image.fromarray((m*255).astype('uint8')).save(folder/'mask.png')
        for arm in ['baseline','adapted']:
            wait_for_studio()
            for _,module in modules:module.enabled=arm=='adapted'
            result=runtime(observed.astype(np.float32)/255,m,seed=SEED+77)
            rgb=np.rint(result*255).clip(0,255).astype('uint8')
            assert np.array_equal(rgb[~m],observed[~m])
            path=folder/f'{arm}.png';Image.fromarray(rgb).save(path)
            rows.append(dict(case_id=case_id,group=group,arm=arm,result=str(path),sha256=sha(path),outside_exact=True))
            write(OUT/'generation.partial.json',dict(rows=rows))
            print('GENERATED',case_id,arm,flush=True)
    write(OUT/'generation.json',dict(rows=rows,seed=SEED+77,selected_step=training['selected_step']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','generate'])
    globals()[parser.parse_args().stage]()
