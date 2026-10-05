"""Fixed official-input encoding diagnostic on already-observed identities."""
import gc
import json
import sys
import time
import zipfile
from pathlib import Path

import torch
from reference_intervention_core import ROOT, load_model, sha

OUT = ROOT / 'outputs/refldm_encoding_v2'
CACHE = Path('C:/Users/rehaa/.cache/facial-inpainting/refldm_official_benchmark_v1')
SEEDS = [17, 29, 43]
ARMS = ['joint_rgb', 'separate_latents']
TARGETS = ['11491.png', '19122.png', '24589.png', '27308.png', '27474.png']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    local = json.loads((ROOT / 'configs/local.json').read_text())
    audit = json.loads((CACHE / 'overlap_audit.json').read_text())
    selected = [r for r in audit['records'] if r['flags'] == ['pilot_development_overlap']]
    assert sorted(r['target'] for r in selected) == TARGETS
    assert {r['identity'] for r in selected} == {'263', '6681'}
    cases = []
    with zipfile.ZipFile(CACHE / 'CelebA-Test-Ref.zip') as archive:
        for row in selected:
            assert set(row['reference_identities']) == {row['identity']}
            dest = OUT / Path(row['target']).stem
            dest.mkdir()
            for kind in ['hq', 'lq']:
                member = f"CelebA-Test-Ref/celeba_test_images/{kind}/{row['target']}"
                (dest / f'{kind}.png').write_bytes(archive.read(member))
            references = [Path(local['hq']) / 'CelebA-HQ-img' / f'{int(Path(r).stem)}.jpg'
                          for r in row['references'][:5]]
            cases.append(dict(identity=row['identity'], target=row['target'],
                              references=[dict(path=str(p), sha256=sha(p)) for p in references],
                              hq_sha256=sha(dest/'hq.png'), lq_sha256=sha(dest/'lq.png')))
    config = dict(scope='Observed-development encoding diagnostic; not full paper benchmark',
                  cases=cases, seeds=SEEDS, arms=ARMS, steps=100, guidance=1.5,
                  precision='float32', schedule='uniform_trailing', adapter=None,
                  reference_selection='First up-to-five in official CSV order',
                  output_conversion='torchvision to_pil_image', backbone='Frozen strict-loaded EMA',
                  script_sha256=sha(__file__), loader_sha256=sha(ROOT/'scripts/reference_intervention_core.py'),
                  preflight_sha256=sha(ROOT/'research/REFLDM_ENCODING_SCREEN_V2_PREFLIGHT.md'))
    write(OUT/'config.json', config)
    sys.path.insert(0, str(Path(local['cache'])/'refldm_source_v1'))
    from inference import read_and_normalize_image
    from torchvision.transforms.functional import to_pil_image
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm import cache_kv
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model = load_model().cuda()
    sampler = DDIMSampler(model, print_tqdm=False, schedule='uniform_trailing')
    rows = []
    for case in cases:
        dest = OUT/Path(case['target']).stem
        for seed in SEEDS:
            for arm in ARMS:
                cache_kv.clear_cache()
                cache_kv.mode = 'save'
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(seed)
                noise = torch.randn(1, 8, 64, 64, device='cuda')
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.monotonic()
                row = dict(identity=case['identity'], target=case['target'], seed=seed, arm=arm,
                           references=len(case['references']), status='running')
                try:
                    image = read_and_normalize_image(dest/'lq.png').unsqueeze(0).cuda()
                    refs = [read_and_normalize_image(r['path']) for r in case['references']]
                    if arm == 'joint_rgb':
                        cond = model.get_learned_conditioning(dict(lq_image=image,
                            ref_image=torch.cat(refs, dim=-1).unsqueeze(0).cuda()))
                    else:
                        cond = model.get_learned_conditioning(dict(lq_image=image))
                        cond['ref_image'] = torch.cat([model.get_learned_conditioning(
                            dict(ref_image=r.unsqueeze(0).cuda()))['ref_image'] for r in refs], dim=-1)
                    null = {k: v.detach().clone() for k, v in cond.items()}
                    null['ref_image'] *= 0
                    latent, intermediates = sampler.sample(S=100, unconditional_guidance_scale=1.5,
                        conditioning=cond, unconditional_conditioning=null, shape=[8, 64, 64],
                        x_T=noise, batch_size=1, verbose=False)
                    result = ((model.decode_first_stage(latent)+1)/2).clamp(0,1).squeeze(0).cpu()
                    assert torch.isfinite(result).all()
                    file = dest/f'{arm}_{seed}.png'
                    to_pil_image(result).save(file)
                    torch.cuda.synchronize()
                    row.update(status='complete', file=str(file.relative_to(OUT)), sha256=sha(file),
                               seconds=time.monotonic()-start, peak_mib=torch.cuda.max_memory_allocated()/2**20)
                    if case['target']=='11491.png' and seed==17:
                        prior = ROOT/'outputs'/('refldm_author_smoke_v1' if arm=='joint_rgb'
                                                else 'refldm_separate_encoding_smoke_v1')/'result.png'
                        row['smoke_replay_identical'] = sha(prior)==row['sha256']
                        assert row['smoke_replay_identical'], 'Prior smoke replay mismatch'
                    del image, refs, cond, null, latent, intermediates, result
                except Exception as error:
                    row.update(status='failed', error=repr(error), seconds=time.monotonic()-start)
                    rows.append(row)
                    write(OUT/'generation.json', rows)
                    raise
                rows.append(row)
                write(OUT/'generation.json', rows)
                print('COMPLETED', len(rows), case['target'], seed, arm, flush=True)
        if len(case['references'])==1:
            for seed in SEEDS:
                pair = [r for r in rows if r['target']==case['target'] and r['seed']==seed]
                assert len(pair)==2 and pair[0]['sha256']==pair[1]['sha256'], 'One-reference equivalence failed'
    write(OUT/'completion.json', dict(status='complete', outputs=len(rows), pairs=len(rows)//2,
          config_sha256=sha(OUT/'config.json'), generation_sha256=sha(OUT/'generation.json')))
    print('ALL GENERATIONS COMPLETE', len(rows), flush=True)


if __name__=='__main__':
    main()
