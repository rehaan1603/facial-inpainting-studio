"""Matched, fixed-budget exploratory training; never selects checkpoints on holdout."""
import json
import time
import torch
from reference_intervention_core import ROOT, ReferenceBias, install_adapter, load_model, predict, set_cache, sha


def main():
    out = ROOT / 'outputs/reference_intervention_pilot_v1'
    manifest = json.loads((out/'manifest.json').read_text())
    cache_receipt = json.loads((out/'cache_receipt.json').read_text())
    assert cache_receipt['manifest_sha256'] == sha(out/'manifest.json')
    data = []
    for row in cache_receipt['records']:
        if row['role'] != 'train':
            continue
        path = out/'latents'/(row['identity']+'.pt')
        assert sha(path) == row['sha256']
        data.append(torch.load(path, weights_only=True))
    assert len(data) == 16
    folder = out/'training'
    folder.mkdir(exist_ok=False)
    model = load_model().cuda()
    model.first_stage_model.cpu()
    torch.cuda.empty_cache()
    torch.manual_seed(manifest['seed'])
    adapter = ReferenceBias().cuda()
    initial = {k:v.detach().clone() for k,v in adapter.state_dict().items()}
    layers = install_adapter(model, adapter)
    settings = manifest['training']
    receipt = dict(manifest_sha256=sha(out/'manifest.json'), script_sha256=sha(__file__),
                   core_sha256=sha(ROOT/'scripts/reference_intervention_core.py'), arms=[],
                   attention_layers=layers, trainable_parameters=sum(p.numel() for p in adapter.parameters()),
                   objective_scope='latent diffusion epsilon prediction; no image-space or identity loss')
    (folder/'signature.json').write_text(json.dumps(receipt, indent=2))
    for arm in settings['arms']:
        adapter.load_state_dict(initial)
        adapter.enabled = True
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=settings['learning_rate'])
        torch.manual_seed(manifest['seed'])
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        trace = []
        for step in range(settings['steps_per_arm']):
            batch = {k:v.cuda() for k,v in data[step % len(data)].items()}
            refs = list(batch['references'].split(1))
            t = torch.tensor([settings['timesteps'][(step // len(data)) % 4]], device='cuda')
            noise = torch.randn_like(batch['target'])
            noisy = model.q_sample(batch['target'], t, noise)
            optimizer.zero_grad(set_to_none=True)
            set_cache(model, refs)
            if step == 0:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    adapter.enabled = False
                    original = predict(model, noisy, batch['lq'], t)
                    adapter.enabled = True
                    neutral = predict(model, noisy, batch['lq'], t)
                    assert torch.equal(original, neutral), 'Neutral attention adapter changed author calculation'
            with torch.autocast('cuda', dtype=torch.bfloat16):
                clean = predict(model, noisy, batch['lq'], t)
                clean_loss = (clean.float()-noise).square().mean()
            clean_target = clean.detach().float()
            (0.5 * clean_loss).backward()
            del clean
            refs[0] = batch['corrupt_reference']
            set_cache(model, refs)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                corrupt = predict(model, noisy, batch['lq'], t)
                corrupt_loss = (corrupt.float()-noise).square().mean()
                consistency = (corrupt.float()-clean_target).square().mean()
                total = 0.5 * corrupt_loss
                if arm == 'intervention':
                    total = total + settings['intervention_weight'] * consistency
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            assert not any(p.grad is not None for p in model.parameters())
            trace.append(dict(step=step+1, clean_loss=clean_loss.item(), corrupt_loss=corrupt_loss.item(), consistency=consistency.item(), gradient_norm=grad.item()))
            if (step+1) % 8 == 0:
                print(arm, step+1, trace[-1], flush=True)
                (folder/(arm+'_trace.json')).write_text(json.dumps(trace, indent=2))
        torch.save(adapter.state_dict(), folder/(arm+'.pt'))
        receipt['arms'].append(dict(arm=arm, steps=len(trace), elapsed_seconds=time.monotonic()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, checkpoint_sha256=sha(folder/(arm+'.pt')),
            neutral_adapter_exact=True, backbone_frozen=True))
        (folder/'receipt.json').write_text(json.dumps(receipt, indent=2))
    print('Both fixed-budget arms complete', flush=True)


if __name__ == '__main__':
    main()
