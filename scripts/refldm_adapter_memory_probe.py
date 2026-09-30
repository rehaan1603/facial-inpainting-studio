"""Synthetic-latent backward feasibility only; never loads dataset images or saves weights."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=int, choices=[1, 4], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an earlier receipt')
    import torch
    cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
    source = cache / 'refldm_source_v1'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != 'af6690c19fdc6421802fd7996510bcfef259bfd1':
        raise ValueError('Unexpected author source revision')
    sys.path.insert(0, str(source))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm import cache_kv
    torch.manual_seed(20260930)
    receipt = dict(scope='Synthetic latent single backward step; not quality evidence or full training feasibility',
                   references=args.references, image_equivalent=512, latent_size=64,
                   source_commit=revision, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                   exclusions=['VQ encoder/decoder', 'image losses', 'paired intervention forward', 'dataset images'],
                   precision='float32 parameters, bfloat16 autocast', checkpointing=False,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    start = time.monotonic()
    try:
        weight_path = cache / 'refldm_weights_v1/refldm.ckpt'
        expected = next(f['sha256'] for f in json.loads((ROOT / 'research/refldm_downloads_v1.json').read_text())['files'] if f['name'] == 'refldm.ckpt')
        with weight_path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError('Checkpoint checksum mismatch')
        receipt['checkpoint_sha256'] = digest
        config = OmegaConf.load(source / 'configs/refldm.yaml')
        model = instantiate_from_config(config.model.params.unet_config)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
        prefix = 'model.diffusion_model.'
        model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        del state
        model.requires_grad_(False).eval().cuda()
        # Neutral value-cache adapter, used only to measure a real gradient path.
        adapter = torch.nn.Sequential(torch.nn.Conv1d(32, 8, 1), torch.nn.SiLU(), torch.nn.Conv1d(8, 1, 1)).cuda()
        torch.nn.init.zeros_(adapter[-1].weight)
        torch.nn.init.zeros_(adapter[-1].bias)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
        x = torch.randn(1, 16, 64, 64, device='cuda')
        t = torch.tensor([500], device='cuda')
        torch.cuda.reset_peak_memory_stats()
        cache_kv.clear_cache()
        cache_kv.mode = 'save'
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(args.references):
                model(torch.randn_like(x), torch.zeros_like(t), is_ref=True)
        cache_kv.mode = 'use'
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            baseline = model(x, t).detach()
        before = [p.detach().clone() for p in adapter.parameters()]
        step_start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for key, values in cache_kv.v.items():
                cache_kv.v[key] = [v.detach() * (2 * adapter(v.detach()).sigmoid()) for v in values]
            prediction = model(x, t)
            neutral_error = (prediction.detach().float() - baseline.float()).abs().max().item()
            if neutral_error != 0:
                raise RuntimeError(f'Neutral adapter changed the baseline: {neutral_error}')
            loss = (prediction.float() - torch.randn_like(prediction).float()).square().mean()
        loss.backward()
        gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite adapter gradients')
        grad_norm = sum(g.float().square().sum() for g in gradients).sqrt().item()
        optimizer.step()
        torch.cuda.synchronize()
        changed = any(not torch.equal(old, new) for old, new in zip(before, adapter.parameters()))
        if not changed or grad_norm == 0:
            raise RuntimeError('Adapter did not update')
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen backbone unexpectedly has gradients')
        receipt.update(status='passed', loss=loss.item(), gradient_norm=grad_norm,
                       neutral_adapter_max_error=neutral_error, frozen_backbone_has_gradients=False,
                       parameter_update=changed, trainable_parameters=sum(p.numel() for p in adapter.parameters()),
                       backward_step_seconds=time.monotonic() - step_start)
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - start,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        cache_kv.clear_cache()
        cache_kv.mode = None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2), flush=True)
    if receipt['status'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
